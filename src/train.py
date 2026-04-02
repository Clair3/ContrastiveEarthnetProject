import os
import torch
import wandb
import yaml
import argparse
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

torch.set_float32_matmul_precision("medium")

from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from data import ContrastiveDataModule, ForecastingDataModule
from modules import ContrastiveTrainingModule, ForecastingTrainModule
from models import TimeSeriesTransformerEncoder, ModelClass


PROJECT_DIR = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BaseExperiment:
    def __init__(self, config, data_config, output_dir):
        self.config = config
        self.data_config = data_config
        self.output_dir = output_dir

    def build_datamodule(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def build_callbacks(self, run_id):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(CHECKPOINT_DIR, run_id),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            save_weights_only=False,  # ensures full reproducibility
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        return [checkpoint, early_stop, lr_monitor]

    def run(self, profile_resources=False):
        datamodule = self.build_datamodule()
        model = self.build_model()

        if profile_resources:
            print("Running dry-run on single batch to estimate resources...")
            self.estimate_batch_resources(model, datamodule)
            return  # skip full training

        self.logger = WandbLogger(
            project="contrastive-earthnet", config=self.config, log_model="best"
        )

        trainer = Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            logger=self.logger,
            callbacks=self.build_callbacks(wandb.run.id),
            log_every_n_steps=32,
            gradient_clip_val=self.config.get("gradient_clip_val", 1.0),
            deterministic=True,
            profiler=None,  # "simple",
            enable_progress_bar=True,
        )

        trainer.fit(model, datamodule)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.test(model=None, datamodule=datamodule, ckpt_path=ckpt_path)

    @staticmethod
    def estimate_batch_resources(model, datamodule, device="cuda"):
        """
        Run a single batch through the model to check memory and speed.
        """
        model = model.to(device)
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        torch.cuda.reset_peak_memory_stats(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(batch) if callable(model) else model.forward(batch)
        if hasattr(output, "sum"):  # make a dummy loss
            loss = output.sum()
            loss.backward()
        end.record()
        torch.cuda.synchronize()

        mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Single-batch GPU memory used: {mem_peak:.1f} MB")
        print(f"Time per batch: {start.elapsed_time(end)/1000:.3f} s")


class ContrastiveExperiment(BaseExperiment):

    def build_datamodule(self):
        return ContrastiveDataModule(
            data_config=self.data_config,
            batch_size=self.config.batch_size,
            num_workers=self.config["num_workers"],
        )

    def build_model(self):
        encoder_veg = TimeSeriesTransformerEncoder(
            input_dim=len(self.data_config["vegetation"]["variables"]),
            sequence_length=self.data_config["vegetation"]["sequence_length"],
            d_model=self.config.d_model,
        )

        encoder_weather = TimeSeriesTransformerEncoder(
            input_dim=len(self.data_config["weather"]["variables"]),
            sequence_length=self.data_config["weather"]["sequence_length"],
            d_model=self.config.d_model,
        )

        self.encoder_veg = encoder_veg
        self.encoder_weather = encoder_weather

        return ContrastiveTrainingModule(
            encoder_veg=encoder_veg,
            encoder_weather=encoder_weather,
            lr=float(self.config.lr),
            temperature=self.config.temperature,
            output_dir=PREDICTIONS_DIR,
        )


class ForecastingExperiment(BaseExperiment):

    def build_datamodule(self):
        return ForecastingDataModule(
            data_config=self.data_config,
            batch_size=self.config.batch_size,
            num_workers=self.config["num_workers"],
        )

    def build_model(self):

        self.model = ModelClass[self.config.model_name](
            data_config=self.data_config,
            config=self.config,
        )

        return ForecastingTrainModule(
            model=self.model,
            lr=float(self.config.lr),
            output_dir=self.output_dir,
        )


@contextmanager
def wandb_run(config, group=None, name=None, project="contrastive-earthnet"):
    wandb.init(project=project, group=group, name=name, config=config)
    try:
        yield wandb.config
    finally:
        wandb.finish()


def run_experiment(config, data_config, output_dir, profile_resources=False):
    task = config["task"]

    if task == "contrastive":
        experiment = ContrastiveExperiment(config, data_config, output_dir)
    elif task == "forecasting":
        experiment = ForecastingExperiment(config, data_config, output_dir)
    else:
        raise ValueError(f"Unknown experiment task: {task}")

    experiment.run(profile_resources=profile_resources)


def get_folds(mode, data_config, predefined_folds=None):
    if mode == "single" or mode == "tune":
        return [(data_config["train"], data_config["test"])]

    elif mode == "kfold":
        if predefined_folds is None:
            raise ValueError("K-fold mode requires predefined folds")
        return predefined_folds

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_pipeline(
    train_config_file="models/mlp.yaml",
    data_config_file="data_config.yaml",
    mode="single",  # "tune", "single", "kfold"
    folds=None,
    profile_resources=False,
):
    """
    Run training and evaluation experiments under different execution modes.

    This function is the main entry point for launching experiments. It supports:
    - single-run training
    - hyperparameter tuning on a single split
    - k-fold cross-validation with fixed hyperparameters

    Parameters
    ----------
    train_config_file : str
        Path (relative to CONFIG_DIR) to the training configuration file
        (e.g., "models/mlp.yaml").
        Must include a "task" field (e.g., "contrastive" or "forecasting")
        to select the experiment type. Defines model architecture,
        training hyperparameters, and related settings; may also specify
        a sweep configuration for tuning.

    data_config_file : str
        Path (relative to CONFIG_DIR) to the data configuration file
        (e.g., "data_config.yaml").


    mode : {"single", "tune", "kfold"}, default="single"
        Execution mode controlling how data splits are used:

        - "single":
            Runs a standard training/validation/test split using the dataset
            configuration as defined in `data_config.yaml`.

        - "tune":
            Runs training on a single fold (same as "single") but stores outputs
            under a dedicated "tuning" directory. Intended for hyperparameter
            search or debugging.

        - "kfold":
            Runs multiple experiments using predefined folds. Each fold defines
            its own train/validation/test years. Typically used after tuning to
            evaluate a fixed configuration across multiple splits.

    folds : list of tuple(list[int], list[int]), optional
        Required when `mode="kfold"`. Each element is a tuple:
            (train_years, test_years)

        Example:
            [
                ([2017, 2018], [2019]),
                ([2017, 2018, 2019], [2020]),
            ]

        Ignored for "single" and "tune" modes.

    profile_resources : bool, default=False
        If True, runs the experiment for a single batch to estimate
        computational resource usage (e.g., memory, runtime).

    Behavior
    --------
    - Loads training and data configurations.
    - Sets a global random seed for reproducibility.
    - Iterates over the selected folds depending on the mode.
    - For each fold:
        - Updates the data configuration (train/val/test splits).
        - Creates an output directory.
        - Starts a Weights & Biases (wandb) run.
        - Executes the experiment.

    Output Structure
    ----------------
    Outputs are stored under:

        PREDICTIONS_DIR / <model_name> /

    With subdirectories depending on the mode:

        - "single":   single/
        - "tune":     tuning/
        - "kfold":    fold_<test_years>/

    Notes
    -----
    - In "tune" mode, only a single fold is used, but results are stored
      separately to avoid overwriting final experiments.
    - In "kfold" mode, the same training configuration is reused across all folds.
    - The function assumes that the configuration files are valid and contain
      all required fields (e.g., "task", "model_name", etc.).
    """
    data_config = load_config(CONFIG_DIR / data_config_file)
    train_config = load_config(CONFIG_DIR / train_config_file)

    if mode == "kfold" and "sweep" in train_config_file.lower():
        raise ValueError(
            f"K-fold evaluation cannot be run with sweep config '{train_config_file}'. "
            "Use a fixed hyperparameter config instead."
        )

    seed_everything(42, workers=True)

    base_output_dir = PREDICTIONS_DIR / train_config["model_name"]

    fold_list = get_folds(mode, data_config, folds)

    for train_years, test_years in fold_list:
        fold_name = f"{test_years[-1]}" if mode != "single" else "single"
        output_dir = base_output_dir / ("tuning" if mode == "tune" else fold_name)
        fold_config = deepcopy(train_config)

        data_config["train"] = train_years
        data_config["validation"] = test_years
        data_config["test"] = test_years

        with wandb_run(
            fold_config,
            group=fold_config["model_name"],
            name=fold_name,
        ) as config:
            run_experiment(
                config,
                data_config,
                output_dir=output_dir,
                profile_resources=profile_resources,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="sweep/mlp.yaml",
        help="Path to training config file (relative to project/configs/)",
    )
    parser.add_argument(
        "--data_config",
        default="data_config.yaml",
        help="Path to data config file (relative to project/configs/)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "tune", "kfold"],
        default="single",
        help="Execution mode: 'single' for standard train/val/test split, 'tune' for hyperparameter tuning on a single fold, 'kfold' for k-fold cross-validation with predefined folds",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run a single batch for resource estimation",
    )  # "store_true" means this flag is False by default and becomes True if --profile isincluded in the command line

    args = parser.parse_args()

    folds = [
        ([2017, 2018], [2018, 2019]),
        ([2017, 2018, 2019], [2019, 2020]),
        ([2017, 2018, 2019, 2020], [2020, 2021]),
        ([2017, 2018, 2019, 2020, 2021], [2021, 2022]),
    ]

    run_pipeline(
        train_config_file=args.config,
        data_config_file=args.data_config,
        mode=args.mode,
        folds=folds if args.mode == "kfold" else None,
        profile_resources=args.profile,
    )
