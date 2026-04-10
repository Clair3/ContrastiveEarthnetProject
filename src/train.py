from datetime import datetime
import os
import sys
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
            devices=self.config.gpu_device,
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
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            use_cls=self.config.use_cls,
            seasonal_positional_encoding=self.config.seasonal_positional_encoding,
        )

        encoder_weather = TimeSeriesTransformerEncoder(
            input_dim=len(self.data_config["weather"]["variables"]),
            sequence_length=self.data_config["weather"]["sequence_length"],
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            use_cls=self.config.use_cls,
            seasonal_positional_encoding=self.config.seasonal_positional_encoding,
        )

        self.encoder_veg = encoder_veg
        self.encoder_weather = encoder_weather

        return ContrastiveTrainingModule(
            encoder_veg=encoder_veg,
            encoder_weather=encoder_weather,
            lr=float(self.config.lr),
            temperature=self.config.temperature,
        )


class ForecastingExperiment(BaseExperiment):

    def build_datamodule(self):
        return ForecastingDataModule(
            data_config=self.data_config,
            batch_size=self.config.batch_size,
            num_workers=self.config["num_workers"],
        )

    def build_model(self):
        encoders = getattr(self, "pretrained_encoders", None)

        self.model = ModelClass[self.config.model_name](
            data_config=self.data_config,
            config=self.config,
            pretrained_encoders=encoders,
        )

        return ForecastingTrainModule(
            model=self.model,
            config=self.config,
            output_dir=self.output_dir,
        )


class PretrainThenForecastExperiment(BaseExperiment):
    """
    Runs contrastive pretraining first, then uses the pretrained encoders
    for forecasting.
    """

    def __init__(self, config, data_config, output_dir):
        super().__init__(config, data_config, output_dir)
        self.contrastive_experiment = ContrastiveExperiment(
            config, data_config, output_dir
        )
        self.forecasting_experiment = ForecastingExperiment(
            config, data_config, output_dir
        )

    def run(self, profile_resources=False):
        # Step 1: Pretrain contrastive model
        print("=== Step 1: Contrastive pretraining ===")
        self.contrastive_experiment.run(profile_resources=profile_resources)

        # Load pretrained encoders
        encoder_veg = self.contrastive_experiment.encoder_veg
        encoder_weather = self.contrastive_experiment.encoder_weather

        # Step 2: Forecasting with pretrained encoders
        print("=== Step 2: Forecasting finetuning ===")
        self.forecasting_experiment.pretrained_encoders = {
            "veg": encoder_veg,
            "weather": encoder_weather,
        }
        self.forecasting_experiment.run(profile_resources=profile_resources)


@contextmanager
def wandb_run(config, group=None, project="contrastive-earthnet"):
    run = wandb.init(project=project, group=group, config=config)
    try:
        yield run.config
    finally:
        run.finish()


def run_experiment(config, data_config, output_dir, profile_resources=False):
    task = config["task"]

    if task == "contrastive":
        experiment = ContrastiveExperiment(config, data_config, output_dir)
    elif task == "forecasting":
        experiment = ForecastingExperiment(config, data_config, output_dir)
    elif task == "pretrain_then_forecast":
        experiment = PretrainThenForecastExperiment(config, data_config, output_dir)
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
    Run training and evaluation experiments in different execution modes.

    This is the main entry point for launching experiments. It supports:
    - single-run training
    - hyperparameter tuning on one split
    - k-fold cross-validation with fixed hyperparameters

    Parameters
    ----------
    train_config_file : str
        Path (relative to CONFIG_DIR) to the training config (e.g., "models/mlp.yaml").
        Must include a "task" field (e.g., "contrastive", "forecasting").
        Defines model architecture, training settings, and optional sweep config.

    data_config_file : str
        Path (relative to CONFIG_DIR) to the data config (e.g., "data_config.yaml").

    mode : {"single", "tune", "kfold"}, default="single"
        Execution mode:

        - "single": standard train/val/test split from data config
        - "tune": same as "single", but outputs stored under "tuning/" (for sweeps/debugging)
        - "kfold": runs across predefined folds with fixed hyperparameters

    folds : list of tuple(list[int], list[int]), optional
        Required for "kfold". Each tuple is (train_years, test_years), e.g.:
            [([2017, 2018], [2019]), ([2017, 2018, 2019], [2020])]
        Ignored otherwise.

    profile_resources : bool, default=False
        If True, runs a single batch to estimate resource usage.

    Behavior
    --------
    - Load configs and set a global seed.
    - Iterate over folds based on mode.
    - For each fold:
        - update data splits
        - create output directory
        - start a wandb run
        - execute training/evaluation

    Outputs
    -------
    Stored under:

        PREDICTIONS_DIR / <model_name> /

    Subdirectories:
        - "single":  single/
        - "tune":    tuning/
        - "kfold":   fold_<test_years>/

    Notes
    -----
    - "tune" uses a single fold but isolates outputs.
    - "kfold" reuses the same config across folds.
    - Assumes valid configs with required fields.
    """

    print(
        f"Loading training config from {train_config_file} and data config from {data_config_file}..."
    )
    data_config = load_config(CONFIG_DIR / data_config_file)
    train_config = load_config(CONFIG_DIR / train_config_file)

    seed_everything(42, workers=True)

    base_output_dir = PREDICTIONS_DIR / train_config["model_name"]
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fold_list = get_folds(mode, data_config, folds)

    for train_years, test_years in fold_list:
        folder = f"{time}/{test_years[-1]}" if mode == "kfold" else f"{mode}"
        output_dir = base_output_dir / folder
        fold_config = deepcopy(train_config)

        data_config["train"] = train_years
        data_config["validation"] = test_years
        data_config["test"] = test_years

        with wandb_run(
            fold_config,
            group=fold_config["model_name"],
        ) as config:
            run_experiment(
                config,
                data_config,
                output_dir=output_dir,
                profile_resources=profile_resources,
            )

    print("All experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_config_file",
        default="defaults/transformer_baseline.yaml",
        help="Path to training config file (relative to project/configs/)",
    )
    parser.add_argument(
        "--data_config_file",
        default="data_config_forecasting.yaml",
        help="Path to data config file (relative to project/configs/)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "tune", "kfold"],
        default="single",
        help="Execution mode: 'single' for standard train/val/test split, 'tune' for hyperparameter tuning on a single fold, 'kfold' for k-fold cross-validation with predefined folds",
    )
    parser.add_argument(
        "--profile_resources",
        action="store_true",
        help="Run a single batch for resource estimation",
    )  # "store_true" means this flag is False by default and becomes True if --profile isincluded in the command line

    args = parser.parse_args()

    if args.mode == "kfold" and "sweep" in args.train_config_file.lower():
        print(
            f"\nERROR: Attempting k-fold evaluation with a sweep config '{args.train_config_file}'!\n Use a fixed hyperparameter config instead to ensure all folds use the same parameters.\n"
        )
        sys.exit(1)

    folds = [
        ([2017, 2018], [2018, 2019]),
        ([2017, 2018, 2019], [2019, 2020]),
        ([2017, 2018, 2019, 2020], [2020, 2021]),
        ([2017, 2018, 2019, 2020, 2021], [2021, 2022]),
    ]
    print(f"args are: {args}")

    run_pipeline(
        train_config_file=args.train_config_file,
        data_config_file=args.data_config_file,
        mode=args.mode,
        folds=folds if args.mode == "kfold" else None,
        profile_resources=args.profile_resources,
    )
