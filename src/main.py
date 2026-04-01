import os
import torch
import wandb
import yaml
import os
from pathlib import Path
import torch

torch.set_float32_matmul_precision("medium")

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from contextlib import contextmanager
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from data import ContrastiveDataModule, ForecastingDataModule
from train import ContrastiveTrainingModule, ForecastingTrainModule
from models import TimeSeriesTransformerEncoder, ModelClass


PROJECT_DIR = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BaseExperiment:
    def __init__(self, config, data_config):
        self.config = config
        self.data_config = data_config
        self.save_path = os.path.join(
            f"{OUTPUT_DIR}/checkpoints", f"{wandb.run.id}_final.pth"
        )

    def build_datamodule(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def build_callbacks(self, run_id):
        os.makedirs("checkpoints", exist_ok=True)

        checkpoint = ModelCheckpoint(
            dirpath=os.path.join("checkpoints", run_id),
            filename="epoch={epoch:02d}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        return [checkpoint, early_stop, lr_monitor]

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

    def run(self, profile_ressources=False):
        datamodule = self.build_datamodule()
        model = self.build_model()

        if profile_ressources:
            print("Running dry-run on single batch to estimate resources...")
            self.estimate_batch_resources(model, datamodule)
            return  # skip full training

        logger = WandbLogger(
            project="contrastive-earthnet", config=self.config, log_model="best"
        )

        trainer = Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            logger=logger,
            callbacks=self.build_callbacks(wandb.run.id),
            log_every_n_steps=32,
            gradient_clip_val=1.0,
            deterministic=True,
            profiler="simple",
            enable_progress_bar=True,
        )

        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)


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
        )

    def run(self, profile_ressources=False):

        super().run(profile_ressources=profile_ressources)

        torch.save(
            {
                "encoder_veg": self.encoder_veg.state_dict(),
                "encoder_weather": self.encoder_weather.state_dict(),
                "config": dict(self.config),
                "run_id": wandb.run.id,
            },
            self.save_path,
        )


class ForecastingExperiment(BaseExperiment):

    def build_datamodule(self):
        print(self.data_config["train"])

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
        )

    def run(self, profile_ressources=False):

        super().run(profile_ressources=profile_ressources)

        torch.save(
            {
                "model": self.model.state_dict(),
                "config": dict(self.config),
                "run_id": wandb.run.id,
            },
            self.save_path,
        )
        # predictions = self.model.predict(data_config["test"])
        # pred_path = PREDICTIONS_DIR / f"predictions_{fold[1][0]}.pkl"


#
# with open(pred_path, "wb") as f:
#    pickle.dump(predictions, f)


def run_experiment(config, data_config, profile_ressources=False):
    task = config["task"]
    if task == "contrastive":
        experiment = ContrastiveExperiment(config, data_config)
    elif task == "forecasting":
        experiment = ForecastingExperiment(config, data_config)
    else:
        raise ValueError(f"Unknown experiment task: {task}")

    experiment.run(profile_ressources=profile_ressources)


@contextmanager
def wandb_run(train_config, group=None, name=None, project="contrastive-earthnet"):
    wandb.init(project=project, group=group, name=name, config=train_config)
    try:
        yield wandb.config
    finally:
        wandb.finish()


def run_k_fold(train_config_file, folds=None, profile_ressources=False):
    data_config = load_config(CONFIG_DIR / "data_config.yaml")
    train_config = load_config(CONFIG_DIR / train_config_file)
    seed_everything(42, workers=True)

    if folds is None:
        with wandb_run(
            train_config,
            group=f"{train_config['model_name']}",
            name=f"fold_{data_config['test']}",
        ) as config:
            run_experiment(config, data_config, profile_ressources=profile_ressources)

    else:
        for train_years, test_years in folds:
            data_config["train"] = train_years
            data_config["validation"] = test_years
            data_config["test"] = test_years

            with wandb_run(
                train_config,
                group=f"{train_config['model_name']}",
                name=f"fold_{test_years}",
            ) as config:
                run_experiment(
                    config, data_config, profile_ressources=profile_ressources
                )


if __name__ == "__main__":
    profile_ressources = False  # True  # Set to True for quick resource estimation
    train_config_file = "models/mlp.yaml"

    folds = [
        ([2017, 2018], [2019]),
        ([2017, 2018, 2019], [2020]),
        ([2017, 2018, 2019, 2020], [2021]),
        ([2017, 2018, 2019, 2020, 2021], [2022]),
    ]
    run_k_fold(train_config_file=train_config_file, folds=None)
