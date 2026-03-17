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


SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = SCRIPT_DIR / "configs"


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BaseExperiment:
    def __init__(self, config, data_config):
        self.config = config
        self.data_config = data_config

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

    def run(self):

        datamodule = self.build_datamodule()
        model = self.build_model()
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
            profiler=None,  # "simple",
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

    def run(self):

        super().run()

        save_path = os.path.join("checkpoints", f"{wandb.run.id}_final.pth")

        torch.save(
            {
                "encoder_veg": self.encoder_veg.state_dict(),
                "encoder_weather": self.encoder_weather.state_dict(),
                "config": dict(self.config),
                "run_id": wandb.run.id,
            },
            save_path,
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
        )

    def run(self):

        super().run()

        save_path = os.path.join("checkpoints", f"{wandb.run.id}_final.pth")

        torch.save(
            {
                "model": self.model.state_dict(),
                "config": dict(self.config),
                "run_id": wandb.run.id,
            },
            save_path,
        )


def run():
    data_config = load_config(CONFIG_DIR / "data_config.yaml")
    default_config = load_config(CONFIG_DIR / "train_config.yaml")

    seed_everything(42, workers=True)

    wandb.init(
        project="contrastive-earthnet",
        config=default_config,
    )
    # Overwrite config with sweep values if running in a sweep
    config = wandb.config

    task = default_config["task"]

    if task == "contrastive":
        experiment = ContrastiveExperiment(config, data_config)

    elif task == "forecasting":
        pass
        experiment = ForecastingExperiment(config, data_config)

    else:
        raise ValueError(f"Unknown experiment task: {task}")

    experiment.run()

    wandb.finish()


if __name__ == "__main__":
    run()
