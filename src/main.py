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

from data import ContrastiveDataModule
from models import TimeSeriesTransformerEncoder
from train import ContrastiveTrainingModule

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = SCRIPT_DIR / "configs"


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(config, data_config):
    """Instantiate encoders and the Lightning module from a W&B config object."""
    encoder_veg = TimeSeriesTransformerEncoder(
        input_dim=len(data_config["vegetation"]["variables"]),
        sequence_length=data_config["vegetation"]["sequence_length"],
        d_model=config.d_model,
    )
    encoder_weather = TimeSeriesTransformerEncoder(
        input_dim=len(data_config["weather"]["variables"]),
        sequence_length=data_config["weather"]["sequence_length"],
        d_model=config.d_model,
    )
    model = ContrastiveTrainingModule(
        encoder_veg=encoder_veg,
        encoder_weather=encoder_weather,
        lr=float(config.lr),
        temperature=config.temperature,
    )
    return model, encoder_veg, encoder_weather


def build_callbacks(run_id: str):
    """Standard set of training callbacks."""
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
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint, early_stop, lr_monitor]


def run():
    """Main training function."""

    # Load configs
    data_config = load_config(CONFIG_DIR / "data_config.yaml")
    train_config = load_config(CONFIG_DIR / "train_config.yaml")
    default_config = {
        **train_config["model"],
        **train_config["training"],
    }

    seed_everything(train_config["system"]["seed"], workers=True)

    # Initialize W&B (automatically handles sweep params)
    wandb.init(project="contrastive-earthnet", config=default_config)
    config = wandb.config

    print(f"\nTraining with:")
    print(f"  lr: {config.lr}")
    print(f"  d_model: {config.d_model}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  temperature: {config.temperature}")
    print(f"  Sweep mode: {os.environ.get('WANDB_SWEEP_ID') is not None}")

    # ── Logger ────────────────────────────────
    logger = WandbLogger(log_model="best")

    # ── Data ──────────────────────────────────
    datamodule = ContrastiveDataModule(
        data_config=data_config,
        batch_size=config.batch_size,
        num_workers=train_config["system"]["num_workers"],
    )

    # ── Model ─────────────────────────────────
    model, encoder_veg, encoder_weather = build_model(config, data_config)

    # ── Trainer ───────────────────────────────
    # Effective batch size is kept roughly constant across batch_size values
    # by adjusting gradient accumulation steps.
    # TARGET_EFFECTIVE_BATCH = 256
    # accumulate = max(1, TARGET_EFFECTIVE_BATCH // config.batch_size)

    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=train_config["system"]["precision"],
        logger=logger,
        callbacks=build_callbacks(wandb.run.id),
        log_every_n_steps=16,
        gradient_clip_val=1.0,
        # accumulate_grad_batches=accumulate,
        deterministic=True,
        profiler="simple",
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)

    # ── Save weights ──────────────────────────
    save_path = os.path.join("checkpoints", f"{wandb.run.id}_final.pth")
    torch.save(
        {
            "encoder_veg": encoder_veg.state_dict(),
            "encoder_weather": encoder_weather.state_dict(),
            "config": dict(config),
            "run_id": wandb.run.id,
        },
        save_path,
    )
    print(f"Weights saved → {save_path}")

    wandb.finish()


if __name__ == "__main__":
    run()
