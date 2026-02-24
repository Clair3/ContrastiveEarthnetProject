import os
import torch
import wandb
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

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DATASET_PATH = (
    "/Net/Groups/BGI/scratch/crobin/PythonProjects/"
    "ContrastiveEarthnetProject/preprocessing/datasets/"
)
CHECKPOINT_DIR = "checkpoints"
SEED = 42


def build_model(config):
    """Instantiate encoders and the Lightning module from a W&B config object."""
    encoder_veg = TimeSeriesTransformerEncoder(
        input_dim=1,
        sequence_length=23,
        d_model=config.d_model,
    )
    encoder_weather = TimeSeriesTransformerEncoder(
        input_dim=2,
        sequence_length=73,
        d_model=config.d_model,
    )
    model = ContrastiveTrainingModule(
        encoder_veg=encoder_veg,
        encoder_weather=encoder_weather,
        lr=config.lr,
        temperature=config.temperature,
    )
    return model, encoder_veg, encoder_weather


def build_callbacks(run_id: str):
    """Standard set of training callbacks."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, run_id),
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
    seed_everything(SEED, workers=True)

    # W&B initialises the run and injects hyperparameters from the sweep
    wandb.init()
    config = wandb.config

    # ── Logger ────────────────────────────────
    logger = WandbLogger(log_model=False)

    # ── Data ──────────────────────────────────
    datamodule = ContrastiveDataModule(
        dataset_path=DATASET_PATH,
        batch_size=config.batch_size,
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
    )

    # ── Model ─────────────────────────────────
    model, encoder_veg, encoder_weather = build_model(config)

    # ── Trainer ───────────────────────────────
    # Effective batch size is kept roughly constant across batch_size values
    # by adjusting gradient accumulation steps.
    TARGET_EFFECTIVE_BATCH = 256
    accumulate = max(1, TARGET_EFFECTIVE_BATCH // config.batch_size)

    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        precision=16,  # free ~2x memory / speed
        logger=logger,
        callbacks=build_callbacks(wandb.run.id),
        log_every_n_steps=16,
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)

    # ── Save weights ──────────────────────────
    save_path = os.path.join(CHECKPOINT_DIR, f"{wandb.run.id}_final.pth")
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
