import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data import ContrastiveDataModule
from models import TimeSeriesTransformerEncoder
from train import ContrastiveTrainingModule


def main():
    logger = TensorBoardLogger("runs", name="contrastive_experiment")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # -----------------------------
    # DataModule / Dataloader
    # -----------------------------

    datamodule = ContrastiveDataModule(
        dataset_path=dataset_path,
        batch_size=128,
        num_workers=16,
    )
    datamodule.setup()

    # Models
    encoder_vegetation = TimeSeriesTransformerEncoder(
        input_dim=1, sequence_length=23, d_model=128
    ).to(device)
    encoder_weather = TimeSeriesTransformerEncoder(
        input_dim=2, sequence_length=73, d_model=128
    ).to(device)

    contrastive_model = ContrastiveTrainingModule(
        encoder_veg=encoder_vegetation,
        encoder_weather=encoder_weather,
        lr=3e-2,
    )

    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",  # automatically uses available GPU
        devices=1,  # number of GPUs to use
        logger=logger,
        accumulate_grad_batches=1,
        log_every_n_steps=16,
        gradient_clip_val=1.0,
        # profiler="simple",
        auto_lr_find=True,
    )

    trainer.fit(contrastive_model, datamodule=datamodule)

    torch.save(
        {
            "encoder_veg": encoder_vegetation.state_dict(),
            "encoder_weather": encoder_weather.state_dict(),
        },
        "veg_weather_contrastive.pth",
    )


if __name__ == "__main__":
    main()
