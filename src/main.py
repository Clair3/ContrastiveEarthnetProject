from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from data import ContrastiveDataModule
from models import TimeSeriesTransformerEncoder
from train import ContrastiveTrainingModule
from loss import info_nce_loss

logger = TensorBoardLogger("tb_logs", name="contrastive_experiment")


device = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------
# DataModule / Dataloader
# -----------------------------

dataset_path = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes/full/"


datamodule = ContrastiveDataModule(
    dataset_path=dataset_path,
    batch_size=1,
    num_workers=4,
    years=range(2017, 2023),
    test_year=2020,  # last year held out
)
datamodule.setup()

# Models
encoder_veg = TimeSeriesTransformerEncoder(
    input_dim=1, sequence_length=23, d_model=128
).to(device)
encoder_weather = TimeSeriesTransformerEncoder(
    input_dim=2, sequence_length=73, d_model=128
).to(device)


contrastive_model = ContrastiveTrainingModule(
    encoder_veg=encoder_veg, encoder_weather=encoder_weather, lr=1e-3
)

trainer = Trainer(
    max_epochs=10,
    accelerator="gpu",  # automatically uses available GPU
    devices=1,  # number of GPUs to use
    logger=logger,
    log_every_n_steps=10,
)

trainer.fit(contrastive_model, datamodule=datamodule)

train_loader = datamodule.train_dataloader()

torch.save(
    {
        "encoder_veg": encoder_veg.state_dict(),
        "encoder_weather": encoder_weather.state_dict(),
    },
    "veg_weather_contrastive.pth",
)
