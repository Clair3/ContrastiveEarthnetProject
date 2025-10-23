from pytorch_lightning import Trainer
from data import ContrastiveDataModule
import os
from pathlib import Path

dataset_path = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes/full/"

dm = ContrastiveDataModule(
    dataset_path=dataset_path,
    batch_size=1,
    num_workers=1,
    years=range(2017, 2023),
    test_year=2020,  # last year held out
)

dm.setup()

for batch in dm.train_dataloader():
    positive, negatives = batch["positive"], batch["negatives"]
    veg, weather = positive

    print("Number of negatives:", len(negatives))
    break

# trainer = Trainer(max_epochs=100)
# trainer.fit(model, datamodule=dm)
