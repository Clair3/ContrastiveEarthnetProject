from pytorch_lightning import Trainer
from data import ContrastiveDataModule
import os
from pathlib import Path
from models import TimeSeriesTransformerEncoder
from loss import contrastive_loss
from torch import nn, optim
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# DataModule / Dataloader
# -----------------------------

dataset_path = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes/full/"

dm = ContrastiveDataModule(
    dataset_path=dataset_path,
    batch_size=1,
    num_workers=1,
    years=range(2017, 2023),
    test_year=2020,  # last year held out
)
dm.setup()

train_loader = dm.train_dataloader()


# -----------------------------
#  Model
# -----------------------------

encoder_veg = TimeSeriesTransformerEncoder(
    input_dim=1, sequence_length=23, d_model=128
).to(device)

encoder_weather = TimeSeriesTransformerEncoder(
    input_dim=2, sequence_length=73, d_model=128
).to(device)

# -----------------------------
#  Optimizer
# -----------------------------
optimizer_veg = optim.Adam(encoder_veg.parameters(), lr=1e-3)
optimizer_weather = optim.Adam(encoder_weather.parameters(), lr=1e-3)

# -----------------------------
# Training loop
# -----------------------------
num_epochs = 10
for epoch in range(num_epochs):
    encoder_veg.train()
    encoder_weather.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        optimizer_veg.zero_grad()
        optimizer_weather.zero_grad()

        vegetation = batch["vegetation"].to(device)
        weather = batch["weather"].to(device)

        # Forward pass
        veg_emb = encoder_veg(vegetation)
        print(veg_emb.shape)
        weather_emb = encoder_weather(vegetation)

        # Compute loss
        loss_v2w = contrastive_loss(veg_emb, weather_emb)
        loss_w2v = contrastive_loss(weather_emb, veg_emb)
        loss = (loss_v2w + loss_w2v) / 2

        # Backprop
        loss.backward()
        optimizer_veg.step()
        optimizer_weather.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} finished | Avg loss: {avg_loss:.4f}")

# -----------------------------
# 5️⃣ (Optional) Save model
# -----------------------------
torch.save(model.state_dict(), "veg_weather_contrastive.pth")
