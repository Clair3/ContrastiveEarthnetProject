from collections.abc import Mapping

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from .contrastive import ContrastiveModule
from .forecasting import ForecastingModule


class ProbingModule(LightningModule):
    def __init__(
        self, model, base_module, config, probe_tasks=["max_evi", "sum_precip"]
    ):
        super().__init__()
        self.model = model
        self.base_module = base_module
        self.config = config

    def forward(self, embedding):
        out = self.base_module(*args, **kwargs)
        return self.model(out)

    def training_step(self, batch, batch_idx):
        return self.base_module.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.base_module.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.base_module.test_step(batch, batch_idx)


class ProbingMixin:
    def setup_probes(self, probing_config, feature_dim):
        self.probe_weight = probing_config.get("weight", 1.0)
        self.detach_probe_features = probing_config.get("detach", True)

        hidden_dim = probing_config.get("hidden_dim", 128)

        self.probes = nn.ModuleDict(
            {
                target: ProbeHead(feature_dim, hidden_dim)
                for target in self.probe_targets
            }
        )

        self.probe_loss_fn = nn.MSELoss()


class ProbingModule(LightningModule):
    """
    Wrap a main training module with optional probing heads.

    The wrapped module still defines the primary task. Probe heads are trained
    from features extracted during the same step and can either be diagnostic
    only, or contribute a weighted auxiliary loss.
    """

    def __init__(self, base_module, config):
        super().__init__()
        self.base_module = base_module
        self.config = config

        probing_config = _config_get(config, "probing", {})
        self.targets = _config_get(probing_config, "targets", ["max_evi", "sum_precip"])
        self.feature_source = _config_get(probing_config, "feature_source", "auto")
        self.detach_features = _config_get(probing_config, "detach", True)
        self.probe_weight = float(_config_get(probing_config, "weight", 1.0))
        hidden_dim = int(_config_get(probing_config, "hidden_dim", 128))
        dropout = float(_config_get(probing_config, "dropout", 0.0))

        self.probes = nn.ModuleDict(
            {
                target: ProbeHead(hidden_dim=hidden_dim, dropout=dropout)
                for target in self.targets
            }
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        if batch is None:
            self.log(
                "train_loss", float("nan"), on_step=True, on_epoch=True, prog_bar=True
            )
            return None

        base_loss, features = self._base_step(batch, batch_idx, stage="train")
        probe_loss = self._probe_step(batch, features, stage="train")
        return base_loss + self.probe_weight * probe_loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        base_loss, features = self._base_step(batch, batch_idx, stage="val")
        self._probe_step(batch, features, stage="val")
        return base_loss

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None

        base_loss, features = self._base_step(batch, batch_idx, stage="test")
        self._probe_step(batch, features, stage="test")
        return base_loss

    def _base_step(self, batch, batch_idx, stage):
        if isinstance(self.base_module, ContrastiveModule):
            return self._contrastive_base_step(batch, stage)

        if isinstance(self.base_module, ForecastingModule):
            return self._forecasting_base_step(batch, stage)

        raise TypeError(
            f"ProbingModule does not know how to wrap {type(self.base_module).__name__}."
        )

    def _contrastive_base_step(self, batch, stage):
        vegetation = batch["vegetation"]
        weather = batch["weather"]
        veg_emb, weather_emb = self.base_module(vegetation, weather)
        loss = self.base_module.loss_fn(veg_emb, weather_emb)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=vegetation.shape[0],
        )

        if stage == "train":
            self.log("veg_emb_std", veg_emb.std(), on_step=True)
            self.log("weather_emb_std", weather_emb.std(), on_step=True)
            self.log("veg_emb_mean", veg_emb.mean().abs(), on_step=True)

        features = self._select_contrastive_features(veg_emb, weather_emb)
        return loss, features

    def _forecasting_base_step(self, batch, stage):
        y_pred = self.base_module(batch)
        y_true = batch["vegetation_forecast"]
        mask = ~torch.isnan(y_true)
        loss = self.base_module.loss_fn(y_pred[mask], y_true[mask])

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true.shape[0],
        )

        features = self._select_forecasting_features(y_pred)
        return loss, features

    def _select_contrastive_features(self, veg_emb, weather_emb):
        if self.feature_source == "vegetation":
            return veg_emb
        if self.feature_source == "weather":
            return weather_emb
        return torch.cat([veg_emb, weather_emb], dim=-1)

    def _select_forecasting_features(self, y_pred):
        if y_pred.ndim == 3:
            return y_pred.flatten(start_dim=1)
        return y_pred

    def _probe_step(self, batch, features, stage):
        if self.detach_features:
            features = features.detach()

        losses = []
        for target_name, probe in self.probes.items():
            if target_name not in batch:
                continue

            y_true = batch[target_name].float()
            y_pred = probe(features)
            mask = ~torch.isnan(y_true)

            if not mask.any():
                continue

            loss = self.loss_fn(y_pred[mask], y_true[mask])
            losses.append(loss)

            prefix = f"{stage}_probe/{target_name}"
            self.log(
                f"{prefix}_loss",
                loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                batch_size=y_true.shape[0],
            )
            self.log(
                f"{prefix}_mae",
                torch.mean(torch.abs(y_pred[mask] - y_true[mask])),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=y_true.shape[0],
            )

        if not losses:
            return features.new_tensor(0.0)

        probe_loss = torch.stack(losses).mean()
        self.log(
            f"{stage}_probe_loss",
            probe_loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
        )
        return probe_loss

    def configure_optimizers(self):
        lr = float(_config_get(self.config, "lr"))
        weight_decay = float(_config_get(self.config, "weight_decay", 0.0))
        warmup_fraction = float(_config_get(self.config, "warmup_fraction", 0.0))

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        if warmup_fraction <= 0:
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(warmup_fraction * total_steps)

        if warmup_steps == 0:
            return optimizer

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=lr * 0.05,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
