import torch
import torch.nn as nn


# ======================================================================
# Vanilla RevIN
# ======================================================================

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    NaN-aware Reversible Instance Normalization.

    Input:
        x: [B, T, C]

    For each sample and feature, statistics are computed only over
    valid (non-NaN) observations.

    Normalization:
        x_norm = (x - mean) / std

    Denormalization:
        x = x_norm * std + mean

    Notes
    -----
    NaNs are ignored when computing mean/std and are replaced with
    zero before being passed to the downstream neural network.

    The normalization statistics are computed from the observed
    context and can therefore be used at inference time without
    accessing future targets.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        print("Initializing RevIN with affine:", affine)

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(
                torch.ones(
                    1,
                    1,
                    num_features,
                )
            )

            self.affine_bias = nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    num_features,
                )
            )

        # Runtime state.
        #
        # These are not model parameters.
        self._mean = None
        self._std = None

    def _check_input(self, x):

        if x.ndim != 3:
            raise ValueError(
                f"Expected input with shape [B, T, C], " f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, " f"got {x.shape[-1]}"
            )

    def _compute_statistics(self, x):
        """
        Compute NaN-aware instance statistics.

        Parameters
        ----------
        x : [B, T, C]

        Returns
        -------
        mean : [B, 1, C]
        std  : [B, 1, C]
        """

        valid = torch.isfinite(x)

        # Number of valid observations per sample/feature.
        count = valid.sum(
            dim=1,
            keepdim=True,
        )

        # Avoid division by zero for completely missing features.
        count_safe = count.clamp_min(1)

        # Replace invalid values only for the summation.
        x_safe = torch.where(
            valid,
            x,
            torch.zeros_like(x),
        )

        mean = (
            x_safe.sum(
                dim=1,
                keepdim=True,
            )
            / count_safe
        )

        # For the variance, only valid observations contribute.
        centered = torch.where(
            valid,
            x - mean,
            torch.zeros_like(x),
        )

        variance = (
            centered.pow(2).sum(
                dim=1,
                keepdim=True,
            )
            / count_safe
        )

        std = torch.sqrt(variance + self.eps)

        # If a feature is completely missing, mean=0 and std=1.
        # This means that its normalized value remains zero after
        # NaNs are replaced below.
        no_observation = count == 0

        mean = torch.where(
            no_observation,
            torch.zeros_like(mean),
            mean,
        )

        std = torch.where(
            no_observation,
            torch.ones_like(std),
            std,
        )

        return mean, std

    def normalize(self, x):
        """
        Compute statistics from x and normalize x.

        NaNs are ignored when computing statistics and replaced by
        zero in the returned tensor.

        Returns
        -------
        x_norm : [B, T, C]
        """

        self._check_input(x)

        mean, std = self._compute_statistics(x)

        # Store statistics for denormalization.
        #
        # Detaching is important: normalization statistics represent
        # instance-level state and should not create an unnecessary
        # computational path through the denormalization operation.
        self._mean = mean.detach()
        self._std = std.detach()

        valid = torch.isfinite(x)

        x_safe = torch.where(
            valid,
            x,
            mean.expand_as(x),
        )

        x_norm = (x_safe - mean) / std

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        # Missing values are represented as zero after normalization.
        #
        # This is preferable to using the raw missing value because
        # the downstream encoder can safely process the tensor.
        x_norm = torch.where(
            valid,
            x_norm,
            torch.zeros_like(x_norm),
        )

        return x_norm

    def normalize_with_statistics(self, x):
        """
        Normalize x using statistics previously computed by
        normalize().

        This is used for future weather:

            weather_history
                -> normalize()
                -> stores mean/std

            weather_forecast
                -> normalize_with_statistics()

        Thus future weather does not determine the normalization
        statistics.
        """

        self._check_input(x)

        if self._mean is None or self._std is None:
            raise RuntimeError(
                "No normalization statistics available. " "Call normalize() first."
            )

        valid = torch.isfinite(x)

        x_safe = torch.where(
            valid,
            x,
            self._mean.expand_as(x),
        )

        x_norm = (x_safe - self._mean) / self._std

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        x_norm = torch.where(
            valid,
            x_norm,
            torch.zeros_like(x_norm),
        )

        return x_norm

    def denormalize(self, x):
        """
        Reverse normalization using statistics computed from the
        corresponding observed history.
        """

        if self._mean is None or self._std is None:
            raise RuntimeError(
                "No normalization statistics available. "
                "Call normalize() before denormalize()."
            )

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)

        return x * self._std + self._mean

    def forward(self, x, mode="norm"):

        if mode == "norm":
            return self.normalize(x)

        if mode == "norm_with_statistics":
            return self.normalize_with_statistics(x)

        if mode == "denorm":
            return self.denormalize(x)

        raise ValueError(f"Unknown RevIN mode: {mode}")


class RevINOld(nn.Module):
    """
    Reversible Instance Normalization.

    Normalization is performed independently for each sample and
    variable over the temporal dimension.

    Input:
        x: [B, T, C]

    The statistics are computed from the current instance and are
    therefore available at inference time without using training-set
    statistics.

    Parameters
    ----------
    num_features:
        Number of input variables.

    eps:
        Numerical stability constant.

    affine:
        Whether to use learnable affine parameters.

    subtract_last:
        If True, subtract the last observation instead of the mean.
        Standard RevIN uses the mean, so this should normally remain False.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        print(
            "Initializing RevIN with affine:", affine, "subtract_last:", subtract_last
        )

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))

            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        # Statistics are stored temporarily between normalize()
        # and denormalize(). They are buffers rather than parameters.
        self._mean = None
        self._std = None

    def _check_input(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(x.shape)}")

        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, " f"got {x.shape[-1]}"
            )

    def _get_statistics(self, x):
        """
        Compute instance statistics.

        Returns
        -------
        mean : [B, 1, C]
        std  : [B, 1, C]
        """

        if self.subtract_last:
            mean = x[:, -1:, :]
        else:
            mean = x.mean(
                dim=1,
                keepdim=True,
            )

        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)

        std = torch.sqrt(variance + self.eps)

        return mean, std

    def normalize(self, x):
        """
        Normalize an input instance.

        Returns
        -------
        x_norm : [B, T, C]
        """

        self._check_input(x)

        # Missing values are handled before statistics are computed.
        x = torch.nan_to_num(
            x,
            nan=0.0,
        )

        mean, std = self._get_statistics(x)

        self._mean = mean.detach()
        self._std = std.detach()

        x_norm = (x - mean) / std

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def denormalize(self, x):
        """
        Reverse the normalization using statistics stored during
        normalize().
        """

        if self._mean is None or self._std is None:
            raise RuntimeError("denormalize() called before normalize().")

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)

        return x * self._std + self._mean

    def forward(self, x, mode="norm"):
        if mode == "norm":
            return self.normalize(x)

        if mode == "denorm":
            return self.denormalize(x)

        raise ValueError(f"Unknown RevIN mode: {mode}")


class ConditionalRevIN(nn.Module):
    """
    Cross-variable conditional reversible normalization.

    Unlike vanilla RevIN, the affine normalization parameters for
    each variable are conditioned on the joint instance-level
    distribution of all variables.

    Given

        s = [mean_1, ..., mean_C,
             log_std_1, ..., log_std_C]

    an MLP predicts variable-specific affine parameters:

        gamma = f(s)
        beta  = g(s)

    so that the normalization of one variable can depend on the
    distributional state of the other variables.

    Input:
        [B, T, C]

    Output:
        [B, T, C]
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        state_dim = 2 * num_features

        self.state_encoder = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        self.gamma_head = nn.Linear(
            hidden_dim,
            num_features,
        )

        self.beta_head = nn.Linear(
            hidden_dim,
            num_features,
        )

        self._mean = None
        self._std = None
        self._gamma = None
        self._beta = None

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(self.gamma_head.weight)

        nn.init.zeros_(self.gamma_head.bias)

        nn.init.xavier_uniform_(self.beta_head.weight)

        nn.init.zeros_(self.beta_head.bias)

    def _statistics(self, x):

        mean = x.mean(
            dim=1,
            keepdim=True,
        )

        variance = (
            (x - mean)
            .pow(2)
            .mean(
                dim=1,
                keepdim=True,
            )
        )

        std = torch.sqrt(variance + self.eps)

        return mean, std

    def normalize(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {x.shape}")

        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, " f"got {x.shape[-1]}"
            )

        x = torch.nan_to_num(
            x,
            nan=0.0,
        )

        mean, std = self._statistics(x)

        # Distributional state.
        #
        # log(std) is preferable to std directly because it
        # represents multiplicative scale changes more naturally.
        state = torch.cat(
            [
                mean.squeeze(1),
                torch.log(std.squeeze(1)),
            ],
            dim=-1,
        )

        state = self.state_encoder(state)

        # Keep gamma close to one initially.
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma_head(state))

        beta = self.beta_head(state)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        x_norm = (x - mean) / std

        x_norm = x_norm * gamma + beta

        self._mean = mean.detach()
        self._std = std.detach()
        self._gamma = gamma
        self._beta = beta

        return x_norm

    def denormalize(self, x):

        if self._mean is None:
            raise RuntimeError("denormalize() called before normalize().")

        x = (x - self._beta) / (self._gamma + self.eps)

        return x * self._std + self._mean

    def forward(self, x, mode="norm"):

        if mode == "norm":
            return self.normalize(x)

        if mode == "denorm":
            return self.denormalize(x)

        raise ValueError(f"Unknown mode: {mode}")
