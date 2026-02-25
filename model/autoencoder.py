"""
model/autoencoder.py — PyTorch denoising autoencoder with configurable
hidden layers, activation, and a 2-D bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "ELU": nn.ELU,
}


class Autoencoder(nn.Module):
    """Symmetric fully-connected autoencoder with a 2-D bottleneck.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_layers : list[int]
        Widths of hidden layers *before* the bottleneck.
        E.g. ``[64, 64, 64]`` → encoder 64→64→64→2, decoder 2→64→64→64→d.
    activation : str
        Name of the activation function (key in ``_ACTIVATIONS``).
    bottleneck_dim : int
        Size of the bottleneck layer (default 2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        activation: str = "ReLU",
        bottleneck_dim: int = 2,
    ):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)

        # ── Encoder ─────────────────────────────────────────────────
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(act_cls())
            prev = h
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        # No activation after bottleneck (linear projection)
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder (mirror) ────────────────────────────────────────
        dec_layers: list[nn.Module] = []
        prev = bottleneck_dim
        for h in reversed(hidden_layers):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(act_cls())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        # No activation on output (reconstruct raw values)
        self.decoder = nn.Sequential(*dec_layers)

    # -----------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottleneck codes for *x*."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, bottleneck_code)``."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
