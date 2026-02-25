"""
model/train.py — Training loop with denoising, early stopping, progress bar,
PCA orthogonalization, and embedding extraction.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from model.autoencoder import Autoencoder

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Optimiser map ──────────────────────────────────────────────────────────
_OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}


# =========================================================================
# Public API
# =========================================================================

def train_autoencoder(
    X: np.ndarray,
    hidden_layers: list[int],
    activation: str = "ReLU",
    optimizer_name: str = "Adam",
    max_epochs: int = 50,
    batch_size: int = 64,
    noise_factor: float = 0.1,
    patience: int = 5,
    progress_callback: Callable[[int, int, float, float], None] | None = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Train a denoising autoencoder and return 2-D embeddings.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Scaled input features.
    progress_callback :
        ``(epoch, max_epochs, train_loss, val_loss) -> None``

    Returns
    -------
    embeddings : ndarray, shape (n, 2)
    train_losses : list[float]
    val_losses : list[float]
    """
    _seed_everything()

    n = len(X)
    split = int(0.8 * n)
    idx = np.random.permutation(n)
    X_train, X_val = X[idx[:split]], X[idx[split:]]

    train_ds = TensorDataset(torch.tensor(X_train))
    val_ds = TensorDataset(torch.tensor(X_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    input_dim = X.shape[1]
    model = Autoencoder(input_dim, hidden_layers, activation)
    opt_cls = _OPTIMIZERS.get(optimizer_name, torch.optim.Adam)
    optimizer = opt_cls(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    wait = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, max_epochs + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_dl:
            noisy = batch + noise_factor * torch.randn_like(batch)
            x_hat, _ = model(noisy)
            loss = criterion(x_hat, batch)  # reconstruct *clean* input
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        train_losses.append(epoch_loss / len(X_train))

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_dl:
                x_hat, _ = model(batch)
                val_loss += criterion(x_hat, batch).item() * len(batch)
        val_losses.append(val_loss / len(X_val))

        if progress_callback:
            progress_callback(epoch, max_epochs, train_losses[-1], val_losses[-1])

        # ── Early stopping ─────────────────────────────────────────
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # ── Extract embeddings for ALL data ────────────────────────────
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(torch.tensor(X)).numpy()

    return embeddings, train_losses, val_losses


def apply_pca(embeddings: np.ndarray) -> np.ndarray:
    """PCA-orthogonalize 2-D bottleneck codes."""
    pca = PCA(n_components=2, random_state=SEED)
    return pca.fit_transform(embeddings)
