"""
viz/scatter.py — Build the Plotly interactive scatter plot for the 2-D
latent space, with highlight / gray logic and hover info.
Supports both categorical (discrete palette) and continuous (gradient) colour modes.
"""

from __future__ import annotations

import plotly.graph_objects as go
import numpy as np
import pandas as pd

# ── Colour palette (10 distinct saturated colours) ─────────────────────
_PALETTE = [
    "#EF553B", "#636EFA", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# Continuous features use a purple → orange → yellow gradient
_CONTINUOUS_COLORSCALE = "Plasma"


def build_scatter(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    color_feature: str,
    highlighted_vals: list[str],
    continuous: bool = False,
) -> go.Figure:
    """Return a Plotly ``Figure`` with the 2-D scatter plot.

    If *continuous* is True, colour by a continuous gradient rather than
    discrete category traces.
    """
    if continuous:
        return _build_continuous(embeddings, metadata, color_feature)
    return _build_categorical(embeddings, metadata, color_feature, highlighted_vals)


# ── Categorical mode ──────────────────────────────────────────────────

def _build_categorical(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    color_feature: str,
    highlighted_vals: list[str],
) -> go.Figure:
    fig = go.Figure()
    categories = metadata[color_feature].astype(str)

    # "Other" trace (non-highlighted)
    other_mask = ~categories.isin([str(v) for v in highlighted_vals])
    if other_mask.any():
        fig.add_trace(go.Scatter(
            x=embeddings[other_mask, 0],
            y=embeddings[other_mask, 1],
            mode="markers",
            marker=dict(color="gray", size=5, opacity=0.3),
            name="Other",
            text=_hover_text(metadata.loc[other_mask]),
            hoverinfo="text",
            customdata=metadata.index[other_mask].values,
        ))

    for i, val in enumerate(highlighted_vals):
        mask = categories == str(val)
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1],
            mode="markers",
            marker=dict(color=_PALETTE[i % len(_PALETTE)], size=6, opacity=0.8),
            name=str(val),
            text=_hover_text(metadata.loc[mask]),
            hoverinfo="text",
            customdata=metadata.index[mask].values,
        ))

    fig.update_layout(
        title=dict(text="Autoencoder 2D Bottleneck — Potential Clusters", font_size=16),
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        legend_title=color_feature,
        template="plotly_white",
        height=550,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ── Continuous mode ───────────────────────────────────────────────────

def _build_continuous(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    color_feature: str,
) -> go.Figure:
    vals = pd.to_numeric(metadata[color_feature], errors="coerce").fillna(0).values

    fig = go.Figure(go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode="markers",
        marker=dict(
            color=vals,
            colorscale=_CONTINUOUS_COLORSCALE,
            size=5,
            opacity=0.8,
            colorbar=dict(title=color_feature, thickness=15),
        ),
        text=_hover_text(metadata, extra_col=color_feature, extra_vals=vals),
        hoverinfo="text",
        customdata=metadata.index.values,
    ))

    fig.update_layout(
        title=dict(text="Autoencoder 2D Bottleneck — Potential Clusters", font_size=16),
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        template="plotly_white",
        height=550,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ── Helpers ────────────────────────────────────────────────────────────

def _hover_text(
    df: pd.DataFrame,
    extra_col: str | None = None,
    extra_vals: np.ndarray | None = None,
) -> list[str]:
    """Build per-point hover strings."""
    texts = []
    for i, (_, r) in enumerate(df.iterrows()):
        parts = [
            f"Fare: ${r.get('fare_amount', 0):.2f}",
            f"Distance: {r.get('trip_distance', 0):.1f} mi",
            f"Duration: {r.get('trip_duration_min', 0):.0f} min",
        ]
        if "PU_zone_name" in r.index:
            parts.append(f"Pickup: {r['PU_zone_name']}")
        if extra_col and extra_vals is not None:
            parts.append(f"{extra_col}: {extra_vals[i]:.2f}")
        texts.append("<br>".join(parts))
    return texts
