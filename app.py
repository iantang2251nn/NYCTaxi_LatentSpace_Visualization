"""
app.py — Streamlit entry point for the NYC Taxi Latent Space Explorer.
"""

from __future__ import annotations

import sys, pathlib
# Ensure project root is on sys.path so `model.*` / `data.*` / `viz.*` resolve
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from streamlit_folium import st_folium

from data.fetch import fetch_taxi_data, fetch_zone_lookup, fetch_zone_centroids
from data.preprocess import clean, engineer_features, build_ae_input
from model.train import train_autoencoder, apply_pca
from viz.scatter import build_scatter
from viz.map import build_trip_map

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="NYC Taxi Latent Space Explorer", layout="wide")

# ── Color feature options ──────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "pickup_dayofweek", "payment_type_label", "PU_borough", "DO_borough",
    "pickup_hour", "pickup_month", "RateCode_label", "is_weekend",
]
CONTINUOUS_FEATURES = [
    "fare_amount", "trip_distance", "trip_duration_min", "speed_mph",
    "tip_pct", "fare_per_mile", "passenger_count", "extra",
    "tolls_amount", "congestion_surcharge", "airport_fee",
]
COLOR_FEATURES = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES

# =========================================================================
# Sidebar
# =========================================================================
with st.sidebar:
    st.title("🚕 NYC Taxi Explorer")

    # ── Section 1: Data ────────────────────────────────────────────
    st.header("1 · Data")
    data_status = st.empty()

    # ── Section 2: Autoencoder Architecture ────────────────────────
    st.header("2 · Autoencoder Architecture")
    optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"], index=0)
    activation = st.selectbox("Nonlinearity", ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "ELU"], index=0)
    max_epochs = st.slider("Epochs (Max)", 10, 200, 50)
    batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
    hidden_input = st.text_input("Hidden Layers", value="64, 64, 64",
                                  help="Comma-separated list of hidden-layer widths")

    # ── Section 3: Denoising & Regularization ─────────────────────
    st.header("3 · Denoising & Regularization")
    noise_factor = st.slider("Input Noise Factor", 0.0, 0.50, 0.10, 0.01,
                              help="Gaussian noise added to inputs during training for denoising")
    patience = st.slider("Early Stopping Patience", 1, 20, 5,
                          help="Stop training when val-loss hasn't improved for this many epochs")
    use_pca = st.checkbox("Orthogonalize Latent Space (PCA)", value=True)

    # ── Section 4: Visualization Setup ────────────────────────────
    st.header("4 · Visualization Setup")
    color_feature = st.selectbox("Color Mapping Feature", COLOR_FEATURES)

    # Placeholder for multiselect — populated after data loads
    ms_placeholder = st.empty()

    # ── Train button ──────────────────────────────────────────────
    train_btn = st.button("🚀 Train Autoencoder", use_container_width=True)


# =========================================================================
# Data loading (cached)
# =========================================================================
raw = fetch_taxi_data()
zone_lookup = fetch_zone_lookup()
centroids = fetch_zone_centroids()

df_clean = clean(raw)
df_feat = engineer_features(df_clean, zone_lookup)

with st.sidebar:
    data_status.success(f"✅ {len(df_feat):,} rows loaded & processed")

# ── Multiselect (needs data — only for categorical features) ─────────────
is_continuous = color_feature in CONTINUOUS_FEATURES
highlighted = []
if not is_continuous:
    unique_vals = sorted(df_feat[color_feature].astype(str).unique())
    with ms_placeholder:
        highlighted = st.multiselect(
            f"Highlight specific {color_feature} values",
            options=unique_vals,
            default=unique_vals,
        )
else:
    with ms_placeholder:
        st.caption("🎨 Continuous feature — coloured by gradient (Plasma)")

# ── Build AE input ─────────────────────────────────────────────────────────
X_scaled, scaler, metadata = build_ae_input(df_feat)

# =========================================================================
# Training
# =========================================================================

def _parse_hidden(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip().isdigit()]


if train_btn:
    hidden = _parse_hidden(hidden_input)
    if not hidden:
        st.error("Hidden Layers must be a comma-separated list of integers, e.g. '64, 64, 64'")
        st.stop()

    progress_bar = st.progress(0, text="Training autoencoder …")
    loss_text = st.empty()

    def _progress(epoch, total, tl, vl):
        progress_bar.progress(epoch / total,
                              text=f"Epoch {epoch}/{total} — train {tl:.4f}  val {vl:.4f}")

    embeddings, train_losses, val_losses = train_autoencoder(
        X_scaled,
        hidden_layers=hidden,
        activation=activation,
        optimizer_name=optimizer,
        max_epochs=max_epochs,
        batch_size=batch_size,
        noise_factor=noise_factor,
        patience=patience,
        progress_callback=_progress,
    )
    progress_bar.empty()

    if use_pca:
        embeddings = apply_pca(embeddings)

    # Store in session state
    st.session_state["embeddings"] = embeddings
    st.session_state["train_losses"] = train_losses
    st.session_state["val_losses"] = val_losses
    st.session_state["metadata"] = metadata


# =========================================================================
# Main area — Tabs
# =========================================================================

tab_vis, tab_analysis = st.tabs(["📊 Latent Space Visualization", "📈 Data Analysis"])

# ── Tab 1: Latent Space Visualization ─────────────────────────────────────
with tab_vis:
    if "embeddings" not in st.session_state:
        st.info("Configure the autoencoder in the sidebar and click **Train Autoencoder** to begin.")
        st.stop()

    embeddings = st.session_state["embeddings"]
    meta = st.session_state["metadata"]

    st.markdown("## Latent Space Visualization")

    fig = build_scatter(embeddings, meta, color_feature, highlighted, continuous=is_continuous)
    event = st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun",
        selection_mode=["points", "box", "lasso"],
        key="scatter",
    )

    # ── Handle click ──────────────────────────────────────────────
    selected_idx = None
    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        # customdata holds the original DataFrame index as a list
        cd = pt.get("customdata")
        if cd is not None:
            # customdata may be [idx] or just idx
            if isinstance(cd, (list, tuple)) and len(cd) > 0:
                selected_idx = int(cd[0])
            else:
                selected_idx = int(cd)
        elif "point_index" in pt:
            selected_idx = int(pt["point_index"])

    if selected_idx is None:
        st.info("ℹ️  Click on any point in the scatter plot above to view its geographic trip and nearest neighbors.")
    else:
        # ── Nearest neighbours ────────────────────────────────────
        nn = NearestNeighbors(n_neighbors=11, metric="euclidean")
        nn.fit(embeddings)
        dists, idxs = nn.kneighbors(embeddings[selected_idx].reshape(1, -1))
        neighbor_idxs = idxs[0][1:]  # skip self

        sel_trip = meta.iloc[selected_idx]
        nb_trips = meta.iloc[neighbor_idxs]

        col_detail, col_map = st.columns([1, 1])

        # ── Left: detail card + neighbour table ───────────────────
        with col_detail:
            st.markdown("### Selected Point & 10 Nearest Neighbors")
            st.markdown("🔴 **Reference Point**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Fare Amount", f"${sel_trip.get('fare_amount', 0):.2f}")
            m2.metric("Trip Distance", f"{sel_trip.get('trip_distance', 0):.1f} mi")
            m3.metric("Total Amount", f"${sel_trip.get('total_amount', 0):.2f}")

            pu_time = ""
            if hasattr(sel_trip.get("tpep_pickup_datetime", None), "strftime"):
                pu_time = f" at {sel_trip['tpep_pickup_datetime'].strftime('%H:%M')}"
            st.markdown(
                f"**Pickup:** {sel_trip.get('PU_borough', '')} "
                f"({sel_trip.get('PU_zone_name', '')}){pu_time}"
            )
            st.markdown(
                f"**Dropoff:** {sel_trip.get('DO_borough', '')} "
                f"({sel_trip.get('DO_zone_name', '')})"
            )

            st.markdown("---")
            st.markdown("**Nearest Neighbors**")
            nb_display = nb_trips[[
                "fare_amount", "trip_distance", "total_amount",
                "PU_borough", "PU_zone_name", "DO_borough", "DO_zone_name",
            ]].copy()
            nb_display.columns = [
                "Fare", "Distance", "Total",
                "PU Borough", "PU Zone", "DO Borough", "DO Zone",
            ]
            nb_display["Fare"] = nb_display["Fare"].apply(lambda x: f"${x:.2f}")
            nb_display["Distance"] = nb_display["Distance"].apply(lambda x: f"{x:.1f} mi")
            nb_display["Total"] = nb_display["Total"].apply(lambda x: f"${x:.2f}")
            st.dataframe(nb_display.reset_index(drop=True), use_container_width=True, hide_index=True)

        # ── Right: Folium map ─────────────────────────────────────
        with col_map:
            st.markdown("### Geographic Trip Mapping")
            m = build_trip_map(sel_trip, nb_trips, centroids)
            st_folium(m, width=None, height=450)


# ── Tab 2: Data Analysis ──────────────────────────────────────────────────
with tab_analysis:
    if "train_losses" not in st.session_state:
        st.info("Train the autoencoder first to see analysis.")
    else:
        import plotly.express as px

        st.markdown("## Training Loss Curve")
        tl = st.session_state["train_losses"]
        vl = st.session_state["val_losses"]
        loss_df = pd.DataFrame({
            "Epoch": list(range(1, len(tl) + 1)) * 2,
            "Loss": tl + vl,
            "Set": ["Train"] * len(tl) + ["Val"] * len(vl),
        })
        fig_loss = px.line(loss_df, x="Epoch", y="Loss", color="Set",
                           title="Reconstruction MSE Loss", template="plotly_white")
        st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown("## Feature Distributions")
        hist_col = st.selectbox("Feature to plot",
                                ["fare_amount", "trip_distance", "trip_duration_min",
                                 "speed_mph", "tip_pct", "fare_per_mile"],
                                key="hist_feat")
        fig_hist = px.histogram(st.session_state["metadata"], x=hist_col, nbins=50,
                                title=f"Distribution of {hist_col}",
                                template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)
