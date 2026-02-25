"""
data/preprocess.py — Cleaning, feature engineering, and autoencoder-input
construction for NYC Yellow Taxi data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


# ── Day-of-week labels ────────────────────────────────────────────────────
DOW_MAP = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
           4: "Friday", 5: "Saturday", 6: "Sunday"}

PAYMENT_MAP = {1: "Credit card", 2: "Cash", 3: "No charge", 4: "Dispute"}

RATECODE_MAP = {1: "Standard", 2: "JFK", 3: "Newark",
                4: "Nassau/Westchester", 5: "Negotiated", 6: "Group ride"}


# =========================================================================
# 1. Cleaning
# =========================================================================

@st.cache_data(show_spinner="Cleaning data …")
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules from PLAN §2.1."""
    df = df.copy()

    # --- Parse datetimes ------------------------------------------------
    for col in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Drop rows with critical nulls ----------------------------------
    critical = ["fare_amount", "trip_distance", "PULocationID",
                "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime"]
    df.dropna(subset=critical, inplace=True)

    # --- Fare filter ----------------------------------------------------
    df = df[(df["fare_amount"] >= 2.50) & (df["fare_amount"] <= 500)]

    # --- Distance filter ------------------------------------------------
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 200)]

    # --- Duration -------------------------------------------------------
    df["trip_duration_min"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
        .dt.total_seconds() / 60
    )
    df = df[(df["trip_duration_min"] > 1) & (df["trip_duration_min"] <= 360)]

    # --- Passenger count ------------------------------------------------
    df["passenger_count"] = df["passenger_count"].fillna(1).clip(lower=1, upper=6)
    df["passenger_count"] = df["passenger_count"].replace(0, 1).astype(int)

    # --- Payment type & RatecodeID --------------------------------------
    df = df[df["payment_type"].isin([1, 2, 3, 4])]
    df = df[df["RatecodeID"].isin([1, 2, 3, 4, 5, 6])]

    # --- Date range (2023) ----------------------------------------------
    df = df[
        (df["tpep_pickup_datetime"] >= "2023-01-01") &
        (df["tpep_pickup_datetime"] < "2024-01-01")
    ]

    return df.reset_index(drop=True)


# =========================================================================
# 2. Feature Engineering
# =========================================================================

@st.cache_data(show_spinner="Engineering features …")
def engineer_features(df: pd.DataFrame, zone_lookup: pd.DataFrame) -> pd.DataFrame:
    """Create all derived columns from PLAN §2.2."""
    df = df.copy()

    pickup = df["tpep_pickup_datetime"]
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_dayofweek_num"] = pickup.dt.dayofweek
    df["pickup_dayofweek"] = df["pickup_dayofweek_num"].map(DOW_MAP)
    df["pickup_month"] = pickup.dt.month
    df["is_weekend"] = (df["pickup_dayofweek_num"] >= 5).astype(int)

    # Duration already computed in clean(); recompute defensively
    if "trip_duration_min" not in df.columns:
        df["trip_duration_min"] = (
            (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
            .dt.total_seconds() / 60
        )

    df["speed_mph"] = (
        df["trip_distance"] / (df["trip_duration_min"] / 60)
    ).clip(0, 80)

    df["fare_per_mile"] = (df["fare_amount"] / df["trip_distance"]).clip(0, 100)

    df["tip_pct"] = (
        df["tip_amount"] / df["fare_amount"] * 100
    ).clip(0, 100).fillna(0)

    # --- Zone lookups ---------------------------------------------------
    zone_map = zone_lookup.set_index("LocationID")
    for prefix, loc_col in [("PU", "PULocationID"), ("DO", "DOLocationID")]:
        ids = df[loc_col].astype(int)
        df[f"{prefix}_borough"] = ids.map(zone_map["Borough"]).fillna("Unknown")
        df[f"{prefix}_zone_name"] = ids.map(zone_map["Zone"]).fillna("Unknown")

    # --- Labels ---------------------------------------------------------
    df["payment_type_label"] = df["payment_type"].map(PAYMENT_MAP).fillna("Unknown")
    df["RateCode_label"] = df["RatecodeID"].map(RATECODE_MAP).fillna("Unknown")

    # Fill potential NaN in surcharge columns with 0
    for col in ("extra", "tolls_amount", "congestion_surcharge", "airport_fee"):
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# =========================================================================
# 3. Build Autoencoder Input Matrix
# =========================================================================

@st.cache_data(show_spinner="Preparing autoencoder input …")
def build_ae_input(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Return (X_scaled, scaler, metadata_df).

    X_scaled is a 2-D ``ndarray`` ready for the autoencoder.
    metadata_df is the original + engineered DataFrame kept in lock-step.
    """
    parts: list[pd.DataFrame] = []

    # -- Continuous features ---------------------------------------------
    cont_cols = [
        "fare_amount", "trip_distance", "trip_duration_min", "speed_mph",
        "tip_pct", "fare_per_mile", "passenger_count",
        "extra", "tolls_amount", "congestion_surcharge", "airport_fee",
    ]
    parts.append(df[cont_cols].astype(float))

    # -- Cyclical encoding (hour, dayofweek) -----------------------------
    hour = df["pickup_hour"].astype(float)
    dow = df["pickup_dayofweek_num"].astype(float)
    cyc = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dow / 7),
        "dow_cos":  np.cos(2 * np.pi * dow / 7),
    }, index=df.index)
    parts.append(cyc)

    # -- Borough one-hot (PU + DO) — ~7 each = 14 ----------------------
    pu_boro = pd.get_dummies(df["PU_borough"], prefix="PU_boro").astype(float)
    do_boro = pd.get_dummies(df["DO_borough"], prefix="DO_boro").astype(float)
    parts.append(pu_boro)
    parts.append(do_boro)

    # -- Payment type one-hot (4 cols) -----------------------------------
    pay_oh = pd.get_dummies(df["payment_type"], prefix="pay").astype(float)
    parts.append(pay_oh)

    # -- RatecodeID one-hot (up to 6 cols) -------------------------------
    rate_oh = pd.get_dummies(df["RatecodeID"], prefix="rate").astype(float)
    parts.append(rate_oh)

    # -- is_weekend (1 col) ----------------------------------------------
    parts.append(df[["is_weekend"]].astype(float))

    X = pd.concat(parts, axis=1).fillna(0).values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled, scaler, df.copy()
