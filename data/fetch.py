"""
data/fetch.py — Download and cache NYC Yellow Taxi trip data + zone lookup.
"""

import os
import pathlib
import random

import numpy as np
import pandas as pd
import requests
import streamlit as st

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

SOCRATA_URL = "https://data.cityofnewyork.us/resource/4b4i-vvec.csv"
PARQUET_FALLBACK = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
)
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

SAMPLE_SIZE = 10_000
SEED = 42

# Socrata API returns all-lowercase column names; the rest of the code
# expects the canonical TLC naming convention.
_COL_RENAME = {
    "vendorid": "VendorID",
    "ratecodeid": "RatecodeID",
    "pulocationid": "PULocationID",
    "dolocationid": "DOLocationID",
    "store_and_fwd_flag": "store_and_fwd_flag",
    "payment_type": "payment_type",
    "fare_amount": "fare_amount",
    "extra": "extra",
    "mta_tax": "mta_tax",
    "tip_amount": "tip_amount",
    "tolls_amount": "tolls_amount",
    "improvement_surcharge": "improvement_surcharge",
    "total_amount": "total_amount",
    "congestion_surcharge": "congestion_surcharge",
    "airport_fee": "airport_fee",
    "tpep_pickup_datetime": "tpep_pickup_datetime",
    "tpep_dropoff_datetime": "tpep_dropoff_datetime",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from all-lowercase (Socrata) to canonical TLC names."""
    rename = {k: v for k, v in _COL_RENAME.items() if k in df.columns}
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Taxi trip data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fetching taxi trip data …")
def fetch_taxi_data(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Return *n* randomly-sampled rows of 2023 Yellow Taxi trip data.

    Strategy:
      1. Try the Socrata API with ``$limit`` (up to 50 000 rows), then sample.
      2. If that fails, fall back to the TLC CloudFront Parquet file.
      3. Cache the result to ``data/raw_sample.parquet``.
    """
    cache_path = DATA_DIR / "raw_sample.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = _try_socrata(n) if _try_socrata else None
    if df is None or df.empty:
        df = _try_parquet_fallback(n)

    df = _normalize_columns(df)
    df.to_parquet(cache_path, index=False)
    return df


def _try_socrata(n: int) -> pd.DataFrame | None:
    """Attempt to pull rows from the Socrata CSV endpoint."""
    fetch_limit = max(n * 4, 20_000)  # grab extra so sample is robust
    params = {"$limit": fetch_limit, "$order": ":id"}
    try:
        resp = requests.get(SOCRATA_URL, params=params, timeout=60)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        if len(df) >= n:
            return df.sample(n=n, random_state=SEED).reset_index(drop=True)
        return df
    except Exception:
        return None


def _try_parquet_fallback(n: int) -> pd.DataFrame:
    """Download a monthly Parquet file and sample from it."""
    local = DATA_DIR / "yellow_tripdata_2023-01.parquet"
    if not local.exists():
        resp = requests.get(PARQUET_FALLBACK, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    df = pd.read_parquet(local)
    return df.sample(n=n, random_state=SEED).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Zone lookup
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading zone lookup …")
def fetch_zone_lookup() -> pd.DataFrame:
    """Return the TLC Taxi Zone Lookup table."""
    cache_path = DATA_DIR / "taxi_zone_lookup.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(ZONE_LOOKUP_URL, timeout=30)
    resp.raise_for_status()
    with open(cache_path, "w") as fh:
        fh.write(resp.text)
    return pd.read_csv(cache_path)


# ---------------------------------------------------------------------------
# Zone centroids (bundled CSV)
# ---------------------------------------------------------------------------

@st.cache_data
def fetch_zone_centroids() -> pd.DataFrame:
    """Load the bundled zone-centroid CSV."""
    path = DATA_DIR / "zone_centroids.csv"
    return pd.read_csv(path)
