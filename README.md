# NYC Taxi Autoencoder Latent Space Explorer

An interactive Streamlit app that trains a denoising autoencoder on 2023 NYC Yellow Taxi trip data and visualizes the 2D latent space with click-to-inspect trip details and geographic mapping.

## Features

- **Configurable autoencoder** — PyTorch denoising AE with adjustable hidden layers, activation, optimizer, noise factor, and early stopping
- **PCA orthogonalization** — optional decorrelation of the 2D bottleneck codes
- **Interactive scatter plot** — colour by categorical features (day of week, borough, payment type) or continuous features (fare, distance, speed) with a Plasma gradient
- **Click-to-inspect** — select points to see trip details, 10 nearest neighbors, and a Folium geographic map
- **Data Analysis tab** — training loss curves and feature distribution histograms

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data

- **Source:** [2023 Yellow Taxi Trip Data](https://data.cityofnewyork.us/Transportation/2023-Yellow-Taxi-Trip-Data/4b4i-vvec) (NYC Open Data, Socrata API)
- **Sample:** 10,000 rows, randomly sampled and cached locally
- **Fallback:** TLC CloudFront Parquet mirror if the API is unavailable

## Project Structure

```
app.py                  # Streamlit entry point
data/
  fetch.py              # API fetch + Parquet fallback + caching
  preprocess.py         # Cleaning, feature engineering, scaling
  zone_centroids.csv    # Taxi zone lat/lon centroids
model/
  autoencoder.py        # PyTorch autoencoder (symmetric, configurable)
  train.py              # Training loop, early stopping, PCA
viz/
  scatter.py            # Plotly scatter (categorical + continuous)
  map.py                # Folium map (CartoDB dark_matter)
```
