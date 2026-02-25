# NYC Taxi Autoencoder Latent Space Explorer — Walkthrough

## What Was Built

A Streamlit application that fetches 2023 NYC Yellow Taxi trip data, trains a configurable denoising autoencoder to produce 2D bottleneck embeddings, and provides an interactive visualization with click-to-inspect trip details and geographic mapping.

## Files Created

| File | Purpose |
|---|---|
| [requirements.txt](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/requirements.txt) | Dependencies (streamlit, torch, scikit-learn, plotly, folium, etc.) |
| [data/fetch.py](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/data/fetch.py) | Socrata API fetch + Parquet fallback + zone lookup + caching |
| [data/preprocess.py](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/data/preprocess.py) | Cleaning, feature engineering, AE input matrix |
| [data/zone_centroids.csv](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/data/zone_centroids.csv) | Bundled lat/lon centroids for 265 taxi zones |
| [model/autoencoder.py](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/model/autoencoder.py) | PyTorch autoencoder (configurable layers, activation, bottleneck=2) |
| [model/train.py](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/model/train.py) | Training loop with denoising, early stopping, PCA |
| [viz/scatter.py](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/viz/scatter.py) | Plotly scatter with highlight/gray logic |
| [viz/map.py](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/viz/map.py) | Folium map (CartoDB dark_matter tiles) |
| [app.py](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/app.py) | Streamlit entry point with sidebar + 2-tab layout |

## Bugs Fixed During Development

1. **Column name mismatch** — Socrata API returns all-lowercase columns (`pulocationid`) but codebase expected PascalCase (`PULocationID`). Added [_normalize_columns()](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/data/fetch.py#50-54) in [fetch.py](file:///Users/dk2251n/Desktop/Spring2026/nycTaxi/data/fetch.py).
2. **Click-to-inspect not triggering** — Plotly's `on_select` needed explicit `selection_mode=["points", "box", "lasso"]` and improved `customdata` parsing.
3. **Duplicate title** — Removed redundant `st.markdown` subtitle since the Plotly chart already has the title.
4. **Variable scope bug** — Data Analysis tab referenced undefined `metadata` variable; fixed to use `st.session_state["metadata"]`.

## Verification Results

### ✅ Data Loading
- Successfully fetches 10000 rows from Socrata API, cleans to ~9620 rows
- Zone lookup and centroids load correctly

### ✅ Training
- Autoencoder trains in < 15 seconds with default settings
- Progress bar updates in real-time

### ✅ Latent Space Scatter Plot
- Clear cluster structure visible in 2D embeddings
- Color mapping and highlight/gray logic work correctly

### ✅ Click-to-Inspect
- Box/lasso selection triggers detail card with fare, distance, total, pickup/dropoff info
- 10 nearest neighbors displayed in table
- Geographic map renders with trip polylines and colored markers

### ✅ Data Analysis Tab
- Training loss curve shows train/val MSE decreasing
- Feature distribution histograms render correctly

## Screenshots

````carousel
![Scatter plot with 2D bottleneck clusters and sidebar controls](/Users/dk2251n/.gemini/antigravity/brain/793518f4-e52f-420b-84aa-3ddcf3a228c2/.system_generated/click_feedback/click_feedback_1772037694845.png)
<!-- slide -->
![Click-to-inspect — detail card, neighbor table, and Folium map](/Users/dk2251n/.gemini/antigravity/brain/793518f4-e52f-420b-84aa-3ddcf3a228c2/final_app_state_1772038019315.png)
````

## How to Run

```bash
cd /Users/dk2251n/Desktop/Spring2026/nycTaxi
source venv/bin/activate
streamlit run app.py
```
