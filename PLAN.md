# PLAN.md — NYC Yellow Taxi Autoencoder Latent Space Explorer

## 0. Project Overview

Build a Streamlit application that:

1. Fetches the 2023 NYC Yellow Taxi Trip dataset from NYC Open Data.
    **Description: /Users/dk2251n/Desktop/Spring2026/nycTaxi/2023_yellow_taxi_columns_description.md**
2. Samples 10000 rows randomly.
3. Preprocesses and feature-engineers the data.
4. Trains a configurable denoising autoencoder to produce a 2D bottleneck embedding.
5. Optionally orthogonalizes the latent space via PCA.
6. Renders an interactive scatter plot of the 2D embeddings, color-coded by a user-chosen categorical feature, with click-to-inspect trip details and a geographic map of the selected trip + nearest neighbors.

The UI must match the reference screenshots (Streamlit sidebar controls, main-area scatter plot with legend, selected-point detail card, Folium/Leaflet geographic map).

---

## 1. Data Acquisition

### 1.1 Source

- **Dataset:** `2023 Yellow Taxi Trip Data`
- **NYC Open Data ID:** `4b4i-vvec`
- **API endpoint (SoQL):** `https://data.cityofnewyork.us/resource/4b4i-vvec.csv`
- **Sampling strategy:** Use SoQL query parameter `$limit=5000&$order=:id&$offset=<random>` OR fetch a larger chunk and call `DataFrame.sample(5000, random_state=42)`. The latter is simpler and more reproducible.
- **Fallback:** If the Socrata API is unreachable, download one monthly Parquet file from the TLC CloudFront mirror (`https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet`) and sample from it.

### 1.2 Taxi Zone Lookup

- Download the zone lookup CSV: `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`
- Columns: `LocationID`, `Borough`, `Zone`, `service_zone`
- Needed to resolve `PULocationID` / `DOLocationID` to human-readable names (e.g., "Manhattan (Upper East Side North)") for the detail card and map tooltips.

### 1.3 Caching

- Cache the raw data to disk (`data/raw_sample.parquet`) so re-runs don't re-fetch.
- Use `@st.cache_data` in Streamlit for both fetch and preprocessing steps.

---

## 2. Data Preprocessing

### 2.1 Cleaning

| Step | Rule |
|---|---|
| Drop nulls | Drop rows where any of `fare_amount`, `trip_distance`, `PULocationID`, `DOLocationID`, `tpep_pickup_datetime`, `tpep_dropoff_datetime` are null. |
| Fare filter | Keep `fare_amount` ∈ [2.50, 500]. Remove $0 and negative fares. |
| Distance filter | Keep `trip_distance` ∈ (0, 200]. Remove zero-distance trips. |
| Duration | Compute `trip_duration_min = (dropoff − pickup).total_seconds() / 60`. Keep ∈ (1, 360]. |
| Passenger count | Clip to [1, 6]. Replace 0 with 1. |
| Payment type | Keep only values 1–4 (Credit card, Cash, No charge, Dispute). Drop Unknown/Voided. |
| RatecodeID | Keep only 1–6. |
| Date range | Drop rows with pickup datetime outside 2023-01-01 to 2023-12-31. |

### 2.2 Feature Engineering

Create derived columns needed for the autoencoder input and for visualization color-mapping:

| Feature | Derivation | Type |
|---|---|---|
| `pickup_hour` | `pickup_dt.hour` | int 0–23 |
| `pickup_dayofweek` | `pickup_dt.dayofweek` → mapped to Monday..Sunday string | categorical |
| `pickup_month` | `pickup_dt.month` | int 1–12 |
| `is_weekend` | dayofweek ∈ {5, 6} | bool → int |
| `trip_duration_min` | (dropoff − pickup) / 60 | float |
| `speed_mph` | `trip_distance / (trip_duration_min / 60)`, clipped to [0, 80] | float |
| `fare_per_mile` | `fare_amount / trip_distance`, clipped to [0, 100] | float |
| `tip_pct` | `tip_amount / fare_amount * 100`, clipped to [0, 100] | float |
| `PU_borough` | Looked up from zone table | categorical |
| `DO_borough` | Looked up from zone table | categorical |
| `PU_zone_name` | Looked up from zone table | string (for display) |
| `DO_zone_name` | Looked up from zone table | string (for display) |
| `payment_type_label` | Map 1→Credit card, 2→Cash, 3→No charge, 4→Dispute | categorical |
| `RateCode_label` | Map 1→Standard, 2→JFK, 3→Newark, 4→Nassau/Westchester, 5→Negotiated, 6→Group ride | categorical |

### 2.3 Autoencoder Input Features

Select a numeric feature vector for the autoencoder. All features below are **standardized** (zero mean, unit variance) using `sklearn.preprocessing.StandardScaler` before feeding to the network.

| # | Feature | Rationale |
|---|---|---|
| 1 | `fare_amount` | Core fare signal |
| 2 | `trip_distance` | Spatial extent |
| 3 | `trip_duration_min` | Temporal extent |
| 4 | `speed_mph` | Derived ratio |
| 5 | `tip_pct` | Tipping behavior |
| 6 | `fare_per_mile` | Rate proxy |
| 7 | `passenger_count` | Occupancy |
| 8 | `pickup_hour` | Temporal (cyclical encoding: sin + cos → 2 dims) |
| 9 | `pickup_dayofweek` (numeric 0–6) | Temporal (cyclical encoding: sin + cos → 2 dims) |
| 10 | `PULocationID` | Spatial (entity-embed or keep as-is if one-hot is too wide) |
| 11 | `DOLocationID` | Spatial |
| 12 | `extra` | Surcharge signal |
| 13 | `tolls_amount` | Toll signal |
| 14 | `congestion_surcharge` | Congestion zone indicator |
| 15 | `airport_fee` | Airport indicator |
| 16 | `payment_type` (one-hot, 4 cols) | Payment modality |
| 17 | `RatecodeID` (one-hot, 6 cols) | Rate category |
| 18 | `is_weekend` | Binary |

**Cyclical encoding** for hour and dayofweek:
```
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
dow_sin  = sin(2π * dow / 7)
dow_cos  = cos(2π * dow / 7)
```

**Location IDs:** One-hot encoding of 263 zones is too wide for 5k rows. Instead, use borough-level one-hot (5 boroughs + EWR + Unknown = 7 cols each for PU/DO = 14 cols) or bin LocationIDs into ~20 clusters via frequency. Borough-level one-hot is simpler and recommended.

**Total input dimensionality estimate:** ~7 continuous + 4 cyclical + 14 borough one-hot + 4 payment one-hot + 6 ratecode one-hot + 1 binary ≈ **36 dimensions**.

### 2.4 Preserve Metadata

Keep a parallel DataFrame of all original + derived columns (not standardized) indexed identically to the autoencoder input matrix. This DataFrame is used for:
- Color mapping in the scatter plot
- Detail card rendering
- Map rendering

---

## 3. Autoencoder Architecture

### 3.1 Network Design

A symmetric, fully-connected denoising autoencoder with a **2D bottleneck**.

```
Input (d) → [Hidden₁] → [Hidden₂] → ... → [Bottleneck (2)] → ... → [Hidden₂] → [Hidden₁] → Output (d)
```

**Configurable via sidebar:**

| Parameter | Sidebar Widget | Default | Range/Options |
|---|---|---|---|
| Optimizer | `st.selectbox` | Adam | Adam, SGD, RMSprop |
| Nonlinearity | `st.selectbox` | ReLU | ReLU, LeakyReLU, Tanh, Sigmoid, ELU |
| Epochs (Max) | `st.slider` | 50 | 10–200 |
| Batch Size | `st.selectbox` | 64 | 32, 64, 128, 256 |
| Hidden Layers | `st.text_input` | "64, 64, 64" | Comma-separated list of ints |
| Input Noise Factor | `st.slider` | 0.10 | 0.0–0.5 |
| Early Stopping Patience | `st.slider` | 5 | 1–20 |
| Orthogonalize Latent Space (PCA) | `st.checkbox` | True | — |

### 3.2 Denoising

- At training time, corrupt input `x` to `x̃ = x + noise_factor * N(0, 1)`.
- Reconstruction target is the clean `x`.
- This acts as regularization and forces the bottleneck to capture salient structure.

### 3.3 Training Loop

1. Split data 80/20 train/val.
2. Use MSE reconstruction loss.
3. Train for up to `max_epochs` with early stopping on validation loss (patience = configurable).
4. After training, extract 2D bottleneck activations for **all** 5,000 points.

### 3.4 PCA Orthogonalization

If the "Orthogonalize Latent Space (PCA)" checkbox is enabled:
- Fit `sklearn.decomposition.PCA(n_components=2)` on the raw 2D bottleneck codes.
- Transform all points. This decorrelates and axis-aligns the latent dimensions.
- The scatter plot axes become `Dim 1` (PC1) and `Dim 2` (PC2).

### 3.5 Implementation

Use **PyTorch** (`torch.nn.Module`):

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation, bottleneck_dim=2):
        # Encoder: input_dim → hidden_layers[0] → ... → hidden_layers[-1] → 2
        # Decoder: 2 → hidden_layers[-1] → ... → hidden_layers[0] → input_dim
```

Wrap training in a function that accepts hyperparams from the sidebar and returns the 2D embeddings + trained model.

---

## 4. Streamlit UI Layout

### 4.1 Page Config

```python
st.set_page_config(page_title="NYC Taxi Latent Space Explorer", layout="wide")
```

### 4.2 Sidebar (Left Panel)

Organized into numbered sections matching the reference screenshots:

**Section 1: Data** (not visible in screenshots but implied)
- "Fetch & Sample Data" button or auto-load with spinner.
- Display row count after load.

**Section 2: Autoencoder Architecture**
- Optimizer dropdown: Adam / SGD / RMSprop
- Nonlinearity dropdown: ReLU / LeakyReLU / Tanh / Sigmoid / ELU
- Epochs (Max) slider: 10–200, default 50
- Batch Size dropdown: 32, 64, 128, 256
- Hidden Layers text input: default "64, 64, 64"

**Section 3: Denoising & Regularization**
- Input Noise Factor slider: 0.00–0.50, step 0.01, default 0.10 (with ⓘ tooltip: "Gaussian noise added to inputs during training for denoising")
- Early Stopping Patience slider: 1–20, default 5 (with ⓘ tooltip)
- Orthogonalize Latent Space (PCA) checkbox: default checked

**Section 4: Visualization Setup**
- Color Mapping Feature dropdown: `pickup_dayofweek`, `payment_type`, `PU_borough`, `DO_borough`, `pickup_hour`, `pickup_month`, `RateCode_label`, `is_weekend`
- Highlight specific `{feature}` values: `st.multiselect` that dynamically populates with the unique values of the selected Color Mapping Feature. Selected values are shown as colored pill/tag chips (red background, "×" to remove). Non-highlighted points appear as semi-transparent gray ("Other" in legend).

**"Train Autoencoder" button** at bottom of sidebar (or top). Triggers training with current params.

### 4.3 Main Area — Tab Layout

Two tabs visible in the reference: **"Latent Space Visualization"** | **"Data Analysis"**

#### Tab 1: Latent Space Visualization

**Top section: Scatter plot**
- Title: `"Latent Space Visualization"` (h1) → `"Autoencoder 2D Bottleneck Potential Clusters"` (bold subtitle)
- Plotly interactive scatter (`px.scatter` or `go.Scatter`):
  - x = `Dim 1`, y = `Dim 2`
  - Color = selected Color Mapping Feature
  - Non-highlighted categories → gray with `opacity=0.3`, legend label "Other"
  - Highlighted categories → distinct saturated colors with `opacity=0.8`
  - Legend title: `_highlight`
  - Point size: ~5–6px
  - Hover: show key trip info (fare, distance, duration, pickup zone)
  - **Click event:** Capture clicked point index via `plotly_events` (use `streamlit-plotly-events` package) or Streamlit's built-in `st.plotly_chart(on_select="rerun")`.

**Bottom section (appears after click): Two-column layout**

Left column — **"Selected Point & 10 Nearest Neighbors"**
- Header card showing:
  - 🔴 **Reference Point** label
  - Metric row: `Fare Amount`, `Trip Distance`, `Total Amount` displayed as `st.metric`-style large numbers
  - **Pickup:** `{Borough} ({Zone Name}) at {HH:MM}`
  - **Dropoff:** `{Borough} ({Zone Name})`
- Below: table or cards for 10 nearest neighbors in the 2D latent space (Euclidean distance). Show same fields per neighbor.

Right column — **"Geographic Trip Mapping"**
- Folium map (`streamlit-folium`) centered on NYC (~40.75, -73.98), zoom ~11.
- Plot the selected trip as a polyline (pickup → dropoff) with colored markers.
- Plot the 10 nearest neighbor trips similarly (different colors or white markers as seen in screenshot).
- Pickup markers: green, Dropoff markers: red (or per the screenshot pattern).
- Tooltip on each marker: zone name, fare, distance.

#### Tab 2: Data Analysis

- Basic EDA views (optional stretch goal):
  - Distribution histograms of key features
  - Correlation heatmap
  - Training loss curve (train + val over epochs)

### 4.4 Styling Notes (from screenshots)

- Sidebar background: default Streamlit white
- Slider accent color: red/coral (#FF4B4B — Streamlit default)
- Multiselect pills: red background, white text, "×" dismiss — Streamlit default `st.multiselect` chip styling
- Scatter plot: white background, thin gray gridlines, compact legend on right side
- Map: dark tile layer (CartoDB dark_matter or similar, matching the dark map in screenshot)
- Info banner (yellow): `st.info("Click on any point in the scatter plot above to view its geographic trip and nearest neighbors.")`

---

## 5. File Structure

```
project/
├── app.py                  # Streamlit entry point
├── data/
│   ├── fetch.py            # Data download + sampling logic
│   └── preprocess.py       # Cleaning, feature engineering, scaling
├── model/
│   ├── autoencoder.py      # PyTorch Autoencoder class
│   └── train.py            # Training loop, early stopping, embedding extraction
├── viz/
│   ├── scatter.py          # Plotly scatter plot builder
│   └── map.py              # Folium map builder
├── requirements.txt
└── PLAN.md
```

---

## 6. Dependencies

```
streamlit>=1.32
torch>=2.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
plotly>=5.18
folium>=0.15
streamlit-folium>=0.18
requests
pyarrow            # for parquet I/O
```

Optional: `streamlit-plotly-events` if Streamlit's native `on_select` callback is insufficient.

---

## 7. Execution Flow (Runtime)

```
1. User opens app → sidebar defaults are populated
2. @st.cache_data: fetch 5k sample from Socrata API (or Parquet fallback)
3. @st.cache_data: preprocess + feature-engineer → numeric matrix X, metadata df
4. User adjusts sidebar params
5. User clicks "Train Autoencoder"
6. Train AE with params → progress bar in main area
7. Extract 2D bottleneck codes → optionally PCA-orthogonalize
8. Render scatter plot (color by selected feature, highlight selected categories)
9. User clicks a point → rerun:
   a. Compute 10 nearest neighbors in 2D latent space
   b. Render detail card (left col)
   c. Render Folium map (right col) with selected trip + neighbors
```

---

## 8. Implementation Notes & Edge Cases

**Reproducibility:** Seed `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` globally.

**Socrata API throttling:** The public endpoint allows unauthenticated requests up to 1,000 rows per call by default. To get 5,000 rows, either pass `$limit=5000` (allowed without app token, may be throttled) or paginate with `$offset`. Add retry logic with exponential backoff.

**Zone ID → lat/lon for map:** The zone lookup CSV has no coordinates. Options:
- Hardcode approximate centroids for the ~263 zones (available from the TLC shapefile or precomputed).
- Download the shapefile, compute centroids with `geopandas`, and cache.
- Use a precomputed zone-centroid CSV (many are available on GitHub).
- Simplest: bundle a `zone_centroids.csv` with columns `LocationID, latitude, longitude` derived from the shapefile.

**Plotly click events in Streamlit:** As of Streamlit 1.36+, `st.plotly_chart` supports `on_select="rerun"` which returns selection data. Use this. Fallback: `streamlit-plotly-events`.

**Training time:** With 5k rows, 36 input dims, 3 hidden layers of 64, and 50 epochs — expect <10 seconds on CPU. Show a `st.progress` bar during training.

**Dynamic multiselect label:** When the user changes "Color Mapping Feature", the multiselect below must update its options and label. Use `st.multiselect(f"Highlight specific {feature_name}s", options=unique_values, default=unique_values)`.

**Nearest neighbors:** Use `sklearn.neighbors.NearestNeighbors(n_neighbors=11, metric='euclidean')` on the 2D embeddings. Index 0 is the point itself; indices 1–10 are the neighbors.

---

## 9. Stretch Goals (if time permits)

- **Animated training:** Show the scatter plot evolving epoch-by-epoch during training.
- **Loss curve panel:** Plot train/val MSE loss in the "Data Analysis" tab.
- **Downloadable embeddings:** Button to export the 2D codes + metadata as CSV.
- **t-SNE / UMAP toggle:** Add alternative DR methods alongside PCA on the bottleneck codes.
- **Adjustable neighbor count:** Slider for k in k-NN (default 10).
