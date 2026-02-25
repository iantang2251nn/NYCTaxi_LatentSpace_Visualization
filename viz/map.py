"""
viz/map.py — Build a Folium map showing a selected taxi trip plus its
nearest neighbours in the 2-D latent space.
"""

from __future__ import annotations

import folium
import pandas as pd


NYC_CENTER = (40.75, -73.98)
DEFAULT_ZOOM = 11


def build_trip_map(
    selected: pd.Series,
    neighbors: pd.DataFrame,
    centroids: pd.DataFrame,
) -> folium.Map:
    """Return a Folium ``Map`` with polylines and markers.

    Parameters
    ----------
    selected : Series  — single trip row from metadata
    neighbors : DataFrame — 10 nearest-neighbour trip rows
    centroids : DataFrame — zone centroid lookup (LocationID, latitude, longitude)
    """
    m = folium.Map(
        location=NYC_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles="CartoDB dark_matter",
    )

    centroid_map = centroids.set_index("LocationID")

    # ── Helper: resolve LocationID → (lat, lon) ───────────────────
    def _coords(location_id: int) -> tuple[float, float] | None:
        if location_id in centroid_map.index:
            row = centroid_map.loc[location_id]
            return (row["latitude"], row["longitude"])
        return None

    # ── Plot selected trip (red) ──────────────────────────────────
    _add_trip(m, selected, centroid_map, _coords, is_selected=True)

    # ── Plot neighbours (white / blue) ────────────────────────────
    for _, nb in neighbors.iterrows():
        _add_trip(m, nb, centroid_map, _coords, is_selected=False)

    return m


def _add_trip(m, trip, centroid_map, coords_fn, *, is_selected: bool):
    pu_id = int(trip.get("PULocationID", 0))
    do_id = int(trip.get("DOLocationID", 0))
    pu = coords_fn(pu_id)
    do = coords_fn(do_id)
    if not pu or not do:
        return

    line_color = "#EF553B" if is_selected else "#7fcdff"
    line_weight = 3 if is_selected else 2

    folium.PolyLine(
        [pu, do], color=line_color, weight=line_weight, opacity=0.8
    ).add_to(m)

    # Pickup marker (green)
    pu_label = trip.get("PU_zone_name", "Unknown")
    folium.CircleMarker(
        location=pu,
        radius=6 if is_selected else 4,
        color="#00CC96",
        fill=True,
        fill_opacity=0.9,
        tooltip=f"PU: {pu_label} | Fare: ${trip.get('fare_amount', 0):.2f}",
    ).add_to(m)

    # Dropoff marker (red / orange)
    do_label = trip.get("DO_zone_name", "Unknown")
    folium.CircleMarker(
        location=do,
        radius=6 if is_selected else 4,
        color="#EF553B" if is_selected else "#FFA15A",
        fill=True,
        fill_opacity=0.9,
        tooltip=f"DO: {do_label} | Dist: {trip.get('trip_distance', 0):.1f} mi",
    ).add_to(m)
