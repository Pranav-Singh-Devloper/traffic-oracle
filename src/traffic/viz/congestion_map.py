"""Interactive folium maps and static maps of speeds and forecast errors."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"red": "#d7191c", "orange": "#fdae61", "green": "#1a9641", "grey": "#999999"}


def speed_color(v: float) -> str:
    if v is None or np.isnan(v) or v <= 0:
        return COLORS["grey"]
    return COLORS["red"] if v < 40 else COLORS["orange"] if v <= 55 else COLORS["green"]


def pick_am_peak_anchor(
    target_ts: np.ndarray, y_true: np.ndarray, mask: np.ndarray, h: int = 3
) -> int:
    """Test weekday anchor (target 07:00-09:00) with the most congested observed sensors."""
    ts = pd.DatetimeIndex(target_ts[:, h - 1])
    hm = ts.hour * 60 + ts.minute
    ok = (ts.day_of_week < 5) & (hm >= 420) & (hm < 540)
    cong = ((y_true[:, h - 1] < 40) & mask[:, h - 1]).sum(1)
    cong = np.where(ok, cong, -1)
    return int(cong.argmax())


def forecast_map(
    layers: dict[str, np.ndarray],
    target_times: dict[int, str],
    locations: pd.DataFrame,
    out: Path,
    title: str,
) -> None:
    """layers: name -> {horizon: [N] speeds}; one toggleable layer per (model, horizon)."""
    import folium

    m = folium.Map(location=[locations.latitude.mean(), locations.longitude.mean()], zoom_start=10)
    for name, per_h in layers.items():
        for h, vals in per_h.items():
            fg = folium.FeatureGroup(name=f"{name} +{h * 5} min ({target_times[h]})", show=(h == 3))
            for k, r in enumerate(locations.itertuples()):
                folium.CircleMarker(
                    [r.latitude, r.longitude],
                    radius=5,
                    color=speed_color(vals[k]),
                    fill=True,
                    fill_opacity=0.85,
                    tooltip=f"{r.sensor_id}: {vals[k]:.1f} mph",
                ).add_to(fg)
            fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    folium.map.Marker(
        [locations.latitude.max(), locations.longitude.min()],
        icon=folium.DivIcon(html=f"<b>{title}</b>"),
    ).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))


def animated_map(series: np.ndarray, times: list[str], locations: pd.DataFrame, out: Path) -> None:
    """series [T, N] speeds -> time-slider map (folium TimestampedGeoJson)."""
    import folium
    from folium.plugins import TimestampedGeoJson

    feats = []
    for t, ts in enumerate(times):
        for k, r in enumerate(locations.itertuples()):
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r.longitude, r.latitude]},
                    "properties": {
                        "time": ts,
                        "popup": f"{r.sensor_id}: {series[t, k]:.1f} mph",
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": speed_color(series[t, k]),
                            "fillOpacity": 0.85,
                            "stroke": "false",
                            "radius": 6,
                        },
                    },
                }
            )
    m = folium.Map(location=[locations.latitude.mean(), locations.longitude.mean()], zoom_start=10)
    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": feats},
        period="PT5M",
        duration="PT5M",
        add_last_point=False,
        auto_play=False,
        loop=False,
    ).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))


def static_scatter(
    locations: pd.DataFrame,
    values: np.ndarray,
    title: str,
    out: Path,
    cmap: str = "viridis",
    center: float | None = None,
    label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    kw = {}
    if center is not None:
        lim = np.nanmax(np.abs(values - center))
        kw = {"vmin": center - lim, "vmax": center + lim}
    sc = ax.scatter(
        locations.longitude,
        locations.latitude,
        c=values,
        cmap=cmap,
        s=40,
        edgecolor="k",
        lw=0.3,
        **kw,
    )
    ax.set_aspect(1 / np.cos(np.radians(locations.latitude.mean())))
    ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(sc, label=label)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
