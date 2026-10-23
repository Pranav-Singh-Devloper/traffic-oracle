"""EDA figures (one function each). All stats exclude missing (speed == 0).

Each function saves PNG + PDF under ``out_dir`` and returns a dict of numbers that
``scripts/run_eda.py`` collects into ``results/metrics/eda_stats.json``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 150
COLORS = {"weekday": "#0072B2", "weekend": "#D55E00", "peak": "#CC79A7", "off": "#009E73"}


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=DPI)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def _nan_speed(speed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, speed, np.nan)


def peak_mask(index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """Weekday timestamps inside the AM/PM peak windows."""
    hm = index.hour * 60 + index.minute
    out = np.zeros(len(index), bool)
    for a, b in cfg["peak_windows"].values():
        lo = int(a[:2]) * 60 + int(a[3:])
        hi = int(b[:2]) * 60 + int(b[3:])
        out |= (hm >= lo) & (hm < hi)
    return out & (index.day_of_week < 5)


def plot_missingness(mask, index, out_dir) -> dict:
    miss = ~mask
    days = pd.Series(miss.mean(1), index=index).groupby(index.date).size().shape[0]
    per_day = miss.reshape(days, 288, -1).mean(1)  # [days, N]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    im = ax[0].imshow(per_day.T, aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    ax[0].set(xlabel="Day (0 = 2012-03-01)", ylabel="Sensor", title="Missing rate per sensor-day")
    fig.colorbar(im, ax=ax[0], label="missing fraction")
    ps = miss.mean(0)
    ax[1].hist(ps * 100, bins=30, color=COLORS["weekday"])
    ax[1].set(
        xlabel="Per-sensor missing rate (%)", ylabel="# sensors", title="Missing-rate histogram"
    )
    _save(fig, out_dir, "01_missingness")
    net = miss.mean(1) >= 0.9
    return {
        "missing_rate_overall": float(miss.mean()),
        "sensor_missing_rate_max": float(ps.max()),
        "sensor_missing_rate_median": float(np.median(ps)),
        "sensors_never_missing": int((ps == 0).sum()),
        "whole_network_outage_timestamps": int(net.sum()),
    }


def plot_daily_profile(speed, mask, index, cfg, out_dir) -> dict:
    sp = _nan_speed(speed, mask)
    slot = ((index.hour * 60 + index.minute) // 5).to_numpy()
    wk = index.day_of_week.to_numpy() < 5
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(288) * 5 / 60
    res = {}
    for name, sel in (("weekday", wk), ("weekend", ~wk)):
        prof = np.array([np.nanmean(sp[sel & (slot == s)]) for s in range(288)])
        per_sensor = np.array([np.nanmean(sp[sel & (slot == s)], axis=0) for s in range(288)])
        q1, q3 = np.nanpercentile(per_sensor, [25, 75], axis=1)
        ax.plot(x, prof, color=COLORS[name], label=name)
        ax.fill_between(x, q1, q3, color=COLORS[name], alpha=0.2)
        res[f"{name}_min_mean_speed"] = float(np.nanmin(prof))
        res[f"{name}_min_speed_hour"] = float(x[np.nanargmin(prof)])
    for a, b in cfg["peak_windows"].values():
        ax.axvspan(int(a[:2]), int(b[:2]), color="grey", alpha=0.12)
    ax.set(
        xlabel="Hour of day",
        ylabel="Mean speed (mph)",
        title="Daily profile (band = IQR across sensors)",
    )
    ax.legend()
    _save(fig, out_dir, "02_daily_profile")
    return res


def plot_tod_dow_heatmap(speed, mask, index, out_dir) -> dict:
    sp = _nan_speed(speed, mask)
    slot = ((index.hour * 60 + index.minute) // 5).to_numpy()
    dow = index.day_of_week.to_numpy()
    grid = np.array(
        [[np.nanmean(sp[(dow == d) & (slot == s)]) for s in range(288)] for d in range(7)]
    )
    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", extent=[0, 24, 6.5, -0.5])
    ax.set(
        xlabel="Hour of day",
        ylabel="Day of week (0 = Mon)",
        title="Mean speed by time-of-day x day-of-week",
    )
    fig.colorbar(im, ax=ax, label="mph")
    _save(fig, out_dir, "03_tod_dow_heatmap")
    return {"heatmap_min_mph": float(np.nanmin(grid)), "heatmap_max_mph": float(np.nanmax(grid))}


def plot_speed_distribution(speed, mask, index, cfg, out_dir) -> dict:
    pk = peak_mask(index, cfg)
    v = speed[mask]
    vp = speed[pk][mask[pk]]
    vo = speed[~pk][mask[~pk]]
    thr = cfg["congestion_threshold_mph"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(v, bins=70, color=COLORS["weekday"], density=True)
    ax[0].set(xlabel="Speed (mph)", ylabel="Density", title="All observed speeds")
    ax[1].hist(vo, bins=70, alpha=0.6, density=True, color=COLORS["off"], label="off-peak")
    ax[1].hist(vp, bins=70, alpha=0.6, density=True, color=COLORS["peak"], label="weekday peak")
    for a in ax:
        a.axvline(thr, color="k", ls="--", lw=1)
    ax[1].set(xlabel="Speed (mph)", title=f"Peak vs off-peak (dashed = {thr} mph)")
    ax[1].legend()
    _save(fig, out_dir, "04_speed_distribution")
    return {
        "mean_speed": float(v.mean()),
        "share_below_threshold_all": float((v < thr).mean()),
        "share_below_threshold_peak": float((vp < thr).mean()),
        "share_below_threshold_offpeak": float((vo < thr).mean()),
        "mean_speed_peak": float(vp.mean()),
        "mean_speed_offpeak": float(vo.mean()),
    }


def plot_acf_pacf(speed, mask, index, out_dir, train_stop: int) -> dict:
    from statsmodels.tsa.stattools import acf, pacf

    sp = pd.DataFrame(_nan_speed(speed[:train_stop], mask[:train_stop]))
    miss = (~mask[:train_stop]).mean(0)
    var = np.nanstd(sp.to_numpy(), axis=0)
    ok = np.where(miss < 0.1)[0]
    order = ok[np.argsort(var[ok])]
    picks = [order[0], order[len(order) // 2], order[-1]]
    fig, ax = plt.subplots(len(picks), 2, figsize=(11, 8))
    res = {}
    for r, j in enumerate(picks):
        # Interpolate for ACF only (statsmodels needs a gap-free series).
        s = sp[j].interpolate(limit_direction="both").to_numpy()
        a = acf(s, nlags=576, fft=True)
        p = pacf(s[:6000], nlags=48)
        ax[r, 0].plot(a, lw=0.8)
        ax[r, 0].axvline(288, color="r", ls=":", lw=1)
        ax[r, 0].set(title=f"ACF, sensor idx {j}", xlabel="lag (5-min steps)")
        ax[r, 1].bar(range(len(p)), p, color=COLORS["weekday"])
        ax[r, 1].set(title=f"PACF (first 6000 steps), sensor idx {j}", xlabel="lag")
        res[f"acf_lag288_sensor{j}"] = float(a[288])
    _save(fig, out_dir, "05_acf_pacf")
    return res


def _hops(w: np.ndarray) -> np.ndarray:
    g = nx.from_numpy_array(((w + w.T) > 0).astype(int) - np.eye(len(w), dtype=int).clip(0, 1))
    n = len(w)
    h = np.full((n, n), -1)
    for s, d in nx.all_pairs_shortest_path_length(g):
        for t, v in d.items():
            h[s, t] = v
    return h


def plot_corr_vs_hops(speed, mask, w, out_dir, train_stop: int) -> dict:
    df = pd.DataFrame(_nan_speed(speed[:train_stop], mask[:train_stop]))
    corr = df.corr(min_periods=500).to_numpy()
    hops = _hops(w)
    iu = np.triu_indices(len(w), 1)
    c, h = corr[iu], hops[iu]
    groups = [("unreach.", c[h < 0])] + [(str(k), c[h == k]) for k in range(1, 9)]
    groups = [(n, g[~np.isnan(g)]) for n, g in groups]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(
        [g for _, g in groups if len(g)],
        tick_labels=[n for n, g in groups if len(g)],
        showfliers=False,
    )
    ax.set(
        xlabel="Shortest-path hops (undirected)",
        ylabel="Pearson correlation",
        title="Correlation vs graph distance (train)",
    )
    _save(fig, out_dir, "06_corr_vs_hops")
    med = {
        f"median_corr_{n}_hops": float(np.median(g))
        for n, g in groups
        if len(g) and n != "unreach."
    }
    med["median_corr_unreachable"] = float(np.median(groups[0][1])) if len(groups[0][1]) else None
    return med


def lagged_xcorr(
    w,
    speed,
    mask,
    index,
    train_stop: int,
    max_lag: int = 12,
    max_missing: float = 0.1,
    time_sel: np.ndarray | None = None,
):
    """Cross-correlation of detrended speed for every directed edge.

    Edge (i, j) with W[i, j] > 0 means the road runs i -> j, so j is DOWNSTREAM of i.
    We compute corr(z_j(t), z_i(t + lag)). A positive peak lag means the upstream
    sensor i reacts AFTER the downstream sensor j, i.e. congestion moves upstream.
    Detrending removes the train time-of-day mean so the daily cycle doesn't dominate.
    """
    from traffic.data.missing import slot_of_day, tod_means

    tm = tod_means(speed, mask, index, slice(0, train_stop))
    z = np.where(mask, speed - tm[slot_of_day(index)], np.nan)[:train_stop]
    ok = (~mask[:train_stop]).mean(0) < max_missing
    lags = np.arange(-max_lag, max_lag + 1)
    rows, curves = [], {}
    for i, j in zip(*np.nonzero(w > 0), strict=True):
        if i == j or not (ok[i] and ok[j]):
            continue
        cs = []
        for lag in lags:
            a = z[:, j][max(0, -lag) : len(z) - max(0, lag)]
            b = z[:, i][max(0, lag) : len(z) - max(0, -lag)]
            m = ~np.isnan(a) & ~np.isnan(b)
            if time_sel is not None:  # restrict by the anchor-side timestamp of ``a``
                m &= time_sel[max(0, -lag) : len(z) - max(0, lag)]
            cs.append(np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 100 else np.nan)
        cs = np.array(cs)
        if np.isnan(cs).all():
            continue
        rows.append(
            {
                "upstream": i,
                "downstream": j,
                "peak_lag": int(lags[np.nanargmax(cs)]),
                "peak_corr": float(np.nanmax(cs)),
                "corr_lag0": float(cs[max_lag]),
            }
        )
        curves[(i, j)] = cs
    return lags, pd.DataFrame(rows), curves


def plot_lagged_xcorr(w, speed, mask, index, cfg, out_dir, results_dir, train_stop: int) -> dict:
    lags, df, curves = lagged_xcorr(w, speed, mask, index, train_stop)
    # Propagation is a congestion phenomenon, so also look at weekday peak hours only.
    pk = peak_mask(index, cfg)[:train_stop]
    _, dfp, _ = lagged_xcorr(w, speed, mask, index, train_stop, time_sel=pk)
    df["peak_lag_peakhours"] = dfp["peak_lag"].to_numpy()
    df.to_csv(Path(results_dir) / "xcorr_peak_lags.csv", index=False)
    # Pick pairs by rule, never by eye, and label the rule. Each unordered pair appears once.
    df["pair"] = [tuple(sorted(t)) for t in zip(df.upstream, df.downstream, strict=True)]
    uniq = df.drop_duplicates("pair")
    top = uniq.sort_values("peak_corr", ascending=False).head(3)
    strong = uniq[(uniq.peak_corr >= 0.5) & (uniq.peak_lag != 0)]
    lagged = strong.reindex(strong.peak_lag.abs().sort_values(ascending=False).index).head(3)
    chosen = [("most correlated", r) for _, r in top.iterrows()] + [
        ("largest |lag|, corr>=0.5", r) for _, r in lagged.iterrows()
    ]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for a, (tag, r) in zip(ax.ravel(), chosen, strict=False):
        c = curves[(int(r.upstream), int(r.downstream))]
        a.plot(lags, c, marker="o", ms=3)
        a.axvline(0, color="grey", lw=0.8)
        a.axvline(r.peak_lag, color="r", ls=":")
        a.set(
            title=f"{tag}\nup {int(r.upstream)} / down {int(r.downstream)}  peak@{int(r.peak_lag)}",
            xlabel="lag (5-min steps)",
        )
    ax[0, 0].set_ylabel("corr(z_down(t), z_up(t+lag))")
    _save(fig, out_dir, "07_lagged_xcorr")
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-12.5, 13.5)
    ax.hist(df["peak_lag"], bins=bins, color=COLORS["weekday"], alpha=0.6, label="all times")
    ax.hist(
        df["peak_lag_peakhours"],
        bins=bins,
        color=COLORS["peak"],
        alpha=0.6,
        label="weekday peak hours",
    )
    ax.legend()
    ax.axvline(0, color="k", lw=0.8)
    ax.set(
        xlabel="Peak lag (steps); >0: upstream sensor lags downstream",
        ylabel="# adjacent pairs",
        title="Peak cross-correlation lag, all adjacent pairs (train)",
    )
    _save(fig, out_dir, "07b_peak_lag_hist")
    pl = df["peak_lag"]
    return {
        "xcorr_pairs": int(len(df)),
        "xcorr_share_peak_lag_positive": float((pl > 0).mean()),
        "xcorr_share_peak_lag_zero": float((pl == 0).mean()),
        "xcorr_share_peak_lag_negative": float((pl < 0).mean()),
        "xcorr_median_peak_lag": float(pl.median()),
        "xcorr_peakhours_share_positive": float((df["peak_lag_peakhours"] > 0).mean()),
        "xcorr_peakhours_share_zero": float((df["peak_lag_peakhours"] == 0).mean()),
        "xcorr_peakhours_share_negative": float((df["peak_lag_peakhours"] < 0).mean()),
    }


def sensor_map(speed, mask, locations, out_html: Path) -> dict:
    import folium

    mean = np.nanmean(_nan_speed(speed, mask), axis=0)
    miss = (~mask).mean(0)
    m = folium.Map(location=[locations.latitude.mean(), locations.longitude.mean()], zoom_start=10)
    for name, vals, cmap in (("Mean speed", mean, "speed"), ("Missing rate", miss, "miss")):
        fg = folium.FeatureGroup(name=name, show=(cmap == "speed"))
        for k, r in locations.reset_index(drop=True).iterrows():
            if cmap == "speed":
                col = "red" if vals[k] < 40 else "orange" if vals[k] < 55 else "green"
            else:
                col = "red" if vals[k] > 0.2 else "orange" if vals[k] > 0.08 else "green"
            folium.CircleMarker(
                [r.latitude, r.longitude],
                radius=5,
                color=col,
                fill=True,
                fill_opacity=0.8,
                tooltip=f"{r.sensor_id}: mean {mean[k]:.1f} mph, missing {miss[k]*100:.1f}%",
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return {"sensors_mean_speed_below_55": int((mean < 55).sum())}


def congestion_frequency_map(speed, mask, index, locations, cfg, out_dir, out_html: Path) -> dict:
    import folium

    pk = peak_mask(index, cfg)
    sp, mk = speed[pk], mask[pk]
    freq = ((sp < cfg["congestion_threshold_mph"]) & mk).sum(0) / np.maximum(mk.sum(0), 1)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(
        locations.longitude, locations.latitude, c=freq, cmap="Reds", s=30, edgecolor="k", lw=0.3
    )
    ax.set_aspect(1 / np.cos(np.radians(locations.latitude.mean())))
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Share of weekday-peak time below 40 mph")
    fig.colorbar(sc, label="fraction")
    _save(fig, out_dir, "09_congestion_frequency_map")
    m = folium.Map(location=[locations.latitude.mean(), locations.longitude.mean()], zoom_start=10)
    for k, r in locations.reset_index(drop=True).iterrows():
        folium.CircleMarker(
            [r.latitude, r.longitude],
            radius=4 + 12 * freq[k],
            color="red",
            fill=True,
            tooltip=f"{r.sensor_id}: {freq[k]*100:.0f}% congested at peak",
        ).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return {
        "peak_congestion_share_max": float(freq.max()),
        "peak_congestion_share_median": float(np.median(freq)),
        "sensors_congested_over_20pct_of_peak": int((freq > 0.2).sum()),
    }


def plot_split_timeline(speed, mask, index, split, cfg, out_dir) -> dict:
    daily = (
        pd.Series(
            (
                _nan_speed(speed, mask).mean(1)
                if False
                else np.nanmean(_nan_speed(speed, mask), axis=1)
            ),
            index=index,
        )
        .resample("D")
        .mean()
    )
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(daily.index, daily.values, color="k", lw=1)
    cols = {"train": "#56B4E9", "val": "#E69F00", "test": "#009E73"}
    pk = peak_mask(index, cfg)
    res = {}
    for name, s in split.as_dict().items():
        ax.axvspan(index[s.start], index[s.stop - 1], color=cols[name], alpha=0.25, label=name)
        sel = np.zeros(len(index), bool)
        sel[s] = True
        res[name] = {
            "weekday_share": float((index[s].day_of_week < 5).mean()),
            "peak_share": float(pk[s].mean()),
            "missing_rate": float((~mask[s]).mean()),
            "mean_speed": float(speed[s][mask[s]].mean()),
        }
    ax.set(ylabel="Daily mean speed (mph)", title="Chronological train / val / test periods")
    ax.legend(loc="lower left")
    _save(fig, out_dir, "10_split_timeline")
    return {"split_stats": res}
