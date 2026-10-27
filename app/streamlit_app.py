"""Traffic Oracle: interactive test bench for the METR-LA forecasting models (Team T042).

Run:  streamlit run app/streamlit_app.py
Needs the dataset (make data) and, for the saved-forecast views, results/predictions/*.npz.
The Outage Lab additionally needs checkpoints/dl/{stgat,gru_nograph}_seed*.pt (git-ignored; run the DL chain).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PRED = ROOT / "results" / "predictions"
MET = ROOT / "results" / "metrics"
CK = ROOT / "checkpoints" / "dl"

NICE = {
    "persistence": "Persistence",
    "historical_average": "Historical average",
    "seasonal_naive": "Seasonal naive",
    "xgb_per_segment": "XGBoost per-segment",
    "xgb_global": "XGBoost pooled",
    "rf_multioutput": "Random Forest",
    "xgb_spatial": "XGBoost + neighbours",
    "gru_nograph": "GRU (no graph)",
    "stgat": "STGAT (graph attention)",
    "hybrid_residual": "Hybrid · residual",
    "hybrid_gating": "Hybrid · gated",
    "hybrid_average": "Hybrid · average",
}
PALETTE = {
    "Truth": "#FFFFFF",
    "persistence": "#6B7794",
    "xgb_global": "#FFB547",
    "xgb_spatial": "#FF8A3D",
    "xgb_per_segment": "#8C9BFF",
    "gru_nograph": "#B28CFF",
    "stgat": "#3DD6C4",
    "hybrid_residual": "#FF5A4E",
    "hybrid_gating": "#FF7AA8",
    "hybrid_average": "#5BD68A",
    "rf_multioutput": "#9AA7C7",
    "historical_average": "#7A869F",
    "seasonal_naive": "#566079",
}

import os
from types import SimpleNamespace

BUNDLE = ROOT / "app" / "demo_data" / "bundle.npz"
FULL = (ROOT / "data/raw/metr-la.h5").exists() and (PRED / "persistence.npz").exists()
DEMO = BUNDLE.exists() and (os.environ.get("TRAFFIC_ORACLE_DEMO") == "1" or not FULL)

st.set_page_config(
    page_title="Traffic Oracle", page_icon="🚦", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
.hero {padding: 26px 32px; border-radius: 18px; margin-bottom: 14px;
  background: radial-gradient(1200px 300px at 85% -40%, rgba(255,90,78,.45), transparent 60%),
              radial-gradient(900px 260px at 10% 120%, rgba(61,214,196,.28), transparent 60%),
              linear-gradient(135deg, #0E1530 0%, #0A0F1E 100%);
  border: 1px solid rgba(255,255,255,.07); box-shadow: 0 10px 40px rgba(0,0,0,.45);}
.hero h1 {font-size: 2.5rem; margin: 0; letter-spacing: -.5px;
  background: linear-gradient(90deg,#fff 20%,#FFB547 60%,#FF5A4E 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {color:#AEB8D0; margin: 6px 0 0 0; font-size: 1.02rem;}
.tag {display:inline-block; font-size:.72rem; letter-spacing:.14em; color:#FFB547; font-weight:700; margin-bottom:6px;}
.kpi {background: linear-gradient(160deg, rgba(21,28,53,.95), rgba(14,19,40,.95)); border:1px solid rgba(255,255,255,.07);
  border-radius: 16px; padding: 16px 18px; box-shadow: 0 6px 24px rgba(0,0,0,.35); height: 100%;}
.kpi .v {font-size: 2rem; font-weight: 800; line-height: 1.1;}
.kpi .l {color:#AEB8D0; font-size:.82rem; margin-top:4px;}
.kpi .d {font-size:.8rem; margin-top:6px;}
.good {color:#5BD68A;} .bad {color:#FF5A4E;} .mut {color:#8791AD;}
div[data-baseweb="tab-list"] {gap: 6px;}
button[data-baseweb="tab"] {background: rgba(255,255,255,.04); border-radius: 12px 12px 0 0; padding: 8px 18px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#0E1530,#0A0F1E); border-right: 1px solid rgba(255,255,255,.06);}
.note {border-left: 3px solid #FFB547; padding: 8px 14px; background: rgba(255,181,71,.08); border-radius: 6px; color:#D8DEEE; font-size:.9rem;}
.sim {border-left: 3px solid #FF5A4E; padding: 8px 14px; background: rgba(255,90,78,.08); border-radius: 6px; color:#D8DEEE; font-size:.9rem;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading the demo bundle …")
def load_bundle():
    return dict(np.load(BUNDLE, allow_pickle=False))


@st.cache_resource(show_spinner="Loading METR-LA and the road graph …")
def load_world():
    if DEMO:  # lightweight stand-in with just what the views need
        b = load_bundle()
        speed = b["slab_speed"].astype(np.float32)
        index = pd.date_range(
            pd.Timestamp(int(b["slab_start_ns"])), periods=len(speed), freq="5min"
        )
        loc_df = pd.DataFrame(
            {"sensor_id": b["sensor_ids"], "latitude": b["lat"], "longitude": b["lon"]}
        )
        prep = SimpleNamespace(
            speed=speed,
            mask=speed > 0,
            index=index,
            windows={"test": SimpleNamespace(anchors=b["anchor_pos"])},
            data=SimpleNamespace(locations=loc_df, sensor_ids=list(b["sensor_ids"])),
            chan=None,
            scaler=None,
        )
        return None, prep
    from traffic.config import load_config
    from traffic.data.pipeline import prepare

    cfg = load_config(str(ROOT / "configs/base.yaml"))
    import os

    os.chdir(ROOT)
    p = prepare(cfg)
    return cfg, p


@st.cache_data(show_spinner=False)
def load_truth():
    if DEMO:
        b = load_bundle()
        y = b["y_true"].astype(np.float32)
        return y, y > 0, b["tts"].astype("datetime64[ns]")
    z = np.load(PRED / "persistence.npz")
    return z["y_true"], z["mask"], z["target_ts"].astype("datetime64[ns]")


@st.cache_data(show_spinner="Reading saved forecasts …", max_entries=8)
def load_model(name: str) -> np.ndarray | None:
    if DEMO:
        return (
            load_bundle().get(f"pred_{name}", None).astype(np.float32)
            if f"pred_{name}" in load_bundle()
            else None
        )
    files = sorted(PRED.glob(f"{name}_seed*.npz")) or (
        [PRED / f"{name}.npz"] if (PRED / f"{name}.npz").exists() else []
    )
    if not files:
        return None
    return np.mean([np.load(f)["y_pred"] for f in files], axis=0).astype(np.float32)  # seed-average


def speed_rgb(v: float) -> list[int]:
    """Red (jam) → amber → teal/green (free flow); grey for missing."""
    if v is None or np.isnan(v) or v <= 0:
        return [110, 118, 140, 120]
    stops = [
        (15, (255, 60, 60)),
        (40, (255, 120, 60)),
        (52, (255, 190, 80)),
        (62, (80, 214, 140)),
        (70, (61, 214, 196)),
    ]
    if v <= stops[0][0]:
        return [*stops[0][1], 235]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return [int(ca[i] + t * (cb[i] - ca[i])) for i in range(3)] + [235]
    return [*stops[-1][1], 235]


def kpi(
    label: str, value: str, delta: str = "", good: bool | None = None, color: str = "#FFFFFF"
) -> str:
    cls = "mut" if good is None else ("good" if good else "bad")
    return f'<div class="kpi"><div class="v" style="color:{color}">{value}</div><div class="l">{label}</div><div class="d {cls}">{delta}</div></div>'


def deck(values: np.ndarray, loc: pd.DataFrame, title: str, mode: str = "speed", height: int = 430):
    df = loc.copy()
    df["v"] = values
    if mode == "speed":
        df["color"] = [speed_rgb(v) for v in values]
        tip = "{sensor_id}: {v} mph"
        df["v"] = df["v"].round(1)
        df["r"] = 480.0
    else:  # error / difference map: blue = small, red = big
        m = max(float(np.nanmax(np.abs(values))), 1e-6)
        df["color"] = [
            [
                int(255 * min(abs(x) / m, 1)),
                int(110 * (1 - min(abs(x) / m, 1))) + 60,
                int(230 * (1 - min(abs(x) / m, 1))) + 20,
                235,
            ]
            for x in values
        ]
        df["v"] = df["v"].round(2)
        tip = "{sensor_id}: {v} mph"
        df["r"] = 300.0 + 700.0 * np.minimum(np.abs(values) / m, 1)
    df["r2"] = df["r"] * 2.6
    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius="r",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 110],
        line_width_min_pixels=1,
        radius_min_pixels=3,
    )
    glow = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius="r2",
        opacity=0.2,
    )
    view = pdk.ViewState(
        latitude=float(loc.latitude.mean()),
        longitude=float(loc.longitude.mean()),
        zoom=9.6,
        pitch=0,
    )
    st.markdown(f"**{title}**")
    st.pydeck_chart(
        pdk.Deck(
            layers=[glow, layer], initial_view_state=view, map_style="dark", tooltip={"text": tip}
        ),
        height=height,
    )


def masked_mae(y, p, m) -> float:
    return float(np.abs(y - p)[m].mean()) if m.any() else float("nan")


def dark_fig(fig: go.Figure, h: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21,28,53,.55)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.18),
        font=dict(family="Arial"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    return fig


# --------------------------------------------------------------------------- guards
if not DEMO and not FULL:
    st.markdown(
        '<div class="hero"><span class="tag">TEAM T042</span><h1>Traffic Oracle</h1><p>Test bench for our METR-LA forecasting models</p></div>',
        unsafe_allow_html=True,
    )
    st.error("Data or saved forecasts not found.")
    st.markdown("Run these once from the repository root, then reload:")
    st.code(
        "python3 join.py            # sets up the environment and downloads the dataset\nmake phase1               # baselines\nbash scripts/run_ml_chain.sh && bash scripts/run_dl_chain.sh && bash scripts/run_phase2_chain.sh",
        language="bash",
    )
    st.stop()

cfg, prep = load_world()
y_true_all, mask_all, tts = load_truth()
loc = prep.data.locations.reset_index(drop=True)
sensor_ids = prep.data.sensor_ids
first_target = pd.DatetimeIndex(tts[:, 0])
anchor_time = first_target - pd.Timedelta(minutes=5)
H, F = 12, 12

# --------------------------------------------------------------------------- hero
st.markdown(
    '<div class="hero"><span class="tag">TEAM T042  ·  LIVE TEST BENCH</span><h1>Traffic Oracle</h1>'
    "<p>Can a road network see congestion coming? Pick a moment from the held-out test period and watch 12 forecasting models fight it out across 207 Los Angeles freeway sensors.</p></div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- sidebar (shared window controls)
P1_MODELS = [
    "persistence",
    "historical_average",
    "seasonal_naive",
    "xgb_per_segment",
    "xgb_global",
    "rf_multioutput",
    "xgb_spatial",
]
P2_MODELS = [
    "persistence",
    "xgb_global",
    "gru_nograph",
    "stgat",
    "hybrid_residual",
    "hybrid_gating",
    "hybrid_average",
]


def has_model(m: str) -> bool:
    if DEMO:
        return f"pred_{m}" in load_bundle()
    return (PRED / f"{m}.npz").exists() or bool(list(PRED.glob(f"{m}_seed*.npz")))


with st.sidebar:
    st.markdown("### 🎛️ Pick a moment")
    days = sorted({t.date() for t in anchor_time})
    if "pick" not in st.session_state:
        wk = (anchor_time.day_of_week < 5) & (anchor_time.hour >= 7) & (anchor_time.hour < 9)
        st.session_state.pick = int(
            np.argmax(((y_true_all[:, 2] < 40) & mask_all[:, 2]).sum(1) * wk)
        )
    c1, c2 = st.columns(2)
    if c1.button("🔥 Worst jam", width="stretch"):
        wk = (anchor_time.day_of_week < 5) & (anchor_time.hour >= 6) & (anchor_time.hour < 19)
        st.session_state.pick = int(
            np.argmax(((y_true_all[:, 2] < 40) & mask_all[:, 2]).sum(1) * wk)
        )
    if c2.button("🎲 Random", width="stretch"):
        st.session_state.pick = int(np.random.default_rng().integers(len(anchor_time)))
    cur = anchor_time[st.session_state.pick]
    day = st.date_input("Test day", value=cur.date(), min_value=days[0], max_value=days[-1])
    idx_day = np.flatnonzero(anchor_time.date == day)
    if len(idx_day) == 0:
        st.warning("No windows on that day (test period ends 27 Jun).")
        idx_day = np.array([st.session_state.pick])
    times = [anchor_time[i].strftime("%H:%M") for i in idx_day]
    cur_str = cur.strftime("%H:%M")
    t_sel = st.select_slider(
        "Time of day (forecast made at)",
        options=times,
        value=cur_str if cur_str in times else times[len(times) // 3],
    )
    s_idx = int(idx_day[times.index(t_sel)])
    horizon = st.select_slider(
        "Forecast horizon",
        options=list(range(1, 13)),
        value=6,
        format_func=lambda h: f"+{h * 5} min",
    )
    st.markdown("---")
    st.caption(
        "Held-out test period: 4 – 27 Jun 2012. The models never saw these days in training. Choose models inside each phase tab."
    )

a_time = anchor_time[s_idx]
y_true = y_true_all[s_idx]  # [F, N]
mk = mask_all[s_idx]
hh = horizon - 1
target_time = pd.Timestamp(tts[s_idx, hh])
anchor_i = prep.windows["test"].anchors[s_idx]
mt_all = pd.read_csv(MET / "main_table.csv") if (MET / "main_table.csv").exists() else None
agg_all = (
    pd.read_csv(MET / "all_metrics_agg.csv") if (MET / "all_metrics_agg.csv").exists() else None
)


# --------------------------------------------------------------------------- shared views
def kpi_row(fc: dict, focus: str) -> None:
    speed_now, mask_now = prep.speed[anchor_i], prep.mask[anchor_i]
    cong_now = int(((speed_now < 40) & mask_now).sum())
    cong_future = int(((y_true[hh] < 40) & mk[hh]).sum())
    k = st.columns(5)
    k[0].markdown(
        kpi(
            f"Forecast made at · {a_time.strftime('%a %d %b')}",
            a_time.strftime("%H:%M"),
            f"predicting {target_time.strftime('%H:%M')}",
            None,
            "#FFB547",
        ),
        unsafe_allow_html=True,
    )
    k[1].markdown(
        kpi(
            "Sensors in a jam now (< 40 mph)",
            f"{cong_now}",
            f"{cong_future} at +{horizon * 5} min",
            cong_future <= cong_now,
            "#FF5A4E",
        ),
        unsafe_allow_html=True,
    )
    if fc.get(focus) is not None:
        mae_f = masked_mae(y_true[hh], fc[focus][s_idx, hh], mk[hh])
        pers = load_model("persistence")
        base_f = masked_mae(y_true[hh], pers[s_idx, hh], mk[hh]) if pers is not None else None
        k[2].markdown(
            kpi(
                f"{NICE[focus]} · MAE this window",
                f"{mae_f:.2f} mph",
                (f"{base_f - mae_f:+.2f} vs persistence" if base_f is not None else ""),
                (base_f is None) or (mae_f <= base_f),
                "#3DD6C4",
            ),
            unsafe_allow_html=True,
        )
    mean_t = float(np.mean(y_true[hh][mk[hh]])) if mk[hh].any() else float("nan")
    k[3].markdown(
        kpi(
            f"Network mean speed at +{horizon * 5} min",
            f"{mean_t:.1f} mph",
            "all observed sensors",
            None,
            "#8C9BFF",
        ),
        unsafe_allow_html=True,
    )
    k[4].markdown(
        kpi(
            "Sensors silent (outage)",
            f"{int((~mk[hh]).sum())}",
            "masked, never scored",
            None,
            "#AEB8D0",
        ),
        unsafe_allow_html=True,
    )
    st.write("")


def map_view(fc: dict, focus: str) -> None:
    cA, cB, cC = st.columns(3)
    with cA:
        deck(
            np.where(mk[hh], y_true[hh], np.nan), loc, f"Reality at {target_time.strftime('%H:%M')}"
        )
    if fc.get(focus) is not None:
        with cB:
            deck(fc[focus][s_idx, hh], loc, f"{NICE[focus]} forecast (+{horizon * 5} min)")
        with cC:
            deck(
                np.where(mk[hh], np.abs(fc[focus][s_idx, hh] - y_true[hh]), 0.0),
                loc,
                "Absolute error (bigger & redder = worse)",
                mode="error",
            )
    st.markdown(
        '<div class="note">Colour = speed: <b style="color:#FF5050">red</b> jam · <b style="color:#FFB547">amber</b> slow · <b style="color:#50D68C">green</b> free flow · <b>grey</b> sensor silent. Hover a dot for its id and value.</div>',
        unsafe_allow_html=True,
    )


def sensor_view(fc: dict, key: str) -> None:
    default_sensor = int(np.nanargmin(np.where(mk[hh], y_true[hh], np.inf))) if mk[hh].any() else 0
    c1, c2 = st.columns([1, 3])
    with c1:
        sidx = st.selectbox(
            "Sensor",
            options=list(range(len(sensor_ids))),
            index=default_sensor,
            key=f"{key}_sensor",
            format_func=lambda i: f"{sensor_ids[i]}  ({loc.latitude[i]:.3f}, {loc.longitude[i]:.3f})",
        )
        st.caption("Default: the slowest sensor at the chosen horizon.")
        rows = [
            {
                "Model": NICE[m],
                "MAE over the hour": round(
                    masked_mae(y_true[:, sidx], a[s_idx, :, sidx], mk[:, sidx]), 2
                ),
            }
            for m, a in fc.items()
            if a is not None and mk[:, sidx].any()
        ]
        if rows:
            st.dataframe(
                pd.DataFrame(rows).sort_values("MAE over the hour"),
                hide_index=True,
                width="stretch",
            )
    with c2:
        a = anchor_i
        hist_x = [prep.index[a - H + 1 + i] for i in range(H)]
        hist_y = np.where(
            prep.mask[a - H + 1 : a + 1, sidx], prep.speed[a - H + 1 : a + 1, sidx], np.nan
        )
        fut_x = [pd.Timestamp(t) for t in tts[s_idx]]
        fig = go.Figure()
        fig.add_vrect(x0=hist_x[-1], x1=fut_x[-1], fillcolor="rgba(255,255,255,.04)", line_width=0)
        fig.add_hline(
            y=40,
            line=dict(color="rgba(255,90,78,.5)", dash="dot"),
            annotation_text="40 mph jam threshold",
            annotation_font_color="#FF5A4E",
        )
        fig.add_trace(
            go.Scatter(
                x=hist_x,
                y=hist_y,
                name="Input (last hour)",
                line=dict(color="#AEB8D0", width=3),
                mode="lines+markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[hist_x[-1]] + fut_x,
                y=[hist_y[-1]] + list(np.where(mk[:, sidx], y_true[:, sidx], np.nan)),
                name="Truth",
                line=dict(color="#FFFFFF", width=4),
                mode="lines+markers",
            )
        )
        for m, arr in fc.items():
            if arr is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[hist_x[-1]] + fut_x,
                        y=[hist_y[-1]] + list(arr[s_idx, :, sidx]),
                        name=NICE[m],
                        mode="lines+markers",
                        line=dict(
                            color=PALETTE.get(m, "#999"),
                            width=2.5,
                            dash="dash" if m == "persistence" else "solid",
                        ),
                    )
                )
        fig.update_layout(
            title=f"Sensor {sensor_ids[sidx]}: one hour of history, then the next hour",
            yaxis_title="Speed (mph)",
            xaxis_title="Time",
        )
        st.plotly_chart(dark_fig(fig, 470), width="stretch")


def score_view(fc: dict, models: list[str]) -> None:
    cL, cR = st.columns([3, 2])
    with cL:
        fig = go.Figure()
        if agg_all is not None:
            a_ = agg_all[(agg_all.slice == "all") & (agg_all.metric == "MAE")]
            for m in models:
                d = a_[a_.model == m].sort_values("horizon")
                if len(d):
                    fig.add_trace(
                        go.Scatter(
                            x=d.horizon * 5,
                            y=d["mean"],
                            name=NICE[m],
                            mode="lines+markers",
                            line=dict(color=PALETTE.get(m), width=3),
                        )
                    )
        fig.update_layout(
            title="Whole test set: error grows with horizon",
            xaxis_title="Horizon (minutes)",
            yaxis_title="MAE (mph)",
        )
        st.plotly_chart(dark_fig(fig, 430), width="stretch")
    with cR:
        rows = []
        for m in models:
            arr = fc.get(m)
            if arr is not None:
                full = None
                if mt_all is not None:
                    r = mt_all[(mt_all.model == m) & (mt_all.horizon == horizon)]
                    full = round(float(r.MAE.iloc[0]), 2) if len(r) else None
                rows.append(
                    {
                        "Model": NICE[m],
                        f"This window +{horizon * 5} min": round(
                            masked_mae(y_true[hh], arr[s_idx, hh], mk[hh]), 2
                        ),
                        "Whole test set": full,
                    }
                )
        st.markdown("**This moment vs. the whole test set** (MAE, mph; lower is better)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(
            '<div class="note">One window is an anecdote; the whole-test-set column is the evidence. "–" means that model only predicts 15/30/60 minutes.</div>',
            unsafe_allow_html=True,
        )
    if mt_all is not None:
        h3 = mt_all[mt_all.horizon.isin([3, 6, 12]) & mt_all.model.isin(models)].copy()
        h3["Model"] = h3.model.map(NICE)
        piv = h3.pivot_table(index="Model", columns="horizon_min", values="MAE").round(2)
        piv.columns = [f"MAE @ {c} min" for c in piv.columns]
        st.markdown("**Headline results (whole test set, mph)**")
        st.dataframe(piv.sort_values("MAE @ 60 min"), width="stretch")


def phase_header(prefix: str, options: list[str], default: list[str], focus_default: str):
    avail = [m for m in options if has_model(m)]
    c1, c2 = st.columns([3, 1])
    models = c1.multiselect(
        "Models to compare",
        options=avail,
        default=[m for m in default if m in avail],
        key=f"{prefix}_models",
        format_func=lambda m: NICE[m],
    )
    models = models or avail[:1]
    focus = c2.selectbox(
        "Map forecast from",
        options=models,
        index=models.index(focus_default) if focus_default in models else 0,
        key=f"{prefix}_focus",
        format_func=lambda m: NICE[m],
    )
    fc = {m: load_model(m) for m in set(models) | {"persistence"}}
    kpi_row(fc, focus)
    return models, focus, fc


# --------------------------------------------------------------------------- PHASES
phase1, phase2 = st.tabs(
    ["①  PHASE 1  ·  Machine learning", "②  PHASE 2  ·  Graph attention & hybrids"]
)

with phase1:
    st.markdown(
        '<div class="note"><b>Phase 1.</b> Naive baselines and tree models. Each sensor is forecast from its own history (plus, for <i>XGBoost + neighbours</i>, hand-made features from the sensors next to it).</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    models, focus, fc = phase_header(
        "p1",
        P1_MODELS,
        ["persistence", "historical_average", "xgb_global", "xgb_spatial"],
        "xgb_spatial",
    )
    t_map, t_sens, t_score, t_gap = st.tabs(
        ["🗺️  Live map", "📈  Sensor forecast", "🏆  Scoreboard", "🤝  Do neighbours help?"]
    )
    with t_map:
        map_view(fc, focus)
    with t_sens:
        sensor_view({m: fc[m] for m in models}, "p1")
    with t_score:
        score_view(fc, models)
    with t_gap:
        gp = MET / "gap_analysis.csv"
        if gp.exists():
            g = pd.read_csv(gp)
            st.markdown(
                "**Adding neighbour features to the same XGBoost model** (positive = neighbours helped; whole test set, 95% intervals from a paired day-block bootstrap)"
            )
            sl = ["all", "peak", "offpeak_weekday", "weekend", "congested", "freeflow"]
            lab = {
                "all": "All",
                "peak": "Peak hours",
                "offpeak_weekday": "Off-peak",
                "weekend": "Weekend",
                "congested": "Jams (<40 mph)",
                "freeflow": "Free flow",
            }
            fig = go.Figure()
            for h, col in ((3, "#3DD6C4"), (6, "#FFB547"), (12, "#FF5A4E")):
                d = g[(g.horizon == h) & g.slice.isin(sl)].set_index("slice").loc[sl]
                fig.add_trace(
                    go.Bar(
                        x=[lab[x] for x in sl],
                        y=d.delta_mae,
                        name=f"+{h * 5} min",
                        marker_color=col,
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=d.ci_hi - d.delta_mae,
                            arrayminus=d.delta_mae - d.ci_lo,
                            color="rgba(255,255,255,.55)",
                        ),
                    )
                )
            fig.update_layout(
                barmode="group",
                yaxis_title="MAE gain (mph)",
                title="Neighbour features help most in jams and at peaks, but not more at longer horizons",
            )
            cL, cR = st.columns([3, 2])
            cL.plotly_chart(dark_fig(fig, 430), width="stretch")
            with cR:
                img = ROOT / "results/figures/ml/shap_xgb_spatial_h12.png"
                if img.exists():
                    st.markdown("**What the model looks at** (TreeSHAP, +60 min)")
                    st.image(str(img), width="stretch")
        else:
            st.info("Run `scripts/ml_analysis.py` to generate the gap analysis.")

with phase2:
    st.markdown(
        '<div class="note"><b>Phase 2.</b> A graph attention network (STGAT) that lets each sensor read its neighbours, its no-graph twin (GRU), and hybrids that combine XGBoost with the graph model.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    models, focus, fc = phase_header(
        "p2",
        P2_MODELS,
        ["persistence", "xgb_global", "gru_nograph", "stgat", "hybrid_gating"],
        "stgat",
    )
    t_map, t_sens, t_score, t_hyb, t_abl, t_lab = st.tabs(
        [
            "🗺️  Live map",
            "📈  Sensor forecast",
            "🏆  Scoreboard",
            "🧬  Hybrids & gate",
            "🔬  Ablations",
            "🧪  Outage Lab",
        ]
    )
    with t_map:
        map_view(fc, focus)
    with t_sens:
        sensor_view({m: fc[m] for m in models}, "p2")
    with t_score:
        score_view(fc, models)
    with t_hyb:
        gw, hg = MET / "gate_weights_summary.csv", MET / "hybrid_gain.csv"
        cL, cR = st.columns(2)
        if gw.exists():
            g = pd.read_csv(gw)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=g.horizon * 5,
                    y=g.mean_w_stgat_peak,
                    name="Peak hours",
                    line=dict(color="#FF5A4E", width=4),
                    mode="lines+markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=g.horizon * 5,
                    y=g.mean_w_stgat_offpeak,
                    name="Off-peak",
                    line=dict(color="#6B7794", width=4),
                    mode="lines+markers",
                )
            )
            fig.update_layout(
                title="Learned gate: how much it trusts STGAT over XGBoost",
                xaxis_title="Horizon (minutes)",
                yaxis_title="Weight on STGAT",
                yaxis_range=[0, 1],
            )
            cL.plotly_chart(dark_fig(fig, 420), width="stretch")
        if hg.exists():
            h = pd.read_csv(hg)
            h = h[
                (h.hybrid == "hybrid_residual")
                & (h.component == "xgb_global")
                & (h.horizon_min == 60)
            ]
            sl = ["all", "peak", "offpeak_weekday", "weekend", "congested", "freeflow"]
            h = h.set_index("slice").loc[sl]
            fig = go.Figure(
                go.Bar(
                    x=["All", "Peak", "Off-peak", "Weekend", "Jams", "Free flow"],
                    y=h.gain_mph,
                    marker_color=["#FF5A4E" if v >= 0 else "#6B7794" for v in h.gain_mph],
                )
            )
            fig.update_layout(
                title="Residual hybrid vs XGBoost at +60 min: gain by situation (mph)",
                yaxis_title="MAE gain (mph)",
            )
            cR.plotly_chart(dark_fig(fig, 420), width="stretch")
        st.markdown(
            '<div class="note">The hybrid learns what XGBoost misses; its gain is almost entirely in jams. The gate, never told our hypothesis, learned to trust the graph model more at peaks and longer horizons.</div>',
            unsafe_allow_html=True,
        )
    with t_abl:
        ap = MET / "ablations.csv"
        if ap.exists():
            ab = pd.read_csv(ap)
            ab = ab[ab.metric == "MAE"].pivot_table(
                index=["ablation", "variant"], columns="horizon_min", values="mean"
            )
            ref = ab.loc[("A0", "default")]
            d = (ab[60] - ref[60]).drop(("A0", "default")).sort_values()
            fig = go.Figure(
                go.Bar(
                    y=[f"{a} {v}" for a, v in d.index],
                    x=d.values,
                    orientation="h",
                    marker_color=["#3DD6C4" if v < 0 else "#FF5A4E" for v in d.values],
                )
            )
            fig.update_layout(
                title="Change in +60 min MAE vs the default STGAT (mph; left = better)",
                xaxis_title="MAE change (mph)",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(dark_fig(fig, 620), width="stretch")
            st.markdown(
                '<div class="note">One seed and 12 epochs per run (laptop budget): differences under about 0.05 mph are noise. Clear signals: removing the node embedding or calendar features hurts; a <i>random</i> graph hurts; removing neighbours entirely did <b>not</b> hurt under this short budget.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Run `scripts/evaluate_ablations.py` first.")
    with t_lab:
        st.markdown(
            '<div class="note"><b>Outage Lab.</b> Silence some sensors in the last hour of input (as a real outage would) and re-run the networks live. A graph model can lean on neighbours; a no-graph model cannot.</div>',
            unsafe_allow_html=True,
        )
        have_ck = (
            (not DEMO)
            and (CK / "stgat_seed0.pt").exists()
            and (CK / "gru_nograph_seed0.pt").exists()
        )
        if DEMO:
            st.info(
                "The Outage Lab re-runs the trained networks live, so it needs the saved checkpoints. Run the app locally (see the README) to use it."
            )
        elif not have_ck:
            st.warning(
                "Checkpoints not found (checkpoints/dl/*.pt are git-ignored). Train them with `bash scripts/run_dl_chain.sh` to enable the lab."
            )
        else:
            import torch

            @st.cache_resource(show_spinner="Loading STGAT and GRU checkpoints …")
            def load_nets():
                from traffic.models.factory import build_model

                nets = {}
                for name in ("stgat", "gru_nograph"):
                    ck = torch.load(CK / f"{name}_seed0.pt", map_location="cpu", weights_only=False)
                    net = build_model(
                        ck["cfg"], prep.data.adj, prep.data.locations, len(sensor_ids)
                    )
                    net.load_state_dict(ck["state"])
                    nets[name] = (net.eval(), ck["cfg"])
                return nets

            nets = load_nets()
            from traffic.models.factory import channel_slice

            def run(name: str, silenced: list[int]) -> np.ndarray:
                net, c = nets[name]
                x = prep.chan[anchor_i - H + 1 : anchor_i + 1].copy()  # [H, N, C]
                if silenced:
                    x[:, silenced, 0] = 0.0  # train mean in scaled space
                    x[:, silenced, 1] = 1.0  # is_missing flag
                x = x[..., channel_slice(c)]
                with torch.no_grad():
                    y = net(torch.from_numpy(x[None]).float())[0].numpy()  # [F, N]
                return y * float(prep.scaler.std) + float(prep.scaler.mean)

            cc1, cc2 = st.columns([1, 2])
            order = np.argsort(np.where(mk[hh], y_true[hh], 99))
            with cc1:
                n_out = st.slider("How many sensors to silence", 0, 40, 8)
                mode = st.radio(
                    "Which sensors", ["Most congested", "Random", "Pick manually"], horizontal=True
                )
                if mode == "Most congested":
                    silenced = [int(i) for i in order[:n_out]]
                elif mode == "Random":
                    silenced = [
                        int(i)
                        for i in np.random.default_rng(7).choice(
                            len(sensor_ids), n_out, replace=False
                        )
                    ]
                else:
                    silenced = st.multiselect(
                        "Sensors",
                        list(range(len(sensor_ids))),
                        default=[int(i) for i in order[:6]],
                        format_func=lambda i: sensor_ids[i],
                    )
                watch = st.selectbox(
                    "Watch this sensor",
                    options=silenced or [int(order[0])],
                    format_func=lambda i: f"{sensor_ids[i]}"
                    + ("  (silenced)" if i in silenced else ""),
                )
            base = {n: run(n, []) for n in nets}
            pert = {n: run(n, silenced) for n in nets}
            saved = load_model("stgat")
            check = (
                float(np.abs(base["stgat"] - saved[s_idx]).max())
                if saved is not None
                else float("nan")
            )
            with cc2:
                drift = {n: float(np.abs(pert[n] - base[n]).mean()) for n in nets}
                kk = st.columns(3)
                kk[0].markdown(
                    kpi("Silenced sensors", str(len(silenced)), "of 207", None, "#FF5A4E"),
                    unsafe_allow_html=True,
                )
                kk[1].markdown(
                    kpi(
                        "STGAT forecast drift",
                        f"{drift['stgat']:.2f} mph",
                        "mean change, all sensors",
                        None,
                        "#3DD6C4",
                    ),
                    unsafe_allow_html=True,
                )
                kk[2].markdown(
                    kpi(
                        "GRU forecast drift",
                        f"{drift['gru_nograph']:.2f} mph",
                        "mean change, all sensors",
                        None,
                        "#B28CFF",
                    ),
                    unsafe_allow_html=True,
                )
            fig = go.Figure()
            fx = [pd.Timestamp(t) for t in tts[s_idx]]
            fig.add_trace(
                go.Scatter(
                    x=fx,
                    y=np.where(mk[:, watch], y_true[:, watch], np.nan),
                    name="Truth",
                    line=dict(color="#FFFFFF", width=4),
                )
            )
            for n, col in (("stgat", "#3DD6C4"), ("gru_nograph", "#B28CFF")):
                fig.add_trace(
                    go.Scatter(
                        x=fx,
                        y=base[n][:, watch],
                        name=f"{NICE[n]} · all sensors",
                        line=dict(color=col, width=2, dash="dot"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=fx,
                        y=pert[n][:, watch],
                        name=f"{NICE[n]} · after outage",
                        line=dict(color=col, width=3.5),
                    )
                )
            fig.update_layout(
                title=f"Sensor {sensor_ids[watch]}: forecast before and after the outage",
                yaxis_title="Speed (mph)",
            )
            st.plotly_chart(dark_fig(fig, 400), width="stretch")
            m1, m2 = st.columns(2)
            with m1:
                deck(
                    np.abs(pert["stgat"][hh] - base["stgat"][hh]),
                    loc,
                    "STGAT: how much each sensor's forecast moved",
                    mode="error",
                    height=360,
                )
            with m2:
                deck(
                    np.abs(pert["gru_nograph"][hh] - base["gru_nograph"][hh]),
                    loc,
                    "GRU: how much each sensor's forecast moved",
                    mode="error",
                    height=360,
                )
            st.caption(
                f"Sanity check: the live STGAT run matches the saved seed-averaged forecast to within {check:.2f} mph (seed 0 vs the 3-seed mean)."
            )
            st.markdown(
                '<div class="note">Reading it: the GRU only changes where its own input was silenced; STGAT also moves at <i>neighbouring</i> sensors, because attention lets it read the damaged inputs of the roads around it. Whether that makes STGAT more or less accurate under outages is a question the full test set, not one window, has to answer.</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption(
    "Traffic Oracle · Team T042 · Pranav Singh, Kunal, Himanshu, Kanchan, Vaibhav · Forecasts are offline research output on METR-LA; the ramp-metering study is a simulation and no real-world traffic control is performed or claimed."
)
