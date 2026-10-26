"""
Halconomics — Real Exchange Rate explorer built on IMF data.

Definitions
-----------
S(t)   = units of Country 2 currency per 1 unit of Country 1 currency
P1, P2 = monthly CPI of Country 1 and Country 2
RER(t) = S(t) * P1(t) / P2(t)
RER*   = average of RER(t) over valid months in the selected sample
Real Index(t) = 100 * RER(t) / RER*

IMF exchange rates are local-currency units per USD (E_i), so
S(t) = E2(t) / E1(t).
"""

from __future__ import annotations

import os
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pycountry
import requests
import streamlit as st
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
IMF_BASE = "https://api.imf.org/external/sdmx/3.0"
AGENCY = "IMF.STA"
CPI_FLOW = "CPI"
ER_FLOW = "ER"
START_MONTH = pd.Timestamp("1976-01-01")
EURO_AREA_CODE = "G163"

ORANGE = "#FF7A00"
ORANGE_LIGHT = "#FFA64D"
BLACK = "#0B0B0B"
PANEL = "#161616"
GREY = "#BFBFBF"


# --------------------------------------------------------------------------
# Page setup and theme
# --------------------------------------------------------------------------
def make_logo(size: int = 128) -> Image.Image:
    """Orange tile with a black 'H', drawn with rectangles (no font needed)."""
    img = Image.new("RGB", (size, size), ORANGE)
    d = ImageDraw.Draw(img)
    m, w = int(size * 0.24), int(size * 0.14)
    d.rectangle([m, m, m + w, size - m], fill=BLACK)
    d.rectangle([size - m - w, m, size - m, size - m], fill=BLACK)
    d.rectangle([m, size // 2 - w // 2, size - m, size // 2 + w // 2], fill=BLACK)
    return img


st.set_page_config(page_title="Halconomics", page_icon=make_logo(64), layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {BLACK}; color: #F2F2F2; }}
      section[data-testid="stSidebar"] {{ background-color: {PANEL};
          border-right: 2px solid {ORANGE}; }}
      h1, h2, h3, h4 {{ color: {ORANGE} !important; }}
      p, label, span, li, div[data-testid="stMarkdownContainer"] {{ color: #F2F2F2; }}
      div[data-testid="stMetric"] {{ background-color: {PANEL};
          border: 1px solid {ORANGE}; border-radius: 10px; padding: 12px 16px; }}
      div[data-testid="stMetricValue"] {{ color: {ORANGE}; }}
      .stButton > button, .stDownloadButton > button {{
          background-color: {ORANGE}; color: {BLACK}; border: none;
          font-weight: 700; border-radius: 8px; }}
      .stButton > button:hover, .stDownloadButton > button:hover {{
          background-color: {ORANGE_LIGHT}; color: {BLACK}; }}
      div[data-baseweb="select"] > div {{ background-color: {PANEL};
          border-color: {ORANGE}; }}
      div[data-baseweb="slider"] div[role="slider"] {{ background-color: {ORANGE}; }}
      .hal-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }}
      .hal-logo {{ width: 60px; height: 60px; border-radius: 12px; background: {ORANGE};
          color: {BLACK}; font-size: 42px; font-weight: 900; display: flex;
          align-items: center; justify-content: center; font-family: Arial, sans-serif; }}
      .hal-title {{ font-size: 40px; font-weight: 800; color: {ORANGE}; margin: 0; }}
      .hal-sub {{ color: {GREY}; margin-top: -6px; }}
      .hal-box {{ background: {PANEL}; border-left: 4px solid {ORANGE};
          padding: 12px 16px; border-radius: 6px; margin: 8px 0 16px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Country list
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def country_options() -> dict[str, str]:
    """Display name -> IMF code. Euro area first, then ISO3 countries A–Z."""
    countries = {}
    for c in pycountry.countries:
        name = getattr(c, "common_name", None) or c.name
        countries[f"{name} ({c.alpha_3})"] = c.alpha_3
    ordered = {f"Euro area ({EURO_AREA_CODE})": EURO_AREA_CODE}
    ordered.update(dict(sorted(countries.items())))
    return ordered


# --------------------------------------------------------------------------
# IMF data access
# --------------------------------------------------------------------------
_PERIOD_RE = re.compile(r"^(\d{4})-?M?(\d{1,2})$")


def parse_period(p: str) -> pd.Timestamp | None:
    """Accepts '2024-M01', '2024M01', '2024-01'. Returns first day of month."""
    m = _PERIOD_RE.match(str(p).strip())
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def _to_float(v) -> float:
    if isinstance(v, list):
        v = v[0] if v else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def parse_sdmx_json(payload: dict) -> pd.Series:
    """Parse an SDMX-JSON (1.0 or 2.0) data message into a monthly Series."""
    data = payload.get("data", payload)
    structures = data.get("structures") or [data.get("structure", {})]
    datasets = data.get("dataSets") or []
    if not datasets or not structures:
        return pd.Series(dtype=float)

    dims = structures[0].get("dimensions", {})
    obs_dims = dims.get("observation", [])
    time_pos = next((i for i, d in enumerate(obs_dims) if d.get("id") == "TIME_PERIOD"), None)
    if time_pos is None:
        return pd.Series(dtype=float)
    time_values = [v.get("id") or v.get("value") for v in obs_dims[time_pos].get("values", [])]

    points: dict[pd.Timestamp, float] = {}
    ds = datasets[0]

    def add(idx_str: str, value):
        parts = idx_str.split(":")
        try:
            t_idx = int(parts[time_pos]) if len(parts) > 1 else int(parts[0])
        except (ValueError, IndexError):
            return
        if t_idx >= len(time_values):
            return
        ts = parse_period(time_values[t_idx])
        val = _to_float(value)
        if ts is not None and not np.isnan(val):
            points[ts] = val

    if "series" in ds:  # TIME_PERIOD at observation level (default)
        for s in ds["series"].values():
            for o_idx, o_val in (s.get("observations") or {}).items():
                add(o_idx, o_val)
    elif "observations" in ds:  # dimensionAtObservation=AllDimensions
        for o_idx, o_val in ds["observations"].items():
            add(o_idx, o_val)

    return pd.Series(points, dtype=float).sort_index()


def parse_sdmx_csv(text: str) -> pd.Series:
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        return pd.Series(dtype=float)
    idx = df["TIME_PERIOD"].map(parse_period)
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values, index=idx)
    s = s[s.index.notna()].dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _headers(accept: str) -> dict[str, str]:
    h = {"Accept": accept, "User-Agent": "Halconomics/1.0", "Cache-Control": "no-cache"}
    key = os.environ.get("IMF_API_KEY")
    if key:
        h["Ocp-Apim-Subscription-Key"] = key
    return h


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_imf_series(flow: str, key: str) -> tuple[pd.Series, str | None]:
    """Download one monthly IMF series. Returns (series, error_message)."""
    url = f"{IMF_BASE}/data/dataflow/{AGENCY}/{flow}/+/{key}"
    attempts = [
        ("application/vnd.sdmx.data+json;version=2.0.0", "json"),
        ("application/json", "json"),
        ("application/vnd.sdmx.data+csv;version=2.0.0", "csv"),
    ]
    last_err = None
    for accept, kind in attempts:
        try:
            r = requests.get(url, headers=_headers(accept), timeout=45)
            if r.status_code == 404:
                return pd.Series(dtype=float), f"No IMF data for {key}"
            r.raise_for_status()
            s = parse_sdmx_json(r.json()) if kind == "json" else parse_sdmx_csv(r.text)
            if not s.empty:
                return s[s.index >= START_MONTH], None
            last_err = f"Empty response for {key}"
        except Exception as exc:  # network, JSON or parsing issue -> try next format
            last_err = f"{key}: {type(exc).__name__}: {exc}"
    return pd.Series(dtype=float), last_err


def get_country_data(code: str) -> dict:
    cpi, cpi_err = fetch_imf_series(CPI_FLOW, f"{code}.CPI._T.IX.M")
    er, er_err = fetch_imf_series(ER_FLOW, f"{code}.XDC_USD.PA_RT.M")
    note = None
    if er.empty and code == "USA":
        # USD per USD is 1 by definition; the IMF may not publish it.
        er = pd.Series(1.0, index=pd.date_range(START_MONTH, pd.Timestamp.today(), freq="MS"))
        er_err, note = None, "USD per USD set to 1.0 by definition."
    return {"cpi": cpi, "er": er, "cpi_err": cpi_err, "er_err": er_err, "note": note}


# --------------------------------------------------------------------------
# Calculations
# --------------------------------------------------------------------------
def build_panel(cpi1, er1, cpi2, er2) -> pd.DataFrame:
    """Align the four series monthly and compute S(t) and RER(t)."""
    df = pd.concat(
        {"E1_per_USD": er1, "E2_per_USD": er2, "P1_CPI": cpi1, "P2_CPI": cpi2}, axis=1
    ).sort_index()
    if df.empty:
        return df
    full = pd.date_range(max(df.index.min(), START_MONTH), df.index.max(), freq="MS")
    df = df.reindex(full)
    df.index.name = "Month"
    with np.errstate(divide="ignore", invalid="ignore"):
        df["S"] = df["E2_per_USD"] / df["E1_per_USD"]
        df["RER"] = df["S"] * df["P1_CPI"] / df["P2_CPI"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df["Valid"] = df["RER"].notna() & (df["RER"] > 0)
    return df


def apply_sample(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    """Slice to the sample, compute RER* and the Real Index."""
    sample = df.loc[start:end].copy()
    valid = sample.loc[sample["Valid"], "RER"]
    rer_star = float(valid.mean()) if not valid.empty else np.nan
    sample["RER_star"] = rer_star
    sample["Real_Index"] = np.where(sample["Valid"], 100 * sample["RER"] / rer_star, np.nan)
    return sample, rer_star, int(valid.size)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def make_chart(sample: pd.DataFrame, view: str, name1: str, name2: str) -> go.Figure:
    s_label = f"{name2} currency per 1 {name1} currency"
    real = go.Scatter(
        x=sample.index, y=sample["Real_Index"], name="Real Index (avg = 100)",
        line=dict(color=ORANGE, width=2.5), connectgaps=False,
        hovertemplate="%{x|%b %Y}<br>Real Index: %{y:.2f}<extra></extra>",
    )
    nominal = go.Scatter(
        x=sample.index, y=sample["S"], name="Nominal S(t)",
        line=dict(color="#F2F2F2", width=1.8), connectgaps=False,
        hovertemplate="%{x|%b %Y}<br>S: %{y:,.4f}<extra></extra>",
    )

    if view == "Both":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(real, secondary_y=False)
        fig.add_trace(nominal, secondary_y=True)
        fig.update_yaxes(title_text="Real Index (sample avg = 100)", secondary_y=False,
                         color=ORANGE, gridcolor="#2A2A2A")
        fig.update_yaxes(title_text=s_label, secondary_y=True, color="#F2F2F2",
                         showgrid=False)
        fig.add_hline(y=100, line_dash="dot", line_color=ORANGE_LIGHT, opacity=0.6)
    elif view == "Real":
        fig = go.Figure(real)
        fig.update_yaxes(title_text="Real Index (sample avg = 100)", gridcolor="#2A2A2A")
        fig.add_hline(y=100, line_dash="dot", line_color=ORANGE_LIGHT, opacity=0.6)
    else:
        fig = go.Figure(nominal)
        fig.update_yaxes(title_text=s_label, gridcolor="#2A2A2A")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BLACK, plot_bgcolor=BLACK,
        font=dict(color="#F2F2F2"), height=520, hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(gridcolor="#2A2A2A",
                     rangeslider=dict(visible=True, bgcolor=PANEL, thickness=0.06))
    return fig


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def header():
    st.markdown(
        """
        <div class="hal-header">
          <div class="hal-logo">H</div>
          <div><p class="hal-title">Halconomics</p>
               <p class="hal-sub">Real exchange rates from IMF data</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    header()
    options = country_options()
    names = list(options.keys())
    default1 = next((i for i, n in enumerate(names) if n.endswith("(USA)")), 0)
    default2 = next((i for i, n in enumerate(names) if n.endswith(f"({EURO_AREA_CODE})")), 1)

    with st.sidebar:
        st.markdown("### 1 · Choose countries")
        c1_label = st.selectbox("Country 1 (base currency)", names, index=default1)
        c2_label = st.selectbox("Country 2 (quote currency)", names, index=default2)
        st.caption("S(t) = units of Country 2 currency per 1 unit of Country 1 currency.")

    code1, code2 = options[c1_label], options[c2_label]
    name1, name2 = c1_label.rsplit(" (", 1)[0], c2_label.rsplit(" (", 1)[0]

    if code1 == code2:
        st.warning("Pick two different countries to compare.")
        st.stop()

    with st.spinner(f"Fetching IMF data for {name1} and {name2}…"):
        d1, d2 = get_country_data(code1), get_country_data(code2)

    problems = []
    for nm, d in ((name1, d1), (name2, d2)):
        if d["cpi"].empty:
            problems.append(f"**{nm}** CPI unavailable — {d['cpi_err']}")
        if d["er"].empty:
            problems.append(f"**{nm}** exchange rate unavailable — {d['er_err']}")
    if problems:
        st.error("The IMF did not return enough data for this pair:\n\n- " + "\n- ".join(problems))
        st.info("Try another country, or check your connection. The IMF does not cover every "
                "country monthly, and some series start well after 1976.")
        st.stop()

    df = build_panel(d1["cpi"], d1["er"], d2["cpi"], d2["er"])
    valid_months = df.index[df["Valid"]]
    if valid_months.empty:
        st.error("The CPI and exchange-rate series for these two countries never overlap, "
                 "so no real exchange rate can be computed.")
        st.stop()

    first_valid, last_valid = valid_months.min(), valid_months.max()
    months = [d.strftime("%Y-%m") for d in pd.date_range(first_valid, last_valid, freq="MS")]

    with st.sidebar:
        st.markdown("### 2 · Choose sample period")
        start_s, end_s = st.select_slider(
            "Start and end month", options=months, value=(months[0], months[-1]),
            format_func=lambda m: pd.Timestamp(m + "-01").strftime("%b %Y"),
        )
        start, end = pd.Timestamp(start_s + "-01"), pd.Timestamp(end_s + "-01")
        st.caption(f"Data with both CPIs and exchange rates: "
                   f"{first_valid:%b %Y} – {last_valid:%b %Y}")
        st.markdown("### 3 · Choose chart")
        view = st.radio("Show", ["Real", "Nominal", "Both"], index=2, horizontal=True)
        for d in (d1, d2):
            if d["note"]:
                st.caption(d["note"])

    sample, rer_star, n_valid = apply_sample(df, start, end)
    if n_valid == 0:
        st.warning("No valid months in this sample. Widen the date range.")
        st.stop()

    last = sample[sample["Valid"]].iloc[-1]
    last_month = sample[sample["Valid"]].index[-1]
    dev = last["Real_Index"] - 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average RER (RER*)", f"{rer_star:,.4f}")
    m2.metric(f"Real Index · {last_month:%b %Y}", f"{last['Real_Index']:.1f}",
              f"{dev:+.1f}% vs average")
    m3.metric(f"Nominal S · {last_month:%b %Y}", f"{last['S']:,.4f}",
              help=f"{name2} currency per 1 {name1} currency")
    m4.metric("Valid months in sample", f"{n_valid:,}",
              help=f"Out of {len(sample):,} months between {start:%b %Y} and {end:%b %Y}")

    if dev > 0:
        reading = (f"{name1}'s currency is **{dev:.1f}% stronger in real terms** than its "
                   f"average over the sample: {name1} goods are relatively expensive versus "
                   f"{name2}. If the real rate reverts to its average, expect a real "
                   f"depreciation of {name1}'s currency.")
    else:
        reading = (f"{name1}'s currency is **{abs(dev):.1f}% weaker in real terms** than its "
                   f"average over the sample: {name1} goods are relatively cheap versus "
                   f"{name2}. If the real rate reverts to its average, expect a real "
                   f"appreciation of {name1}'s currency.")
    st.markdown(f'<div class="hal-box">{reading}</div>', unsafe_allow_html=True)

    st.plotly_chart(make_chart(sample, view, name1, name2), width="stretch")

    gaps = len(sample) - n_valid
    if gaps:
        st.caption(f"⚠️ {gaps} month(s) in this sample are missing CPI or exchange-rate data "
                   f"and are excluded from RER* (shown as gaps in the chart).")

    # Download
    out = sample.reset_index().rename(columns={
        "E1_per_USD": f"E1_{code1}_per_USD",
        "E2_per_USD": f"E2_{code2}_per_USD",
        "P1_CPI": f"P1_CPI_{code1}",
        "P2_CPI": f"P2_CPI_{code2}",
        "S": f"S_{code2}_per_{code1}",
    })
    out["Month"] = out["Month"].dt.strftime("%Y-%m")
    out.insert(1, "Country1", name1)
    out.insert(2, "Country2", name2)
    st.download_button(
        "⬇ Download CSV (sample period)",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"halconomics_{code1}_{code2}_{start:%Y%m}_{end:%Y%m}.csv",
        mime="text/csv",
    )

    with st.expander("See the data table"):
        st.dataframe(out, width="stretch", hide_index=True)

    with st.expander("How it's calculated"):
        st.markdown(
            f"""
- **IMF exchange rates** are local currency per USD: E1 ({code1}) and E2 ({code2}).
- **Nominal rate:** S(t) = E2(t) / E1(t) = {name2} currency per 1 {name1} currency.
- **Real exchange rate:** RER(t) = S(t) × P1(t) / P2(t), using monthly CPIs.
- **RER\\*** = average of RER(t) over the valid months between your start and end dates
  (recomputed whenever you move the slider).
- **Real Index(t)** = 100 × RER(t) / RER\\*. Above 100 means {name1}'s currency is
  stronger in real terms than its sample average; below 100, weaker.
- Sources: IMF SDMX 3.0 API, dataflows `{AGENCY}/{CPI_FLOW}` (key `ISO3.CPI._T.IX.M`) and
  `{AGENCY}/{ER_FLOW}` (key `ISO3.XDC_USD.PA_RT.M`). Euro area = `{EURO_AREA_CODE}`.
            """
        )

    st.caption(f"Data: International Monetary Fund · Retrieved {date.today():%d %b %Y}")


if __name__ == "__main__":
    main()
