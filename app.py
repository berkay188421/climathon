"""OptiHydro AI — transparent, constraint-based industrial fuel-blending demo.

Run locally with: streamlit run app.py

This application is a scenario simulator. It does not connect to Siemens systems,
read real plant telemetry, certify a fuel train, or issue commands to a PLC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="OptiHydro AI | Endüstriyel Enerji Dijital İkizi",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Design system
# -----------------------------------------------------------------------------
TEAL = "#009999"
NAVY = "#002B49"
INK = "#17324D"
MUTED = "#62758A"
PALE = "#F2F4F7"
GREEN = "#1E9B72"
AMBER = "#D99119"
RED = "#C94B50"

st.markdown(
    f"""
    <style>
      :root {{ --oh-teal: {TEAL}; --oh-navy: {NAVY}; --oh-pale: {PALE}; }}
      .stApp {{ background: {PALE}; color: {INK}; }}
      [data-testid="stHeader"] {{ background: rgba(242,244,247,.92); }}
      [data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #e4eaf0; }}
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {{ color: {NAVY}; }}
      .block-container {{ padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1500px; }}
      [data-testid="stMetric"] {{ background: #fff; border: 1px solid #e5ebf0;
        border-radius: 15px; padding: 17px 18px; box-shadow: 0 4px 18px rgba(0,43,73,.045); }}
      [data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 650; }}
      [data-testid="stMetricValue"] {{ color: {NAVY}; }}
      [data-testid="stMetricDelta"] {{ font-size: .83rem; }}
      div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {TEAL}; border-bottom-color: {TEAL}; font-weight: 700;
      }}
      .hero {{ background: linear-gradient(112deg, {NAVY} 0%, #064b68 65%, {TEAL} 145%);
        color: white; border-radius: 22px; padding: 27px 32px; margin: 3px 0 18px;
        box-shadow: 0 12px 30px rgba(0,43,73,.13); }}
      .hero-kicker {{ color: #9ee6df; text-transform: uppercase; letter-spacing: .15em;
        font-size: .73rem; font-weight: 800; }}
      .hero h1 {{ color: white; font-size: clamp(2rem, 3.3vw, 3rem); margin: 7px 0 4px; }}
      .hero p {{ color: #e1edf2; font-size: 1.02rem; margin: 0; max-width: 900px; }}
      .hero-pill {{ display: inline-block; border: 1px solid rgba(255,255,255,.3);
        background: rgba(255,255,255,.09); border-radius: 999px; padding: 6px 11px;
        margin: 16px 7px 0 0; font-size: .78rem; color: white; }}
      .section-label {{ color: {TEAL}; text-transform: uppercase; font-size: .72rem;
        letter-spacing: .13em; font-weight: 800; margin-bottom: 3px; }}
      .soft-card {{ background: white; border: 1px solid #e5ebf0; border-radius: 15px;
        padding: 17px 19px; height: 100%; }}
      .soft-card h4 {{ color: {NAVY}; margin: 0 0 7px; font-size: 1rem; }}
      .soft-card p {{ color: {MUTED}; margin: 0; line-height: 1.5; font-size: .91rem; }}
      .small-note {{ color: {MUTED}; font-size: .82rem; line-height: 1.5; }}
      .status-chip {{ display: inline-block; border-radius: 999px; padding: 5px 10px;
        font-weight: 750; font-size: .75rem; background: #eaf5f3; color: #087c72; }}
      .status-chip.amber {{ background: #fff4df; color: #9a650b; }}
      .status-chip.gray {{ background: #edf1f5; color: #526577; }}
      [data-testid="stDownloadButton"] button {{ border-color: {TEAL}; color: {NAVY}; }}
      hr {{ border-color: #e3e9ef; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Constants and transparent calculation model
# -----------------------------------------------------------------------------
HOURS = np.arange(24)
H2_DENSITY_KG_PER_NM3 = 0.08988  # H2 at the defined normal-volume reference
H2_LHV_KWH_PER_KG = 33.3
H2_LHV_MJ_PER_NM3 = H2_LHV_KWH_PER_KG * 3.6 * H2_DENSITY_KG_PER_NM3
H2_WATER_STOICH_KG_PER_KG = 9.0
H2_OXYGEN_STOICH_KG_PER_KG = 8.0
IPCC_NG_CO2_KG_PER_GJ = 56.1
DEFAULT_NG_LHV_MJ_PER_NM3 = 35.8
DEFAULT_NG_CO2_KG_PER_NM3 = (
    IPCC_NG_CO2_KG_PER_GJ * DEFAULT_NG_LHV_MJ_PER_NM3 / 1000.0
)

SECTOR_DEFAULTS = {
    "Cam": 1200,
    "Çelik": 1450,
    "Seramik": 800,
    "Çimento": 1800,
}


def blend_flows(
    h2_volume_fraction: float,
    baseline_ng_nm3_h: float,
    ng_lhv_mj_nm3: float,
    h2_lhv_mj_nm3: float = H2_LHV_MJ_PER_NM3,
) -> tuple[float, float, float]:
    """Return total blend, H2, and NG flows at the same thermal input as baseline."""
    x = float(np.clip(h2_volume_fraction, 0.0, 1.0))
    mixed_lhv = (1.0 - x) * ng_lhv_mj_nm3 + x * h2_lhv_mj_nm3
    if baseline_ng_nm3_h <= 0 or mixed_lhv <= 0:
        return 0.0, 0.0, 0.0
    total_blend = baseline_ng_nm3_h * ng_lhv_mj_nm3 / mixed_lhv
    h2_flow = x * total_blend
    ng_flow = (1.0 - x) * total_blend
    return total_blend, h2_flow, ng_flow


def blend_cap_from_h2_flow(
    h2_limit_nm3_h: float,
    baseline_ng_nm3_h: float,
    ng_lhv_mj_nm3: float,
    h2_lhv_mj_nm3: float = H2_LHV_MJ_PER_NM3,
) -> float:
    """Invert the constant-thermal-input blend relation for a supply cap."""
    if h2_limit_nm3_h <= 0 or baseline_ng_nm3_h <= 0:
        return 0.0
    ratio = h2_limit_nm3_h / baseline_ng_nm3_h
    denominator = ng_lhv_mj_nm3 + ratio * (ng_lhv_mj_nm3 - h2_lhv_mj_nm3)
    if denominator <= 0:
        return 0.0
    return float(np.clip(ratio * ng_lhv_mj_nm3 / denominator, 0.0, 1.0))


def renewable_availability_profile(peak_index_pct: float) -> np.ndarray:
    """Illustrative normalized daily availability curve, not a grid forecast."""
    solar_shape = np.clip(np.sin((HOURS - 6.0) * np.pi / 12.0), 0.0, 1.0) ** 1.55
    wind_shape = np.clip(
        0.58 + 0.22 * np.cos((HOURS - 2.0) * 2.0 * np.pi / 24.0), 0.18, 0.92
    )
    composite = 0.72 * solar_shape + 0.28 * wind_shape
    normalized = composite / composite.max() if composite.max() else composite
    return np.clip(normalized * peak_index_pct / 100.0, 0.0, 1.0)


def optimize_blend_schedule(
    availability: np.ndarray,
    safety_cap_fraction: float,
    supply_cap_fraction: float,
    ng_lhv_mj_nm3: float,
    h2_lhv_mj_nm3: float = H2_LHV_MJ_PER_NM3,
) -> np.ndarray:
    """Grid-search dispatch with availability, blend, supply and ramp penalty.

    The objective rewards energy-equivalent natural-gas displacement during
    higher-availability hours and applies a modest penalty to setpoint changes.
    This is an explainable deterministic optimizer, not a trained AI forecast.
    """
    upper_bound = max(0.0, min(safety_cap_fraction, supply_cap_fraction))
    step = 0.0025  # 0.25 vol-percentage-point resolution
    candidates = np.arange(0.0, upper_bound + step / 2.0, step)
    if candidates.size == 0 or candidates[-1] < upper_bound - 1e-9:
        candidates = np.append(candidates, upper_bound)
    candidates = np.unique(np.append(candidates, upper_bound))
    state_count = len(candidates)
    hour_count = len(availability)
    scores = np.full((hour_count, state_count), -np.inf)
    parents = np.full((hour_count, state_count), -1, dtype=int)
    ramp_penalty = 0.10

    for hour in range(hour_count):
        allowed = min(upper_bound, safety_cap_fraction * float(availability[hour]))
        feasible = candidates <= allowed + 1e-10
        energy_share = np.zeros(state_count)
        denominator = candidates * h2_lhv_mj_nm3 + (1.0 - candidates) * ng_lhv_mj_nm3
        np.divide(
            candidates * h2_lhv_mj_nm3,
            denominator,
            out=energy_share,
            where=denominator > 0,
        )
        reward = (0.25 + 0.75 * float(availability[hour])) * energy_share * 100.0

        if hour == 0:
            scores[hour, feasible] = (
                reward[feasible] - ramp_penalty * np.abs(candidates[feasible]) * 100.0
            )
            continue

        transition = scores[hour - 1, :, None] - ramp_penalty * np.abs(
            candidates[None, :] - candidates[:, None]
        ) * 100.0
        best_parent = np.argmax(transition, axis=0)
        best_score = transition[best_parent, np.arange(state_count)]
        scores[hour, feasible] = reward[feasible] + best_score[feasible]
        parents[hour, feasible] = best_parent[feasible]

    schedule = np.zeros(hour_count)
    state = int(np.argmax(scores[-1]))
    for hour in range(hour_count - 1, -1, -1):
        schedule[hour] = candidates[state]
        if hour > 0:
            state = int(parents[hour, state])
    return schedule


def create_scenario(
    *,
    sector: str,
    gas_flow_nm3_h: float,
    peak_index_pct: float,
    safety_cap_pct: float,
    external_h2_supply_kg_h: float,
    condensate_l_h: float,
    water_intensity_l_kg: float,
    ng_lhv_mj_nm3: float,
    ng_co2_kg_nm3: float,
    operating_days: int,
    electrolyzer_kwh_kg: float,
    oxygen_demand_kg_h: float,
    oxygen_use_efficiency: float,
    aeration_fan_kw: float,
    furnace_temperature_c: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    availability = renewable_availability_profile(peak_index_pct)
    safety_cap = safety_cap_pct / 100.0

    # External H2 availability and recovered-water treatment capacity both bind.
    water_limited_h2_kg_h = condensate_l_h / water_intensity_l_kg if water_intensity_l_kg else 0.0
    available_h2_kg_h = min(external_h2_supply_kg_h, water_limited_h2_kg_h)
    available_h2_nm3_h = available_h2_kg_h / H2_DENSITY_KG_PER_NM3
    supply_cap = blend_cap_from_h2_flow(
        available_h2_nm3_h,
        gas_flow_nm3_h,
        ng_lhv_mj_nm3,
    )
    blend = optimize_blend_schedule(availability, safety_cap, supply_cap, ng_lhv_mj_nm3)

    rows: list[dict[str, Any]] = []
    for hour, blend_fraction in enumerate(blend):
        total_mix, h2_nm3_h, ng_nm3_h = blend_flows(
            blend_fraction, gas_flow_nm3_h, ng_lhv_mj_nm3
        )
        h2_kg_h = h2_nm3_h * H2_DENSITY_KG_PER_NM3
        ng_avoided_nm3_h = max(0.0, gas_flow_nm3_h - ng_nm3_h)
        process_water_l_h = h2_kg_h * water_intensity_l_kg
        oxygen_generated_kg_h = h2_kg_h * H2_OXYGEN_STOICH_KG_PER_KG
        oxygen_credited_kg_h = min(
            oxygen_generated_kg_h * oxygen_use_efficiency,
            oxygen_demand_kg_h,
        )
        oxygen_displacement_fraction = (
            oxygen_credited_kg_h / oxygen_demand_kg_h if oxygen_demand_kg_h > 0 else 0.0
        )
        rows.append(
            {
                "Saat": f"{hour:02d}:00",
                "Saat indeksi": hour,
                "Yeşil enerji erişilebilirlik indeksi (%)": availability[hour] * 100.0,
                "H2 harmanlama (vol %)": blend_fraction * 100.0,
                "Baz doğalgaz (Nm³/saat)": gas_flow_nm3_h,
                "Senaryo doğalgazı (Nm³/saat)": ng_nm3_h,
                "Önlenen doğalgaz (Nm³/saat)": ng_avoided_nm3_h,
                "Karışım toplamı (Nm³/saat)": total_mix,
                "H2 beslemesi (Nm³/saat)": h2_nm3_h,
                "H2 tüketimi (kg/saat)": h2_kg_h,
                "Proses suyu (L/saat)": process_water_l_h,
                "Üretilen O2 (kg/saat)": oxygen_generated_kg_h,
                "Arıtmada kredilendirilen O2 (kg/saat)": oxygen_credited_kg_h,
                "Elektroliz elektrik ihtiyacı (kWh/saat)": h2_kg_h * electrolyzer_kwh_kg,
                "Havalandırma fan gücü senaryosu (kW)": aeration_fan_kw
                * oxygen_displacement_fraction,
                "Önlenen doğrudan CO2 (kg/saat)": ng_avoided_nm3_h * ng_co2_kg_nm3,
            }
        )

    frame = pd.DataFrame(rows)
    baseline_gas_day = gas_flow_nm3_h * 24.0
    saved_gas_day = float(frame["Önlenen doğalgaz (Nm³/saat)"].sum())
    h2_day_kg = float(frame["H2 tüketimi (kg/saat)"].sum())
    water_day_l = float(frame["Proses suyu (L/saat)"].sum())
    oxygen_day_kg = float(frame["Üretilen O2 (kg/saat)"].sum())
    oxygen_used_day_kg = float(frame["Arıtmada kredilendirilen O2 (kg/saat)"].sum())
    fan_saved_kwh_day = float(frame["Havalandırma fan gücü senaryosu (kW)"].sum())
    avoided_co2_day_kg = saved_gas_day * ng_co2_kg_nm3
    baseline_emissions_day_kg = baseline_gas_day * ng_co2_kg_nm3
    projected_gas_day = baseline_gas_day - saved_gas_day
    mean_blend_pct = float(frame["H2 harmanlama (vol %)"].mean())
    peak_blend_pct = float(frame["H2 harmanlama (vol %)"].max())
    green_h2_power_day_kwh = h2_day_kg * electrolyzer_kwh_kg

    summary = {
        "baseline_gas_day_nm3": baseline_gas_day,
        "saved_gas_day_nm3": saved_gas_day,
        "projected_gas_day_nm3": projected_gas_day,
        "gas_saving_pct": saved_gas_day / baseline_gas_day * 100.0 if baseline_gas_day else 0.0,
        "h2_day_kg": h2_day_kg,
        "water_day_l": water_day_l,
        "water_available_day_l": condensate_l_h * 24.0,
        "water_coverage_pct": water_day_l / (condensate_l_h * 24.0) * 100.0
        if condensate_l_h > 0
        else 0.0,
        "oxygen_day_kg": oxygen_day_kg,
        "oxygen_used_day_kg": oxygen_used_day_kg,
        "oxygen_demand_day_kg": oxygen_demand_kg_h * 24.0,
        "fan_saved_kwh_day": fan_saved_kwh_day,
        "fan_saved_avg_kw": fan_saved_kwh_day / 24.0,
        "fan_saving_pct": fan_saved_kwh_day / (aeration_fan_kw * 24.0) * 100.0
        if aeration_fan_kw > 0
        else 0.0,
        "avoided_co2_day_kg": avoided_co2_day_kg,
        "avoided_co2_year_t": avoided_co2_day_kg * operating_days / 1000.0,
        "baseline_co2_day_t": baseline_emissions_day_kg / 1000.0,
        "projected_co2_day_t": (baseline_emissions_day_kg - avoided_co2_day_kg) / 1000.0,
        "mean_blend_pct": mean_blend_pct,
        "peak_blend_pct": peak_blend_pct,
        "green_h2_power_day_mwh": green_h2_power_day_kwh / 1000.0,
        "operating_days": float(operating_days),
        "furnace_temperature_c": float(furnace_temperature_c),
        "available_h2_kg_h": available_h2_kg_h,
        "water_limited_h2_kg_h": water_limited_h2_kg_h,
    }
    return frame, summary


def format_number(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=12),
        margin=dict(l=10, r=18, t=58, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor=NAVY, font_color="white", bordercolor=NAVY),
    )
    fig.update_xaxes(showgrid=False, linecolor="#dfe7ed", zeroline=False)
    fig.update_yaxes(gridcolor="#edf1f5", zeroline=False)
    return fig


# -----------------------------------------------------------------------------
# Sidebar — scenario inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='section-label'>OPTİMİZASYON GİRDİLERİ</div>", unsafe_allow_html=True)
    st.title("Senaryo ayarları")
    st.caption("Değerleri değiştirin; simülasyon ve KPI'lar anında güncellenir.")

    sector = st.selectbox("Fırın sektörü", list(SECTOR_DEFAULTS.keys()), index=0)
    gas_flow = st.number_input(
        "Baz doğalgaz tüketimi (Nm³/saat)",
        min_value=0.0,
        max_value=10000.0,
        value=float(SECTOR_DEFAULTS[sector]),
        step=50.0,
        key=f"gas_flow_{sector}",
        help="Saatlik normal metreküp cinsinden baz yakıt debisi. Tesis ölçümüyle değiştirin.",
    )
    green_peak = st.slider(
        "Yeşil enerji erişim piki (indeks, %)",
        min_value=0,
        max_value=100,
        value=72,
        help="Sentetik günlük erişilebilirlik eğrisinin tepe değeridir; gerçek şebeke tahmini değildir.",
    )
    safety_cap_pct = st.slider(
        "Simülasyon harman üst sınırı (H₂ vol %)",
        min_value=0,
        max_value=20,
        value=20,
        help="Model limiti. Fırın/brülör OEM onayı ve proses emniyet doğrulamasının yerine geçmez.",
    )
    h2_supply = st.number_input(
        "Kullanılabilir düşük karbonlu H₂ beslemesi (kg/saat)",
        min_value=0.0,
        max_value=2000.0,
        value=25.0,
        step=1.0,
        help="Harici H₂ üretim/tedarik sisteminden gelen varsayılan kullanılabilir debi.",
    )
    condensate = st.number_input(
        "Kullanılabilir chiller yoğuşma suyu (L/saat)",
        min_value=0.0,
        max_value=100000.0,
        value=220.0,
        step=10.0,
    )

    with st.expander("Gelişmiş mühendislik varsayımları", expanded=False):
        ng_lhv = st.number_input(
            "Doğalgaz alt ısıl değeri (MJ/Nm³)",
            min_value=20.0,
            max_value=50.0,
            value=DEFAULT_NG_LHV_MJ_PER_NM3,
            step=0.1,
        )
        ng_co2_factor = st.number_input(
            "Doğalgaz doğrudan CO₂ faktörü (kg/Nm³)",
            min_value=0.0,
            max_value=5.0,
            value=DEFAULT_NG_CO2_KG_PER_NM3,
            step=0.01,
            format="%.3f",
            help="Varsayılan: IPCC 56,1 kg CO₂/GJ × 35,8 MJ/Nm³. Yerel yakıt analiziyle güncelleyin.",
        )
        water_intensity = st.number_input(
            "Toplam elektroliz suyu ihtiyacı (L/kg H₂)",
            min_value=9.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help="9 L/kg yaklaşık stokiyometrik alt sınırdır; seçilen değer proses/pürifikasyon payını temsil eder.",
        )
        electrolyzer_kwh_kg = st.number_input(
            "Elektroliz elektrik tüketimi (kWh/kg H₂)",
            min_value=35.0,
            max_value=90.0,
            value=55.0,
            step=1.0,
        )
        operating_days = st.number_input(
            "Yıllık işletme günü", min_value=1, max_value=365, value=330, step=1
        )
        furnace_temperature = st.number_input(
            "Fırın sıcaklığı / senaryo (°C)", min_value=500, max_value=1800, value=1050, step=10
        )
        st.markdown("**Atık su arıtma oksijen senaryosu**")
        oxygen_demand = st.number_input(
            "Tesis çözünmüş O₂ ihtiyacı (kg/saat)",
            min_value=0.0,
            max_value=100000.0,
            value=450.0,
            step=10.0,
        )
        oxygen_use_pct = st.slider(
            "Üretilen O₂ için etkin kullanım varsayımı (%)",
            min_value=0,
            max_value=100,
            value=40,
            help="Saf O₂'nin transfer/kullanım verimini temsil eden senaryo girdisi.",
        )
        aeration_fan_kw = st.number_input(
            "Arıtma havalandırma fanlarının baz gücü (kW)",
            min_value=0.0,
            max_value=100000.0,
            value=350.0,
            step=10.0,
        )

    st.divider()
    st.markdown(
        "<span class='status-chip gray'>Senaryo modu · canlı tesis bağlantısı yok</span>",
        unsafe_allow_html=True,
    )


scenario, totals = create_scenario(
    sector=sector,
    gas_flow_nm3_h=gas_flow,
    peak_index_pct=green_peak,
    safety_cap_pct=safety_cap_pct,
    external_h2_supply_kg_h=h2_supply,
    condensate_l_h=condensate,
    water_intensity_l_kg=water_intensity,
    ng_lhv_mj_nm3=ng_lhv,
    ng_co2_kg_nm3=ng_co2_factor,
    operating_days=int(operating_days),
    electrolyzer_kwh_kg=electrolyzer_kwh_kg,
    oxygen_demand_kg_h=oxygen_demand,
    oxygen_use_efficiency=oxygen_use_pct / 100.0,
    aeration_fan_kw=aeration_fan_kw,
    furnace_temperature_c=int(furnace_temperature),
)


# -----------------------------------------------------------------------------
# Header and headline KPIs
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <section class="hero">
      <div class="hero-kicker">Industrial energy digital twin · decision support</div>
      <h1>OptiHydro AI</h1>
      <p>Fırın yakıt karışımını; yeşil enerji erişimi, güvenlik için tanımlanmış harman limiti,
      H₂ beslemesi ve geri kazanılmış su kapasitesiyle birlikte senaryolaştırın.</p>
      <span class="hero-pill">{sector} · {format_number(gas_flow)} Nm³/saat baz yük</span>
      <span class="hero-pill">Açıklanabilir, kısıtlı dispatch</span>
      <span class="hero-pill">Canlı saha kontrolü değildir</span>
    </section>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "**Emniyet sınırı:** Bu MVP, fırın/brülör güvenlik sistemi değildir ve PLC'ye komut göndermez. "
    "H₂ üst sınırı yalnızca senaryo parametresidir; gerçek harman oranı OEM uygunluğu, "
    "gaz treni tasarımı, alev denetimi, tehlikeli alan sınıflandırması ve yetkin mühendislik "
    "onaylarıyla belirlenmelidir.",
    icon="⚠️",
)

st.markdown("<div class='section-label'>BUGÜNÜN SENARYOSU</div>", unsafe_allow_html=True)
st.subheader("Ölçülebilir işletme etkisi")

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Ortalama H₂ karışımı",
    f"{format_number(totals['mean_blend_pct'], 1)} vol %",
    f"pik {format_number(totals['peak_blend_pct'], 1)} vol %",
)
k2.metric(
    "Günlük doğalgaz ikamesi",
    f"{format_number(totals['saved_gas_day_nm3'])} Nm³",
    f"bazın %{format_number(totals['gas_saving_pct'], 1)}'i",
)
k3.metric(
    "Yıllık önlenen doğrudan CO₂",
    f"{format_number(totals['avoided_co2_year_t'])} ton",
    f"{int(operating_days)} işletme günü/yıl",
)
k4.metric(
    "Arıtma fanı senaryo tasarrufu",
    f"{format_number(totals['fan_saving_pct'], 1)}%",
    f"ortalama {format_number(totals['fan_saved_avg_kw'], 1)} kW",
)

st.caption(
    "CO₂ metriği doğrudan yanma emisyonu farkıdır; H₂'nin yaşam döngüsü emisyonlarını, "
    "elektroliz elektriğini, NOₓ'i ve ekipman verim değişimini içermez."
)

tab_dispatch, tab_circular, tab_edge, tab_model = st.tabs(
    ["24 saat dispatch", "Çifte dairesel ekonomi", "Edge telemetri görünümü", "Model ve kabuller"]
)


# -----------------------------------------------------------------------------
# Tab 1 — dispatch and emissions
# -----------------------------------------------------------------------------
with tab_dispatch:
    left, right = st.columns([1.65, 1])
    with left:
        st.markdown("<div class='section-label'>ENERJİ DENGESİ</div>", unsafe_allow_html=True)
        st.subheader("Saatlik yakıt ikamesi")
        chart = make_subplots(specs=[[{"secondary_y": True}]])
        chart.add_trace(
            go.Scatter(
                x=scenario["Saat"],
                y=scenario["Baz doğalgaz (Nm³/saat)"],
                name="Baz doğalgaz debisi",
                mode="lines",
                line=dict(color="#9aa9b6", width=2, dash="dot"),
                hovertemplate="%{x}<br>Baz: %{y:,.0f} Nm³/saat<extra></extra>",
            ),
            secondary_y=False,
        )
        chart.add_trace(
            go.Scatter(
                x=scenario["Saat"],
                y=scenario["Senaryo doğalgazı (Nm³/saat)"],
                name="Senaryo doğalgazı",
                mode="lines",
                line=dict(color=TEAL, width=3),
                fill="tonexty",
                fillcolor="rgba(0,153,153,.10)",
                hovertemplate="%{x}<br>Doğalgaz: %{y:,.1f} Nm³/saat<extra></extra>",
            ),
            secondary_y=False,
        )
        chart.add_trace(
            go.Scatter(
                x=scenario["Saat"],
                y=scenario["H2 harmanlama (vol %)"],
                name="H₂ hacimsel oranı",
                mode="lines+markers",
                marker=dict(size=6, color=AMBER),
                line=dict(color=AMBER, width=2.5),
                hovertemplate="%{x}<br>H₂: %{y:.2f} vol %<extra></extra>",
            ),
            secondary_y=True,
        )
        chart.update_yaxes(title_text="Doğalgaz debisi (Nm³/saat)", secondary_y=False)
        chart.update_yaxes(title_text="H₂ oranı (vol %)", secondary_y=True, rangemode="tozero")
        chart.update_xaxes(title_text="Saat")
        chart.update_layout(title="Aynı termal girdiyi koruyan 24 saatlik senaryo")
        st.plotly_chart(style_figure(chart, 410), width="stretch")

    with right:
        st.markdown("<div class='section-label'>EMİSYON KARŞILAŞTIRMASI</div>", unsafe_allow_html=True)
        st.subheader("Doğrudan yanma CO₂'si")
        emissions_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Baz", "H₂ senaryosu"],
                    y=[totals["baseline_co2_day_t"], totals["projected_co2_day_t"]],
                    marker_color=["#9aa9b6", TEAL],
                    text=[
                        f"{totals['baseline_co2_day_t']:.1f} t",
                        f"{totals['projected_co2_day_t']:.1f} t",
                    ],
                    textposition="outside",
                    hovertemplate="%{x}<br>%{y:.2f} ton CO₂/gün<extra></extra>",
                )
            ]
        )
        emissions_fig.update_layout(title="ton CO₂ / gün", showlegend=False, yaxis_title="ton CO₂/gün")
        emissions_fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(style_figure(emissions_fig, 330), width="stretch")
        st.markdown(
            f"<div class='soft-card'><h4>H₂ için elektrik gereksinimi</h4>"
            f"<p>{format_number(totals['green_h2_power_day_mwh'], 2)} MWh/gün · "
            f"{format_number(totals['h2_day_kg'], 1)} kg H₂/gün × {format_number(electrolyzer_kwh_kg)} kWh/kg</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Saatlik dispatch verisi")
    st.dataframe(
        scenario[
            [
                "Saat",
                "Yeşil enerji erişilebilirlik indeksi (%)",
                "H2 harmanlama (vol %)",
                "Senaryo doğalgazı (Nm³/saat)",
                "Önlenen doğalgaz (Nm³/saat)",
                "H2 tüketimi (kg/saat)",
                "Proses suyu (L/saat)",
            ]
        ].round(2),
        width="stretch",
        hide_index=True,
    )
    csv_bytes = scenario.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Saatlik senaryoyu CSV indir",
        data=csv_bytes,
        file_name="optihydro_24_saat_senaryosu.csv",
        mime="text/csv",
        icon="⬇️",
    )


# -----------------------------------------------------------------------------
# Tab 2 — water and oxygen mass balance
# -----------------------------------------------------------------------------
with tab_circular:
    st.markdown("<div class='section-label'>KÜTLE DENGESİ</div>", unsafe_allow_html=True)
    st.subheader("Yoğuşma suyu → H₂ → O₂ değerlendirme senaryosu")

    water_used = totals["water_day_l"]
    water_available = totals["water_available_day_l"]
    water_shortfall = max(0.0, water_used - water_available)
    coverage_caption = (
        f"Yoğuşma suyu ihtiyacının %{format_number(totals['water_coverage_pct'], 1)}'i kullanılıyor."
        if water_available > 0
        else "Yoğuşma suyu yok; dispatch H₂ kullanımını sıfıra indirir."
    )
    water_col, oxygen_col = st.columns(2)
    with water_col:
        water_fig = go.Figure(
            go.Bar(
                x=["Geri kazanılabilir yoğuşma", "Senaryo proses suyu"],
                y=[water_available, water_used],
                marker_color=["#a9d9d4", TEAL],
                text=[f"{water_available:,.0f} L", f"{water_used:,.0f} L"],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:,.0f} L/gün<extra></extra>",
            )
        )
        water_fig.update_layout(title="Su bütçesi · L/gün", showlegend=False, yaxis_title="L/gün")
        water_fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(style_figure(water_fig, 330), width="stretch")
        st.caption(coverage_caption)
        st.metric("Kullanılmayan / eksik su", f"{format_number(abs(water_available-water_used))} L/gün")

    with oxygen_col:
        oxygen_fig = go.Figure(
            data=[
                go.Bar(
                    name="Elektroliz O₂ üretimi",
                    x=["O₂ kütle dengesi"],
                    y=[totals["oxygen_day_kg"]],
                    marker_color="#9bc9e2",
                    hovertemplate="Üretilen: %{y:,.0f} kg/gün<extra></extra>",
                ),
                go.Bar(
                    name="Arıtmada kredilendirilen",
                    x=["O₂ kütle dengesi"],
                    y=[totals["oxygen_used_day_kg"]],
                    marker_color=TEAL,
                    hovertemplate="Kredilendirilen: %{y:,.0f} kg/gün<extra></extra>",
                ),
            ]
        )
        oxygen_fig.update_layout(
            title="Saf O₂ · kg/gün",
            barmode="group",
            yaxis_title="kg/gün",
            legend=dict(orientation="h", y=1.13, x=0),
        )
        oxygen_fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(style_figure(oxygen_fig, 330), width="stretch")
        st.caption(
            f"Arıtma ihtiyacına göre O₂ kullanımı: {format_number(totals['oxygen_used_day_kg'], 0)} / "
            f"{format_number(totals['oxygen_demand_day_kg'], 0)} kg/gün. Fan etkisi doğrusal ikame varsayımıyla sınırlandırılmıştır."
        )
        st.metric(
            "Fan enerjisi senaryo farkı",
            f"{format_number(totals['fan_saved_kwh_day'])} kWh/gün",
            f"%{format_number(totals['fan_saving_pct'], 1)} model tahmini",
        )

    if water_shortfall > 1e-6:
        st.error("Su bütçesi bu senaryoda yetersiz. H₂ dispatch su kapasitesiyle sınırlandırılmıştır.")
    elif h2_supply <= 0:
        st.info("Harici H₂ beslemesi 0 kg/saat seçildiği için su geri kazanımı tek başına H₂ üretimi oluşturmaz.")
    else:
        st.success("H₂ dispatch, tanımlı harici besleme ve saatlik yoğuşma suyu üst sınırlarını aşmıyor.")

    st.markdown(
        "<div class='soft-card'><h4>Kapsam notu</h4><p>Elektrolizör ve oksijen kullanım altyapısı bu yazılım senaryosunun dışında ayrı "
        "proses ekipmanlarıdır. Yoğuşma suyu doğrudan elektrolizör kalitesinde varsayılmaz; ön arıtma, "
        "iletkenlik kontrolü ve su şartlandırma gerekir. Fan tasarrufu saha kütle transferi ve blower eğrileriyle doğrulanmalıdır.</p></div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Tab 3 — Edge-facing model view, carefully separated from live telemetry
# -----------------------------------------------------------------------------
with tab_edge:
    st.markdown("<div class='section-label'>SİEMENS EDGE'E UYGUN ÜRÜN KONSEPTİ</div>", unsafe_allow_html=True)
    st.subheader("Dijital ikiz görünümü")
    st.markdown(
        "<span class='status-chip amber'>Sentetik senaryo verisi</span> "
        "<span class='status-chip gray'>Industrial Edge / Building X bağlantısı etkin değil</span>",
        unsafe_allow_html=True,
    )
    st.write(
        "Bu ekranda gösterilenler kullanıcı girdilerinden hesaplanan senaryo çıktılarıdır. "
        "Bir PLC, burner management system, UV alev dedektörü veya Building X tenant'ından veri okunmuyor."
    )

    edge_a, edge_b, edge_c = st.columns(3)
    edge_a.metric("Fırın senaryo sıcaklığı", f"{int(furnace_temperature)} °C", "kullanıcı girdisi")
    edge_b.metric("H₂ dispatch ortalaması", f"{format_number(totals['mean_blend_pct'], 2)} vol %", "model çıktısı")
    edge_c.metric("Öngörülen doğalgaz debisi", f"{format_number(totals['projected_gas_day_nm3']/24, 1)} Nm³/saat", "sabit termal girdi")

    st.markdown("#### Model çıkışları / entegrasyon gereksinimleri")
    telemetry_rows = [
        ("Fırın sıcaklığı", f"{int(furnace_temperature)} °C", "Senaryo girdisi", "Gerçek zamanlı sıcaklık transmitter'i"),
        ("H₂ hacimsel setpoint", f"{format_number(totals['mean_blend_pct'], 2)} vol % ortalama", "Optimizer çıktısı", "Gaz analizörü + OEM izinli ratio control"),
        ("Doğalgaz debisi", f"{format_number(totals['projected_gas_day_nm3']/24, 1)} Nm³/saat senaryo", "Enerji dengesi", "Kalibre edilmiş gaz debimetresi"),
        ("H₂ beslemesi", f"{format_number(totals['available_h2_kg_h'], 1)} kg/saat üst sınır", "Kullanıcı girdisi", "H₂ debimetresi, basınç ve kalite ölçümü"),
        ("Wobbe indeksi", "Hesaplanmıyor", "Gaz bileşimi yok", "Doğal gaz kompozisyonu ve bağıl yoğunluk"),
        ("UV alev / flashback durumu", "Değerlendirilmiyor", "Emniyet sistemi dışında", "Sertifikalı alev denetimi ve burner OEM limitleri"),
        ("PLC yanıtı", "Ölçülmüyor", "PLC bağlantısı yok", "Edge connector ve onaylı read-only veri sözleşmesi"),
        ("Chiller suyu", f"{format_number(condensate)} L/saat kapasite girdisi", "Kullanıcı girdisi", "Debi + iletkenlik + su kalite analizörü"),
    ]
    telemetry_df = pd.DataFrame(
        telemetry_rows,
        columns=["Sinyal", "Bu demodaki değer", "Veri durumu", "Saha entegrasyonu için gereken"]
    )
    st.dataframe(telemetry_df, width="stretch", hide_index=True)

    st.markdown("#### Önerilen veri akışı")
    flow_cols = st.columns(4)
    flow_cards = [
        ("01 · Saha verisi", "Read-only OPC UA / MQTT gateway; zaman damgası, kalite bayrağı ve birimler."),
        ("02 · Edge dijital ikiz", "Yerel enerji ve kütle dengesi; bağlantı kesilince güvenli sabit senaryo."),
        ("03 · Operatör onayı", "Sınırlar yetkili mühendislik değişiklik kontrolüyle tanımlanır."),
        ("04 · Building X", "Varlık ve enerji KPI'ları ancak yetkili tenant/API entegrasyonuyla yayımlanır."),
    ]
    for col, (title, description) in zip(flow_cols, flow_cards):
        with col:
            st.markdown(
                f"<div class='soft-card'><h4>{title}</h4><p>{description}</p></div>",
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Tab 4 — assumptions and equations
# -----------------------------------------------------------------------------
with tab_model:
    st.markdown("<div class='section-label'>ŞEFFAF HESAPLAMA</div>", unsafe_allow_html=True)
    st.subheader("Model, kabuller ve doğrulama sınırları")
    st.markdown(
        "**Optimizasyon yöntemi:** 0,25 yüzde puan çözünürlüklü bir aday harman ızgarası üzerinde "
        "24 saatlik dinamik programlama. Amaç, yeşil enerji erişim indeksi yüksek saatlerde "
        "doğalgazın enerji eşdeğerini azaltırken harman üst sınırını, H₂ tedarik debisini, "
        "yoğuşma suyu debisini ve setpoint değişim cezasını uygulamaktır. Bu MVP'de eğitilmiş "
        "bir TFT veya başka bir makine öğrenmesi tahmin modeli bulunmuyor; geçmiş tesis verisi "
        "olmadan model tahmini iddiası üretilmiyor."
    )

    st.markdown("#### Kullanılan temel denklemler")
    st.latex(r"L_{mix} = x_{H_2} L_{H_2} + (1-x_{H_2})L_{NG}")
    st.latex(r"\dot V_{mix} = \dot V_{NG,base}\, L_{NG}/L_{mix}")
    st.markdown(
        "Karışım debileri, doğalgaz baz senaryosuyla aynı alt ısıl değer girdisini koruyacak şekilde hesaplanır. "
        "H₂ **hacim yüzdesi**, ikame edilen doğalgaz **enerji yüzdesi** değildir; H₂'nin hacim başına LHV'si daha düşüktür."
    )
    st.latex(r"m_{H_2}=\dot V_{H_2}\rho_{H_2};\quad W_{process}=m_{H_2}\,w_{L/kg};\quad m_{O_2,stoich}=8m_{H_2}")
    st.latex(r"CO_{2,avoided}=\Delta V_{NG}\,EF_{NG}")

    st.markdown("#### Varsayımlar")
    assumptions = [
        f"Doğalgaz LHV: {format_number(ng_lhv, 1)} MJ/Nm³; H₂ LHV: {format_number(H2_LHV_MJ_PER_NM3, 2)} MJ/Nm³ (DOE H₂ LHV × normal yoğunluk).",
        f"Doğrudan CO₂ faktörü: {format_number(ng_co2_factor, 3)} kg/Nm³; varsayılan türetilmiştir IPCC 56,1 kg/GJ ve 35,8 MJ/Nm³ değerlerinden.",
        f"H₂ elektroliz su bütçesi: {format_number(water_intensity, 1)} L/kg kullanıcı varsayımı; 9 kg/kg kimyasal stokiyometrik su ihtiyacını ve varsa ek proses payını ayırır.",
        f"O₂ yan ürün kütlesi: yaklaşık 8 kg O₂/kg H₂. Arıtmada faydalı sayılan miktar, ayarlanabilir etkin kullanım oranı ve tesis O₂ ihtiyacının küçüğüdür.",
        "Fan elektrik tasarrufu, kredilendirilen O₂ ihtiyacının fan gücüyle doğrusal ölçeklendiği basit senaryodur; blower performans eğrisi veya kütle transfer modeli değildir.",
        "Yeşil enerji profili, sentetik ve normalize edilmiş günlük şekildir. Şebeke/RES/GES tahmini, üretim optimizasyonu veya gerçek zamanlı hava durumu kullanılmıyor.",
        "Scope 1 metriği yalnızca doğalgazın doğrudan yanma CO₂'sini kapsar. H₂ üretim elektriği ve yaşam döngüsü emisyonları düşülmez; NOₓ ve proses etkileri modellenmez.",
        "Software retrofit, fırın yenilemesini gerektirmeyebilir; ancak elektrolizör, su şartlandırma, H₂ depolama/sıkıştırma, gaz treni ve emniyet altyapısının CAPEX/OPEX'i bu uygulamada hesaplanmaz.",
    ]
    for item in assumptions:
        st.markdown(f"- {item}")

    st.markdown("#### Referanslar")
    st.markdown(
        "- [IPCC 2006 Guidelines, Volume 2, Chapter 1 — Table 1.4](https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_1_Ch1_Introduction.pdf): doğal gaz için 56.1 t CO₂/TJ varsayılan yanma faktörü.\n"
        "- [U.S. DOE — Hydrogen production: electrolysis](https://www.energy.gov/cmei/fuels/hydrogen-production-electrolysis): elektroliz suyu H₂ ve O₂'ye ayırır; gerçek sistem enerji tüketimi teknolojiye bağlıdır.\n"
        "- [U.S. DOE — hydrogen technical targets](https://www.energy.gov/cmei/fuels/doe-technical-targets-onboard-hydrogen-storage-light-duty-vehicles): H₂ LHV için 33.3 kWh/kg değeri."
    )

st.divider()
st.markdown(
    "<div class='small-note'>OptiHydro AI · demonstrasyon arayüzü · senaryo hesapları saha verisiyle kalibre edilmeden "
    "yatırım kararı, proses emniyeti veya emisyon doğrulaması için kullanılmamalıdır.</div>",
    unsafe_allow_html=True,
)
