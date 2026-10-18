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
H2_SIMULATION_HARD_CAP_PCT = 20.0
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


def waste_heat_availability_profile(profile_name: str) -> np.ndarray:
    """Return a clearly synthetic availability shape for the thermal-store demo."""
    if profile_name == "Parti üretim (sentetik)":
        profile = np.full(24, 0.10, dtype=float)
        profile[8:16] = 1.0
        return profile
    return np.ones(24, dtype=float)


def simulate_thermal_storage(
    *,
    source_peak_kwth: float,
    source_profile: np.ndarray,
    source_temperature_c: float,
    sink_temperature_c: float,
    heat_demand_kwth: float,
    furnace_duty_kwth: float,
    storage_capacity_kwhth: float,
    charge_limit_kwth: float,
    discharge_limit_kwth: float,
    heat_exchanger_efficiency: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    standing_loss_pct_day: float,
) -> pd.DataFrame:
    """Dispatch an illustrative sensible-heat store for a lower-temperature preheat load.

    One row represents one hour. Heat is first used directly, then from existing storage,
    and only then is surplus charged. It is not a model of the furnace hot zone.
    """
    hours = len(source_profile)
    storage_capacity = max(0.0, float(storage_capacity_kwhth))
    demand_limit = min(max(0.0, float(heat_demand_kwth)), max(0.0, float(furnace_duty_kwth)))
    source_enabled = (
        float(source_temperature_c) > float(sink_temperature_c)
        and float(source_peak_kwth) > 0.0
        and demand_limit > 0.0
    )
    hx_eff = float(np.clip(heat_exchanger_efficiency, 0.0, 1.0))
    charge_eff = float(np.clip(charge_efficiency, 0.0, 1.0))
    discharge_eff = float(np.clip(discharge_efficiency, 0.0, 1.0))
    hourly_loss = float(np.clip(standing_loss_pct_day, 0.0, 100.0)) / 100.0 / max(hours, 1)
    state_kwh = 0.0
    rows: list[dict[str, float | str]] = []

    for hour in range(hours):
        state_kwh *= 1.0 - hourly_loss
        source_kwth = (
            max(0.0, float(source_peak_kwth)) * float(source_profile[hour])
            if source_enabled
            else 0.0
        )
        usable_source_kwh = source_kwth * hx_eff  # one-hour interval
        demand_kwh = demand_limit

        direct_heat_kwh = min(usable_source_kwh, demand_kwh)
        remaining_demand_kwh = max(0.0, demand_kwh - direct_heat_kwh)
        stored_energy_available_kwh = state_kwh * discharge_eff
        discharged_heat_kwh = min(
            remaining_demand_kwh,
            max(0.0, float(discharge_limit_kwth)),
            stored_energy_available_kwh,
        )
        if discharge_eff > 0.0:
            state_kwh -= discharged_heat_kwh / discharge_eff
        else:
            discharged_heat_kwh = 0.0

        source_surplus_kwh = max(0.0, usable_source_kwh - direct_heat_kwh)
        charge_space_kwh = max(0.0, storage_capacity - state_kwh)
        if charge_eff > 0.0:
            charged_input_kwh = min(
                source_surplus_kwh,
                max(0.0, float(charge_limit_kwth)),
                charge_space_kwh / charge_eff,
            )
            state_kwh += charged_input_kwh * charge_eff
        else:
            charged_input_kwh = 0.0
        state_kwh = float(np.clip(state_kwh, 0.0, storage_capacity))
        useful_heat_kwh = direct_heat_kwh + discharged_heat_kwh
        rows.append(
            {
                "Saat": f"{hour:02d}:00",
                "Atık ısı kaynağı (kWth)": source_kwth,
                "Doğrudan kullanılan ısı (kWhth)": direct_heat_kwh,
                "Depoya giren ısı (kWhth)": charged_input_kwh,
                "Depodan çıkan ısı (kWhth)": discharged_heat_kwh,
                "Faydalı ısı (kWhth)": useful_heat_kwh,
                "Depo doluluğu (kWhth)": state_kwh,
                "Kullanılmayan ısı (kWhth)": max(
                    0.0, usable_source_kwh - direct_heat_kwh - charged_input_kwh
                ),
            }
        )
    return pd.DataFrame(rows)


def h2_blend_for_gas_saving(
    gas_saving_fraction: float,
    ng_lhv_mj_nm3: float,
    h2_lhv_mj_nm3: float = H2_LHV_MJ_PER_NM3,
) -> float:
    """Solve the volume blend needed for a natural-gas volume reduction at equal duty."""
    target = float(np.clip(gas_saving_fraction, 0.0, 0.999999))
    denominator = h2_lhv_mj_nm3 + target * (ng_lhv_mj_nm3 - h2_lhv_mj_nm3)
    if target <= 0.0 or denominator <= 0.0:
        return 0.0
    return float(np.clip(target * ng_lhv_mj_nm3 / denominator, 0.0, 1.0))


def optimize_blend_schedule(
    availability: np.ndarray,
    safety_cap_fraction: float,
    hourly_supply_cap_fractions: np.ndarray,
    ng_lhv_mj_nm3: float,
    max_ramp_fraction_per_hour: float = 0.03,
    h2_lhv_mj_nm3: float = H2_LHV_MJ_PER_NM3,
) -> np.ndarray:
    """Grid-search dispatch with availability, blend, supply and ramp penalty.

    The objective rewards energy-equivalent natural-gas displacement during
    higher-availability hours and applies a modest penalty to setpoint changes.
    This is an explainable deterministic optimizer, not a trained AI forecast.
    """
    max_hourly_supply_cap = (
        float(np.max(hourly_supply_cap_fractions)) if hourly_supply_cap_fractions.size else 0.0
    )
    upper_bound = max(0.0, min(safety_cap_fraction, max_hourly_supply_cap))
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
        allowed = min(
            upper_bound,
            safety_cap_fraction * float(availability[hour]),
            float(hourly_supply_cap_fractions[hour]),
        )
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

        ramp = np.abs(candidates[None, :] - candidates[:, None])
        transition = scores[hour - 1, :, None] - ramp_penalty * ramp * 100.0
        transition[ramp > max(0.0, max_ramp_fraction_per_hour) + 1e-10] = -np.inf
        best_parent = np.argmax(transition, axis=0)
        best_score = transition[best_parent, np.arange(state_count)]
        scores[hour, feasible] = reward[feasible] + best_score[feasible]
        parents[hour, feasible] = best_parent[feasible]

    schedule = np.zeros(hour_count)
    terminal_scores = scores[-1] - ramp_penalty * np.abs(candidates) * 100.0
    state = int(np.argmax(terminal_scores))
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
    renewable_surplus_peak_kw: float,
    condensate_l_h: float,
    water_intensity_l_kg: float,
    ng_lhv_mj_nm3: float,
    ng_co2_kg_nm3: float,
    furnace_thermal_efficiency: float,
    operating_days: int,
    electrolyzer_kwh_kg: float,
    oxygen_demand_kg_h: float,
    oxygen_use_efficiency: float,
    aeration_fan_kw: float,
    furnace_temperature_c: int,
    max_ramp_pct_per_hour: float,
    heat_source_peak_kwth: float,
    heat_source_profile_name: str,
    heat_source_temperature_c: float,
    heat_sink_temperature_c: float,
    heat_demand_kwth: float,
    heat_storage_capacity_kwhth: float,
    heat_charge_limit_kwth: float,
    heat_discharge_limit_kwth: float,
    heat_exchanger_efficiency: float,
    heat_charge_efficiency: float,
    heat_discharge_efficiency: float,
    heat_storage_loss_pct_day: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    availability = renewable_availability_profile(peak_index_pct)
    # A project-configured demo ceiling, not a universal standard limit.
    safety_cap = min(float(safety_cap_pct), H2_SIMULATION_HARD_CAP_PCT) / 100.0

    heat_profile = waste_heat_availability_profile(heat_source_profile_name)
    furnace_efficiency = float(np.clip(furnace_thermal_efficiency, 0.01, 1.0))
    furnace_duty_kwth = gas_flow_nm3_h * ng_lhv_mj_nm3 / 3.6 * furnace_efficiency
    heat_frame = simulate_thermal_storage(
        source_peak_kwth=heat_source_peak_kwth,
        source_profile=heat_profile,
        source_temperature_c=heat_source_temperature_c,
        sink_temperature_c=heat_sink_temperature_c,
        heat_demand_kwth=heat_demand_kwth,
        furnace_duty_kwth=furnace_duty_kwth,
        storage_capacity_kwhth=heat_storage_capacity_kwhth,
        charge_limit_kwth=heat_charge_limit_kwth,
        discharge_limit_kwth=heat_discharge_limit_kwth,
        heat_exchanger_efficiency=heat_exchanger_efficiency,
        charge_efficiency=heat_charge_efficiency,
        discharge_efficiency=heat_discharge_efficiency,
        standing_loss_pct_day=heat_storage_loss_pct_day,
    )
    heat_used_kwh = heat_frame["Faydalı ısı (kWhth)"].to_numpy(dtype=float)
    heat_gas_equivalent_nm3_h = np.minimum(
        gas_flow_nm3_h,
        heat_used_kwh * 3.6 / (ng_lhv_mj_nm3 * furnace_efficiency)
        if ng_lhv_mj_nm3 > 0
        else 0.0,
    )
    burner_ng_reference_nm3_h = np.maximum(0.0, gas_flow_nm3_h - heat_gas_equivalent_nm3_h)

    # External H2 availability and recovered-water treatment capacity both bind.
    water_limited_h2_kg_h = condensate_l_h / water_intensity_l_kg if water_intensity_l_kg else 0.0
    available_h2_kg_h = min(external_h2_supply_kg_h, water_limited_h2_kg_h)
    renewable_power_available_kw = renewable_surplus_peak_kw * availability
    power_limited_h2_kg_h = renewable_power_available_kw / electrolyzer_kwh_kg
    hourly_h2_supply_kg_h = np.minimum(available_h2_kg_h, power_limited_h2_kg_h)
    hourly_supply_cap_fractions = np.array(
        [
            blend_cap_from_h2_flow(
                h2_kg_h / H2_DENSITY_KG_PER_NM3,
                burner_ng_reference_nm3_h[hour],
                ng_lhv_mj_nm3,
            )
            for hour, h2_kg_h in enumerate(hourly_h2_supply_kg_h)
        ]
    )
    blend = optimize_blend_schedule(
        availability,
        safety_cap,
        hourly_supply_cap_fractions,
        ng_lhv_mj_nm3,
        max_ramp_fraction_per_hour=max(0.0, float(max_ramp_pct_per_hour)) / 100.0,
    )

    rows: list[dict[str, Any]] = []
    for hour, blend_fraction in enumerate(blend):
        total_mix, h2_nm3_h, ng_nm3_h = blend_flows(
            blend_fraction, burner_ng_reference_nm3_h[hour], ng_lhv_mj_nm3
        )
        h2_kg_h = h2_nm3_h * H2_DENSITY_KG_PER_NM3
        ng_avoided_by_heat_nm3_h = heat_gas_equivalent_nm3_h[hour]
        ng_avoided_by_h2_nm3_h = max(0.0, burner_ng_reference_nm3_h[hour] - ng_nm3_h)
        ng_avoided_nm3_h = ng_avoided_by_heat_nm3_h + ng_avoided_by_h2_nm3_h
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
                "Atık ısı sonrası brülör doğalgaz referansı (Nm³/saat)": burner_ng_reference_nm3_h[hour],
                "Senaryo doğalgazı (Nm³/saat)": ng_nm3_h,
                "Atık ısıyla önlenen doğalgaz (Nm³/saat)": ng_avoided_by_heat_nm3_h,
                "H2 ile önlenen doğalgaz (Nm³/saat)": ng_avoided_by_h2_nm3_h,
                "Toplam önlenen doğalgaz (Nm³/saat)": ng_avoided_nm3_h,
                "Karışım toplamı (Nm³/saat)": total_mix,
                "H2 beslemesi (Nm³/saat)": h2_nm3_h,
                "H2 tüketimi (kg/saat)": h2_kg_h,
                "Proses suyu (L/saat)": process_water_l_h,
                "Üretilen O2 (kg/saat)": oxygen_generated_kg_h,
                "Arıtmada kredilendirilen O2 (kg/saat)": oxygen_credited_kg_h,
                "Elektroliz elektrik ihtiyacı (kWh/saat)": h2_kg_h * electrolyzer_kwh_kg,
                "Yeşil enerji kullanılabilir gücü (kW)": renewable_power_available_kw[hour],
                "Havalandırma fan gücü senaryosu (kW)": aeration_fan_kw
                * oxygen_displacement_fraction,
                "Önlenen doğrudan CO2 (kg/saat)": ng_avoided_nm3_h * ng_co2_kg_nm3,
                "Faydalı atık ısı (kWhth/saat)": heat_used_kwh[hour],
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.merge(heat_frame, on="Saat", how="left", validate="one_to_one")
    baseline_gas_day = gas_flow_nm3_h * 24.0
    saved_gas_day = float(frame["Toplam önlenen doğalgaz (Nm³/saat)"].sum())
    h2_saved_gas_day = float(frame["H2 ile önlenen doğalgaz (Nm³/saat)"].sum())
    heat_saved_gas_day = float(frame["Atık ısıyla önlenen doğalgaz (Nm³/saat)"].sum())
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
    green_power_available_day_kwh = float(renewable_power_available_kw.sum())
    useful_heat_day_kwh = float(heat_frame["Faydalı ısı (kWhth)"].sum())
    heat_charged_day_kwh = float(heat_frame["Depoya giren ısı (kWhth)"].sum())
    heat_discharged_day_kwh = float(heat_frame["Depodan çıkan ısı (kWhth)"].sum())
    thermal_store_end_kwh = float(heat_frame["Depo doluluğu (kWhth)"].iloc[-1])
    safety_ceiling_flows = blend_flows(safety_cap, gas_flow_nm3_h, ng_lhv_mj_nm3)
    theoretical_safety_gas_saving_pct = (
        max(0.0, gas_flow_nm3_h - safety_ceiling_flows[2]) / gas_flow_nm3_h * 100.0
        if gas_flow_nm3_h > 0
        else 0.0
    )
    effective_peak_power_kw = float(np.max(renewable_power_available_kw)) if len(renewable_power_available_kw) else 0.0
    peak_power_h2_kg_h = effective_peak_power_kw / electrolyzer_kwh_kg if electrolyzer_kwh_kg > 0 else 0.0
    safety_h2_flow_kg_h = safety_ceiling_flows[1] * H2_DENSITY_KG_PER_NM3

    summary = {
        "baseline_gas_day_nm3": baseline_gas_day,
        "saved_gas_day_nm3": saved_gas_day,
        "projected_gas_day_nm3": projected_gas_day,
        "gas_saving_pct": saved_gas_day / baseline_gas_day * 100.0 if baseline_gas_day else 0.0,
        "h2_saved_gas_day_nm3": h2_saved_gas_day,
        "heat_saved_gas_day_nm3": heat_saved_gas_day,
        "h2_saving_pct": h2_saved_gas_day / baseline_gas_day * 100.0 if baseline_gas_day else 0.0,
        "heat_saving_pct": heat_saved_gas_day / baseline_gas_day * 100.0 if baseline_gas_day else 0.0,
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
        "green_power_available_day_mwh": green_power_available_day_kwh / 1000.0,
        "green_power_utilization_pct": green_h2_power_day_kwh / green_power_available_day_kwh * 100.0
        if green_power_available_day_kwh > 0
        else 0.0,
        "useful_heat_day_kwh": useful_heat_day_kwh,
        "heat_charged_day_kwh": heat_charged_day_kwh,
        "heat_discharged_day_kwh": heat_discharged_day_kwh,
        "thermal_store_end_kwh": thermal_store_end_kwh,
        "thermal_store_capacity_kwh": float(heat_storage_capacity_kwhth),
        "thermal_heat_enabled": float(
            heat_source_temperature_c > heat_sink_temperature_c
            and heat_source_peak_kwth > 0.0
            and heat_demand_kwth > 0.0
        ),
        "theoretical_safety_gas_saving_pct": theoretical_safety_gas_saving_pct,
        "effective_peak_power_kw": effective_peak_power_kw,
        "peak_power_h2_kg_h": peak_power_h2_kg_h,
        "peak_h2_flow_kg_h": float(frame["H2 tüketimi (kg/saat)"].max()),
        "safety_h2_flow_kg_h": safety_h2_flow_kg_h,
        "max_ramp_pct_per_hour": float(max_ramp_pct_per_hour),
        "operating_days": float(operating_days),
        "furnace_temperature_c": float(furnace_temperature_c),
        "furnace_thermal_efficiency_pct": furnace_efficiency * 100.0,
        "available_h2_kg_h": available_h2_kg_h,
        "water_limited_h2_kg_h": water_limited_h2_kg_h,
    }
    return frame, heat_frame, summary


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
    renewable_surplus_peak_kw = st.number_input(
        "Yeşil elektrik fazlası pik gücü (kW)",
        min_value=0.0,
        max_value=100000.0,
        value=1800.0,
        step=50.0,
        help="Model saatlik elektrolizör tüketimini bu güç × sentetik erişilebilirlik profiliyle sınırlar.",
    )
    safety_cap_pct = st.slider(
        "İstenen H₂ harman üst sınırı (vol %)",
        min_value=0,
        max_value=30,
        value=20,
        help="Bu prototipin sabit dispatch tavanı %20'dir; daha yüksek girdiler simülasyonda bu tavana kesilir.",
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
    target_savings_pct = st.slider(
        "Doğalgaz ikamesi hedefi (%)",
        min_value=1,
        max_value=20,
        value=10,
        help="H₂ ve ısı geri kazanım kısıtlarıyla karşılaştırılan planlama hedefidir; gerçekleşme garantisi değildir.",
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
        furnace_efficiency_pct = st.number_input(
            "Baz fırın yararlı ısı verimi (%)",
            min_value=1.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
            help="Isı geri kazanımının yararlı proses ısısından doğalgaz girdisine çevrimi için senaryo varsayımıdır.",
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
        max_ramp_pct_per_hour = st.number_input(
            "H₂ setpoint değişim üst sınırı (yüzde puan/saat)",
            min_value=0.25,
            max_value=20.0,
            value=3.0,
            step=0.25,
            help="Optimizasyona uygulanan demo kısıtıdır; gerçek brülör rampası OEM ve saha mühendisliğiyle belirlenmelidir.",
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

    with st.expander("Atık ısı depolama senaryosu", expanded=False):
        st.caption(
            "Varsayılanlar yalnızca görsel demo senaryosudur; ölçülmüş saha verisi değildir. "
            "Düşük/orta sıcaklıklı ısı, fırının 1.000°C+ sıcak bölgesinin yerine değil, uygun bir ön ısıtma yüküne yöneltilir."
        )
        heat_source_profile_name = st.selectbox(
            "Atık ısı erişim profili",
            ["Parti üretim (sentetik)", "Sürekli (sentetik)"],
            index=0,
            help="Gerçek pilotta bu sentetik profil saatlik ölçüm serisiyle değiştirilmelidir.",
        )
        heat_source_peak_kwth = st.number_input(
            "Geri kazanılabilir atık ısı piki (kWth)",
            min_value=0.0,
            max_value=100000.0,
            value=400.0,
            step=25.0,
        )
        heat_demand_kwth = st.number_input(
            "Uygun ön ısıtma ısı talebi (kWth)",
            min_value=0.0,
            max_value=100000.0,
            value=250.0,
            step=25.0,
            help="Fırın sıcak bölgesi değil; eşleşen sıcaklık seviyesindeki ölçülebilir ön ısıtma/yardımcı proses yükü.",
        )
        heat_storage_capacity_kwhth = st.number_input(
            "Isıl depo kapasitesi (kWhth)",
            min_value=0.0,
            max_value=10000000.0,
            value=2000.0,
            step=100.0,
        )
        heat_charge_limit_kwth = st.number_input(
            "Maksimum depo şarj gücü (kWth)",
            min_value=0.0,
            max_value=100000.0,
            value=400.0,
            step=25.0,
        )
        heat_discharge_limit_kwth = st.number_input(
            "Maksimum depo deşarj gücü (kWth)",
            min_value=0.0,
            max_value=100000.0,
            value=250.0,
            step=25.0,
        )
        heat_source_temperature_c = st.number_input(
            "Atık ısı kaynağı sıcaklığı (°C)",
            min_value=0.0,
            max_value=1800.0,
            value=400.0,
            step=10.0,
        )
        heat_sink_temperature_c = st.number_input(
            "Ön ısıtma yükü hedef sıcaklığı (°C)",
            min_value=0.0,
            max_value=1800.0,
            value=250.0,
            step=10.0,
        )
        heat_exchanger_efficiency_pct = st.slider(
            "Isı eşanjörü etkinliği (%)", min_value=0, max_value=100, value=85
        )
        heat_charge_efficiency_pct = st.slider(
            "Depo şarj etkinliği (%)", min_value=0, max_value=100, value=90
        )
        heat_discharge_efficiency_pct = st.slider(
            "Depo deşarj etkinliği (%)", min_value=0, max_value=100, value=90
        )
        heat_storage_loss_pct_day = st.number_input(
            "Depo bekleme kaybı (%/gün)",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.5,
        )

    with st.expander("Finansal senaryo varsayımları", expanded=False):
        natural_gas_price_try_nm3 = st.number_input(
            "Doğalgaz birim fiyatı (TL/Nm³)",
            min_value=0.0,
            max_value=10000.0,
            value=15.0,
            step=0.5,
        )
        eur_try = st.number_input(
            "EUR/TRY senaryo kuru (TL/€)",
            min_value=0.01,
            max_value=1000.0,
            value=38.0,
            step=0.5,
            help="Sabit örnek senaryo girdisidir; güncel piyasa kuru değildir.",
        )
        cbam_price_eur_t = st.number_input(
            "CBAM karbon fiyatı varsayımı (€/tCO₂)",
            min_value=0.0,
            max_value=5000.0,
            value=80.0,
            step=1.0,
            help="Model varsayımıdır; CBAM sertifika fiyatı dönemsel olarak EU ETS'e göre belirlenir.",
        )
        cbam_eligible_share_pct = st.slider(
            "Uygun AB CBAM ürünlerine tahsis edilen emisyon payı (%)",
            min_value=0,
            max_value=100,
            value=0,
            help="Ürün kapsamı ve doğrulanmış ürün emisyon tahsisi yoksa 0 bırakın.",
        )
        software_setup_cost_eur = st.number_input(
            "Tek seferlik yazılım/entegrasyon bedeli (€)",
            min_value=0.0,
            max_value=100000000.0,
            value=0.0,
            step=5000.0,
        )
        electrolyzer_capex_eur = st.number_input(
            "Elektrolizör ve güç elektroniği yatırım bedeli (€)",
            min_value=0.0,
            max_value=1000000000.0,
            value=0.0,
            step=10000.0,
        )
        hydrogen_system_capex_eur = st.number_input(
            "H₂ şartlandırma / sıkıştırma / depolama bedeli (€)",
            min_value=0.0,
            max_value=1000000000.0,
            value=0.0,
            step=10000.0,
        )
        water_system_capex_eur = st.number_input(
            "Su arıtma ve kalite kontrol yatırımı (€)",
            min_value=0.0,
            max_value=1000000000.0,
            value=0.0,
            step=5000.0,
        )
        thermal_storage_capex_eur = st.number_input(
            "Isı eşanjörü ve ısıl depo yatırımı (€)",
            min_value=0.0,
            max_value=1000000000.0,
            value=0.0,
            step=10000.0,
        )
        other_site_capex_eur = st.number_input(
            "Mühendislik / saha entegrasyonu / diğer CAPEX (€)",
            min_value=0.0,
            max_value=1000000000.0,
            value=0.0,
            step=5000.0,
        )
        green_electricity_price_eur_mwh = st.number_input(
            "Yeşil elektriğin fırsat maliyeti (€/MWh)",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=5.0,
            help="0 değeri yalnızca gerçekten yedek/curtailed enerji varsayımında kullanılmalıdır.",
        )
        annual_software_opex_eur = st.number_input(
            "Yıllık lisans ve yazılım OPEX (€)",
            min_value=0.0,
            max_value=100000000.0,
            value=0.0,
            step=1000.0,
        )
        annual_plant_opex_eur = st.number_input(
            "Yıllık bakım / su / saha işletme OPEX (€)",
            min_value=0.0,
            max_value=100000000.0,
            value=0.0,
            step=1000.0,
        )

    st.divider()
    st.markdown(
        "<span class='status-chip gray'>Senaryo modu · canlı tesis bağlantısı yok</span>",
        unsafe_allow_html=True,
    )


scenario, heat_storage, totals = create_scenario(
    sector=sector,
    gas_flow_nm3_h=gas_flow,
    peak_index_pct=green_peak,
    safety_cap_pct=safety_cap_pct,
    external_h2_supply_kg_h=h2_supply,
    renewable_surplus_peak_kw=renewable_surplus_peak_kw,
    condensate_l_h=condensate,
    water_intensity_l_kg=water_intensity,
    ng_lhv_mj_nm3=ng_lhv,
    ng_co2_kg_nm3=ng_co2_factor,
    furnace_thermal_efficiency=furnace_efficiency_pct / 100.0,
    operating_days=int(operating_days),
    electrolyzer_kwh_kg=electrolyzer_kwh_kg,
    oxygen_demand_kg_h=oxygen_demand,
    oxygen_use_efficiency=oxygen_use_pct / 100.0,
    aeration_fan_kw=aeration_fan_kw,
    furnace_temperature_c=int(furnace_temperature),
    max_ramp_pct_per_hour=max_ramp_pct_per_hour,
    heat_source_peak_kwth=heat_source_peak_kwth,
    heat_source_profile_name=heat_source_profile_name,
    heat_source_temperature_c=heat_source_temperature_c,
    heat_sink_temperature_c=heat_sink_temperature_c,
    heat_demand_kwth=heat_demand_kwth,
    heat_storage_capacity_kwhth=heat_storage_capacity_kwhth,
    heat_charge_limit_kwth=heat_charge_limit_kwth,
    heat_discharge_limit_kwth=heat_discharge_limit_kwth,
    heat_exchanger_efficiency=heat_exchanger_efficiency_pct / 100.0,
    heat_charge_efficiency=heat_charge_efficiency_pct / 100.0,
    heat_discharge_efficiency=heat_discharge_efficiency_pct / 100.0,
    heat_storage_loss_pct_day=heat_storage_loss_pct_day,
)

target_blend_fraction = (
    h2_blend_for_gas_saving(target_savings_pct / 100.0, ng_lhv)
    if gas_flow > 0.0
    else 0.0
)
target_blend_total_nm3_h, target_h2_nm3_h, target_ng_nm3_h = blend_flows(
    target_blend_fraction, gas_flow, ng_lhv
)
target_h2_kg_h = target_h2_nm3_h * H2_DENSITY_KG_PER_NM3
target_electrolyzer_kw = target_h2_kg_h * electrolyzer_kwh_kg
target_water_l_h = target_h2_kg_h * water_intensity
effective_safety_cap_pct = min(float(safety_cap_pct), H2_SIMULATION_HARD_CAP_PCT)
target_h2_only_feasible_under_cap = (
    gas_flow <= 0.0 or target_blend_fraction * 100.0 <= effective_safety_cap_pct + 1e-9
)
target_resource_coverages = {
    "H₂ tedariki": h2_supply / target_h2_kg_h if target_h2_kg_h > 0 else float("inf"),
    "Chiller suyu": condensate / target_water_l_h if target_water_l_h > 0 else float("inf"),
    "Yeşil elektrik piki": totals["effective_peak_power_kw"] / target_electrolyzer_kw
    if target_electrolyzer_kw > 0
    else float("inf"),
}
target_bottleneck_name = (
    "H₂ harman tavanı"
    if not target_h2_only_feasible_under_cap
    else min(target_resource_coverages, key=target_resource_coverages.get)
)
target_bottleneck_coverage_pct = (
    min(100.0, target_resource_coverages[target_bottleneck_name] * 100.0)
    if target_bottleneck_name in target_resource_coverages
    and np.isfinite(target_resource_coverages[target_bottleneck_name])
    else 100.0
)
target_savings_nm3_day = totals["baseline_gas_day_nm3"] * target_savings_pct / 100.0
target_gap_after_scenario_nm3_day = max(
    0.0, target_savings_nm3_day - totals["saved_gas_day_nm3"]
)
target_gap_heat_mwh_day = (
    target_gap_after_scenario_nm3_day
    * ng_lhv
    / 3.6
    * furnace_efficiency_pct
    / 100.0
    / 1000.0
)
target_required_green_peak_kw = (
    target_electrolyzer_kw / (green_peak / 100.0)
    if green_peak > 0
    else float("inf")
)
target_required_green_peak_display = (
    f"{format_number(target_required_green_peak_kw)} kW"
    if np.isfinite(target_required_green_peak_kw)
    else "yeşil enerji piki gerekli"
)

annual_fuel_savings_try = (
    totals["saved_gas_day_nm3"] * int(operating_days) * natural_gas_price_try_nm3
)
annual_fuel_savings_eur = annual_fuel_savings_try / eur_try if eur_try > 0 else 0.0
cbam_eligible_avoided_t = (
    totals["avoided_co2_year_t"] * cbam_eligible_share_pct / 100.0
)
annual_cbam_scenario_eur = cbam_eligible_avoided_t * cbam_price_eur_t
annual_gross_financial_gain_eur = annual_fuel_savings_eur + annual_cbam_scenario_eur
annual_hydrogen_electricity_mwh = totals["green_h2_power_day_mwh"] * int(operating_days)
annual_electricity_cost_eur = annual_hydrogen_electricity_mwh * green_electricity_price_eur_mwh
annual_total_opex_eur = (
    annual_software_opex_eur + annual_plant_opex_eur + annual_electricity_cost_eur
)
total_system_capex_eur = sum(
    [
        software_setup_cost_eur,
        electrolyzer_capex_eur,
        hydrogen_system_capex_eur,
        water_system_capex_eur,
        thermal_storage_capex_eur,
        other_site_capex_eur,
    ]
)
annual_net_financial_gain_eur = annual_gross_financial_gain_eur - annual_total_opex_eur
payback_years = (
    total_system_capex_eur / annual_net_financial_gain_eur
    if total_system_capex_eur > 0 and annual_net_financial_gain_eur > 0
    else None
)


# -----------------------------------------------------------------------------
# Header and headline KPIs
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <section class="hero">
      <div class="hero-kicker">Industrial energy digital twin · decision support</div>
      <h1>OptiHydro AI</h1>
      <p>Fırın yakıtını ve uygun ön ısıtma yüklerini; yeşil enerji, H₂, geri kazanılmış su ve ısıl depo sınırlarıyla birlikte senaryolaştırın.</p>
      <span class="hero-pill">{sector} · {format_number(gas_flow)} Nm³/saat baz yük</span>
      <span class="hero-pill">Açıklanabilir, kısıtlı dispatch</span>
      <span class="hero-pill">Canlı saha kontrolü değildir</span>
    </section>
    """,
    unsafe_allow_html=True,
)

if float(safety_cap_pct) > H2_SIMULATION_HARD_CAP_PCT:
    st.warning(
        f"⚠️ **AI Safety Override Aktif (MVP kuralı):** Girilen %{format_number(float(safety_cap_pct), 1)} vol sınırı, "
        f"bu prototipte tanımlı %{format_number(H2_SIMULATION_HARD_CAP_PCT, 1)} vol dispatch tavanını aşıyor. "
        f"Simülasyon H₂ oranını %{format_number(H2_SIMULATION_HARD_CAP_PCT, 1)} vol ile sınırlar. "
        "Bu deterministik demo tavanı EN 746-2/ATEX için evrensel bir izin veya emniyet sınırı değildir; gerçek limitler "
        "uygulama, brülör üreticisi ve risk değerlendirmesiyle doğrulanır. Bu yazılım freni flashback'i "
        "algılamaz veya önlemez.",
        icon="⚠️",
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
    f"elektroliz elektriğini, NOₓ'i ve ekipman verim değişimini içermez. Günlük tasarrufun "
    f"H₂ payı {format_number(totals['h2_saved_gas_day_nm3'])} Nm³, atık ısı senaryo payı "
    f"{format_number(totals['heat_saved_gas_day_nm3'])} Nm³'dir."
)

tab_dispatch, tab_circular, tab_heat, tab_edge, tab_targets, tab_finance, tab_model = st.tabs(
    [
        "24 saat dispatch",
        "Çifte dairesel ekonomi",
        "Atık ısı deposu",
        "Edge telemetri görünümü",
        "Hedef ve darboğaz",
        "💶 Finansal analiz",
        "Model ve kabuller",
    ]
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
                y=scenario["Atık ısı sonrası brülör doğalgaz referansı (Nm³/saat)"],
                name="Atık ısı sonrası NG ihtiyacı",
                mode="lines",
                line=dict(color=AMBER, width=2.5, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(217,145,25,.10)",
                hovertemplate="%{x}<br>Isı geri kazanımı sonrası: %{y:,.1f} Nm³/saat<extra></extra>",
            ),
            secondary_y=False,
        )
        chart.add_trace(
            go.Scatter(
                x=scenario["Saat"],
                y=scenario["Senaryo doğalgazı (Nm³/saat)"],
                name="OptiHydro senaryo doğalgazı",
                mode="lines",
                line=dict(color=TEAL, width=3),
                fill="tonexty",
                fillcolor="rgba(0,153,153,.12)",
                hovertemplate="%{x}<br>Senaryo NG: %{y:,.1f} Nm³/saat<extra></extra>",
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
        chart.update_layout(title="Yararlı proses ısısını koruyan 24 saatlik senaryo")
        st.plotly_chart(style_figure(chart, 410), width="stretch")

    with right:
        st.markdown("<div class='section-label'>EMİSYON KARŞILAŞTIRMASI</div>", unsafe_allow_html=True)
        st.subheader("Doğrudan yanma CO₂'si")
        emissions_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Baz", "H₂ + atık ısı"],
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
                "H2 ile önlenen doğalgaz (Nm³/saat)",
                "Atık ısıyla önlenen doğalgaz (Nm³/saat)",
                "Toplam önlenen doğalgaz (Nm³/saat)",
                "H2 tüketimi (kg/saat)",
                "Faydalı atık ısı (kWhth/saat)",
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

    st.markdown("#### 🔄 Çifte dairesel akış şeması")
    st.caption(
        "Su ve elektrik farklı fiziksel birimler olduğu için akışları iki Sankey'de ayrı tutuyoruz: "
        "solda kg/saat kütle dengesi, sağda kW enerji dengesi. Elektrik profili bu demoda sentetik erişilebilirlik indeksidir."
    )
    h2_average_kg_h = totals["h2_day_kg"] / 24.0
    water_average_kg_h = totals["water_day_l"] / 24.0  # liquid water density ≈ 1 kg/L
    oxygen_average_kg_h = totals["oxygen_day_kg"] / 24.0
    process_residual_kg_h = max(
        0.0, water_average_kg_h - h2_average_kg_h - oxygen_average_kg_h
    )
    electrolyzer_average_kw = totals["green_h2_power_day_mwh"] * 1000.0 / 24.0
    hydrogen_lhv_output_kw = h2_average_kg_h * H2_LHV_KWH_PER_KG
    electrolyzer_loss_kw = max(0.0, electrolyzer_average_kw - hydrogen_lhv_output_kw)

    mass_sankey_col, energy_sankey_col = st.columns(2)
    with mass_sankey_col:
        if water_average_kg_h > 0:
            mass_labels = [
                f"Chiller kondensatı<br>{water_average_kg_h:.2f} kg/saat",
                "Elektrolizör",
                f"Yeşil H₂ · fırın yakıtı<br>{h2_average_kg_h:.2f} kg/saat",
                f"Saf O₂ · arıtma yan ürünü<br>{oxygen_average_kg_h:.2f} kg/saat",
                f"Proses suyu farkı<br>{process_residual_kg_h:.2f} kg/saat",
            ]
            mass_sources = [0, 1, 1, 1]
            mass_targets = [1, 2, 3, 4]
            mass_values = [
                water_average_kg_h,
                h2_average_kg_h,
                oxygen_average_kg_h,
                process_residual_kg_h,
            ]
            mass_colors = [
                "rgba(0,153,153,.28)",
                "rgba(0,153,153,.48)",
                "rgba(122,181,216,.48)",
                "rgba(150,160,170,.25)",
            ]
            mass_keep = [i for i, value in enumerate(mass_values) if value > 1e-9]
            mass_sankey = go.Figure(
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=20,
                        thickness=19,
                        line=dict(color="rgba(0,43,73,.18)", width=1),
                        label=mass_labels,
                        color=["#88cfc8", NAVY, TEAL, "#7ab5d8", "#d8dee5"],
                    ),
                    link=dict(
                        source=[mass_sources[i] for i in mass_keep],
                        target=[mass_targets[i] for i in mass_keep],
                        value=[mass_values[i] for i in mass_keep],
                        color=[mass_colors[i] for i in mass_keep],
                        hovertemplate="%{value:.2f} kg/saat<extra></extra>",
                    ),
                )
            )
            mass_sankey.update_layout(title="Su ve yan ürün kütle dengesi · kg/saat")
            st.plotly_chart(style_figure(mass_sankey, 350), width="stretch")
        else:
            st.info("Bu senaryoda elektrolizör üretimi yok; kütle akışları sıfır.")

    with energy_sankey_col:
        if electrolyzer_average_kw > 0:
            energy_labels = [
                f"GES/RES tahsisi · %{green_peak} tepe indeks<br>{electrolyzer_average_kw:.1f} kW",
                "Elektrolizör",
                f"H₂ kimyasal enerjisi<br>{hydrogen_lhv_output_kw:.1f} kW",
                f"Dönüşüm kaybı<br>{electrolyzer_loss_kw:.1f} kW",
            ]
            energy_sources = [0, 1, 1]
            energy_targets = [1, 2, 3]
            energy_values = [
                electrolyzer_average_kw,
                hydrogen_lhv_output_kw,
                electrolyzer_loss_kw,
            ]
            energy_colors = [
                "rgba(217,145,25,.34)",
                "rgba(0,153,153,.48)",
                "rgba(150,160,170,.25)",
            ]
            energy_keep = [i for i, value in enumerate(energy_values) if value > 1e-9]
            energy_sankey = go.Figure(
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=20,
                        thickness=19,
                        line=dict(color="rgba(0,43,73,.18)", width=1),
                        label=energy_labels,
                        color=["#f1c56c", NAVY, TEAL, "#d8dee5"],
                    ),
                    link=dict(
                        source=[energy_sources[i] for i in energy_keep],
                        target=[energy_targets[i] for i in energy_keep],
                        value=[energy_values[i] for i in energy_keep],
                        color=[energy_colors[i] for i in energy_keep],
                        hovertemplate="%{value:.1f} kW<extra></extra>",
                    ),
                )
            )
            energy_sankey.update_layout(title="Yenilenebilir elektrik ve H₂ enerji dengesi · kW")
            st.plotly_chart(style_figure(energy_sankey, 350), width="stretch")
        else:
            st.info("Bu senaryoda elektrolizör elektrik talebi yok; enerji akışları sıfır.")

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
# Tab 3 — waste heat recovery and thermal storage
# -----------------------------------------------------------------------------
with tab_heat:
    st.markdown("<div class='section-label'>DÜŞÜK SICAKLIKLI ISI GERİ KAZANIMI</div>", unsafe_allow_html=True)
    st.subheader("Atık ısı deposu ve ön ısıtma yükü")
    st.caption(
        "Model, sıcak bölgeden bağımsız bir ön ısıtma yüküne doğrudan kullanım ve depolama sırasıyla ısı verir. "
        "Varsayılan saatlik parti profili sentetiktir; sonuçlar ölçüm veya ekipman tasarımı yerine geçmez."
    )

    if heat_source_temperature_c <= heat_sink_temperature_c:
        st.warning(
            "Isı kaynağı sıcaklığı hedef yük sıcaklığından yüksek değil; bu senaryoda kullanılabilir ısı sıfıra indirilmiştir. "
            "Gerçek eşanjör için yaklaşım sıcaklığı ve malzeme limitleri ayrıca doğrulanmalıdır.",
            icon="🌡️",
        )
    elif heat_source_peak_kwth <= 0.0 or heat_demand_kwth <= 0.0:
        st.info("Atık ısı kaynağı veya eşleşen ön ısıtma talebi sıfır; bu nedenle doğalgaz tasarrufuna ısı katkısı eklenmiyor.")

    heat_demand_day_kwh = min(
        heat_demand_kwth,
        gas_flow * ng_lhv / 3.6 * furnace_efficiency_pct / 100.0,
    ) * 24.0
    heat_coverage_pct = (
        totals["useful_heat_day_kwh"] / heat_demand_day_kwh * 100.0
        if heat_demand_day_kwh > 0
        else 0.0
    )
    heat_k1, heat_k2, heat_k3, heat_k4 = st.columns(4)
    heat_k1.metric("Faydalı ısı", f"{format_number(totals['useful_heat_day_kwh'] / 1000.0, 2)} MWhth/gün")
    heat_k2.metric("Isı ile önlenen NG", f"{format_number(totals['heat_saved_gas_day_nm3'])} Nm³/gün")
    heat_k3.metric("Ön ısıtma yükü karşılama", f"{format_number(min(100.0, heat_coverage_pct), 1)}%")
    store_capacity = max(0.0, totals["thermal_store_capacity_kwh"])
    heat_k4.metric(
        "Gün sonu depo doluluğu",
        f"{format_number(totals['thermal_store_end_kwh'])} kWhth",
        f"kapasitenin %{format_number(totals['thermal_store_end_kwh'] / store_capacity * 100.0 if store_capacity else 0.0, 1)}'i",
    )

    heat_fig = make_subplots(specs=[[{"secondary_y": True}]])
    heat_fig.add_trace(
        go.Scatter(
            x=heat_storage["Saat"],
            y=heat_storage["Atık ısı kaynağı (kWth)"],
            name="Erişilebilir atık ısı",
            line=dict(color=AMBER, width=2, dash="dot"),
            hovertemplate="%{x}<br>Kaynak: %{y:.1f} kWth<extra></extra>",
        ),
        secondary_y=False,
    )
    heat_fig.add_trace(
        go.Scatter(
            x=heat_storage["Saat"],
            y=heat_storage["Faydalı ısı (kWhth)"],
            name="Ön ısıtmaya verilen ısı",
            line=dict(color=TEAL, width=3),
            fill="tozeroy",
            fillcolor="rgba(0,153,153,.10)",
            hovertemplate="%{x}<br>Faydalı ısı: %{y:.1f} kWhth<extra></extra>",
        ),
        secondary_y=False,
    )
    heat_fig.add_trace(
        go.Scatter(
            x=heat_storage["Saat"],
            y=heat_storage["Depo doluluğu (kWhth)"],
            name="Depo doluluğu",
            line=dict(color=NAVY, width=2.5),
            hovertemplate="%{x}<br>Depo: %{y:.0f} kWhth<extra></extra>",
        ),
        secondary_y=True,
    )
    heat_fig.update_yaxes(title_text="Isı gücü / saatlik enerji (kWth / kWhth)", secondary_y=False, rangemode="tozero")
    heat_fig.update_yaxes(title_text="Depo enerjisi (kWhth)", secondary_y=True, rangemode="tozero")
    heat_fig.update_xaxes(title_text="Saat")
    heat_fig.update_layout(title="Atık ısı kullanımı ve ısıl depo durumu")
    st.plotly_chart(style_figure(heat_fig, 410), width="stretch")

    heat_table = heat_storage[
        [
            "Saat",
            "Atık ısı kaynağı (kWth)",
            "Doğrudan kullanılan ısı (kWhth)",
            "Depoya giren ısı (kWhth)",
            "Depodan çıkan ısı (kWhth)",
            "Faydalı ısı (kWhth)",
            "Depo doluluğu (kWhth)",
            "Kullanılmayan ısı (kWhth)",
        ]
    ].round(2)
    st.dataframe(heat_table, width="stretch", hide_index=True)
    st.download_button(
        "Isıl depo saatlik dengesini CSV indir",
        data=heat_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="optihydro_thermal_storage_24h.csv",
        mime="text/csv",
        icon="⬇️",
    )
    st.markdown(
        "<div class='soft-card'><h4>Model sınırı</h4><p>Isı depolama bu MVP'de yalnızca sıcaklığı uygun bir ön ısıtma talebini besler. "
        "Eşanjör basınç kaybı, kirlenme, gerçek sıcaklık profili, termal ortam seçimi, saha yerleşimi ve 1.000°C+ sıcak bölgenin güvenliği modellenmez. "
        "Varsayılan kaynak/depo değerleri örnek girdilerdir; doğrulanmış tesis tasarımı veya tasarruf garantisi değildir.</p></div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Tab 4 — Edge-facing model view, carefully separated from live telemetry
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
    edge_c.metric("Öngörülen doğalgaz debisi", f"{format_number(totals['projected_gas_day_nm3']/24, 1)} Nm³/saat", "eşlenmiş faydalı ısı yükü")

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
# Tab 5 — target feasibility and constraint attribution
# -----------------------------------------------------------------------------
with tab_targets:
    st.markdown("<div class='section-label'>HEDEFİN FİZİKSEL UYGULANABİLİRLİĞİ</div>", unsafe_allow_html=True)
    st.subheader("Hedef tasarrufa hangi kaynaklarla ulaşılır?")
    st.caption(
        "H₂ gereksinimi, aynı proses ısısını koruyan doğalgaz/hidrojen alt ısıl değer dengesiyle hesaplanır. "
        "Isı geri kazanımının mevcut senaryodaki katkısı ayrıca gösterilir."
    )

    target_col1, target_col2, target_col3, target_col4 = st.columns(4)
    target_col1.metric(
        "Hedef için H₂ harmanı*",
        f"{format_number(target_blend_fraction * 100.0, 1)} vol %",
        "H₂ tek başına",
    )
    target_col2.metric(
        "Sürekli H₂ ihtiyacı*",
        f"{format_number(target_h2_kg_h, 1)} kg/saat",
        f"mevcut besleme {format_number(h2_supply, 1)} kg/saat",
    )
    target_col3.metric(
        "Elektrolizör gücü*",
        f"{format_number(target_electrolyzer_kw)} kW",
        f"gerekli yeşil pik ≈ {target_required_green_peak_display}",
    )
    target_col4.metric(
        "Toplam H₂ su ihtiyacı*",
        f"{format_number(target_water_l_h)} L/saat",
        f"mevcut kondensat {format_number(condensate)} L/saat",
    )

    if gas_flow <= 0.0:
        st.info(
            "Baz doğalgaz debisi 0 seçildiği için hedef ikame için H₂, elektrik ve su gereksinimi oluşmuyor. "
            "Kaynak darboğazı değerlendirmesi yapabilmek için pozitif bir fırın yükü girin."
        )
    elif not target_h2_only_feasible_under_cap:
        st.warning(
            f"Seçilen %{format_number(target_savings_pct)} doğalgaz ikamesi H₂ ile tek başına hedeflenirse yaklaşık "
            f"%{format_number(target_blend_fraction * 100.0, 1)} vol H₂ gerekir; bu, bu demodaki "
            f"%{format_number(effective_safety_cap_pct, 1)} harman tavanını aşar. H₂ dışı ısı verimliliği, "
            "uygun atık ısı geri kazanımı veya farklı proses önlemleri gerekir. Bu tavan standart onayı değildir."
        )
    else:
        st.info(
            "Hedef blend oranı demo tavanının altında; hedef yine de H₂ beslemesi, elektrik, su, saatlik profil "
            "ve proses uygunluğu kısıtlarıyla ayrıca doğrulanmalıdır."
        )
    if gas_flow > 0.0 and target_bottleneck_name == "H₂ harman tavanı":
        st.info(
            f"Seçilen hedef için ilk fiziksel engel H₂ hacim tavanıdır. H₂ dışı ısı geri kazanımı mevcut senaryoda "
            f"doğalgaz ikamesine %{format_number(totals['heat_saving_pct'], 1)} katkı sağlıyor; "
            f"mevcut toplam sonuç %{format_number(totals['gas_saving_pct'], 1)}."
        )
    elif gas_flow > 0.0:
        st.info(
            f"H₂ ile hedef gereksinimindeki en dar kaynak: {target_bottleneck_name} "
            f"(%{format_number(target_bottleneck_coverage_pct, 1)} kapasite karşılaması)."
        )

    target_gap_col, max_h2_col, actual_col = st.columns(3)
    target_gap_col.metric(
        "Mevcut senaryodan hedefe kalan",
        f"{format_number(target_gap_after_scenario_nm3_day)} Nm³/gün",
        f"yaklaşık {format_number(target_gap_heat_mwh_day, 2)} MWhth/gün faydalı ısı eşdeğeri",
    )
    max_h2_col.metric(
        "Demo tavanında H₂'nin teorik üst sınırı",
        f"%{format_number(totals['theoretical_safety_gas_saving_pct'], 1)} NG ikamesi",
        "sürekli ve diğer kaynaklar sınırsız varsayımı",
    )
    actual_col.metric(
        "Girdilerle hesaplanan toplam sonuç",
        f"%{format_number(totals['gas_saving_pct'], 1)} NG ikamesi",
        f"H₂ %{format_number(totals['h2_saving_pct'], 1)} + ısı %{format_number(totals['heat_saving_pct'], 1)}",
    )

    target_fig = go.Figure(
        data=[
            go.Bar(
                x=["Mevcut senaryo\nH₂ + atık ısı", "H₂ tavanı\nteorik üst sınır", "Seçilen hedef"],
                y=[
                    totals["gas_saving_pct"],
                    totals["theoretical_safety_gas_saving_pct"],
                    float(target_savings_pct),
                ],
                marker_color=[TEAL, AMBER, NAVY],
                text=[
                    f"{totals['gas_saving_pct']:.1f}%",
                    f"{totals['theoretical_safety_gas_saving_pct']:.1f}%",
                    f"{float(target_savings_pct):.1f}%",
                ],
                textposition="outside",
                hovertemplate="%{x}<br>Doğalgaz ikamesi: %{y:.2f}%<extra></extra>",
            )
        ]
    )
    target_fig.update_layout(title="Gerçek senaryo, H₂ tavanı ve hedef karşılaştırması", showlegend=False, yaxis_title="Baz NG debisine göre ikame (%)")
    target_fig.update_yaxes(
        rangemode="tozero",
        range=[
            0,
            max(
                12.0,
                target_savings_pct * 1.25,
                totals["theoretical_safety_gas_saving_pct"] * 1.25,
                totals["gas_saving_pct"] * 1.25,
            ),
        ],
    )
    st.plotly_chart(style_figure(target_fig, 350), width="stretch")

    resource_rows = [
        ("Gerekli H₂ harmanı (yalnız H₂ ile)", f"%{format_number(target_blend_fraction * 100.0, 1)} vol", f"demo tavanı %{format_number(effective_safety_cap_pct, 1)} vol"),
        ("Sürekli H₂ beslemesi", f"{format_number(target_h2_kg_h, 1)} kg/saat", f"{format_number(h2_supply, 1)} kg/saat mevcut"),
        ("Elektrolizör elektrik yükü", f"{format_number(target_electrolyzer_kw)} kW", f"{format_number(totals['effective_peak_power_kw'])} kW mevcut tepe"),
        ("Yeşil elektrik pik gücü girdisi", target_required_green_peak_display, f"{format_number(renewable_surplus_peak_kw)} kW senaryo girdisi"),
        ("Sürekli elektroliz suyu", f"{format_number(target_water_l_h)} L/saat", f"{format_number(condensate)} L/saat kondensat"),
        ("Mevcut senaryodaki atık ısı katkısı", f"{format_number(totals['heat_saved_gas_day_nm3'])} Nm³/gün", f"{format_number(totals['useful_heat_day_kwh'] / 1000.0, 2)} MWhth/gün faydalı"),
    ]
    st.dataframe(
        pd.DataFrame(resource_rows, columns=["Kısıt", "Hedef gereksinimi", "Mevcut senaryo"]),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "* Bu gereksinimler seçilen gaz ikamesini yalnızca H₂ ile ve sürekli yükte karşılama varsayımıdır. "
        "Gerçek planlama saatlik yeşil enerji, gaz bileşimi, brülör/OEM limitleri, ölçülmüş ısı profili ve proses talebiyle yapılmalıdır."
    )


# -----------------------------------------------------------------------------
# Tab 6 — financial scenario, with conditional CBAM exposure proxy
# -----------------------------------------------------------------------------
with tab_finance:
    st.markdown("<div class='section-label'>FİNANSAL SENARYO · KULLANICI GİRDİLERİ</div>", unsafe_allow_html=True)
    st.subheader("Yakıt tasarrufu ve koşullu CBAM etkisi")
    st.write(
        "Aşağıdaki değerler sidebar'daki fiyat/kapsam varsayımlarından hesaplanır. CBAM kalemi, "
        "sadece AB'ye ihraç edilen kapsam içi ürünlerin bu fırına tahsis edilebilen emisyon payı için "
        "basitleştirilmiş bir fiyat senaryosudur; hukuki vergi veya sertifika yükümlülüğü hesabı değildir."
    )

    fin1, fin2, fin3, fin4 = st.columns(4)
    fin1.metric(
        "Yıllık doğalgaz maliyeti tasarrufu",
        f"€{format_number(annual_fuel_savings_eur)}",
        f"{format_number(annual_fuel_savings_try)} TL senaryo",
    )
    fin2.metric(
        "CBAM senaryo farkı",
        f"€{format_number(annual_cbam_scenario_eur)}",
        f"{format_number(cbam_eligible_avoided_t, 1)} uygun ürün tCO₂/yıl",
    )
    fin3.metric(
        "Brüt yıllık finansal fırsat",
        f"€{format_number(annual_gross_financial_gain_eur)}",
        "yakıt + CBAM senaryo farkı",
    )
    if payback_years is None:
        payback_value = "Maliyet girdisi gerekli"
        payback_delta = "ROI hedefi: ≤ 1,2 yıl"
    else:
        payback_value = f"{format_number(payback_years, 2)} yıl"
        payback_delta = "1,2 yıl hedefinin altında" if payback_years <= 1.2 else "1,2 yıl hedefinin üzerinde"
    fin4.metric("Basit toplam sistem geri ödeme süresi", payback_value, payback_delta)

    financial_rows = [
        ("Günlük toplam doğalgaz tasarrufu", f"{format_number(totals['saved_gas_day_nm3'])} Nm³/gün"),
        ("H₂ kaynaklı tasarruf", f"{format_number(totals['h2_saved_gas_day_nm3'])} Nm³/gün"),
        ("Atık ısı kaynaklı tasarruf", f"{format_number(totals['heat_saved_gas_day_nm3'])} Nm³/gün"),
        ("Yıllık yakıt tasarrufu", f"{format_number(totals['saved_gas_day_nm3'] * int(operating_days))} Nm³/yıl"),
        ("Doğalgaz fiyat varsayımı", f"{format_number(natural_gas_price_try_nm3, 2)} TL/Nm³"),
        ("EUR/TRY senaryo kuru", f"{format_number(eur_try, 2)} TL/€"),
        ("Yıllık yakıt tasarrufu", f"€{format_number(annual_fuel_savings_eur)}"),
        ("Yıllık doğrudan CO₂ azaltımı", f"{format_number(totals['avoided_co2_year_t'], 1)} t/yıl"),
        ("CBAM fiyat varsayımı", f"€{format_number(cbam_price_eur_t, 2)}/tCO₂"),
        ("Uygun AB ürünü emisyon tahsisi", f"%{format_number(cbam_eligible_share_pct)}"),
        ("CBAM senaryo farkı", f"€{format_number(annual_cbam_scenario_eur)} / yıl"),
        ("Yazılım / entegrasyon CAPEX", f"€{format_number(software_setup_cost_eur)}"),
        ("Elektrolizör CAPEX", f"€{format_number(electrolyzer_capex_eur)}"),
        ("H₂ şartlandırma / depolama CAPEX", f"€{format_number(hydrogen_system_capex_eur)}"),
        ("Su arıtma CAPEX", f"€{format_number(water_system_capex_eur)}"),
        ("Isıl depo ve eşanjör CAPEX", f"€{format_number(thermal_storage_capex_eur)}"),
        ("Diğer saha CAPEX", f"€{format_number(other_site_capex_eur)}"),
        ("Toplam sistem CAPEX", f"€{format_number(total_system_capex_eur)}"),
        ("Yıllık H₂ elektriği tüketimi", f"{format_number(annual_hydrogen_electricity_mwh)} MWh/yıl"),
        ("Elektrik fırsat maliyeti", f"€{format_number(annual_electricity_cost_eur)} / yıl"),
        ("Yıllık lisans / OPEX", f"€{format_number(annual_software_opex_eur)}"),
        ("Yıllık saha bakım / işletme OPEX", f"€{format_number(annual_plant_opex_eur)}"),
        ("Toplam yıllık OPEX (elektrik dahil)", f"€{format_number(annual_total_opex_eur)}"),
        ("Yıllık net senaryo faydası", f"€{format_number(annual_net_financial_gain_eur)}"),
    ]
    st.dataframe(
        pd.DataFrame(financial_rows, columns=["Kalem", "Senaryo değeri"]),
        width="stretch",
        hide_index=True,
    )

    if cbam_eligible_share_pct <= 0:
        st.info(
            "CBAM payı %0 olduğu için toplam faydaya CBAM eklenmedi. Kapsam içi AB ihracatı ve "
            "ürün bazında doğrulanmış emisyon tahsisi varsa bu payı senaryolaştırın."
        )
    else:
        st.warning(
            "CBAM rakamı yalnızca yaklaşık fiyat farkı senaryosudur. Gerçek CBAM; kapsam içi ithal "
            "ürün, ürünün gömülü emisyonu, serbest tahsis düzeltmesi, üçüncü ülkede ödenen karbon "
            "fiyatı ve doğrulama koşullarına bağlıdır. "
            f"Buradaki €{format_number(cbam_price_eur_t, 2)}/t girdisi senaryo varsayımıdır, güncel resmi fiyat değildir.",
            icon="⚠️",
        )

    if total_system_capex_eur <= 0:
        st.caption(
            "1,2 yıl değeri bir hedef eşiğidir; toplam sistem CAPEX'i girilmediği için gerçekleşen "
            "geri ödeme hesaplanmamıştır. Yazılım retrofit yaklaşımı elektrolizör, H₂ depolama, "
            "su arıtma ve saha emniyet altyapısı maliyetlerini sıfırlamaz."
        )
    elif annual_net_financial_gain_eur <= 0:
        st.error("Yıllık lisans/OPEX, senaryo faydasına eşit veya daha yüksek; geri ödeme oluşmuyor.")

    st.markdown(
        "**Hesap:** Yakıt € = günlük toplam ikame Nm³ × işletme günü × TL/Nm³ ÷ TL/€; "
        "CBAM senaryo € = yıllık önlenen doğrudan CO₂ × uygun ürün tahsis payı × €/tCO₂; "
        "net fayda = yakıt + CBAM senaryosu − elektrik fırsat maliyeti − yıllık OPEX. "
        "Elektrik fiyatı 0 ise yalnızca gerçekten kısıntı/fazla enerji kabulünde kullanın."
    )


# -----------------------------------------------------------------------------
# Tab 7 — assumptions and equations
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
    st.latex(r"s_{NG}=1-\frac{(1-x)L_{NG}}{xL_{H_2}+(1-x)L_{NG}}")
    st.latex(r"SOC_{t+1}=\min(C,\ (SOC_t(1-\lambda)-Q_{dis}/\eta_{dis})_+ + Q_{ch}\eta_{ch})")
    st.latex(r"CO_{2,avoided}=(\Delta V_{NG,H_2}+\Delta V_{NG,heat})EF_{NG}")
    st.markdown(
        "Hedef sekmesi, istenen gaz hacmi ikamesi için gereken H₂ hacim oranını ve sürekli H₂/su/güç ihtiyacını "
        "ayrı hesaplar. Seçili demo harman tavanında H₂ tek başına teorik olarak en fazla "
        f"%{format_number(totals['theoretical_safety_gas_saving_pct'], 1)} doğalgaz hacim ikamesi sağlar; "
        "bu değer tesis güvenlik onayı değildir."
    )

    st.markdown("#### Varsayımlar")
    assumptions = [
        f"Doğalgaz LHV: {format_number(ng_lhv, 1)} MJ/Nm³; H₂ LHV: {format_number(H2_LHV_MJ_PER_NM3, 2)} MJ/Nm³ (DOE H₂ LHV × normal yoğunluk).",
        f"Doğrudan CO₂ faktörü: {format_number(ng_co2_factor, 3)} kg/Nm³; varsayılan türetilmiştir IPCC 56,1 kg/GJ ve 35,8 MJ/Nm³ değerlerinden.",
        f"H₂ elektroliz su bütçesi: {format_number(water_intensity, 1)} L/kg kullanıcı varsayımı; 9 kg/kg kimyasal stokiyometrik su ihtiyacını ve varsa ek proses payını ayırır.",
        f"O₂ yan ürün kütlesi: yaklaşık 8 kg O₂/kg H₂. Arıtmada faydalı sayılan miktar, ayarlanabilir etkin kullanım oranı ve tesis O₂ ihtiyacının küçüğüdür.",
        "Fan elektrik tasarrufu, kredilendirilen O₂ ihtiyacının fan gücüyle doğrusal ölçeklendiği basit senaryodur; blower performans eğrisi veya kütle transfer modeli değildir.",
        f"H₂ dispatch tavanı, kullanıcı girişinin min({format_number(H2_SIMULATION_HARD_CAP_PCT, 1)} vol %, girilen değer) olarak uygulanır. %20, bu MVP için seçilmiş demo kuralıdır; EN 746-2 veya ATEX'in tüm tesislere koyduğu evrensel harman limiti değildir ve flashback emniyet sistemi sayılmaz.",
        f"H₂ setpoint değişimi saatte en fazla {format_number(max_ramp_pct_per_hour, 2)} yüzde puanla sınırlandırılır; bu demo varsayımı OEM veya emniyet PLC limitinin yerine geçmez.",
        f"Yeşil enerji profili sentetik ve normalize edilmiş günlük şekildir; saatlik elektroliz gücü ayrıca {format_number(renewable_surplus_peak_kw)} kW pik giriş × bu profil ile kısıtlanır. Gerçek şebeke/RES/GES tahmini veya hava durumu kullanılmıyor.",
        f"Isı kaynağı profili: {heat_source_profile_name}; pik {format_number(heat_source_peak_kwth)} kWth, hedef ön ısıtma yükü {format_number(heat_demand_kwth)} kWth, depo {format_number(heat_storage_capacity_kwhth)} kWhth. Bunlar değiştirilebilir demo varsayımlarıdır; ölçüm veya eşanjör tasarımı değildir.",
        f"Atık ısı yalnızca kaynak sıcaklığı ({format_number(heat_source_temperature_c)} °C) hedef yükten ({format_number(heat_sink_temperature_c)} °C) yüksekse kullanılabilir kabul edilir; gerçek sıcaklık yaklaşımı, malzeme ve kirlenme sınırları modellenmez.",
        f"Ön ısıtma ısısının doğalgaz eşdeğerine çevrimi baz fırın yararlı ısı verimi %{format_number(furnace_efficiency_pct, 1)} varsayımını kullanır.",
        "Isı deposu saatlik modelinde önce doğrudan ısı kullanılır, sonra mevcut stok deşarj edilir ve kalan kaynakla depo şarj edilir. Depo gün sonu enerjisi gaz tasarrufu sayılmaz; yalnızca prosese verilen faydalı ısı NG eşdeğerine çevrilir.",
        "Scope 1 metriği yalnızca doğalgazın doğrudan yanma CO₂'sini kapsar. H₂ üretim elektriği ve yaşam döngüsü emisyonları düşülmez; NOₓ ve proses etkileri modellenmez.",
        "Isı geri kazanımı ve H₂ kaynaklı doğalgaz ikamesi ayrı hesaplanır; yalnızca kullanılan faydalı ısı toplam tasarrufa eklenir. Isı etkisi saha testiyle doğrulanmalıdır.",
        f"Finans varsayımları kullanıcı girdisidir: {format_number(natural_gas_price_try_nm3, 2)} TL/Nm³, {format_number(eur_try, 2)} TL/€, {format_number(cbam_price_eur_t, 2)} €/tCO₂ ve {int(operating_days)} işletme günü/yıl. Güncel piyasa veya CBAM sertifika fiyatı iddiası değildir.",
        f"CBAM senaryo faydası yalnızca kullanıcı tarafından tahsis edilen %{format_number(cbam_eligible_share_pct, 1)} emisyon payına uygulanır; kapsam içi AB ithalatı/ürünü ve doğrulanmış gömülü emisyonlar yoksa uygun pay %0 olmalıdır. Uygulama, gerçek mevzuat yükümlülüğünü hesaplamaz.",
        "Toplam CAPEX; yazılım, elektrolizör, H₂, su, termal depo ve saha girdilerinin toplamıdır. Net yıllık faydadan H₂ elektriği fırsat maliyeti ve yıllık OPEX düşülür.",
        "Basit geri ödeme, toplam CAPEX'in (yakıt + CBAM senaryo faydası − elektrik fırsat maliyeti − yıllık OPEX) değerine bölünmesidir. ≤1,2 yıl hedef olarak gösterilir; maliyet girdisi ve saha teklifi olmadan ROI iddiası değildir.",
    ]
    for item in assumptions:
        st.markdown(f"- {item}")

    st.markdown("#### Referanslar")
    st.markdown(
        "- [IPCC 2006 Guidelines, Volume 2, Chapter 1 — Table 1.4](https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_1_Ch1_Introduction.pdf): doğal gaz için 56.1 t CO₂/TJ varsayılan yanma faktörü.\n"
        "- [U.S. DOE — Hydrogen production: electrolysis](https://www.energy.gov/cmei/fuels/hydrogen-production-electrolysis): elektroliz suyu H₂ ve O₂'ye ayırır; gerçek sistem enerji tüketimi teknolojiye bağlıdır.\n"
        "- [U.S. DOE — hydrogen technical targets](https://www.energy.gov/cmei/fuels/doe-technical-targets-onboard-hydrogen-storage-light-duty-vehicles): H₂ LHV için 33.3 kWh/kg değeri.\n"
        "- [DIN EN ISO 13577-2:2023](https://www.dinmedia.de/en/standard/din-en-iso-13577-2/353012773): endüstriyel fırın sistemlerinde yanma ve yakıt elleçleme emniyet gereklilikleri; bu referans tek başına evrensel bir H₂ harman yüzdesi belirlemez.\n"
        "- [European Commission — CBAM definitive regime](https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism/cbam-definitive-regime_en): CBAM, kapsam içindeki AB'ye ithal edilen ürünlerin gömülü emisyonlarına uygulanır.\n"
        "- [European Commission — CBAM certificate pricing](https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism/price-cbam-certificates_en): sertifika fiyatı EU ETS açık artırma fiyatlarına bağlıdır; uygulamadaki sabit fiyat yalnızca senaryo girdisidir."
    )

st.divider()
st.markdown(
    "<div class='small-note'>OptiHydro AI · demonstrasyon arayüzü · senaryo hesapları saha verisiyle kalibre edilmeden "
    "yatırım kararı, proses emniyeti veya emisyon doğrulaması için kullanılmamalıdır.</div>",
    unsafe_allow_html=True,
)
