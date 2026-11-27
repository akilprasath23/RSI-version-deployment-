import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="EcoRSI Calculator", page_icon="🌿", layout="wide")

# Exact Dongri Buzurg Benchmarks (D = UMD mean, R = RE mean, AMD = default pre-fill)
BENCHMARKS = {
    "Vegetation": {
        "Floral_Richness": {"label": "Floral Richness (species count)", "dir": "pos", "D": 1.8813, "R": 76.6013, "AMD": 39.36},
    },
    "Fauna": {
        "Faunal_Richness": {"label": "Faunal Richness (bird species)", "dir": "pos", "D": 2.6400, "R": 41.3200, "AMD": 30.12},
    },
    "Soil nutrients": {
        "OC_pct":   {"label": "Organic Carbon (%)",      "dir": "pos", "D": 0.1474, "R": 0.4567, "AMD": 0.3698},
        "AN_mg_kg": {"label": "Available N (mg/kg)",     "dir": "pos", "D": 23.7567, "R": 52.8683, "AMD": 44.0517},
        "Ap_mg_kg": {"label": "Available P (mg/kg)",     "dir": "pos", "D": 74.5782, "R": 129.0868, "AMD": 110.7818},
        "AK_mg_kg": {"label": "Available K (mg/kg)",     "dir": "pos", "D": 29.4733, "R": 87.4983, "AMD": 73.8133},
    },
    "Heavy metals": {
        "Al_mg_kg": {"label": "Al (mg/kg)", "dir": "neg", "D": 16757.6050, "R": 6544.8900, "AMD": 10039.2183},
        "As_mg_kg": {"label": "As (mg/kg)", "dir": "neg", "D": 51.1892,    "R": 11.7710,   "AMD": 25.3228},
        "B_mg_kg":  {"label": "B (mg/kg)",  "dir": "neg", "D": 59.2405,    "R": 18.4518,   "AMD": 28.6518},
        "Ba_mg_kg": {"label": "Ba (mg/kg)", "dir": "neg", "D": 826.0850,   "R": 243.1600,  "AMD": 446.1933},
        "Cd_mg_kg": {"label": "Cd (mg/kg)", "dir": "neg", "D": 0.3929,     "R": 0.1104,    "AMD": 0.2079},
        "Co_mg_kg": {"label": "Co (mg/kg)", "dir": "neg", "D": 28.2715,    "R": 8.2313,    "AMD": 15.0055},
        "Cr_mg_kg": {"label": "Cr (mg/kg)", "dir": "neg", "D": 93.0850,    "R": 27.8670,   "AMD": 49.1562},
        "Cu_mg_kg": {"label": "Cu (mg/kg)", "dir": "neg", "D": 30.8957,    "R": 8.9282,    "AMD": 16.1928},
        "Fe_mg_kg": {"label": "Fe (mg/kg)", "dir": "neg", "D": 89418.4533, "R": 23473.6017,"AMD": 44672.1017},
        "Mn_mg_kg": {"label": "Mn (mg/kg)", "dir": "neg", "D": 7258.9033,  "R": 1799.1950, "AMD": 3371.2433},
        "Ni_mg_kg": {"label": "Ni (mg/kg)", "dir": "neg", "D": 51.5627,    "R": 13.2920,   "AMD": 25.7773},
        "Pb_mg_kg": {"label": "Pb (mg/kg)", "dir": "neg", "D": 19.4012,    "R": 5.2825,    "AMD": 9.6697},
        "Se_mg_kg": {"label": "Se (mg/kg)", "dir": "neg", "D": 29.0908,    "R": 7.6227,    "AMD": 13.8920},
        "V_mg_kg":  {"label": "V (mg/kg)",  "dir": "neg", "D": 57.4982,    "R": 26.8078,   "AMD": 35.8327},
        "Zn_mg_kg": {"label": "Zn (mg/kg)", "dir": "neg", "D": 73.5838,    "R": 21.0850,   "AMD": 35.8568},
    },
}


def normalize(val, d_val, r_val, direction):
    if direction == "pos":
        score = (val - d_val) / (r_val - d_val)
    else:
        score = (d_val - val) / (d_val - r_val)
    return float(np.clip(score, 0.0, 1.0))


st.title("🌿 Restoration State Index (RSI) Calculator")
st.markdown("Insert your **Flora**, **Fauna**, and **Soil** measurements below to calculate the exact Domain Sub-Indices and Composite RSI.")

with st.form("rsi_calc_form"):
    user_vals = {}

    st.subheader("1. Biotic Parameters")
    c1, c2 = st.columns(2)
    with c1:
        meta = BENCHMARKS["Vegetation"]["Floral_Richness"]
        user_vals["Floral_Richness"] = st.number_input(meta["label"], min_value=0.0, value=meta["AMD"], step=1.0, format="%.2f")
    with c2:
        meta = BENCHMARKS["Fauna"]["Faunal_Richness"]
        user_vals["Faunal_Richness"] = st.number_input(meta["label"], min_value=0.0, value=meta["AMD"], step=1.0, format="%.2f")

    st.subheader("2. Soil Nutrients (Positive Indicators)")
    nut_cols = st.columns(4)
    for idx, (key, meta) in enumerate(BENCHMARKS["Soil nutrients"].items()):
        with nut_cols[idx]:
            user_vals[key] = st.number_input(meta["label"], min_value=0.0, value=meta["AMD"], format="%.3f")

    st.subheader("3. Soil Heavy Metals (Negative Indicators)")
    hm_cols = st.columns(5)
    for idx, (key, meta) in enumerate(BENCHMARKS["Heavy metals"].items()):
        with hm_cols[idx % 5]:
            user_vals[key] = st.number_input(meta["label"], min_value=0.0, value=meta["AMD"], format="%.2f")

    submitted = st.form_submit_button("Calculate RSI", use_container_width=True)

# Compute Indicator & Domain Scores
indicator_rows = []
domain_scores = {}

for domain, indicators in BENCHMARKS.items():
    scores = []
    for key, meta in indicators.items():
        val = user_vals[key]
        sc = normalize(val, meta["D"], meta["R"], meta["dir"])
        scores.append(sc)
        indicator_rows.append({
            "Domain": domain,
            "Indicator": meta["label"],
            "Inserted Value": val,
            "UMD Benchmark (0)": meta["D"],
            "RE Benchmark (1)": meta["R"],
            "Normalized Score": round(sc, 4),
        })
    domain_scores[domain] = float(np.mean(scores))

composite_rsi = float(np.mean(list(domain_scores.values())))

# Display Results
st.markdown("---")
col_gauge, col_metrics = st.columns([1.2, 1])

with col_gauge:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=composite_rsi,
        number={"valueformat": ".4f"},
        title={"text": "Composite Restoration State Index (RSI)"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#1a9850"},
            "steps": [
                {"range": [0.00, 0.25], "color": "#f8d7da"},
                {"range": [0.25, 0.55], "color": "#fff3cd"},
                {"range": [0.55, 0.85], "color": "#d4edda"},
                {"range": [0.85, 1.00], "color": "#cce5ff"},
            ],
        },
    ))
    fig.update_layout(height=320, margin=dict(t=50, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

with col_metrics:
    st.subheader("Domain Sub-Indices (Equal Weight = 0.25)")
    m1, m2 = st.columns(2)
    m1.metric("Vegetation Sub-RSI", f"{domain_scores['Vegetation']:.4f}")
    m2.metric("Fauna Sub-RSI", f"{domain_scores['Fauna']:.4f}")
    m1.metric("Soil Nutrients Sub-RSI", f"{domain_scores['Soil nutrients']:.4f}")
    m2.metric("Heavy Metals Sub-RSI", f"{domain_scores['Heavy metals']:.4f}")

with st.expander("View Full Indicator-Level Breakdown"):
    st.dataframe(pd.DataFrame(indicator_rows), use_container_width=True, hide_index=True)