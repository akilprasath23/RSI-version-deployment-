import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoRSI Dashboard | Post-Mining Restoration",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

SITE_LEVELS = ["UMD", "PMD", "AMD", "RE"]
SITE_COLORS = {
    "UMD": "#d7301f",
    "PMD": "#fdae61",
    "AMD": "#1a9850",
    "RE": "#2c7bb6",
}
DOMAIN_LEVELS = ["Vegetation", "Fauna", "Soil nutrients", "Heavy metals"]

DEFAULT_SOIL_POS = ["OC_pct", "AN_mg_kg", "Ap_mg_kg", "AK_mg_kg"]
DEFAULT_SOIL_NEG = [
    "Al_mg_kg", "As_mg_kg", "B_mg_kg", "Ba_mg_kg", "Cd_mg_kg",
    "Co_mg_kg", "Cr_mg_kg", "Cu_mg_kg", "Fe_mg_kg", "Mn_mg_kg",
    "Ni_mg_kg", "Pb_mg_kg", "Se_mg_kg", "V_mg_kg", "Zn_mg_kg",
]


# ------------------------------------------------------------------------------
# CORE EcoRSI FUNCTIONS
# ------------------------------------------------------------------------------
def compute_eco_rsi(df, site_col, umd_label, re_label, pos_cols, neg_cols):
    """Linear benchmark normalization between UMD (0) and RE (1), clipped to [0, 1]."""
    umd_sub = df[df[site_col] == umd_label]
    re_sub = df[df[site_col] == re_label]

    result = pd.DataFrame({site_col: df[site_col].values})
    benchmarks = []

    for col in pos_cols + neg_cols:
        x = pd.to_numeric(df[col], errors="coerce").values
        d_mean = pd.to_numeric(umd_sub[col], errors="coerce").mean()
        r_mean = pd.to_numeric(re_sub[col], errors="coerce").mean()

        if abs(r_mean - d_mean) < 1e-9:
            continue

        if col in pos_cols:
            norm = (x - d_mean) / (r_mean - d_mean)
            direction = "positive"
        else:
            norm = (d_mean - x) / (d_mean - r_mean)
            direction = "negative"

        result[f"{col}_RSI"] = np.clip(norm, 0.0, 1.0)
        benchmarks.append({
            "Indicator": col,
            "Direction": direction,
            "D_UMD_Mean": d_mean,
            "R_RE_Mean": r_mean,
        })

    rsi_cols = [c for c in result.columns if c.endswith("_RSI")]
    result["RSI"] = result[rsi_cols].mean(axis=1)
    return result, pd.DataFrame(benchmarks)


@st.cache_data
def run_bootstrap(domain_scores_df, weights, n_boot=2000, seed=2026):
    """Bootstrap composite RSI across sites (n=2,000 replicates)."""
    rng = np.random.default_rng(seed)
    dom_dict = {
        (s, d): grp["I"].values
        for (s, d), grp in domain_scores_df.groupby(["Site", "Domain"])
    }

    boot_rows = []
    for s in SITE_LEVELS:
        total = np.zeros(n_boot)
        for d in DOMAIN_LEVELS:
            vals = dom_dict.get((s, d), np.array([0.0]))
            draws = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
            total += weights[d] * draws
        for val in total:
            boot_rows.append({"Site": s, "RSI": val})

    boot_df = pd.DataFrame(boot_rows)
    boot_df["Site"] = pd.Categorical(boot_df["Site"], categories=SITE_LEVELS, ordered=True)
    return boot_df


@st.cache_data
def run_soil_loio_sensitivity(soil_df, pos_cols, neg_cols):
    """Leave-One-Indicator-Out (LOIO) sensitivity analysis for soil indicators."""
    base_res, _ = compute_eco_rsi(soil_df, "Site", "UMD", "RE", pos_cols, neg_cols)
    base_rank = base_res.groupby("Site")["RSI"].mean().sort_values().index.tolist()

    rows = []
    for ind in pos_cols + neg_cols:
        new_pos = [c for c in pos_cols if c != ind]
        new_neg = [c for c in neg_cols if c != ind]
        new_res, _ = compute_eco_rsi(soil_df, "Site", "UMD", "RE", new_pos, new_neg)

        corr = np.corrcoef(base_res["RSI"], new_res["RSI"])[0, 1]
        mad = np.mean(np.abs(base_res["RSI"] - new_res["RSI"]))
        pct_change = np.mean(
            np.abs(base_res["RSI"] - new_res["RSI"]) / np.maximum(base_res["RSI"], 1e-6)
        ) * 100
        new_rank = new_res.groupby("Site")["RSI"].mean().sort_values().index.tolist()

        rows.append({
            "Removed_Indicator": ind,
            "Domain": "Soil nutrients" if ind in pos_cols else "Heavy metals",
            "Correlation": round(corr, 4),
            "MAD": round(mad, 4),
            "Percent_Change": round(pct_change, 2),
            "Ranking_Preserved": base_rank == new_rank,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def train_soil_rf_model(soil_df, feature_cols, target_series):
    """Train Random Forest Regressor to predict Soil RSI from raw soil parameters."""
    X = soil_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = target_series.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2026, stratify=soil_df["Site"]
    )
    model = RandomForestRegressor(n_estimators=300, random_state=2026)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
    }
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    return model, metrics, importance


def load_csv_input(uploaded_file, default_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None


# ------------------------------------------------------------------------------
# SIDEBAR: DATA UPLOAD & WEIGHT CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.title("🌿 EcoRSI Controls")
st.sidebar.markdown("Upload your datasets or use the default files in `./data/`.")

up_soil = st.sidebar.file_uploader("Soil Raw Data (CSV)", type=["csv"])
up_flora = st.sidebar.file_uploader("Flora RSI Data (CSV)", type=["csv"])
up_fauna = st.sidebar.file_uploader("Fauna RSI Data (CSV)", type=["csv"])

soil_df = load_csv_input(up_soil, "data/1.2_Soil_Data_with_Correct_Dates.csv")
flora_df = load_csv_input(up_flora, "data/28_flora_rsi.csv")
fauna_df = load_csv_input(up_fauna, "data/28_fauna_rsi.csv")

if soil_df is None or flora_df is None or fauna_df is None:
    st.warning(
        "Please upload all three CSV files in the sidebar (or place them in the `data/` folder) to launch the dashboard."
    )
    st.stop()

# Clean site labels
for d in (soil_df, flora_df, fauna_df):
    d["Site"] = d["Site"].astype(str).str.strip()

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Domain Weights")
w_veg = st.sidebar.slider("Vegetation", 0.0, 1.0, 0.25, 0.05)
w_fau = st.sidebar.slider("Fauna", 0.0, 1.0, 0.25, 0.05)
w_nut = st.sidebar.slider("Soil nutrients", 0.0, 1.0, 0.25, 0.05)
w_hm = st.sidebar.slider("Heavy metals", 0.0, 1.0, 0.25, 0.05)

w_sum = w_veg + w_fau + w_nut + w_hm
if w_sum == 0:
    st.sidebar.error("At least one domain weight must be > 0.")
    st.stop()

weights = {
    "Vegetation": w_veg / w_sum,
    "Fauna": w_fau / w_sum,
    "Soil nutrients": w_nut / w_sum,
    "Heavy metals": w_hm / w_sum,
}
st.sidebar.caption(
    f"Normalized weights: V={weights['Vegetation']:.2f}, F={weights['Fauna']:.2f}, "
    f"S={weights['Soil nutrients']:.2f}, H={weights['Heavy metals']:.2f}"
)

# Indicator selection
avail_pos = [c for c in DEFAULT_SOIL_POS if c in soil_df.columns]
avail_neg = [c for c in DEFAULT_SOIL_NEG if c in soil_df.columns]

sel_pos = st.sidebar.multiselect("Positive Soil Indicators", avail_pos, default=avail_pos)
sel_neg = st.sidebar.multiselect("Negative Soil Indicators", avail_neg, default=avail_neg)

if not sel_pos or not sel_neg:
    st.error("Please select at least one positive and one negative soil indicator.")
    st.stop()


# ------------------------------------------------------------------------------
# COMPUTE RSI PIPELINE
# ------------------------------------------------------------------------------
soil_nut_res, bench_pos = compute_eco_rsi(soil_df, "Site", "UMD", "RE", sel_pos, [])
soil_hm_res, bench_neg = compute_eco_rsi(soil_df, "Site", "UMD", "RE", [], sel_neg)
soil_all_res, _ = compute_eco_rsi(soil_df, "Site", "UMD", "RE", sel_pos, sel_neg)
benchmarks_df = pd.concat([bench_pos, bench_neg], ignore_index=True)

# Assemble unified domain_scores table
flora_col = "floral_richness_RSI" if "floral_richness_RSI" in flora_df.columns else "RSI"
fauna_col = "faunal_richness_RSI" if "faunal_richness_RSI" in fauna_df.columns else "RSI"

domain_scores = pd.concat([
    pd.DataFrame({"Site": flora_df["Site"], "Domain": "Vegetation", "I": flora_df[flora_col]}),
    pd.DataFrame({"Site": fauna_df["Site"], "Domain": "Fauna", "I": fauna_df[fauna_col]}),
    pd.DataFrame({"Site": soil_nut_res["Site"], "Domain": "Soil nutrients", "I": soil_nut_res["RSI"]}),
    pd.DataFrame({"Site": soil_hm_res["Site"], "Domain": "Heavy metals", "I": soil_hm_res["RSI"]}),
], ignore_index=True)

domain_scores["Site"] = pd.Categorical(domain_scores["Site"], categories=SITE_LEVELS, ordered=True)
domain_scores["Domain"] = pd.Categorical(domain_scores["Domain"], categories=DOMAIN_LEVELS, ordered=True)

domain_summary = (
    domain_scores.groupby(["Site", "Domain"], observed=False)["I"]
    .agg(n="count", Mean="mean", Median="median", SD="std")
    .reset_index()
)

boot_df = run_bootstrap(domain_scores, weights, n_boot=2000)

site_summary = (
    boot_df.groupby("Site", observed=False)["RSI"]
    .agg(
        Mean="mean",
        Median="median",
        SD="std",
        CI95_Low=lambda x: np.percentile(x, 2.5),
        CI95_High=lambda x: np.percentile(x, 97.5),
    )
    .reset_index()
)


# ------------------------------------------------------------------------------
# DASHBOARD HEADER & TABS
# ------------------------------------------------------------------------------
st.title("🌿 Restoration State Index (RSI) Dashboard")
st.markdown(
    "Interactive ecological recovery assessment, sensitivity analysis, and restoration scenario predictor "
    "for post-mining landscapes (**Dongri Buzurg Manganese Mine**)."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & Composite RSI",
    "🔍 Domain & Indicator Profiles",
    "🧪 Sensitivity Analysis",
    "🔮 RSI Predictor & Simulator",
])

# ==============================================================================
# TAB 1: OVERVIEW & COMPOSITE RSI
# ==============================================================================
with tab1:
    st.subheader("Site-Wise Composite RSI Summary")
    cols = st.columns(4)
    for i, s in enumerate(SITE_LEVELS):
        row = site_summary[site_summary["Site"] == s].iloc[0]
        cols[i].metric(
            label=f"{s} Composite RSI",
            value=f"{row['Mean']:.3f}",
            delta=f"± {row['SD']:.4f} (SD)",
            delta_color="off",
        )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig_bar = px.bar(
            site_summary,
            x="Site",
            y="Mean",
            error_y="SD",
            color="Site",
            color_discrete_map=SITE_COLORS,
            text=site_summary["Mean"].map(lambda v: f"{v:.2f}"),
            title="Composite Restoration State Index (Mean ± Bootstrap SD)",
            labels={"Mean": "Restoration State Index (RSI)"},
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(yaxis_range=[0, 1.12], showlegend=False, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        fig_hist = px.histogram(
            boot_df,
            x="RSI",
            color="Site",
            facet_col="Site",
            facet_col_wrap=2,
            color_discrete_map=SITE_COLORS,
            nbins=40,
            histnorm="probability density",
            title="Bootstrap Distribution (n = 2,000)",
        )
        fig_hist.update_layout(showlegend=False, template="plotly_white")
        fig_hist.update_xaxes(range=[0, 1])
        st.plotly_chart(fig_hist, use_container_width=True)

    st.dataframe(site_summary.round(4), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Site RSI Summary (CSV)",
        site_summary.to_csv(index=False).encode("utf-8"),
        "site_RSI_summary.csv",
        "text/csv",
    )

# ==============================================================================
# TAB 2: DOMAIN & INDICATOR PROFILES
# ==============================================================================
with tab2:
    st.subheader("Domain-Level Recovery Trajectories")
    fig_dom = px.box(
        domain_scores,
        x="Site",
        y="I",
        color="Site",
        facet_col="Domain",
        color_discrete_map=SITE_COLORS,
        title="Domain-Wise RSI Distribution Across Sites",
        labels={"I": "Domain RSI Score"},
    )
    fig_dom.update_layout(yaxis_range=[0, 1.05], showlegend=False, template="plotly_white")
    st.plotly_chart(fig_dom, use_container_width=True)

    st.subheader("Soil Indicator-Level Normalized Scores")
    soil_rsi_cols = [c for c in soil_all_res.columns if c.endswith("_RSI")]
    soil_melt = soil_all_res.melt(
        id_vars=["Site"], value_vars=soil_rsi_cols, var_name="Indicator", value_name="Score"
    )
    soil_melt["Indicator"] = soil_melt["Indicator"].str.replace("_RSI", "", regex=False)
    soil_melt["Site"] = pd.Categorical(soil_melt["Site"], categories=SITE_LEVELS, ordered=True)

    selected_ind = st.selectbox("Inspect specific soil indicator:", sorted(soil_melt["Indicator"].unique()))
    fig_ind = px.box(
        soil_melt[soil_melt["Indicator"] == selected_ind],
        x="Site",
        y="Score",
        color="Site",
        color_discrete_map=SITE_COLORS,
        points="all",
        title=f"Normalized RSI Score: {selected_ind}",
    )
    fig_ind.update_layout(yaxis_range=[0, 1.05], showlegend=False, template="plotly_white")
    st.plotly_chart(fig_ind, use_container_width=True)

    with st.expander("View Benchmark Values (D = UMD Mean, R = RE Mean)"):
        st.dataframe(benchmarks_df.round(4), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: SENSITIVITY ANALYSIS
# ==============================================================================
with tab3:
    st.subheader("Leave-One-Indicator-Out (LOIO) Soil Sensitivity Analysis")
    sens_df = run_soil_loio_sensitivity(soil_df, sel_pos, sel_neg)

    fig_sens = px.bar(
        sens_df.sort_values("Correlation"),
        x="Correlation",
        y="Removed_Indicator",
        color="Domain",
        orientation="h",
        text="Correlation",
        title="Pearson Correlation with Baseline Soil RSI After Removing Each Indicator",
    )
    fig_sens.update_layout(xaxis_range=[0.90, 1.01], template="plotly_white", height=600)
    st.plotly_chart(fig_sens, use_container_width=True)

    st.dataframe(sens_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Sensitivity Results (CSV)",
        sens_df.to_csv(index=False).encode("utf-8"),
        "soil_sensitivity_analysis.csv",
        "text/csv",
    )

# ==============================================================================
# TAB 4: RSI PREDICTOR & SCENARIO SIMULATOR
# ==============================================================================
with tab4:
    st.subheader("🔮 Interactive Restoration Scenario Predictor")
    st.markdown(
        "Adjust biotic scores and raw soil parameters below to simulate a restoration intervention "
        "and predict the resulting **Composite RSI** (exact benchmark formula + Random Forest soil predictor)."
    )

    # Train Random Forest model on raw soil parameters
    all_soil_feats = sel_pos + sel_neg
    rf_model, rf_metrics, rf_imp = train_soil_rf_model(soil_df, all_soil_feats, soil_all_res["RSI"])

    col_in, col_out = st.columns([1.3, 1])

    with col_in:
        st.markdown("**1. Biotic Domain Inputs (0–1 Normalized)**")
        in_veg = st.slider("Vegetation Sub-RSI", 0.0, 1.0, 0.50, 0.01)
        in_fau = st.slider("Fauna Sub-RSI", 0.0, 1.0, 0.71, 0.01)

        st.markdown("**2. Raw Soil Parameter Inputs (Pre-filled with AMD Means)**")
        amd_means = soil_df[soil_df["Site"] == "AMD"][all_soil_feats].mean()
        user_soil_vals = {}

        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.caption("Soil Nutrients (Positive)")
            for col in sel_pos:
                min_v = float(soil_df[col].min())
                max_v = float(soil_df[col].max())
                user_soil_vals[col] = st.number_input(
                    col, min_value=0.0, max_value=max_v * 1.5, value=float(amd_means[col]), format="%.2f"
                )
        with c_neg:
            st.caption("Heavy Metals (Negative — Key Metals)")
            # Show top 6 metals as inputs; keep remaining at AMD mean for clean UI
            key_metals = [m for m in ["Fe_mg_kg", "Mn_mg_kg", "Cr_mg_kg", "Pb_mg_kg", "Cd_mg_kg", "Ni_mg_kg"] if m in sel_neg]
            for col in sel_neg:
                if col in key_metals:
                    max_v = float(soil_df[col].max())
                    user_soil_vals[col] = st.number_input(
                        col, min_value=0.0, max_value=max_v * 1.5, value=float(amd_means[col]), format="%.2f"
                    )
                else:
                    user_soil_vals[col] = float(amd_means[col])

    # Compute Exact Mathematical RSI for user scenario
    bench_map = benchmarks_df.set_index("Indicator").to_dict(orient="index")
    nut_scores = []
    for col in sel_pos:
        d, r = bench_map[col]["D_UMD_Mean"], bench_map[col]["R_RE_Mean"]
        nut_scores.append(np.clip((user_soil_vals[col] - d) / (r - d), 0.0, 1.0))

    hm_scores = []
    for col in sel_neg:
        d, r = bench_map[col]["D_UMD_Mean"], bench_map[col]["R_RE_Mean"]
        hm_scores.append(np.clip((d - user_soil_vals[col]) / (d - r), 0.0, 1.0))

    pred_nut = float(np.mean(nut_scores))
    pred_hm = float(np.mean(hm_scores))
    pred_composite = (
        weights["Vegetation"] * in_veg
        + weights["Fauna"] * in_fau
        + weights["Soil nutrients"] * pred_nut
        + weights["Heavy metals"] * pred_hm
    )

    # Random Forest prediction for overall Soil RSI
    rf_input = pd.DataFrame([user_soil_vals])[all_soil_feats]
    rf_soil_pred = float(rf_model.predict(rf_input)[0])

    with col_out:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred_composite,
                number={"valueformat": ".3f"},
                title={"text": "Predicted Composite RSI"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "#1a9850"},
                    "steps": [
                        {"range": [0, 0.25], "color": "#f8d7da"},
                        {"range": [0.25, 0.55], "color": "#fff3cd"},
                        {"range": [0.55, 0.85], "color": "#d4edda"},
                        {"range": [0.85, 1.0], "color": "#cce5ff"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("**Predicted Domain Sub-Indices:**")
        st.write(f"- **Vegetation:** `{in_veg:.3f}`")
        st.write(f"- **Fauna:** `{in_fau:.3f}`")
        st.write(f"- **Soil Nutrients (Exact):** `{pred_nut:.3f}`")
        st.write(f"- **Heavy Metals (Exact):** `{pred_hm:.3f}`")
        st.info(
            f"**Random Forest ML Soil RSI Prediction:** `{rf_soil_pred:.3f}`  \n"
            f"*(Model Test $R^2$ = `{rf_metrics['r2']:.3f}`, MAE = `{rf_metrics['mae']:.4f}`)*"
        )

    st.markdown("---")
    st.subheader("Random Forest Feature Importance (Soil RSI Drivers)")
    fig_imp = px.bar(
        rf_imp.head(12).sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 12 Soil Parameters Driving Soil RSI",
    )
    fig_imp.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_imp, use_container_width=True)