"""
Insurance Reserving Model  —  Streamlit App
Chain-Ladder + Bornhuetter-Ferguson | 50/50 Blend
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Insurance Reserving Model")
st.title("Insurance Reserving Model")
st.markdown("**Chain-Ladder + Bornhuetter-Ferguson | 50/50 Blend**")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Model Parameters")

valuation_year = st.sidebar.number_input(
    "Valuation Year", value=2023, step=1,
    help="Claims settled after this year are excluded from the triangle.",
)

tail_factor = st.sidebar.number_input(
    "Tail Factor", min_value=1.0, max_value=10.0, value=1.0, step=0.01,
    help="Multiplied onto the last CDF. 1.0 = no tail.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Premium / BF Settings")

premium_source = st.sidebar.radio(
    "Premium Source for BF",
    ["Derive from data (mean CL ultimates)", "Enter manually"],
)

manual_premium = None
elr = None
if premium_source == "Enter manually":
    manual_premium = st.sidebar.number_input(
        "Annual Premium ($)", min_value=0.0, value=1_000_000.0, step=10_000.0,
    )
    elr = st.sidebar.number_input(
        "Expected Loss Ratio", min_value=0.0, max_value=5.0, value=0.65, step=0.01,
    )
    st.sidebar.caption(f"BF Expected Ultimate = ${elr * manual_premium:,.0f}")
else:
    st.sidebar.caption("BF Expected Ultimate = mean of CL ultimates.")


# ── Core calculation ───────────────────────────────────────────────────────────
def compute_reserves(df, valuation_year, tail=1.0,
                     premium_source="derive", manual_premium=None,
                     elr=None, premium_series=None):

    df = df[df["Settlement_Year"] <= valuation_year].copy()
    if df.empty:
        return None

    latest_paid = df.groupby("Accident_Year")["Amount"].sum()
    accident_years = sorted(latest_paid.index.tolist())
    diag_lag = {ay: int(valuation_year - ay) for ay in accident_years}

    grouped = df.groupby(["Accident_Year", "Development_Lag"])["Amount"].sum().reset_index()
    tri_inc = grouped.pivot(index="Accident_Year", columns="Development_Lag", values="Amount").fillna(0)
    tri_inc.columns = [int(c) for c in tri_inc.columns]
    tri_inc = tri_inc.sort_index().sort_index(axis=1)
    tri_cum = tri_inc.cumsum(axis=1)
    lags = sorted(tri_cum.columns.tolist())

    # Volume-weighted LDFs
    ldfs = {}
    for i in range(len(lags) - 1):
        c_lag, n_lag = lags[i], lags[i + 1]
        eligible = [ay for ay in tri_cum.index if diag_lag.get(ay, -1) >= n_lag]
        if not eligible:
            ldfs[f"{c_lag}-{n_lag}"] = 1.0
            continue
        sub = tri_cum.loc[eligible]
        mask = sub[c_lag].notna() & sub[n_lag].notna() & (sub[c_lag] > 0)
        ldfs[f"{c_lag}-{n_lag}"] = (
            sub.loc[mask, n_lag].sum() / sub.loc[mask, c_lag].sum()
            if mask.sum() > 0 else 1.0
        )

    ldf_values = list(ldfs.values())
    cdfs = {}
    for si, lag in enumerate(lags):
        cdf = tail
        for j in range(si, len(ldf_values)):
            cdf *= ldf_values[j]
        cdfs[lag] = cdf
    max_lag = max(lags) if lags else 0

    def get_cdf(d):
        if d in cdfs: return cdfs[d]
        if d > max_lag: return tail
        return cdfs.get(min(lags), tail) * 100

    # CL
    cl_ultimates, cl_reserves = {}, {}
    for ay in accident_years:
        latest = float(latest_paid[ay])
        cdf = get_cdf(diag_lag[ay])
        ult = latest * cdf
        cl_ultimates[ay] = ult
        cl_reserves[ay] = ult - latest

    # BF expected ultimate
    if premium_source == "manual" and manual_premium is not None and elr is not None:
        def get_prem(ay):
            if premium_series is not None and ay in premium_series.index:
                return float(premium_series[ay])
            return float(manual_premium)
        eu_by_ay = {ay: elr * get_prem(ay) for ay in accident_years}
        eu_display = elr * float(manual_premium)
    else:
        valid = [v for v in cl_ultimates.values() if not np.isnan(v)]
        eu_val = float(np.mean(valid)) if valid else 0.0
        eu_by_ay = {ay: eu_val for ay in accident_years}
        eu_display = eu_val

    # BF
    bf_ultimates, bf_reserves = {}, {}
    for ay in accident_years:
        latest = float(latest_paid[ay])
        cdf = get_cdf(diag_lag[ay])
        pct_dev = 1.0 / cdf if cdf > 0 else 1.0
        ult = latest + eu_by_ay[ay] * (1.0 - pct_dev)
        bf_ultimates[ay] = ult
        bf_reserves[ay] = ult - latest

    # Results table
    rows = []
    for ay in accident_years:
        rows.append({
            "Accident_Year":   ay,
            "Latest_Paid":     float(latest_paid[ay]),
            "Diagonal_Lag":    diag_lag[ay],
            "CDF":             get_cdf(diag_lag[ay]),
            "CL_Ultimate":     cl_ultimates[ay],
            "CL_Reserve":      cl_reserves[ay],
            "BF_Ultimate":     bf_ultimates[ay],
            "BF_Reserve":      bf_reserves[ay],
            "Blend_50_50_Ult": (cl_ultimates[ay] + bf_ultimates[ay]) / 2,
            "Blend_50_50_Res": (cl_reserves[ay]  + bf_reserves[ay])  / 2,
        })
    results = pd.DataFrame(rows).set_index("Accident_Year")

    # Cumulative paid triangle (upper only)
    paid_tri = tri_cum.copy().astype(float)
    for ay in paid_tri.index:
        d = diag_lag.get(ay, -1)
        for lag in paid_tri.columns:
            if lag > d:
                paid_tri.loc[ay, lag] = np.nan
    # string columns so Styler never chokes
    paid_tri.columns = [f"Lag {c}" for c in paid_tri.columns]
    paid_tri.index.name = "Accident_Year"

    # Reserves triangle: CL Ultimate − cumulative paid at each observed lag
    res_tri = pd.DataFrame(index=paid_tri.index,
                           columns=paid_tri.columns, dtype=float)
    res_tri.index.name = "Accident_Year"
    for ay in res_tri.index:
        d = diag_lag.get(ay, -1)
        ult_cl = float(cl_ultimates.get(ay, 0.0))
        for col in paid_tri.columns:
            lag = int(col.split()[1])
            if lag <= d:
                paid = paid_tri.loc[ay, col]
                paid = 0.0 if pd.isna(paid) else float(paid)
                res_tri.loc[ay, col] = round(ult_cl - paid, 2)
            else:
                res_tri.loc[ay, col] = np.nan
    res_tri["CL_Reserve"] = [
        round(float(cl_reserves[ay]), 2) if ay in cl_reserves else np.nan
        for ay in res_tri.index
    ]

    return {
        "results":    results,
        "ldfs":       ldfs,
        "cdfs":       cdfs,
        "eu":         eu_display,
        "paid_tri":   paid_tri,
        "res_tri":    res_tri,
    }


# ── Data Input ─────────────────────────────────────────────────────────────────
st.header("Data Input")
tab_upload, tab_ref = st.tabs(["📁 Upload File", "📋 Column Reference"])

with tab_upload:
    st.markdown(
        "Upload individual claim records. Required columns: "
        "**Accident_Year**, **Development_Lag**, **Amount**, **Settlement_Year**. "
        "Optional: **Premium**. Accepts `.xlsx`, `.xls`, `.csv`."
    )
    uploaded = st.file_uploader("Upload claim data file", type=["xlsx", "xls", "csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") \
                     else pd.read_excel(uploaded)

            raw_df = raw_df.dropna(how="all")
            num_cols = raw_df.select_dtypes(include="number").columns
            if len(num_cols):
                raw_df = raw_df.loc[~(raw_df[num_cols] == 0).all(axis=1)]

            st.subheader("Raw Data (first 10 rows)")
            st.dataframe(raw_df.head(10), use_container_width=True)

            col_map = {}
            for c in raw_df.columns:
                cl = c.lower().replace(" ", "_")
                if "accident" in cl or cl in ("year", "ay"):
                    col_map.setdefault("Accident_Year", c)
                if "development" in cl or "lag" in cl:
                    col_map.setdefault("Development_Lag", c)
                if "amount" in cl or "loss" in cl or ("paid" in cl and "latest" not in cl):
                    col_map.setdefault("Amount", c)
                if "settlement" in cl:
                    col_map.setdefault("Settlement_Year", c)
                if "premium" in cl or "prem" in cl:
                    col_map.setdefault("Premium", c)

            required = ["Accident_Year", "Development_Lag", "Amount", "Settlement_Year"]
            missing = [r for r in required if r not in col_map]

            if missing:
                st.error(f"Could not auto-detect columns: **{missing}**. "
                         "Please rename your columns to match.")
            else:
                work_df = raw_df.rename(columns={v: k for k, v in col_map.items()})
                for col in ["Accident_Year", "Development_Lag", "Settlement_Year"]:
                    work_df[col] = pd.to_numeric(work_df[col], errors="coerce").astype("Int64")
                work_df["Amount"] = pd.to_numeric(work_df["Amount"], errors="coerce").fillna(0)
                work_df = work_df.dropna(subset=required)

                premium_series = None
                if "Premium" in work_df.columns:
                    premium_series = work_df.groupby("Accident_Year")["Premium"].first().astype(float)
                    st.info("Premium column detected in file.")

                st.session_state["work_df"] = work_df
                st.session_state["premium_series"] = premium_series

                n_ays = work_df["Accident_Year"].nunique()
                sy_min = int(work_df["Settlement_Year"].min())
                sy_max = int(work_df["Settlement_Year"].max())
                st.success(
                    f"✓ Loaded **{len(work_df):,}** records | "
                    f"**{n_ays}** accident years | "
                    f"Settlement years: {sy_min}–{sy_max}"
                )
                st.info(f"Valuation Year = **{int(valuation_year)}**. "
                        "Claims with Settlement_Year > Valuation Year will be excluded.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab_ref:
    st.markdown("""
| Column | Type | Description |
|---|---|---|
| `Accident_Year` | Integer | Year the loss event occurred |
| `Development_Lag` | Integer | Settlement_Year − Accident_Year |
| `Amount` | Numeric | Claim payment amount |
| `Settlement_Year` | Integer | Year the claim was settled |
| `Premium` *(optional)* | Numeric | Earned premium for that accident year |
    """)

# ── Run button ─────────────────────────────────────────────────────────────────
st.markdown("---")
run = st.button("🚀 RUN RESERVING MODEL", type="primary", use_container_width=True)

if run:
    if "work_df" not in st.session_state:
        st.error("No data loaded. Please upload a file first.")
        st.stop()

    out = compute_reserves(
        df             = st.session_state["work_df"],
        valuation_year = int(valuation_year),
        tail           = float(tail_factor),
        premium_source = "manual" if premium_source == "Enter manually" else "derive",
        manual_premium = float(manual_premium) if manual_premium is not None else None,
        elr            = float(elr) if elr is not None else None,
        premium_series = st.session_state.get("premium_series"),
    )

    if out is None:
        st.error("No claims remain after applying the valuation filter. "
                 "Try increasing the Valuation Year in the sidebar.")
        st.stop()

    st.session_state["out"] = out   # persist so output stays visible

# ── Output (rendered outside button block so it always shows) ─────────────────
if "out" in st.session_state:
    out     = st.session_state["out"]
    results = out["results"]
    ldfs    = out["ldfs"]
    cdfs    = out["cdfs"]
    eu      = out["eu"]
    paid_tri = out["paid_tri"]
    res_tri  = out["res_tri"]

    # 1. Reserves triangle
    st.subheader("📐 Reserves Triangle (Chain-Ladder)")
    st.caption(
        "Each cell = CL Ultimate − Cumulative Paid at that development lag. "
        "Read across a row to see the reserve shrink as claims develop. "
        "CL_Reserve = outstanding reserve at the current valuation diagonal."
    )
    st.dataframe(res_tri.style.format("${:,.2f}", na_rep="—"), use_container_width=True)

    st.markdown("---")

    # 2. Cumulative paid triangle (collapsible)
    with st.expander("💰 Cumulative Paid Loss Triangle", expanded=False):
        st.caption("— = not yet observed at the valuation date.")
        st.dataframe(paid_tri.style.format("${:,.2f}", na_rep="—"), use_container_width=True)

    st.markdown("---")

    # 3. Summary
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chain-Ladder Reserve", f"${results['CL_Reserve'].sum():,.0f}")
    c2.metric("BF Reserve",           f"${results['BF_Reserve'].sum():,.0f}")
    c3.metric("50/50 Blend Reserve",  f"${results['Blend_50_50_Res'].sum():,.0f}")
    c4.metric("BF Expected Ultimate", f"${eu:,.0f}")

    # 4. Results by AY
    st.subheader("Results by Accident Year")
    money_cols = [c for c in results.columns if c not in ("Diagonal_Lag", "CDF")]
    st.dataframe(
        results.style.format({c: "${:,.2f}" for c in money_cols} | {"CDF": "{:.6f}"}),
        use_container_width=True,
    )

    # Totals row
    totals_df = results[money_cols].sum().to_frame("Total").T
    st.dataframe(totals_df.style.format("${:,.2f}"), use_container_width=True)

    # 5. Development factors
    with st.expander("Development Factors", expanded=False):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Age-to-Age LDFs (volume-weighted)**")
            ldf_df = pd.DataFrame(list(ldfs.items()), columns=["Transition", "LDF"])
            st.dataframe(ldf_df.style.format({"LDF": "{:.6f}"}), use_container_width=True)
        with col_r:
            st.markdown("**Cumulative Development Factors (to Ultimate)**")
            cdf_df = pd.DataFrame(list(cdfs.items()), columns=["Lag", "CDF"]).sort_values("Lag")
            st.dataframe(cdf_df.style.format({"CDF": "{:.6f}"}), use_container_width=True)

    # 6. Downloads
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇ Download Results (CSV)", results.to_csv(),
            file_name=f"reserves_{int(valuation_year)}.csv", mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            "⬇ Download Reserves Triangle (CSV)", res_tri.to_csv(),
            file_name=f"reserves_triangle_{int(valuation_year)}.csv", mime="text/csv",
        )