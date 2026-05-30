import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Insurance Reserving Model")
st.title("Insurance Reserving Model")
st.markdown("**Chain-Ladder + Bornhuetter-Ferguson | 50/50 Blend**")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Model Parameters")
valuation_year = st.sidebar.number_input("Valuation Year", value=2023, step=1)
tail_factor = st.sidebar.number_input("Tail Factor", min_value=1.0, max_value=10.0, value=1.0, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Premium / BF Settings")
premium_source = st.sidebar.radio(
    "Premium Source for BF",
    ["Derive from data (mean CL ultimates)", "Enter manually"]
)

manual_premium = None
elr = None
if premium_source == "Enter manually":
    manual_premium = st.sidebar.number_input("Annual Premium ($)", min_value=0.0, value=1_000_000.0, step=10_000.0)
    elr = st.sidebar.number_input("Expected Loss Ratio", min_value=0.0, max_value=5.0, value=0.65, step=0.01)
    st.sidebar.caption(f"BF Expected Ultimate = ${elr * manual_premium:,.0f}")
else:
    st.sidebar.caption("BF Expected Ultimate = mean of CL ultimates.")

# ── Data Input ────────────────────────────────────────────────────────────────
st.header("Data Input")
tab_upload, tab_ref = st.tabs(["📁 Upload File", "📋 Column Reference"])

with tab_upload:
    st.markdown("Required: **Accident_Year**, **Development_Lag**, **Amount**, **Settlement_Year**. Optional: **Premium**.")
    uploaded = st.file_uploader("Upload claim data file", type=["xlsx", "xls", "csv"])
    
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
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
                st.error(f"Could not auto-detect columns: {missing}")
            else:
                work_df = raw_df.rename(columns={v: k for k, v in col_map.items()})
                for col in ["Accident_Year", "Development_Lag", "Settlement_Year"]:
                    work_df[col] = pd.to_numeric(work_df[col], errors="coerce").astype(int)
                work_df["Amount"] = pd.to_numeric(work_df["Amount"], errors="coerce").fillna(0)
                work_df = work_df.dropna(subset=required)

                st.session_state["work_df"] = work_df
                st.session_state["premium_series"] = (
                    work_df.groupby("Accident_Year")["Premium"].first().astype(float)
                    if "Premium" in work_df.columns else None
                )

                st.success(f"✓ Loaded {len(work_df):,} records | {work_df['Accident_Year'].nunique()} accident years")
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab_ref:
    st.markdown("""
    | Column            | Description |
    |-------------------|-------------|
    | Accident_Year     | Year loss occurred |
    | Development_Lag   | Settlement_Year - Accident_Year |
    | Amount            | Claim payment |
    | Settlement_Year   | Year claim was paid |
    | Premium (optional)| Earned premium |
    """)

# ── Run Button ────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🚀 RUN RESERVING MODEL", type="primary", use_container_width=True):
    if "work_df" not in st.session_state:
        st.error("No data loaded. Please upload a file first.")
        st.stop()

    try:
        # === CALCULATION STARTS HERE ===
        df = st.session_state["work_df"].copy()
        premium_series = st.session_state.get("premium_series")
        vy = int(valuation_year)
        tail = float(tail_factor)

        df = df[df["Settlement_Year"] <= vy].copy()
        if df.empty:
            st.error("No claims remain after filtering.")
            st.stop()

        for col in ["Accident_Year", "Development_Lag", "Settlement_Year"]:
            df[col] = df[col].astype(int)

        latest_paid = df.groupby("Accident_Year")["Amount"].sum()
        accident_years = sorted(int(ay) for ay in latest_paid.index)
        diag_lag = {ay: vy - ay for ay in accident_years}

        grouped = df.groupby(["Accident_Year", "Development_Lag"])["Amount"].sum().reset_index()
        tri_inc = grouped.pivot(index="Accident_Year", columns="Development_Lag", values="Amount").fillna(0)
        tri_inc.columns = [int(c) for c in tri_inc.columns]
        tri_inc = tri_inc.sort_index().sort_index(axis=1)
        tri_cum = tri_inc.cumsum(axis=1)
        lags = sorted(int(c) for c in tri_cum.columns)

        # LDFs
        ldfs = {}
        for i in range(len(lags) - 1):
            c_lag, n_lag = lags[i], lags[i + 1]
            eligible = [ay for ay in accident_years if diag_lag[ay] >= n_lag]
            key = f"{c_lag}-{n_lag}"
            if not eligible:
                ldfs[key] = 1.0
                continue
            sub = tri_cum.loc[eligible, [c_lag, n_lag]]
            mask = sub[c_lag].notna() & sub[n_lag].notna() & (sub[c_lag] > 0)
            ldfs[key] = (sub.loc[mask, n_lag].sum() / sub.loc[mask, c_lag].sum() if mask.any() else 1.0)

        ldf_vals = list(ldfs.values())

        # CDFs
        cdfs = {}
        for si, lag in enumerate(lags):
            cdf = tail
            for j in range(si, len(ldf_vals)):
                cdf *= ldf_vals[j]
            cdfs[lag] = cdf

        def _cdf(d):
            return cdfs.get(d, tail)

        # Chain Ladder
        cl_ult, cl_res = {}, {}
        for ay in accident_years:
            lp = float(latest_paid[ay])
            u = lp * _cdf(diag_lag[ay])
            cl_ult[ay] = u
            cl_res[ay] = u - lp

        # Bornhuetter-Ferguson
        if premium_source == "Enter manually" and manual_premium and elr:
            def _eu(ay):
                if premium_series is not None and ay in premium_series.index:
                    return elr * float(premium_series[ay])
                return elr * float(manual_premium)
            eu_display = elr * float(manual_premium)
        else:
            _mean = float(np.mean(list(cl_ult.values())))
            def _eu(_): return _mean
            eu_display = _mean

        bf_ult, bf_res = {}, {}
        for ay in accident_years:
            lp = float(latest_paid[ay])
            pct_dev = 1.0 / _cdf(diag_lag[ay])
            u = lp + _eu(ay) * (1.0 - pct_dev)
            bf_ult[ay] = u
            bf_res[ay] = u - lp

        # Results Table
        rows = []
        for ay in accident_years:
            rows.append({
                "Accident_Year": ay,
                "Latest_Paid": float(latest_paid[ay]),
                "Diagonal_Lag": diag_lag[ay],
                "CDF": _cdf(diag_lag[ay]),
                "CL_Ultimate": cl_ult[ay],
                "CL_Reserve": cl_res[ay],
                "BF_Ultimate": bf_ult[ay],
                "BF_Reserve": bf_res[ay],
                "Blend_50_50_Ult": (cl_ult[ay] + bf_ult[ay]) / 2,
                "Blend_50_50_Res": (cl_res[ay] + bf_res[ay]) / 2,
            })
        results = pd.DataFrame(rows).set_index("Accident_Year")

        # Paid Triangle
        paid_data = {}
        for ay in accident_years:
            d = diag_lag[ay]
            row = {}
            for lag in lags:
                col_name = f"Lag {lag}"
                if lag <= d and ay in tri_cum.index:
                    raw = tri_cum.at[ay, lag]
                    row[col_name] = float(raw) if pd.notna(raw) else float("nan")
                else:
                    row[col_name] = float("nan")
            paid_data[ay] = row
        paid_tri = pd.DataFrame.from_dict(paid_data, orient="index", dtype=float)
        paid_tri.index.name = "Accident_Year"

        # Reserves Triangle
        res_data = {}
        for ay in accident_years:
            d = diag_lag[ay]
            u_cl = cl_ult[ay]
            row = {}
            for lag in lags:
                col_name = f"Lag {lag}"
                if lag <= d:
                    paid = paid_data[ay][col_name]
                    row[col_name] = u_cl - (0.0 if np.isnan(paid) else paid)
                else:
                    row[col_name] = float("nan")
            row["CL_Reserve"] = cl_res[ay]
            res_data[ay] = row
        res_tri = pd.DataFrame.from_dict(res_data, orient="index", dtype=float)
        res_tri.index.name = "Accident_Year"

        # Save to session state
        st.session_state["model_output"] = {
            "results": results,
            "res_tri": res_tri,
            "paid_tri": paid_tri,
            "ldfs": ldfs,
            "cdfs": cdfs,
            "eu_display": eu_display,
            "vy": vy
        }

        st.success("✅ Calculation completed successfully!")

    except Exception as e:
        st.error(f"❌ Error during calculation: {str(e)}")
        st.exception(e)

# ── OUTPUT SECTION ───────────────────────────────────────────────────────────
if "model_output" in st.session_state:
    o = st.session_state["model_output"]
    
    st.markdown("---")
    
    # Reserves Triangle (Main Output)
    st.subheader("📐 Reserves Triangle (Chain-Ladder)")
    st.caption("Each cell = CL Ultimate − Cumulative Paid at that lag. CL_Reserve = reserve at diagonal.")
    
    try:
        st.info(f"Debug: Triangle shape = {o['res_tri'].shape} | Columns: {list(o['res_tri'].columns)}")
        st.dataframe(
            o["res_tri"].style.format("${:,.2f}", na_rep="—"), 
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Failed to display Reserves Triangle: {e}")
        st.dataframe(o["res_tri"])   # Fallback without styling

    st.markdown("---")

    # Paid Triangle
    with st.expander("💰 Cumulative Paid Loss Triangle", expanded=False):
        st.dataframe(o["paid_tri"].style.format("${:,.2f}", na_rep="—"), use_container_width=True)

    # Summary
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chain-Ladder Reserve", f"${o['results']['CL_Reserve'].sum():,.0f}")
    c2.metric("BF Reserve", f"${o['results']['BF_Reserve'].sum():,.0f}")
    c3.metric("50/50 Blend Reserve", f"${o['results']['Blend_50_50_Res'].sum():,.0f}")
    c4.metric("BF Expected Ultimate", f"${o['eu_display']:,.0f}")

    # Results by AY
    st.subheader("Results by Accident Year")
    st.dataframe(o["results"].style.format("${:,.2f}"), use_container_width=True)

    # Development Factors
    with st.expander("Development Factors", expanded=False):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Age-to-Age LDFs**")
            st.dataframe(pd.DataFrame(list(o["ldfs"].items()), columns=["Transition", "LDF"]), use_container_width=True)
        with col_r:
            st.markdown("**CDFs to Ultimate**")
            st.dataframe(pd.DataFrame(list(o["cdfs"].items()), columns=["Lag", "CDF"]).sort_values("Lag"), use_container_width=True)

    # Downloads
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇ Download Results", o["results"].to_csv(), f"reserves_{o['vy']}.csv", mime="text/csv")
    with col2:
        st.download_button("⬇ Download Reserves Triangle", o["res_tri"].to_csv(), f"reserves_triangle_{o['vy']}.csv", mime="text/csv")