import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Insurance Reserving Model")
st.title("Insurance Reserving Model")
st.markdown("**Chain-Ladder + Bornhuetter-Ferguson | 50/50 Blend**")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Model Parameters")

valuation_year = st.sidebar.number_input("Valuation Year", value=2023, step=1)
tail_factor    = st.sidebar.number_input("Tail Factor", min_value=1.0, max_value=10.0, value=1.0, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Premium / BF Settings")
premium_source = st.sidebar.radio(
    "Premium Source for BF",
    ["Derive from data (mean CL ultimates)", "Enter manually"],
)
manual_premium = None
elr = None
if premium_source == "Enter manually":
    manual_premium = st.sidebar.number_input("Annual Premium ($)", min_value=0.0, value=1_000_000.0, step=10_000.0)
    elr = st.sidebar.number_input("Expected Loss Ratio", min_value=0.0, max_value=5.0, value=0.65, step=0.01)
    st.sidebar.caption(f"BF Expected Ultimate = ${elr * manual_premium:,.0f}")
else:
    st.sidebar.caption("BF Expected Ultimate = mean of CL ultimates.")

# ── Data upload ────────────────────────────────────────────────────────────────
st.header("Data Input")
tab_upload, tab_ref = st.tabs(["📁 Upload File", "📋 Column Reference"])

with tab_upload:
    st.markdown("Required columns: **Accident_Year**, **Development_Lag**, **Amount**, **Settlement_Year**. Optional: **Premium**.")
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
                if "accident" in cl or cl in ("year","ay"):           col_map.setdefault("Accident_Year", c)
                if "development" in cl or "lag" in cl:                col_map.setdefault("Development_Lag", c)
                if "amount" in cl or "loss" in cl or ("paid" in cl and "latest" not in cl): col_map.setdefault("Amount", c)
                if "settlement" in cl:                                col_map.setdefault("Settlement_Year", c)
                if "premium" in cl or "prem" in cl:                   col_map.setdefault("Premium", c)

            required = ["Accident_Year","Development_Lag","Amount","Settlement_Year"]
            missing  = [r for r in required if r not in col_map]

            if missing:
                st.error(f"Could not auto-detect columns: {missing}. Please rename them.")
            else:
                work_df = raw_df.rename(columns={v:k for k,v in col_map.items()})
                for col in ["Accident_Year","Development_Lag","Settlement_Year"]:
                    work_df[col] = pd.to_numeric(work_df[col], errors="coerce").astype("Int64")
                work_df["Amount"] = pd.to_numeric(work_df["Amount"], errors="coerce").fillna(0)
                work_df = work_df.dropna(subset=required)

                # store raw bytes + metadata so we can re-run on button click
                st.session_state["work_df"]        = work_df
                st.session_state["premium_series"] = (
                    work_df.groupby("Accident_Year")["Premium"].first().astype(float)
                    if "Premium" in work_df.columns else None
                )

                n_ays  = work_df["Accident_Year"].nunique()
                sy_min = int(work_df["Settlement_Year"].min())
                sy_max = int(work_df["Settlement_Year"].max())
                st.success(f"✓ {len(work_df):,} records | {n_ays} accident years | Settlement years: {sy_min}–{sy_max}")
                st.info(f"Valuation Year = {int(valuation_year)}. Claims with Settlement_Year > {int(valuation_year)} will be excluded.")
        except Exception as e:
            st.error(f"Error: {e}")

with tab_ref:
    st.markdown("""
| Column | Description |
|---|---|
| `Accident_Year` | Year the loss occurred |
| `Development_Lag` | Settlement_Year − Accident_Year |
| `Amount` | Claim payment |
| `Settlement_Year` | Year the claim was paid |
| `Premium` *(optional)* | Earned premium for that AY |
""")

# ── Run ────────────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🚀 RUN RESERVING MODEL", type="primary", use_container_width=True):
    if "work_df" not in st.session_state:
        st.error("No data loaded.")
        st.stop()

    df             = st.session_state["work_df"]
    premium_series = st.session_state.get("premium_series")
    vy             = int(valuation_year)
    tail           = float(tail_factor)

    # ── Filter ──────────────────────────────────────────────────────────────
    df = df[df["Settlement_Year"] <= vy].copy()
    if df.empty:
        st.error("No claims remain after filtering. Increase the Valuation Year.")
        st.stop()

    # ── Triangle ─────────────────────────────────────────────────────────────
    latest_paid    = df.groupby("Accident_Year")["Amount"].sum()
    accident_years = sorted(latest_paid.index.tolist())
    diag_lag       = {ay: int(vy - ay) for ay in accident_years}

    grouped = df.groupby(["Accident_Year","Development_Lag"])["Amount"].sum().reset_index()
    tri_inc = grouped.pivot(index="Accident_Year", columns="Development_Lag", values="Amount").fillna(0)
    tri_inc.columns = [int(c) for c in tri_inc.columns]
    tri_inc  = tri_inc.sort_index().sort_index(axis=1)
    tri_cum  = tri_inc.cumsum(axis=1)
    lags     = sorted(tri_cum.columns.tolist())

    # ── LDFs ──────────────────────────────────────────────────────────────────
    ldfs = {}
    for i in range(len(lags)-1):
        c_lag, n_lag = lags[i], lags[i+1]
        eligible = [ay for ay in tri_cum.index if diag_lag.get(ay,-1) >= n_lag]
        if not eligible: ldfs[f"{c_lag}-{n_lag}"] = 1.0; continue
        sub  = tri_cum.loc[eligible]
        mask = sub[c_lag].notna() & sub[n_lag].notna() & (sub[c_lag] > 0)
        ldfs[f"{c_lag}-{n_lag}"] = sub.loc[mask,n_lag].sum()/sub.loc[mask,c_lag].sum() if mask.sum()>0 else 1.0

    ldf_vals = list(ldfs.values())
    cdfs = {}
    for si, lag in enumerate(lags):
        cdf = tail
        for j in range(si, len(ldf_vals)): cdf *= ldf_vals[j]
        cdfs[lag] = cdf

    def get_cdf(d):
        return cdfs.get(d, tail)

    # ── CL ────────────────────────────────────────────────────────────────────
    cl_ult = {}; cl_res = {}
    for ay in accident_years:
        latest = float(latest_paid[ay])
        ult    = latest * get_cdf(diag_lag[ay])
        cl_ult[ay] = ult
        cl_res[ay] = ult - latest

    # ── BF ────────────────────────────────────────────────────────────────────
    if premium_source == "Enter manually" and manual_premium and elr:
        def get_prem(ay):
            if premium_series is not None and ay in premium_series.index:
                return float(premium_series[ay])
            return float(manual_premium)
        eu_by_ay  = {ay: elr * get_prem(ay) for ay in accident_years}
        eu_display = elr * float(manual_premium)
    else:
        eu_val    = float(np.mean(list(cl_ult.values())))
        eu_by_ay  = {ay: eu_val for ay in accident_years}
        eu_display = eu_val

    bf_ult = {}; bf_res = {}
    for ay in accident_years:
        latest  = float(latest_paid[ay])
        pct_dev = 1.0 / get_cdf(diag_lag[ay])
        ult     = latest + eu_by_ay[ay] * (1.0 - pct_dev)
        bf_ult[ay] = ult
        bf_res[ay] = ult - latest

    # ── Results table ─────────────────────────────────────────────────────────
    rows = []
    for ay in accident_years:
        rows.append({
            "Accident_Year":   ay,
            "Latest_Paid":     float(latest_paid[ay]),
            "Diagonal_Lag":    diag_lag[ay],
            "CDF":             get_cdf(diag_lag[ay]),
            "CL_Ultimate":     cl_ult[ay],
            "CL_Reserve":      cl_res[ay],
            "BF_Ultimate":     bf_ult[ay],
            "BF_Reserve":      bf_res[ay],
            "Blend_50_50_Ult": (cl_ult[ay] + bf_ult[ay]) / 2,
            "Blend_50_50_Res": (cl_res[ay] + bf_res[ay]) / 2,
        })
    results = pd.DataFrame(rows).set_index("Accident_Year")

    # ── Paid triangle (string cols) ───────────────────────────────────────────
    paid_tri = tri_cum.copy().astype(float)
    for ay in paid_tri.index:
        d = diag_lag.get(ay, -1)
        for lag in paid_tri.columns:
            if lag > d:
                paid_tri.loc[ay, lag] = np.nan
    paid_tri.columns     = [f"Lag {c}" for c in paid_tri.columns]
    paid_tri.index.name  = "Accident_Year"

    # ── Reserves triangle (string cols) ──────────────────────────────────────
    res_tri = pd.DataFrame(
        index   = paid_tri.index,
        columns = paid_tri.columns,
        dtype   = float
    )
    res_tri.index.name = "Accident_Year"
    for ay in res_tri.index:
        d      = diag_lag.get(ay, -1)
        ult_cl = float(cl_ult.get(ay, 0.0))
        for col in paid_tri.columns:
            lag = int(col.split()[1])
            if lag <= d:
                paid = paid_tri.loc[ay, col]
                res_tri.loc[ay, col] = round(ult_cl - (0.0 if pd.isna(paid) else float(paid)), 2)
            else:
                res_tri.loc[ay, col] = np.nan
    res_tri["CL_Reserve"] = [round(float(cl_res[ay]), 2) for ay in res_tri.index]

    # ════════════════════════════════════════════════════════════════════════
    # OUTPUT — all rendered here, same run, no session_state needed
    # ════════════════════════════════════════════════════════════════════════

    # 1. Reserves triangle
    # Convert index to plain int to avoid Int64/object rendering bugs in st.dataframe
    res_tri.index  = res_tri.index.astype(int)
    paid_tri.index = paid_tri.index.astype(int)
    # Convert all values to plain float (replace pd.NA with np.nan)
    res_tri  = res_tri.astype(object).where(res_tri.notna(), other=None)
    res_tri  = res_tri.apply(pd.to_numeric, errors="coerce")
    paid_tri = paid_tri.astype(object).where(paid_tri.notna(), other=None)
    paid_tri = paid_tri.apply(pd.to_numeric, errors="coerce")

    st.subheader("📐 Reserves Triangle (Chain-Ladder)")
    st.caption("Each cell = CL Ultimate − Cumulative Paid at that development lag. CL_Reserve = reserve at the valuation diagonal.")
    st.write(f"Triangle shape: {res_tri.shape} | dtype sample: {res_tri.dtypes.iloc[0]} | index type: {type(res_tri.index[0])}")
    try:
        st.dataframe(res_tri, use_container_width=True)
    except Exception as e:
        st.error(f"st.dataframe failed: {e}")
        st.write(res_tri.to_dict())

    st.markdown("---")

    # 2. Paid triangle
    with st.expander("💰 Cumulative Paid Loss Triangle", expanded=False):
        st.dataframe(paid_tri, use_container_width=True)

    st.markdown("---")

    # 3. Summary
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chain-Ladder Reserve", f"${results['CL_Reserve'].sum():,.0f}")
    c2.metric("BF Reserve",           f"${results['BF_Reserve'].sum():,.0f}")
    c3.metric("50/50 Blend Reserve",  f"${results['Blend_50_50_Res'].sum():,.0f}")
    c4.metric("BF Expected Ultimate", f"${eu_display:,.0f}")

    # 4. Results by AY
    st.subheader("Results by Accident Year")
    st.dataframe(results, use_container_width=True)
    totals = results[[c for c in results.columns if c not in ("Diagonal_Lag","CDF")]].sum().to_frame("Total").T
    st.dataframe(totals, use_container_width=True)

    # 5. Dev factors
    with st.expander("Development Factors", expanded=False):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Age-to-Age LDFs**")
            st.dataframe(pd.DataFrame(list(ldfs.items()), columns=["Transition","LDF"]), use_container_width=True)
        with col_r:
            st.markdown("**CDFs to Ultimate**")
            st.dataframe(pd.DataFrame(list(cdfs.items()), columns=["Lag","CDF"]).sort_values("Lag"), use_container_width=True)

    # 6. Downloads
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇ Download Results (CSV)", results.to_csv(),
                           file_name=f"reserves_{vy}.csv", mime="text/csv")
    with col_dl2:
        st.download_button("⬇ Download Reserves Triangle (CSV)", res_tri.to_csv(),
                           file_name=f"reserves_triangle_{vy}.csv", mime="text/csv")