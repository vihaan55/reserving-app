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
    st.markdown("Required: **Accident_Year**, **Development_Lag**, **Amount**, **Settlement_Year**. Optional: **Premium**.")
    uploaded = st.file_uploader("Upload claim data file", type=["xlsx", "xls", "csv"])

    if uploaded is not None:
        try:
            raw_df = (pd.read_csv(uploaded)
                      if uploaded.name.lower().endswith(".csv")
                      else pd.read_excel(uploaded))
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
            missing  = [r for r in required if r not in col_map]

            if missing:
                st.error(f"Could not auto-detect columns: {missing}. Please rename them.")
            else:
                work_df = raw_df.rename(columns={v: k for k, v in col_map.items()})
                for col in ["Accident_Year", "Development_Lag", "Settlement_Year"]:
                    work_df[col] = pd.to_numeric(work_df[col], errors="coerce").astype("Int64")
                work_df["Amount"] = pd.to_numeric(work_df["Amount"], errors="coerce").fillna(0)
                work_df = work_df.dropna(subset=required)

                st.session_state["work_df"] = work_df
                st.session_state["premium_series"] = (
                    work_df.groupby("Accident_Year")["Premium"].first().astype(float)
                    if "Premium" in work_df.columns else None
                )

                n_ays  = work_df["Accident_Year"].nunique()
                sy_min = int(work_df["Settlement_Year"].min())
                sy_max = int(work_df["Settlement_Year"].max())
                st.success(f"✓ {len(work_df):,} records | {n_ays} accident years | "
                           f"Settlement years: {sy_min}–{sy_max}")
                st.info(f"Valuation Year = {int(valuation_year)}. "
                        f"Claims with Settlement_Year > {int(valuation_year)} will be excluded.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

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

    df             = st.session_state["work_df"].copy()
    premium_series = st.session_state.get("premium_series")
    vy             = int(valuation_year)
    tail           = float(tail_factor)

    # 1. Filter to valuation year
    df = df[df["Settlement_Year"] <= vy].copy()
    if df.empty:
        st.error("No claims remain after filtering. Increase the Valuation Year.")
        st.stop()

    # 2. *** KEY FIX: convert nullable Int64 → plain Python int BEFORE any dict/index operations ***
    #    Int64 scalars from .iterrows() / index iteration can silently miss dict lookups
    for col in ["Accident_Year", "Development_Lag", "Settlement_Year"]:
        df[col] = df[col].astype(int)

    # 3. Build incremental → cumulative triangle
    latest_paid    = df.groupby("Accident_Year")["Amount"].sum()
    accident_years = sorted(int(ay) for ay in latest_paid.index)
    diag_lag       = {ay: vy - ay for ay in accident_years}

    grouped = df.groupby(["Accident_Year", "Development_Lag"])["Amount"].sum().reset_index()
    tri_inc = grouped.pivot(index="Accident_Year", columns="Development_Lag",
                            values="Amount").fillna(0)
    tri_inc.columns = [int(c) for c in tri_inc.columns]
    tri_inc = tri_inc.sort_index().sort_index(axis=1)
    tri_cum = tri_inc.cumsum(axis=1)
    lags    = sorted(int(c) for c in tri_cum.columns)

    # 4. Volume-weighted LDFs (upper triangle only)
    ldfs = {}
    for i in range(len(lags) - 1):
        c_lag, n_lag = lags[i], lags[i + 1]
        eligible = [ay for ay in accident_years if diag_lag[ay] >= n_lag]
        key      = f"{c_lag}-{n_lag}"
        if not eligible:
            ldfs[key] = 1.0
            continue
        sub  = tri_cum.loc[eligible, [c_lag, n_lag]]
        mask = sub[c_lag].notna() & sub[n_lag].notna() & (sub[c_lag] > 0)
        ldfs[key] = (sub.loc[mask, n_lag].sum() / sub.loc[mask, c_lag].sum()
                     if mask.any() else 1.0)

    ldf_vals = list(ldfs.values())

    # 5. CDFs to ultimate
    cdfs = {}
    for si, lag in enumerate(lags):
        cdf = tail
        for j in range(si, len(ldf_vals)):
            cdf *= ldf_vals[j]
        cdfs[lag] = cdf

    def _cdf(d: int) -> float:
        return cdfs.get(d, tail)

    # 6. Chain-Ladder ultimates & reserves
    cl_ult, cl_res = {}, {}
    for ay in accident_years:
        lp         = float(latest_paid[ay])
        u          = lp * _cdf(diag_lag[ay])
        cl_ult[ay] = u
        cl_res[ay] = u - lp

    # 7. BF ultimates & reserves
    if premium_source == "Enter manually" and manual_premium and elr:
        def _eu(ay):
            if premium_series is not None and ay in premium_series.index:
                return elr * float(premium_series[ay])
            return elr * float(manual_premium)
        eu_display = elr * float(manual_premium)
    else:
        _mean = float(np.mean(list(cl_ult.values())))
        def _eu(_):
            return _mean
        eu_display = _mean

    bf_ult, bf_res = {}, {}
    for ay in accident_years:
        lp         = float(latest_paid[ay])
        pct_dev    = 1.0 / _cdf(diag_lag[ay])
        u          = lp + _eu(ay) * (1.0 - pct_dev)
        bf_ult[ay] = u
        bf_res[ay] = u - lp

    # 8. Results by AY
    rows = []
    for ay in accident_years:
        rows.append({
            "Accident_Year":   ay,
            "Latest_Paid":     float(latest_paid[ay]),
            "Diagonal_Lag":    diag_lag[ay],
            "CDF":             _cdf(diag_lag[ay]),
            "CL_Ultimate":     cl_ult[ay],
            "CL_Reserve":      cl_res[ay],
            "BF_Ultimate":     bf_ult[ay],
            "BF_Reserve":      bf_res[ay],
            "Blend_50_50_Ult": (cl_ult[ay] + bf_ult[ay]) / 2,
            "Blend_50_50_Res": (cl_res[ay] + bf_res[ay]) / 2,
        })
    results = pd.DataFrame(rows).set_index("Accident_Year")

    # 9. Paid triangle — built from explicit Python dicts, NO pandas nullable types
    paid_data = {}
    for ay in accident_years:
        d   = diag_lag[ay]
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

    # 10. Reserves triangle — same dict-based approach, guaranteed float64
    res_data = {}
    for ay in accident_years:
        d    = diag_lag[ay]
        u_cl = cl_ult[ay]
        row  = {}
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

    # ── Store everything; rendering happens OUTSIDE this block ────────────────
    st.session_state["model_output"] = dict(
        results    = results,
        res_tri    = res_tri,
        paid_tri   = paid_tri,
        ldfs       = ldfs,
        cdfs       = cdfs,
        eu_display = eu_display,
        vy         = vy,
    )

# ── OUTPUT — rendered unconditionally below the button ────────────────────────
# This block runs on EVERY script execution once results exist, so the triangle
# always appears in a predictable position regardless of scroll state or button
# click timing.
if "model_output" in st.session_state:
    o          = st.session_state["model_output"]
    results    = o["results"]
    res_tri    = o["res_tri"]
    paid_tri   = o["paid_tri"]
    ldfs       = o["ldfs"]
    cdfs       = o["cdfs"]
    eu_display = o["eu_display"]
    vy         = o["vy"]

    _fmt = lambda x: f"{x:,.2f}" if (isinstance(x, float) and not np.isnan(x)) else ""

    st.markdown("---")

    # ── 1. Reserves Triangle ──────────────────────────────────────────────────
    st.subheader("📐 Reserves Triangle (Chain-Ladder)")
    st.caption("Each cell = CL Ultimate − Cumulative Paid at that lag. "
               "CL_Reserve = reserve at the valuation diagonal.")
    st.dataframe(res_tri.style.format(_fmt), use_container_width=True)

    st.markdown("---")

    # ── 2. Paid triangle ──────────────────────────────────────────────────────
    with st.expander("💰 Cumulative Paid Loss Triangle", expanded=False):
        st.dataframe(paid_tri.style.format(_fmt), use_container_width=True)

    st.markdown("---")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chain-Ladder Reserve", f"${results['CL_Reserve'].sum():,.0f}")
    c2.metric("BF Reserve",           f"${results['BF_Reserve'].sum():,.0f}")
    c3.metric("50/50 Blend Reserve",  f"${results['Blend_50_50_Res'].sum():,.0f}")
    c4.metric("BF Expected Ultimate", f"${eu_display:,.0f}")

    # ── 4. Results by AY ──────────────────────────────────────────────────────
    st.subheader("Results by Accident Year")
    st.dataframe(results, use_container_width=True)
    totals = (results[[c for c in results.columns if c not in ("Diagonal_Lag", "CDF")]]
              .sum().to_frame("Total").T)
    st.dataframe(totals, use_container_width=True)

    # ── 5. Development Factors ────────────────────────────────────────────────
    with st.expander("Development Factors", expanded=False):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Age-to-Age LDFs**")
            st.dataframe(pd.DataFrame(list(ldfs.items()), columns=["Transition", "LDF"]),
                         use_container_width=True)
        with col_r:
            st.markdown("**CDFs to Ultimate**")
            st.dataframe(pd.DataFrame(list(cdfs.items()), columns=["Lag", "CDF"])
                         .sort_values("Lag"), use_container_width=True)

    # ── 6. Downloads ──────────────────────────────────────────────────────────
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇ Download Results (CSV)", results.to_csv(),
                           file_name=f"reserves_{vy}.csv", mime="text/csv")
    with dl2:
        st.download_button("⬇ Download Reserves Triangle (CSV)", res_tri.to_csv(),
                           file_name=f"reserves_triangle_{vy}.csv", mime="text/csv")