"""
Insurance Reserving Model  —  Streamlit App
Chain-Ladder + Bornhuetter-Ferguson | 50/50 Blend

Actuarial design:
  1. Input = individual claim records (one row per claim settlement).
  2. Valuation date is user-controlled; claims settled after it are excluded.
  3. Latest paid per AY = total filtered claims for that AY.
  4. Diagonal lag per AY = valuation_year - Accident_Year.
  5. LDFs are volume-weighted (sum-of-next / sum-of-current).
  6. CDFs = cumulative product of LDFs from a given lag to the last observed lag,
     multiplied by an optional tail factor.
  7. BF Expected Ultimate = mean of CL ultimates (data-driven),
     OR user-supplied ELR × Premium when manual premium is provided.
  8. BF % Developed = 1 / CDF(diagonal lag).
  9. 50/50 blend = simple average of CL and BF results.
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
    "Valuation Year",
    value=2023, step=1,
    help="Claims settled after this year are excluded from the triangle.",
)

tail_factor = st.sidebar.number_input(
    "Tail Factor",
    min_value=1.0, max_value=10.0, value=1.0, step=0.01,
    help="Multiplied onto the last CDF to account for development beyond observed data. 1.0 = no tail.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Premium / BF Settings")

premium_source = st.sidebar.radio(
    "Premium Source for BF",
    ["Derive from data (mean CL ultimates)", "Enter manually"],
    help=(
        "If 'Derive from data', BF Expected Ultimate = mean of CL ultimates.\n\n"
        "If 'Enter manually', BF Expected Ultimate = ELR × Premium."
    ),
)

manual_premium = None
elr = None

if premium_source == "Enter manually":
    manual_premium = st.sidebar.number_input(
        "Annual Premium ($)",
        min_value=0.0, value=1_000_000.0, step=10_000.0,
        help="Applied uniformly to all accident years unless a Premium column is in your data.",
    )
    elr = st.sidebar.number_input(
        "Expected Loss Ratio",
        min_value=0.0, max_value=5.0, value=0.65, step=0.01,
    )
    st.sidebar.caption(
        f"BF Expected Ultimate = ELR × Premium = "
        f"${elr * manual_premium:,.0f}"
    )
else:
    st.sidebar.caption(
        "BF Expected Ultimate will be computed as the mean of the CL ultimates."
    )


# ── Core calculation ────────────────────────────────────────────────────────────

def compute_reserves(
    df: pd.DataFrame,
    valuation_year: int,
    tail: float = 1.0,
    premium_source: str = "derive",   # "derive" or "manual"
    manual_premium: float = None,
    elr: float = None,
    premium_series: pd.Series = None, # AY-indexed premiums from uploaded data
):
    """
    Returns (results_df, ldfs_dict, cdfs_dict, eu_float).
    results_df has one row per accident year.
    """

    # 1. Filter to valuation date
    df = df[df["Settlement_Year"] <= valuation_year].copy()
    if df.empty:
        return None, {}, {}, np.nan

    # 2. Total paid per AY (latest diagonal value)
    latest_paid = df.groupby("Accident_Year")["Amount"].sum()
    accident_years = sorted(latest_paid.index.tolist())

    # 3. Diagonal lag per AY
    diag_lag = {ay: int(valuation_year - ay) for ay in accident_years}

    # 4. Cumulative triangle
    grouped = (
        df.groupby(["Accident_Year", "Development_Lag"])["Amount"]
        .sum()
        .reset_index()
    )
    tri_inc = grouped.pivot(
        index="Accident_Year", columns="Development_Lag", values="Amount"
    ).fillna(0)
    tri_inc.columns = [int(c) for c in tri_inc.columns]
    tri_inc = tri_inc.sort_index().sort_index(axis=1)
    tri_cum = tri_inc.cumsum(axis=1)
    lags = sorted(tri_cum.columns.tolist())

    # 5. Volume-weighted LDFs (upper triangle only)
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

    # 6. CDFs to ultimate
    ldf_values = list(ldfs.values())
    cdfs = {}
    for si, lag in enumerate(lags):
        cdf = tail
        for j in range(si, len(ldf_values)):
            cdf *= ldf_values[j]
        cdfs[lag] = cdf
    max_lag = max(lags) if lags else 0

    def get_cdf(d_lag):
        if d_lag in cdfs:
            return cdfs[d_lag]
        if d_lag > max_lag:
            return tail          # beyond observed — just the tail
        # for negative lags (AY > valuation year), return very high CDF
        return cdfs.get(min(lags), tail) * 100

    # 7. CL ultimates and reserves
    cl_ultimates, cl_reserves = {}, {}
    for ay in accident_years:
        latest = float(latest_paid[ay])
        cdf = get_cdf(diag_lag[ay])
        ult = latest * cdf
        cl_ultimates[ay] = ult
        cl_reserves[ay] = ult - latest

    # 8. BF Expected Ultimate
    if premium_source == "manual" and manual_premium is not None and elr is not None:
        # Per-AY premium: use uploaded premium_series if available, else manual value
        def get_premium(ay):
            if premium_series is not None and ay in premium_series.index:
                return float(premium_series[ay])
            return float(manual_premium)
        eu_by_ay = {ay: elr * get_premium(ay) for ay in accident_years}
        eu_display = elr * float(manual_premium)  # for display
    else:
        # Data-driven: mean of CL ultimates
        valid = [v for v in cl_ultimates.values() if not np.isnan(v)]
        eu_val = float(np.mean(valid)) if valid else 0.0
        eu_by_ay = {ay: eu_val for ay in accident_years}
        eu_display = eu_val

    # 9. BF ultimates and reserves
    bf_ultimates, bf_reserves = {}, {}
    for ay in accident_years:
        latest = float(latest_paid[ay])
        cdf = get_cdf(diag_lag[ay])
        pct_dev = 1.0 / cdf if cdf > 0 else 1.0
        ult = latest + eu_by_ay[ay] * (1.0 - pct_dev)
        bf_ultimates[ay] = ult
        bf_reserves[ay] = ult - latest

    # 10. Assemble results
    rows = []
    for ay in accident_years:
        cl_r = cl_reserves[ay]
        bf_r = bf_reserves[ay]
        rows.append({
            "Accident_Year":   ay,
            "Latest_Paid":     float(latest_paid[ay]),
            "Diagonal_Lag":    diag_lag[ay],
            "CDF":             get_cdf(diag_lag[ay]),
            "CL_Ultimate":     cl_ultimates[ay],
            "CL_Reserve":      cl_r,
            "BF_Ultimate":     bf_ultimates[ay],
            "BF_Reserve":      bf_r,
            "Blend_50_50_Ult": (cl_ultimates[ay] + bf_ultimates[ay]) / 2,
            "Blend_50_50_Res": (cl_r + bf_r) / 2,
        })
    results = pd.DataFrame(rows).set_index("Accident_Year")
    return results, ldfs, cdfs, eu_display


# ── Data Input ──────────────────────────────────────────────────────────────────
st.header("Data Input")

tab_upload, tab_ref = st.tabs(["📁 Upload File", "📋 Column Reference"])

with tab_upload:
    st.markdown(
        "Upload individual claim records. "
        "Required columns: **Accident_Year**, **Development_Lag**, **Amount**, **Settlement_Year**. "
        "Optional: **Premium**. Accepts `.xlsx`, `.xls`, or `.csv`."
    )
    uploaded = st.file_uploader("Upload claim data file", type=["xlsx", "xls", "csv"])

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded)
            else:
                raw_df = pd.read_excel(uploaded)

            # Drop fully null/zero rows (common in Excel exports)
            raw_df = raw_df.dropna(how="all")
            num_cols = raw_df.select_dtypes(include="number").columns
            if len(num_cols):
                raw_df = raw_df.loc[~(raw_df[num_cols] == 0).all(axis=1)]

            st.subheader("Raw Data (first 10 rows)")
            st.dataframe(raw_df.head(10), use_container_width=True)

            # Auto-detect columns (case-insensitive)
            col_map = {}
            for c in raw_df.columns:
                cl = c.lower().replace(" ", "_")
                if ("accident" in cl) or cl in ("year", "ay"):
                    col_map.setdefault("Accident_Year", c)
                if "development" in cl or cl in ("lag", "dev_lag", "development_lag"):
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
                st.error(
                    f"Could not auto-detect columns: **{missing}**. "
                    "Please rename your file's columns to match the required names."
                )
            else:
                rename_map = {v: k for k, v in col_map.items()}
                work_df = raw_df.rename(columns=rename_map)

                for col in ["Accident_Year", "Development_Lag", "Settlement_Year"]:
                    work_df[col] = pd.to_numeric(work_df[col], errors="coerce").astype("Int64")
                work_df["Amount"] = pd.to_numeric(work_df["Amount"], errors="coerce").fillna(0)
                work_df = work_df.dropna(subset=required)

                # Extract per-AY premiums from the data if the column exists
                premium_series = None
                if "Premium" in work_df.columns:
                    premium_series = (
                        work_df.groupby("Accident_Year")["Premium"].first().astype(float)
                    )
                    st.info("📌 Premium column detected in file. "
                            "It will be used when 'Enter manually' mode is active "
                            "and your manual value will serve as a fallback for AYs missing a premium.")

                st.session_state["work_df"] = work_df
                st.session_state["premium_series"] = premium_series

                n_ays = work_df["Accident_Year"].nunique()
                ay_min = int(work_df["Accident_Year"].min())
                ay_max = int(work_df["Accident_Year"].max())
                sy_min = int(work_df["Settlement_Year"].min())
                sy_max = int(work_df["Settlement_Year"].max())
                st.success(
                    f"✓ Loaded **{len(work_df):,}** claim records | "
                    f"**{n_ays}** accident years ({ay_min}–{ay_max}) | "
                    f"Settlement years: {sy_min}–{sy_max}"
                )
                st.info(
                    f"Valuation Year is set to **{int(valuation_year)}** in the sidebar. "
                    "Claims with Settlement_Year > Valuation Year will be excluded."
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab_ref:
    st.markdown("""
    **Required columns** (column names are detected automatically, case-insensitive):

    | Column | Type | Description |
    |---|---|---|
    | `Accident_Year` | Integer | Year the accident / loss event occurred |
    | `Development_Lag` | Integer | Years between Accident_Year and Settlement_Year |
    | `Amount` | Numeric | Claim payment amount |
    | `Settlement_Year` | Integer | Year the claim was settled / paid |

    **Optional columns:**

    | Column | Type | Description |
    |---|---|---|
    | `Premium` | Numeric | Earned premium for that accident year |

    **Notes:**
    - `Development_Lag` = `Settlement_Year` − `Accident_Year`
    - One row per claim. The model aggregates internally.
    - The Valuation Year filter is set in the sidebar.
    - If a `Premium` column is present *and* you choose "Enter manually" mode in the sidebar,
      the per-AY premiums from the file take priority; your manual value is used as a fallback.
    """)


# ── Run Model ───────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🚀 RUN RESERVING MODEL", type="primary", use_container_width=True):

    if "work_df" not in st.session_state:
        st.error("No data loaded. Please upload a file first.")
        st.stop()

    work_df      = st.session_state["work_df"]
    prem_series  = st.session_state.get("premium_series", None)

    results, ldfs, cdfs, eu = compute_reserves(
        df              = work_df,
        valuation_year  = int(valuation_year),
        tail            = float(tail_factor),
        premium_source  = "manual" if premium_source == "Enter manually" else "derive",
        manual_premium  = float(manual_premium) if manual_premium is not None else None,
        elr             = float(elr) if elr is not None else None,
        premium_series  = prem_series,
    )

    if results is None or results.empty:
        st.error(
            "No claims remain after applying the valuation filter. "
            "Try increasing the Valuation Year in the sidebar, "
            "or check that your Settlement_Year column is populated correctly."
        )
        st.stop()

    # ── Summary metrics ─────────────────────────────────────────────────────
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chain-Ladder Reserve", f"${results['CL_Reserve'].sum():,.0f}")
    c2.metric("BF Reserve",           f"${results['BF_Reserve'].sum():,.0f}")
    c3.metric("50/50 Blend Reserve",  f"${results['Blend_50_50_Res'].sum():,.0f}")
    c4.metric(
        "BF Expected Ultimate",
        f"${eu:,.0f}",
        help="Mean of CL ultimates (data-driven) or ELR × Premium (manual mode).",
    )

    # ── Results table ────────────────────────────────────────────────────────
    st.subheader("Results by Accident Year")
    money_cols = [c for c in results.columns if c not in ("Diagonal_Lag", "CDF")]
    styled = results.style.format(
        {c: "${:,.2f}" for c in money_cols} | {"CDF": "{:.6f}"}
    )
    st.dataframe(styled, use_container_width=True)

    # ── Totals row ───────────────────────────────────────────────────────────
    totals = results[money_cols].sum()
    totals_df = totals.to_frame("Total").T
    st.dataframe(
        totals_df.style.format("${:,.2f}"),
        use_container_width=True,
    )

    # ── Development factors ──────────────────────────────────────────────────
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

    # ── Download ─────────────────────────────────────────────────────────────
    st.download_button(
        "⬇ Download Results (CSV)",
        results.to_csv(),
        file_name=f"reserves_{int(valuation_year)}.csv",
        mime="text/csv",
    )
