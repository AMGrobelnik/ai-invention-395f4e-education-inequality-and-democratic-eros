#!/usr/bin/env python3
"""PSE Moderator Comparison + Causal Mediation with Quadrant Diagnostics.

Re-runs DD triple interaction replacing SocProt with ILO PSE share as moderator,
compares β₇_PSE vs β₇_SocProt, runs Baron-Kenny causal mediation via PSE→judicial
compliance, quadrant missing-data table, extreme-values ACME sensitivity, and
reverse causality tests for Gini→LDem and IMF SAP descriptive correlation.
"""

import gc
import json
import resource
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore", category=RuntimeWarning)

def safe_float(x: Any) -> float | None:
    """Convert value to float, returning None for NaN/None/non-finite."""
    if x is None:
        return None
    try:
        v = float(x)
        return None if not np.isfinite(v) else v
    except (TypeError, ValueError):
        return None


WORKSPACE = Path(__file__).parent
DS1 = WORKSPACE / "../../../iter_1/gen_art/gen_art_dataset_1/full_data_out.json"
DS2 = WORKSPACE / "../../../iter_1/gen_art/gen_art_dataset_2/full_data_out.json"
OUT = WORKSPACE / "method_out.json"
OUT_MINI = WORKSPACE / "method_out_mini.json"
LOGS = WORKSPACE / "logs"
LOGS.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS / "run.log"), rotation="30 MB", level="DEBUG")

# 8 GB RAM budget — well above actual needs for this in-memory panel analysis
_BUDGET = 8 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (_BUDGET, _BUDGET))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — LOAD AND PARSE DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def load_panel(ds1_path: Path) -> pd.DataFrame:
    """Parse gen_art_dataset_1 full_data_out.json → flat panel DataFrame (790 rows)."""
    logger.info(f"Loading DS1 from {ds1_path}")
    raw = json.loads(ds1_path.read_text())
    rows: list[dict] = []
    for ex in raw["datasets"][0]["examples"]:
        features = json.loads(ex["input"])
        features["v2x_libdem"] = (
            float(ex["output"]) if ex["output"] not in ("null", None) else None
        )
        features["country_name"] = ex.get("metadata_country_name")
        features["has_all_core"] = ex.get("metadata_has_all_core_vars", False)
        rows.append(features)
    df = pd.DataFrame(rows)
    logger.info(f"DS1 loaded: {len(df)} rows, {df['country_iso3'].nunique()} countries")
    logger.debug(f"DS1 columns: {list(df.columns)}")
    del raw
    gc.collect()
    return df


def load_pse_imf(ds2_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse gen_art_dataset_2 → (pse_annual, imf_annual) DataFrames."""
    logger.info(f"Loading DS2 from {ds2_path}")
    raw = json.loads(ds2_path.read_text())

    # Dataset 0: PSE
    pse_rows: list[dict] = []
    for ex in raw["datasets"][0]["examples"]:
        iso3 = ex["metadata_iso3"]
        year = int(ex["metadata_year"])
        out = ex["output"]
        pse_share = None
        try:
            share_str = out.split("pse_share_of_total_employment:")[1].strip().rstrip("%").strip()
            pse_share = float(share_str)
        except (IndexError, ValueError):
            pass
        pse_rows.append({"country_iso3": iso3, "year": year, "pse_share": pse_share})

    pse_annual = pd.DataFrame(pse_rows).dropna(subset=["pse_share"])
    logger.info(f"PSE annual: {len(pse_annual)} rows, {pse_annual['country_iso3'].nunique()} countries")

    # Dataset 1: IMF SAP
    imf_rows: list[dict] = []
    for ex in raw["datasets"][1]["examples"]:
        iso3 = ex["metadata_iso3"]
        year = int(ex["metadata_year"])
        active = 1 if "imf_sap_active: 1" in ex["output"] else 0
        imf_rows.append({"country_iso3": iso3, "year": year, "imf_sap_active": active})

    imf_annual = pd.DataFrame(imf_rows)
    logger.info(f"IMF SAP annual: {len(imf_annual)} rows, {imf_annual['country_iso3'].nunique()} countries")

    del raw
    gc.collect()
    return pse_annual, imf_annual


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PERIOD AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def year_to_period(y: int) -> int | None:
    """Map annual year → 5-year period_start bucket matching DS1 conventions."""
    if y < 1990:
        return None
    return ((y - 1990) // 5) * 5 + 1990


def aggregate_to_periods(pse_annual: pd.DataFrame, imf_annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pse_annual = pse_annual.copy()
    pse_annual["period_start"] = pse_annual["year"].apply(year_to_period)
    pse_period = (
        pse_annual.dropna(subset=["period_start"])
        .groupby(["country_iso3", "period_start"])["pse_share"]
        .mean()
        .reset_index()
        .rename(columns={"pse_share": "pse_share_period"})
    )
    logger.info(f"PSE period aggregated: {len(pse_period)} country-period pairs")

    imf_annual = imf_annual.copy()
    imf_annual["period_start"] = imf_annual["year"].apply(year_to_period)
    imf_period = (
        imf_annual.dropna(subset=["period_start"])
        .groupby(["country_iso3", "period_start"])["imf_sap_active"]
        .mean()
        .reset_index()
        .rename(columns={"imf_sap_active": "imf_sap_share"})
    )
    logger.info(f"IMF period aggregated: {len(imf_period)} country-period pairs")

    return pse_period, imf_period


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MERGE + DERIVE PSE DEVIATIONS
# ─────────────────────────────────────────────────────────────────────────────

def merge_datasets(panel: pd.DataFrame, pse_period: pd.DataFrame, imf_period: pd.DataFrame) -> pd.DataFrame:
    panel = panel.merge(pse_period, on=["country_iso3", "period_start"], how="left")
    panel = panel.merge(imf_period, on=["country_iso3", "period_start"], how="left")

    panel["mean_pse"] = panel.groupby("country_iso3")["pse_share_period"].transform("mean")
    panel["e_pse"] = panel["pse_share_period"] - panel["mean_pse"]

    n_pse = panel["pse_share_period"].notna().sum()
    logger.info(f"After merge: {len(panel)} rows, {n_pse} with PSE data ({100*n_pse/len(panel):.1f}%)")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — QUADRANT MISSING-DATA DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_quadrant_coverage_table(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["gini_disp", "socprot_coverage"]).copy()
    gini_med = sub["gini_disp"].median()
    socprot_med = sub["socprot_coverage"].median()
    sub["quad_highgini"] = (sub["gini_disp"] > gini_med).astype(int)
    sub["quad_highsocprot"] = (sub["socprot_coverage"] > socprot_med).astype(int)

    table: dict[str, Any] = {}
    for hg in [0, 1]:
        for hs in [0, 1]:
            cell = sub[(sub["quad_highgini"] == hg) & (sub["quad_highsocprot"] == hs)]
            n_total = len(cell)
            n_pse = int(cell["pse_share_period"].notna().sum())
            label = f"highGini={hg}_highSocProt={hs}"
            table[label] = {
                "n_observations": n_total,
                "n_with_pse": n_pse,
                "pse_coverage_pct": round(100 * n_pse / n_total, 1) if n_total > 0 else 0.0,
                "countries": sorted(cell["country_iso3"].unique().tolist()),
            }
    logger.info(f"Quadrant table computed; gini_med={gini_med:.2f}, socprot_med={socprot_med:.2f}")
    for k, v in table.items():
        logger.info(f"  {k}: n={v['n_observations']}, pse_coverage={v['pse_coverage_pct']}%")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DD TRIPLE INTERACTION ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_dd_triple(
    df: pd.DataFrame,
    e_educ_col: str,
    e_gini_col: str,
    e_mod_col: str,
    outcome_col: str,
    country_col: str,
    label: str,
) -> dict:
    """DD triple interaction: Education × Gini × Moderator with Giesselmann-Schmidt-Catran correction.

    Falls back to cross-sectional OLS when within-country variation is absent (singleton countries).
    Returns dict with β₇ (triple interaction), SEs, t-stats, p-values, marginal effects.
    """
    required = [e_educ_col, e_gini_col, e_mod_col, outcome_col, country_col, "period_start"]
    df = df.dropna(subset=required).copy()
    n_obs_raw = len(df)
    n_ctry = df[country_col].nunique()

    if n_obs_raw < 10:
        logger.warning(f"{label}: insufficient obs ({n_obs_raw})")
        return {"error": f"Insufficient obs ({n_obs_raw}) for {label} spec", "n_obs": n_obs_raw}

    # Detect singleton countries (each has only 1 observation) — within-DD is not feasible
    obs_per_country = df.groupby(country_col).size()
    multi_period_countries = obs_per_country[obs_per_country >= 2].index
    n_multi = len(multi_period_countries)

    use_dd = n_multi >= 5  # Need at least 5 countries with 2+ periods for within DD

    if not use_dd:
        logger.warning(
            f"{label}: only {n_multi} countries have 2+ periods — "
            f"within-country DD not feasible; using cross-sectional OLS fallback"
        )
        return _run_cross_sectional_fallback(df, e_educ_col, e_gini_col, e_mod_col, outcome_col, label)

    # Step 1: raw interaction products of within-country residuals
    df["int_eg"]  = df[e_educ_col] * df[e_gini_col]
    df["int_em"]  = df[e_educ_col] * df[e_mod_col]
    df["int_gm"]  = df[e_gini_col] * df[e_mod_col]
    df["int_egm"] = df[e_educ_col] * df[e_gini_col] * df[e_mod_col]

    # Step 2: Giesselmann–Schmidt-Catran: within-demean the interaction products
    for col in ["int_eg", "int_em", "int_gm", "int_egm"]:
        mean_col = df.groupby(country_col)[col].transform("mean")
        df[f"dd_{col}"] = df[col] - mean_col

    # Step 3: within-demean the outcome (absorb country FE)
    df["y_within"] = df[outcome_col] - df.groupby(country_col)[outcome_col].transform("mean")

    # Step 4: OLS with period dummies (absorb period FE)
    period_dummies = pd.get_dummies(df["period_start"], prefix="pd", drop_first=True).astype(float)
    x_cols = [e_educ_col, e_gini_col, e_mod_col,
               "dd_int_eg", "dd_int_em", "dd_int_gm", "dd_int_egm"]
    X = pd.concat([df[x_cols].reset_index(drop=True), period_dummies.reset_index(drop=True)], axis=1)
    X = add_constant(X)
    y = df["y_within"].reset_index(drop=True)

    # Check rank; drop collinear terms iteratively
    for drop_col in ["dd_int_gm", "dd_int_em"]:
        rank = np.linalg.matrix_rank(X.values.astype(float))
        if rank < X.shape[1]:
            if drop_col in X.columns:
                logger.warning(f"{label}: rank deficiency (rank={rank}, ncols={X.shape[1]}), dropping {drop_col}")
                X = X.drop(columns=[drop_col])

    model = OLS(y, X, missing="drop").fit(cov_type="HC3")

    beta7 = safe_float(model.params.get("dd_int_egm"))
    se7   = safe_float(model.bse.get("dd_int_egm"))
    t7    = safe_float(model.tvalues.get("dd_int_egm"))
    p7    = safe_float(model.pvalues.get("dd_int_egm"))

    # Marginal effect of education at 25th vs 75th moderator percentile (Gini held at 75th)
    p25m = float(df[e_mod_col].quantile(0.25))
    p75m = float(df[e_mod_col].quantile(0.75))
    p75g = float(df[e_gini_col].quantile(0.75))

    b1 = safe_float(model.params.get(e_educ_col)) or 0.0
    b4 = safe_float(model.params.get("dd_int_eg"))  or 0.0
    b5 = safe_float(model.params.get("dd_int_em"))  or 0.0
    b7 = beta7 or 0.0

    me_low_m  = b1 + b4 * p75g + b5 * p25m + b7 * p75g * p25m
    me_high_m = b1 + b4 * p75g + b5 * p75m + b7 * p75g * p75m

    def _safe_coef(k: str) -> dict | None:
        c = safe_float(model.params.get(k))
        p = safe_float(model.pvalues.get(k))
        if c is None:
            return None
        return {"coef": round(c, 6), "p": round(p, 4) if p is not None else None}

    all_coefs = {k: _safe_coef(k) for k in model.params.index if _safe_coef(k) is not None}

    # Compute level-space fitted values: fittedvalues are in y_within space; add back country means
    country_means_s = df.groupby(country_col)[outcome_col].mean()
    country_mean_arr = df[country_col].map(country_means_s).values
    fv_raw = model.fittedvalues.values  # 0..nobs-1 after reset_index; missing="drop" means all rows used
    fitted_level = fv_raw + country_mean_arr[:len(fv_raw)]
    fitted_values = {
        int(orig_idx): round(float(fv), 6) if np.isfinite(fv) else None
        for orig_idx, fv in zip(df.index[:len(fv_raw)], fitted_level)
    }

    result = {
        "moderator": label,
        "estimation": "within_DD",
        "n_obs": int(model.nobs),
        "n_countries": n_ctry,
        "n_multi_period_countries": n_multi,
        "beta7": round(beta7, 6) if beta7 is not None else None,
        "se7":   round(se7,   6) if se7   is not None else None,
        "t7":    round(t7,    4) if t7    is not None else None,
        "p7":    round(p7,    4) if p7    is not None else None,
        "sign_reversal": {
            "me_at_low_moderator":  round(me_low_m,  6),
            "me_at_high_moderator": round(me_high_m, 6),
            "signs_differ": bool(me_low_m * me_high_m < 0),
        },
        "rsquared": safe_float(model.rsquared),
        "all_coefs": all_coefs,
        "_fitted_values": fitted_values,
    }
    logger.info(
        f"{label}: n={result['n_obs']}, β₇={result['beta7']}, p={result['p7']}, "
        f"sign_reversal={result['sign_reversal']['signs_differ']}"
    )
    return result


def _run_cross_sectional_fallback(
    df: pd.DataFrame,
    e_educ_col: str,
    e_gini_col: str,
    e_mod_col: str,
    outcome_col: str,
    label: str,
) -> dict:
    """Cross-sectional OLS fallback when within-country DD is not feasible.

    Uses level (not within-demeaned) variables with period dummies.
    Reports β for triple interaction; note this does NOT control for country FE.
    """
    period_dummies = pd.get_dummies(df["period_start"], prefix="pd", drop_first=True).astype(float)
    # Raw triple interactions (not demeaned)
    df = df.copy()
    df["int_eg"]  = df[e_educ_col] * df[e_gini_col]
    df["int_em"]  = df[e_educ_col] * df[e_mod_col]
    df["int_gm"]  = df[e_gini_col] * df[e_mod_col]
    df["int_egm"] = df[e_educ_col] * df[e_gini_col] * df[e_mod_col]
    x_cols = [e_educ_col, e_gini_col, e_mod_col, "int_eg", "int_em", "int_gm", "int_egm"]
    X = pd.concat([df[x_cols].reset_index(drop=True), period_dummies.reset_index(drop=True)], axis=1)
    X = add_constant(X)
    y = df[outcome_col].reset_index(drop=True)

    # Drop collinear columns
    for drop_col in ["int_gm", "int_em"]:
        rank = np.linalg.matrix_rank(X.values.astype(float))
        if rank < X.shape[1] and drop_col in X.columns:
            logger.warning(f"{label} (XS): dropping {drop_col} due to rank deficiency")
            X = X.drop(columns=[drop_col])

    try:
        model = OLS(y, X, missing="drop").fit(cov_type="HC3")
    except Exception as exc:
        logger.error(f"{label} cross-sectional OLS failed: {exc}")
        return {"error": str(exc), "n_obs": len(df), "estimation": "cross_sectional_failed"}

    beta7 = safe_float(model.params.get("int_egm"))
    se7   = safe_float(model.bse.get("int_egm"))
    t7    = safe_float(model.tvalues.get("int_egm"))
    p7    = safe_float(model.pvalues.get("int_egm"))

    p25m = float(df[e_mod_col].quantile(0.25))
    p75m = float(df[e_mod_col].quantile(0.75))
    p75g = float(df[e_gini_col].quantile(0.75))
    b1   = safe_float(model.params.get(e_educ_col))  or 0.0
    b4   = safe_float(model.params.get("int_eg"))    or 0.0
    b5   = safe_float(model.params.get("int_em"))    or 0.0
    b7   = beta7 or 0.0
    me_low_m  = b1 + b4 * p75g + b5 * p25m + b7 * p75g * p25m
    me_high_m = b1 + b4 * p75g + b5 * p75m + b7 * p75g * p75m

    def _safe_coef(k: str) -> dict | None:
        c = safe_float(model.params.get(k))
        p = safe_float(model.pvalues.get(k))
        if c is None:
            return None
        return {"coef": round(c, 6), "p": round(p, 4) if p is not None else None}

    all_coefs = {k: _safe_coef(k) for k in model.params.index if _safe_coef(k) is not None}

    # Fitted values are already in level space (outcome_col not demeaned)
    fv_raw = model.fittedvalues.values
    fitted_values = {
        int(orig_idx): round(float(fv), 6) if np.isfinite(fv) else None
        for orig_idx, fv in zip(df.index[:len(fv_raw)], fv_raw)
    }

    result = {
        "moderator": label,
        "estimation": "cross_sectional_OLS_no_country_FE",
        "caveat": (
            "DD within-country estimator not feasible (all countries are singletons in PSE data). "
            "Cross-sectional OLS without country fixed effects. "
            "Coefficients reflect between-country variation and may be confounded."
        ),
        "n_obs": int(model.nobs),
        "n_countries": df["country_iso3"].nunique() if "country_iso3" in df.columns else None,
        "n_multi_period_countries": 0,
        "beta7": round(beta7, 6) if beta7 is not None else None,
        "se7":   round(se7,   6) if se7   is not None else None,
        "t7":    round(t7,    4) if t7    is not None else None,
        "p7":    round(p7,    4) if p7    is not None else None,
        "sign_reversal": {
            "me_at_low_moderator":  round(me_low_m,  6),
            "me_at_high_moderator": round(me_high_m, 6),
            "signs_differ": bool(me_low_m * me_high_m < 0),
        },
        "rsquared": safe_float(model.rsquared),
        "all_coefs": all_coefs,
        "_fitted_values": fitted_values,
    }
    logger.info(
        f"{label} (XS): n={result['n_obs']}, β₇={result['beta7']}, p={result['p7']}, "
        f"sign_reversal={result['sign_reversal']['signs_differ']}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BARON-KENNY CAUSAL MEDIATION
# ─────────────────────────────────────────────────────────────────────────────

def run_baron_kenny_mediation(
    df: pd.DataFrame,
    outcome_col: str,
    mediator_col: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Baron-Kenny mediation: exposure T = e_edu × e_gini × (−e_socprot) → mediator → outcome.

    All variables within-country demeaned; period FEs included.
    Returns ACME, bootstrap 95% CI, proportion mediated.
    """
    required = ["e_education", "e_gini_disp", "e_socprot_coverage", mediator_col, outcome_col,
                "country_iso3", "period_start"]
    df = df.dropna(subset=required).copy()
    n = len(df)

    if n < 15:
        return {
            "error": "Insufficient obs for mediation",
            "insufficient_data": True,
            "n_obs": n,
        }

    # Construct exposure T = e_edu × e_gini × (−e_socprot) then within-demean per country
    df["T_raw"] = df["e_education"] * df["e_gini_disp"] * (-df["e_socprot_coverage"])
    df["T"] = df["T_raw"] - df.groupby("country_iso3")["T_raw"].transform("mean")

    # Detect singleton-per-period sample: T collapses to 0 when every country appears once.
    # Fall back to level exposure so cross-sectional OLS has variation to estimate.
    cross_sectional_fallback = df["T"].abs().max() < 1e-10
    if cross_sectional_fallback:
        logger.warning(
            f"Mediation ({outcome_col}): T within-demeaned to 0 (singleton-period sample). "
            f"Using level triple product education×gini×(−socprot) as cross-sectional exposure."
        )
        df["T"] = df["education"] * df["gini_disp"] * (-df["socprot_coverage"])
        # Standardize T to unit variance so coefficients are interpretable
        t_std = df["T"].std()
        if t_std > 1e-10:
            df["T"] = df["T"] / t_std

    # Within-demean mediator and outcome (no-op for singletons, but harmless)
    df["M"] = df[mediator_col] - df.groupby("country_iso3")[mediator_col].transform("mean")
    df["Y"] = df[outcome_col]  - df.groupby("country_iso3")[outcome_col].transform("mean")

    # For singleton samples: M and Y within-demeaning = 0; use levels instead
    if cross_sectional_fallback:
        df["M"] = df[mediator_col]
        df["Y"] = df[outcome_col]

    period_dummies = pd.get_dummies(df["period_start"], prefix="pd", drop_first=True).astype(float)

    def fit_ols(y_data: pd.Series, regressors: pd.DataFrame) -> Any:
        X = pd.concat(
            [regressors.reset_index(drop=True), period_dummies.reset_index(drop=True)], axis=1
        )
        X = add_constant(X)
        y = y_data.reset_index(drop=True)
        valid = y.notna() & X.notna().all(axis=1)
        return OLS(y[valid], X[valid], missing="drop").fit(cov_type="HC3")

    T_df = df[["T"]].reset_index(drop=True)
    M_df = df[["M"]].reset_index(drop=True)

    # Step a: total effect c — Y ~ T
    mod_c  = fit_ols(df["Y"], T_df)
    c_coef = safe_float(mod_c.params.get("T"))
    c_p    = safe_float(mod_c.pvalues.get("T"))

    # Step b: mediator equation a — M ~ T
    mod_a  = fit_ols(df["M"], T_df)
    a_coef = safe_float(mod_a.params.get("T"))
    a_p    = safe_float(mod_a.pvalues.get("T"))

    # Step c: direct effect c' and b — Y ~ T + M
    TM_df  = pd.concat([T_df, M_df], axis=1)
    mod_c2 = fit_ols(df["Y"], TM_df)
    b_coef  = safe_float(mod_c2.params.get("M"))
    b_p     = safe_float(mod_c2.pvalues.get("M"))
    cp_coef = safe_float(mod_c2.params.get("T"))
    cp_p    = safe_float(mod_c2.pvalues.get("T"))

    acme = (a_coef * b_coef) if (a_coef is not None and b_coef is not None) else None
    prop = (acme / c_coef) if (c_coef is not None and abs(c_coef) > 1e-12 and acme is not None) else None

    # Bootstrap CIs for ACME (block bootstrap over rows; adequate for N<100)
    rng = np.random.default_rng(seed)
    acme_boot: list[float] = []
    idx_all = np.arange(n)
    pd_arr  = period_dummies.values

    for _ in range(n_bootstrap):
        bi = rng.choice(idx_all, size=n, replace=True)
        T_b  = df["T"].values[bi].reshape(-1, 1)
        M_b  = df["M"].values[bi].reshape(-1, 1)
        Y_b  = df["Y"].values[bi]
        pd_b = pd_arr[bi]
        try:
            Xa = np.column_stack([np.ones(n), T_b, pd_b])
            Xc = np.column_stack([np.ones(n), T_b, M_b, pd_b])
            valid_a = np.isfinite(Xa).all(axis=1) & np.isfinite(M_b.ravel())
            valid_c = np.isfinite(Xc).all(axis=1) & np.isfinite(Y_b)
            if valid_a.sum() < 5 or valid_c.sum() < 5:
                continue
            a_b = np.linalg.lstsq(Xa[valid_a], M_b.ravel()[valid_a], rcond=None)[0][1]
            b_b = np.linalg.lstsq(Xc[valid_c], Y_b[valid_c], rcond=None)[0][2]
            acme_boot.append(float(a_b * b_b))
        except np.linalg.LinAlgError:
            pass

    n_valid_boot = len(acme_boot)
    acme_ci_low  = float(np.percentile(acme_boot, 2.5))  if acme_boot else None
    acme_ci_high = float(np.percentile(acme_boot, 97.5)) if acme_boot else None
    bootstrap_unstable = n_valid_boot < 50

    result = {
        "outcome": outcome_col,
        "mediator": mediator_col,
        "n_obs": n,
        "n_countries": int(df["country_iso3"].nunique()),
        "estimation": "cross_sectional_OLS_no_country_FE" if cross_sectional_fallback else "within_country_FE",
        "caveat": (
            "All PSE obs are period_start=2015; within-country demeaning collapses T→0. "
            "Cross-sectional mediation using level education×gini×(−socprot) as exposure. "
            "Captures cross-country differences, not within-country changes."
        ) if cross_sectional_fallback else None,
        "total_effect_c":     round(c_coef,  6) if c_coef  is not None else None,
        "total_effect_c_p":   round(c_p,     4) if c_p     is not None else None,
        "a_coef_T_to_M":      round(a_coef,  6) if a_coef  is not None else None,
        "a_p":                round(a_p,     4) if a_p     is not None else None,
        "b_coef_M_to_Y":      round(b_coef,  6) if b_coef  is not None else None,
        "b_p":                round(b_p,     4) if b_p     is not None else None,
        "direct_effect_cp":   round(cp_coef, 6) if cp_coef is not None else None,
        "direct_effect_cp_p": round(cp_p,    4) if cp_p    is not None else None,
        "acme": round(acme, 6) if acme is not None else None,
        "acme_ci_95": (
            [round(acme_ci_low, 6), round(acme_ci_high, 6)]
            if acme_ci_low is not None else None
        ),
        "n_bootstrap_valid": n_valid_boot,
        "bootstrap_unstable": bootstrap_unstable,
        "proportion_mediated": round(prop, 4) if prop is not None else None,
        "baron_kenny_satisfied": bool(
            (c_p is not None and c_p < 0.05) and
            (a_p is not None and a_p < 0.05) and
            (b_p is not None and b_p < 0.05)
        ),
    }
    logger.info(
        f"Mediation ({outcome_col}): ACME={result['acme']}, a_p={result['a_p']}, "
        f"b_p={result['b_p']}, BK_satisfied={result['baron_kenny_satisfied']}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EXTREME-VALUES ACME SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_extreme_sensitivity(
    complete: pd.DataFrame,
    med_sample: pd.DataFrame,
    acme_observed: float | None,
) -> dict:
    missing_pse = complete[complete["pse_share_period"].isna()]["country_iso3"].unique().tolist()
    n_missing   = len(missing_pse)
    n_with_pse  = int(med_sample["country_iso3"].nunique())
    n_total     = int(complete["country_iso3"].nunique())

    if acme_observed is None or abs(acme_observed) < 1e-15:
        return {
            "n_countries_with_pse_data": n_with_pse,
            "n_countries_total_complete": n_total,
            "n_countries_missing_pse": n_missing,
            "missing_pse_country_list": sorted(missing_pse),
            "acme_observed": None,
            "acme_weighted_conservative_assuming_zeros": None,
            "pct_attenuation": None,
            "survives_conservative_bound": False,
        }

    acme_weighted = float(acme_observed) * n_with_pse / n_total

    return {
        "n_countries_with_pse_data": n_with_pse,
        "n_countries_total_complete": n_total,
        "n_countries_missing_pse": n_missing,
        "missing_pse_country_list": sorted(missing_pse),
        "acme_observed": round(float(acme_observed), 6),
        "acme_weighted_conservative_assuming_zeros": round(acme_weighted, 6),
        "pct_attenuation": round(100 * (1 - acme_weighted / float(acme_observed)), 1),
        "survives_conservative_bound": bool(abs(acme_weighted) > 0.001),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SEQUENTIAL IGNORABILITY CHECK (sensitivity parameter ρ)
# ─────────────────────────────────────────────────────────────────────────────

def sequential_ignorability_check(
    df: pd.DataFrame,
    outcome_col: str = "v2jucomp",
    mediator_col: str = "pse_share_period",
) -> dict:
    required = ["e_education", "e_gini_disp", "e_socprot_coverage",
                mediator_col, outcome_col, "country_iso3", "period_start"]
    df = df.dropna(subset=required).copy()
    if len(df) < 10:
        return {"error": "Insufficient obs for sequential ignorability check", "n_obs": len(df)}

    df["T_raw"] = df["e_education"] * df["e_gini_disp"] * (-df["e_socprot_coverage"])
    df["T"] = df["T_raw"] - df.groupby("country_iso3")["T_raw"].transform("mean")

    cross_sectional = df["T"].abs().max() < 1e-10
    if cross_sectional:
        df["T"] = df["education"] * df["gini_disp"] * (-df["socprot_coverage"])
        t_std = df["T"].std()
        if t_std > 1e-10:
            df["T"] = df["T"] / t_std
        df["M"] = df[mediator_col]
        df["Y"] = df[outcome_col]
    else:
        df["M"] = df[mediator_col] - df.groupby("country_iso3")[mediator_col].transform("mean")
        df["Y"] = df[outcome_col]  - df.groupby("country_iso3")[outcome_col].transform("mean")

    period_dummies = pd.get_dummies(df["period_start"], prefix="pd", drop_first=True).astype(float)

    T_df = df[["T"]].reset_index(drop=True)
    pd_r = period_dummies.reset_index(drop=True)

    Xa = add_constant(pd.concat([T_df, pd_r], axis=1))
    mod_a = OLS(df["M"].reset_index(drop=True), Xa, missing="drop").fit()
    resid_m = mod_a.resid

    Xc = add_constant(pd.concat([T_df, pd_r], axis=1))
    mod_c = OLS(df["Y"].reset_index(drop=True), Xc, missing="drop").fit()
    resid_y = mod_c.resid

    min_len = min(len(resid_m), len(resid_y))
    rm = resid_m.values[:min_len]
    ry = resid_y.values[:min_len]
    rho_val = rho_p_val = None
    if np.std(rm) > 1e-12 and np.std(ry) > 1e-12:
        rho_raw, rho_p_raw = stats.pearsonr(rm, ry)
        rho_val   = safe_float(rho_raw)
        rho_p_val = safe_float(rho_p_raw)

    result = {
        "rho_residual_correlation": round(rho_val, 4) if rho_val is not None else None,
        "rho_p_value": round(rho_p_val, 4) if rho_p_val is not None else None,
        "sequential_ignorability_concern": bool(
            rho_val is not None and rho_p_val is not None and abs(rho_val) > 0.3 and rho_p_val < 0.05
        ),
        "interpretation": (
            "Non-zero rho suggests unobserved confounders affect both PSE selection and outcome; "
            "this would violate sequential ignorability and bias ACME estimates."
        ),
    }
    logger.info(
        f"Sequential ignorability: rho={result['rho_residual_correlation']}, "
        f"concern={result['sequential_ignorability_concern']}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — REVERSE CAUSALITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_reverse_causality_gini_ldem(df: pd.DataFrame) -> dict:
    """Regress gini_disp on lagged v2x_libdem — does past democracy reduce inequality?"""
    df = df.sort_values(["country_iso3", "period_start"]).copy()
    df["ldem_lag1"] = df.groupby("country_iso3")["v2x_libdem"].shift(1)
    df = df.dropna(subset=["gini_disp", "ldem_lag1"])
    if len(df) < 10:
        return {"error": "Insufficient obs for reverse causality test", "n_obs": len(df)}

    df["gini_w"] = df["gini_disp"] - df.groupby("country_iso3")["gini_disp"].transform("mean")
    df["ldem_w"] = df["ldem_lag1"] - df.groupby("country_iso3")["ldem_lag1"].transform("mean")
    period_dummies = pd.get_dummies(df["period_start"], prefix="pd", drop_first=True).astype(float)
    X = add_constant(pd.concat([df[["ldem_w"]].reset_index(drop=True), period_dummies.reset_index(drop=True)], axis=1))
    y = df["gini_w"].reset_index(drop=True)
    mod = OLS(y, X, missing="drop").fit(cov_type="HC3")
    coef = safe_float(mod.params.get("ldem_w"))
    p    = safe_float(mod.pvalues.get("ldem_w"))

    result = {
        "path_a_description": "Regress gini on lagged LDem (country FE + period FE)",
        "n_obs": int(mod.nobs),
        "beta_ldem_on_gini": round(coef, 6) if coef is not None else None,
        "p_value": round(p, 4) if p is not None else None,
        "reverse_causality_concern": bool(p < 0.05) if p is not None else None,
        "interpretation": (
            "Significant negative effect means democracies reduce inequality; "
            "if so, Gini is endogenous to LDem and must be instrumented or sample restricted."
        ),
    }
    logger.info(f"Reverse causality Gini←LDem: β={result['beta_ldem_on_gini']}, p={result['p_value']}, concern={result['reverse_causality_concern']}")
    return result


def test_imf_sap_socprot_correlation(df: pd.DataFrame) -> dict:
    """Descriptive: regress ΔSocProt on lagged IMF SAP share."""
    df = df.sort_values(["country_iso3", "period_start"]).copy()
    df["dsocprot"] = df.groupby("country_iso3")["socprot_coverage"].diff()
    df["imf_lag"]  = df.groupby("country_iso3")["imf_sap_share"].shift(1)
    sub = df.dropna(subset=["dsocprot", "imf_lag"])
    if len(sub) < 10:
        return {"error": "Insufficient obs for IMF SAP test", "n_obs": len(sub)}

    period_dummies = pd.get_dummies(sub["period_start"], prefix="pd", drop_first=True).astype(float)
    X = add_constant(pd.concat([sub[["imf_lag"]].reset_index(drop=True), period_dummies.reset_index(drop=True)], axis=1))
    y = sub["dsocprot"].reset_index(drop=True)
    mod  = OLS(y, X, missing="drop").fit(cov_type="HC3")
    coef = safe_float(mod.params.get("imf_lag"))
    p    = safe_float(mod.pvalues.get("imf_lag"))
    corr = safe_float(sub[["dsocprot", "imf_lag"]].corr().iloc[0, 1])

    result = {
        "path_b_description": "Correlation of lagged IMF SAP share with SocProt change (descriptive, not IV)",
        "n_obs": int(mod.nobs),
        "raw_correlation": round(corr, 4) if corr is not None else None,
        "beta_imfsap_on_dsocprot": round(coef, 6) if coef is not None else None,
        "p_value": round(p, 4) if p is not None else None,
        "interpretation": (
            "Negative beta means IMF programs precede SocProt reductions; "
            "positive means programs coincide with coverage expansions. "
            "Descriptive only — not used as IV per hypothesis exclusion restriction decision."
        ),
    }
    logger.info(f"IMF SAP→SocProt: β={result['beta_imfsap_on_dsocprot']}, p={result['p_value']}")
    return result


def run_lagged_socprot_sensitivity(complete: pd.DataFrame) -> dict:
    """Replace contemporaneous socprot with one-period lag in DD triple interaction."""
    df = complete.sort_values(["country_iso3", "period_start"]).copy()
    df["socprot_lag1"] = df.groupby("country_iso3")["socprot_coverage"].shift(1)
    df["mean_sp_lag"]  = df.groupby("country_iso3")["socprot_lag1"].transform("mean")
    df["e_socprot_lag1"] = df["socprot_lag1"] - df["mean_sp_lag"]
    sub = df.dropna(subset=["education", "gini_disp", "socprot_lag1", "v2x_libdem"])

    if len(sub) < 10:
        return {
            "error": "Insufficient obs for lagged SocProt sensitivity",
            "lagged_sensitivity_underpowered": True,
            "n_obs": len(sub),
        }

    return run_dd_triple(
        df=sub,
        e_educ_col="e_education",
        e_gini_col="e_gini_disp",
        e_mod_col="e_socprot_lag1",
        outcome_col="v2x_libdem",
        country_col="country_iso3",
        label="SocProt_lagged_t5",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — OUTPUT FORMATTING (exp_gen_sol_out schema)
# ─────────────────────────────────────────────────────────────────────────────

def build_output(
    pse_missing_table: dict,
    complete: pd.DataFrame,
    pse_sample: pd.DataFrame,
    med_sample: pd.DataFrame,
    result_socprot: dict,
    result_pse: dict,
    med_judicial: dict,
    med_ngo: dict,
    med_libdem: dict,
    extreme_sens: dict,
    seq_ign: dict,
    rev_gini: dict,
    rev_imf: dict,
    lag_sens: dict,
) -> dict:
    """Wrap all results into exp_gen_sol_out schema: per-row predictions with predict_* fields."""

    # Extract fitted values (pop so they don't appear in JSON output of stats)
    fitted_socprot: dict[int, float | None] = result_socprot.pop("_fitted_values", {})
    fitted_pse: dict[int, float | None] = result_pse.pop("_fitted_values", {})

    # ── Summary verdict ───────────────────────────────────────────────────────
    sp_sig = result_socprot.get("p7") is not None and result_socprot["p7"] < 0.1
    pse_sig = result_pse.get("p7") is not None and result_pse["p7"] < 0.1
    sp_b7 = result_socprot.get("beta7") or 0
    pse_b7 = result_pse.get("beta7") or 0
    opp_cost_channel = sp_sig and (not pse_sig or abs(sp_b7) > abs(pse_b7))
    bk_jud = med_judicial.get("baron_kenny_satisfied", False)
    acme_surv = extreme_sens.get("survives_conservative_bound", False)
    rev_concern = rev_gini.get("reverse_causality_concern", False)

    verdict = (
        f"SocProt β₇={sp_b7:.4f} (p={result_socprot.get('p7')}), "
        f"PSE β₇={pse_b7:.4f} (p={result_pse.get('p7')}); "
        f"opp-cost={'YES' if opp_cost_channel else 'NO'}; "
        f"BK={bk_jud}; ACME_survives={acme_surv}; rev_concern={rev_concern}"
    )

    # ── Dataset 1: per-row DD SocProt predictions (complete panel, 161 rows) ─
    sp_summary = (
        f"DD SocProt β₇={result_socprot.get('beta7')}, p={result_socprot.get('p7')}, "
        f"n={result_socprot.get('n_obs')}, sign_reversal={result_socprot.get('sign_reversal', {}).get('signs_differ')}"
    )
    pse_summary = (
        f"DD PSE β₇={result_pse.get('beta7')}, p={result_pse.get('p7')}, "
        f"n={result_pse.get('n_obs')}, estimation={result_pse.get('estimation')}"
    )

    panel_examples = []
    for orig_idx, row in complete.reset_index().iterrows():
        df_idx = int(row["index"]) if "index" in row else orig_idx
        inp = {
            "country_iso3": row["country_iso3"],
            "period_start": int(row["period_start"]),
            "education": round(float(row["education"]), 4) if pd.notna(row.get("education")) else None,
            "gini_disp": round(float(row["gini_disp"]), 4) if pd.notna(row.get("gini_disp")) else None,
            "socprot_coverage": round(float(row["socprot_coverage"]), 4) if pd.notna(row.get("socprot_coverage")) else None,
            "e_education": round(float(row["e_education"]), 6) if pd.notna(row.get("e_education")) else None,
            "e_gini_disp": round(float(row["e_gini_disp"]), 6) if pd.notna(row.get("e_gini_disp")) else None,
            "e_socprot_coverage": round(float(row["e_socprot_coverage"]), 6) if pd.notna(row.get("e_socprot_coverage")) else None,
        }
        actual = round(float(row["v2x_libdem"]), 6) if pd.notna(row.get("v2x_libdem")) else None
        fv = fitted_socprot.get(df_idx)
        panel_examples.append({
            "input": json.dumps(inp),
            "output": str(actual) if actual is not None else "",
            "predict_dd_socprot": str(fv) if fv is not None else "",
            "metadata_country_iso3": str(row["country_iso3"]),
            "metadata_period_start": str(int(row["period_start"])),
            "metadata_model_summary": sp_summary,
        })

    # ── Dataset 2: per-row PSE comparison predictions (pse_sample, ~25 rows) ─
    pse_examples = []
    for orig_idx, row in pse_sample.reset_index().iterrows():
        df_idx = int(row["index"]) if "index" in row else orig_idx
        inp = {
            "country_iso3": row["country_iso3"],
            "period_start": int(row["period_start"]),
            "education": round(float(row["education"]), 4) if pd.notna(row.get("education")) else None,
            "gini_disp": round(float(row["gini_disp"]), 4) if pd.notna(row.get("gini_disp")) else None,
            "pse_share_period": round(float(row["pse_share_period"]), 4) if pd.notna(row.get("pse_share_period")) else None,
        }
        actual = round(float(row["v2x_libdem"]), 6) if pd.notna(row.get("v2x_libdem")) else None
        fv = fitted_pse.get(df_idx)
        pse_examples.append({
            "input": json.dumps(inp),
            "output": str(actual) if actual is not None else "",
            "predict_dd_pse": str(fv) if fv is not None else "",
            "metadata_country_iso3": str(row["country_iso3"]),
            "metadata_period_start": str(int(row["period_start"])),
            "metadata_model_summary": pse_summary,
        })

    # ── Dataset 3: mediation analysis (one example per outcome) ──────────────
    mediation_examples = []
    for med_res, outcome_label in [
        (med_judicial, "v2jucomp"),
        (med_ngo, "v2cseeorgs"),
        (med_libdem, "v2x_libdem"),
    ]:
        acme_val = med_res.get("acme")
        mediation_examples.append({
            "input": json.dumps({
                "analysis": "Baron-Kenny_causal_mediation",
                "outcome": outcome_label,
                "mediator": "pse_share_period",
                "exposure": "education×gini×(−socprot)_within_demeaned",
                "n_obs": med_res.get("n_obs"),
            }),
            "output": json.dumps({
                "acme": acme_val,
                "acme_ci_95": med_res.get("acme_ci_95"),
                "baron_kenny_satisfied": med_res.get("baron_kenny_satisfied"),
                "a_p": med_res.get("a_p"),
                "b_p": med_res.get("b_p"),
            }, default=str),
            "predict_dd_socprot": (
                f"ACME={round(acme_val, 6)}" if acme_val is not None else "ACME=N/A"
            ),
            "metadata_outcome": outcome_label,
            "metadata_bk_satisfied": str(med_res.get("baron_kenny_satisfied", False)),
        })

    # ── Dataset 4: diagnostics (quadrant, sensitivity, seq. ignorability) ────
    diag_examples = [
        {
            "input": json.dumps({
                "analysis": "quadrant_missing_data",
                "quadrants": list(pse_missing_table.keys()),
            }),
            "output": json.dumps(pse_missing_table, default=str),
            "predict_dd_socprot": f"n_complete={len(complete)}, n_with_pse={int(pse_sample['country_iso3'].nunique())}",
            "metadata_diagnostic": "quadrant_coverage",
        },
        {
            "input": json.dumps({
                "analysis": "extreme_values_ACME_sensitivity",
                "assumption": "ACME=0 for countries without PSE data",
            }),
            "output": json.dumps({
                "acme_observed": extreme_sens.get("acme_observed"),
                "acme_conservative_bound": extreme_sens.get("acme_weighted_conservative_assuming_zeros"),
                "survives_conservative_bound": extreme_sens.get("survives_conservative_bound"),
                "attenuation_factor": extreme_sens.get("attenuation_factor"),
            }, default=str),
            "predict_dd_socprot": f"survives_bound={extreme_sens.get('survives_conservative_bound')}",
            "metadata_diagnostic": "extreme_sensitivity",
        },
        {
            "input": json.dumps({
                "analysis": "sequential_ignorability_check",
                "test": "residual_correlation_between_mediator_and_outcome_equations",
            }),
            "output": json.dumps({
                "rho_residual_correlation": seq_ign.get("rho_residual_correlation"),
                "p_value": seq_ign.get("p_value"),
                "sequential_ignorability_concern": seq_ign.get("sequential_ignorability_concern"),
            }, default=str),
            "predict_dd_socprot": f"rho={seq_ign.get('rho_residual_correlation')}, concern={seq_ign.get('sequential_ignorability_concern')}",
            "metadata_diagnostic": "sequential_ignorability",
        },
    ]

    # ── Dataset 5: reverse causality & robustness tests ───────────────────────
    robustness_examples = [
        {
            "input": json.dumps({
                "test": "reverse_causality_Gini_LDem",
                "hypothesis": "Does lagged LDem reduce Gini (path a)?",
            }),
            "output": json.dumps({
                "beta_ldem_on_gini": rev_gini.get("beta_ldem_on_gini"),
                "p_value": rev_gini.get("p_value"),
                "reverse_causality_concern": rev_gini.get("reverse_causality_concern"),
            }, default=str),
            "predict_dd_socprot": f"reverse_concern={rev_gini.get('reverse_causality_concern')}",
            "metadata_test": "gini_ldem_reverse",
        },
        {
            "input": json.dumps({
                "test": "IMF_SAP_socprot_association",
                "hypothesis": "Does IMF SAP drive SocProt changes (IV candidate)?",
            }),
            "output": json.dumps({
                "beta_imfsap_on_dsocprot": rev_imf.get("beta_imfsap_on_dsocprot"),
                "p_value": rev_imf.get("p_value"),
            }, default=str),
            "predict_dd_socprot": f"beta_imf={rev_imf.get('beta_imfsap_on_dsocprot')}, p={rev_imf.get('p_value')}",
            "metadata_test": "imf_sap_socprot",
        },
        {
            "input": json.dumps({
                "test": "lagged_socprot_sensitivity",
                "hypothesis": "Lagged SocProt (t-5) in DD triple — consistent sign?",
            }),
            "output": json.dumps({
                "beta7": lag_sens.get("beta7"),
                "p7": lag_sens.get("p7"),
                "n_obs": lag_sens.get("n_obs"),
                "estimation": lag_sens.get("estimation"),
            }, default=str),
            "predict_dd_socprot": f"lagged_beta7={lag_sens.get('beta7')}, p={lag_sens.get('p7')}",
            "metadata_test": "lagged_socprot",
        },
        {
            "input": json.dumps({
                "test": "overall_verdict",
                "analyses": ["DD_triple", "mediation", "sensitivity", "reverse_causality"],
            }),
            "output": verdict,
            "predict_dd_socprot": f"opp_cost_confirmed={opp_cost_channel}",
            "metadata_opp_cost_confirmed": str(opp_cost_channel),
            "metadata_bk_satisfied": str(bk_jud),
        },
    ]

    datasets = [
        {"dataset": "DD_SocProt_Panel_Predictions", "examples": panel_examples},
        {"dataset": "DD_PSE_Comparison_Predictions", "examples": pse_examples},
        {"dataset": "Causal_Mediation_BaronKenny", "examples": mediation_examples},
        {"dataset": "Missing_Data_Diagnostics", "examples": diag_examples},
        {"dataset": "Reverse_Causality_Robustness", "examples": robustness_examples},
    ]

    n_total_examples = sum(len(ds["examples"]) for ds in datasets)
    logger.info(f"Total examples across all datasets: {n_total_examples}")

    return {
        "metadata": {
            "experiment_id": "experiment_iter2_dir4",
            "description": "PSE Moderator Comparison + Causal Mediation with Quadrant Diagnostics",
            "n_panel_rows_total": len(complete),
            "n_complete_rows": len(complete),
            "n_total_examples": n_total_examples,
        },
        "datasets": datasets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main() -> None:
    logger.info("=== PSE Moderator Comparison + Causal Mediation ===")

    # ── Load data ──────────────────────────────────────────────────────────
    panel = load_panel(DS1)
    pse_annual, imf_annual = load_pse_imf(DS2)

    # ── Period aggregate ───────────────────────────────────────────────────
    pse_period, imf_period = aggregate_to_periods(pse_annual, imf_annual)
    del pse_annual, imf_annual
    gc.collect()

    # ── Merge ──────────────────────────────────────────────────────────────
    panel = merge_datasets(panel, pse_period, imf_period)
    del pse_period, imf_period
    gc.collect()

    # ── Subsamples ─────────────────────────────────────────────────────────
    complete = panel[panel["has_all_core"] == True].copy()
    logger.info(f"Complete panel (has_all_core=True): {len(complete)} rows, {complete['country_iso3'].nunique()} countries")

    # PSE moderator sample: education + gini + PSE all non-null (no SocProt required)
    pse_sample = panel.dropna(subset=["education", "gini_disp", "pse_share_period", "v2x_libdem"]).copy()
    # Recompute within-country deviations on the PSE sample (different composition → different means)
    pse_sample["mean_edu_pse"]  = pse_sample.groupby("country_iso3")["education"].transform("mean")
    pse_sample["mean_gini_pse"] = pse_sample.groupby("country_iso3")["gini_disp"].transform("mean")
    pse_sample["e_edu_pse"]     = pse_sample["education"] - pse_sample["mean_edu_pse"]
    pse_sample["e_gini_pse"]    = pse_sample["gini_disp"] - pse_sample["mean_gini_pse"]
    logger.info(f"PSE moderator sample: {len(pse_sample)} rows, {pse_sample['country_iso3'].nunique()} countries")

    # Mediation sample: education + gini + socprot + PSE + v2jucomp all non-null
    med_sample = panel.dropna(
        subset=["education", "gini_disp", "socprot_coverage", "pse_share_period",
                "v2jucomp", "v2x_libdem"]
    ).copy()
    logger.info(f"Mediation sample: {len(med_sample)} rows, {med_sample['country_iso3'].nunique()} countries")

    # ── Quadrant missing-data diagnostic ──────────────────────────────────
    logger.info("--- Quadrant missing-data diagnostic ---")
    pse_missing_table = compute_quadrant_coverage_table(complete)

    # ── DD triple interactions ─────────────────────────────────────────────
    logger.info("--- DD triple interaction: SocProt moderator (reference) ---")
    result_socprot = run_dd_triple(
        df=complete,
        e_educ_col="e_education",
        e_gini_col="e_gini_disp",
        e_mod_col="e_socprot_coverage",
        outcome_col="v2x_libdem",
        country_col="country_iso3",
        label="SocProt_coverage",
    )

    logger.info("--- DD triple interaction: PSE moderator ---")
    # Use level columns: cross-sectional fallback will be triggered (all PSE obs are period_start=2015 singletons)
    # and level variables give meaningful β₇ as a cross-country comparison
    result_pse = run_dd_triple(
        df=pse_sample,
        e_educ_col="education",
        e_gini_col="gini_disp",
        e_mod_col="pse_share_period",
        outcome_col="v2x_libdem",
        country_col="country_iso3",
        label="PSE_share",
    )

    # ── Causal mediation ──────────────────────────────────────────────────
    logger.info("--- Baron-Kenny mediation: judicial compliance ---")
    if len(med_sample) >= 15:
        med_judicial = run_baron_kenny_mediation(
            med_sample, outcome_col="v2jucomp", mediator_col="pse_share_period"
        )
        med_ngo = run_baron_kenny_mediation(
            med_sample, outcome_col="v2cseeorgs", mediator_col="pse_share_period"
        )
        med_libdem = run_baron_kenny_mediation(
            med_sample, outcome_col="v2x_libdem", mediator_col="pse_share_period"
        )
    else:
        logger.warning(f"Mediation sample too small ({len(med_sample)}), using simplified mediation on PSE sample")
        # Fallback: simpler education→PSE→outcome without triple T
        med_judicial = {"error": "mediation_sample_too_small", "n_obs": len(med_sample), "insufficient_data": True}
        med_ngo      = {"error": "mediation_sample_too_small", "n_obs": len(med_sample), "insufficient_data": True}
        med_libdem   = {"error": "mediation_sample_too_small", "n_obs": len(med_sample), "insufficient_data": True}

    # ── Extreme-values sensitivity ─────────────────────────────────────────
    logger.info("--- Extreme-values ACME sensitivity ---")
    acme_obs = med_judicial.get("acme") if isinstance(med_judicial, dict) else None
    extreme_sens = compute_extreme_sensitivity(complete, med_sample, acme_obs)

    # ── Sequential ignorability ────────────────────────────────────────────
    logger.info("--- Sequential ignorability check ---")
    if len(med_sample) >= 10:
        seq_ign = sequential_ignorability_check(med_sample, outcome_col="v2jucomp", mediator_col="pse_share_period")
    else:
        seq_ign = {"error": "Insufficient obs", "n_obs": len(med_sample)}

    # ── Reverse causality tests ────────────────────────────────────────────
    logger.info("--- Reverse causality tests ---")
    rev_gini = test_reverse_causality_gini_ldem(panel)
    rev_imf  = test_imf_sap_socprot_correlation(panel)
    lag_sens = run_lagged_socprot_sensitivity(complete)

    # ── Phase 4 confirmation signals ──────────────────────────────────────
    logger.info("--- Phase 4 confirmation signals ---")
    logger.info(f"result_socprot n_obs ≥ 40: {result_socprot.get('n_obs', 0)} {'>=' if result_socprot.get('n_obs', 0) >= 40 else '<'} 40")
    logger.info(f"result_pse n_obs ≥ 60: {result_pse.get('n_obs', 0)} {'>=' if result_pse.get('n_obs', 0) >= 60 else '<'} 60")
    logger.info(f"med_judicial n_obs ≥ 10: {med_judicial.get('n_obs', 0)}")
    logger.info(f"extreme_sens acme_observed: {extreme_sens.get('acme_observed')}")
    logger.info(f"rev_gini n_obs ≥ 30: {rev_gini.get('n_obs', 0)}")
    logger.info(f"rev_imf n_obs ≥ 10: {rev_imf.get('n_obs', 0)}")

    # Phase 5: interpretation sanity checks
    logger.info("--- Phase 5 interpretation sanity ---")
    sp_sr = result_socprot.get("sign_reversal", {})
    pse_sr = result_pse.get("sign_reversal", {})
    logger.info(f"SocProt sign reversal: me_low={sp_sr.get('me_at_low_moderator')}, me_high={sp_sr.get('me_at_high_moderator')}, differs={sp_sr.get('signs_differ')}")
    logger.info(f"PSE sign reversal: me_low={pse_sr.get('me_at_low_moderator')}, me_high={pse_sr.get('me_at_high_moderator')}, differs={pse_sr.get('signs_differ')}")
    logger.info(f"baron_kenny_satisfied (any): {any([med_judicial.get('baron_kenny_satisfied'), med_ngo.get('baron_kenny_satisfied'), med_libdem.get('baron_kenny_satisfied')])}")
    logger.info(f"rho_residual_correlation: {seq_ign.get('rho_residual_correlation')}")
    logger.info(f"path_a reverse causality concern: {rev_gini.get('reverse_causality_concern')}")

    # ── Assemble and write output ──────────────────────────────────────────
    logger.info("--- Assembling output ---")
    output = build_output(
        pse_missing_table=pse_missing_table,
        complete=complete,
        pse_sample=pse_sample,
        med_sample=med_sample,
        result_socprot=result_socprot,
        result_pse=result_pse,
        med_judicial=med_judicial,
        med_ngo=med_ngo,
        med_libdem=med_libdem,
        extreme_sens=extreme_sens,
        seq_ign=seq_ign,
        rev_gini=rev_gini,
        rev_imf=rev_imf,
        lag_sens=lag_sens,
    )

    OUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Total examples: {output['metadata']['n_total_examples']}")
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
