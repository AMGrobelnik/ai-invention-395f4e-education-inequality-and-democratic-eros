#!/usr/bin/env python3
"""DD Triple-Interaction Estimation: State-Dependent Education Trap (OECD-Excluded Panel).

Implements Giesselmann-Schmidt-Catran (2022) double-demeaning DD estimator on a 161-obs
post-1990 democratizer panel, with OECD exclusion, naïve-vs-DD bias documentation,
delta-method marginal effect plots, sub-index mechanism tests, lagged education spec,
and Oster (2019) coefficient-stability bounds.
"""

from __future__ import annotations

import gc
import json
import math
import os
import resource
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

# ── Setup ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATASET_WS = Path(
    "/home/adrian/projects/ai-inventor/aii_data/users/adrian.marina.photos"
    "/runs/run_zXdSkAIIk5J3/3_invention_loop/iter_1/gen_art/gen_art_dataset_1"
)
FULL_DATA = DATASET_WS / "full_data_out.json"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(
    str(WORKSPACE / "logs" / "run.log"),
    rotation="30 MB", level="DEBUG",
)

try:
    _ram_budget = 6 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (_ram_budget * 3, _ram_budget * 3))
except Exception:
    pass


def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


NUM_CPUS = _detect_cpus()
logger.info(f"Hardware: {NUM_CPUS} CPUs, no GPU — CPU-bound stats experiment")


# ── OLS FE helper ─────────────────────────────────────────────────────────────
def _fit_ols_fe(
    df_clean: pd.DataFrame,
    dep_col: str,
    exog_cols: list[str],
    *,
    include_entity_fe: bool = True,
    include_time_fe: bool = True,
) -> object | None:
    """Statsmodels OLS with explicit country/period dummies + cluster-robust SE.

    Using linearmodels PanelOLS raises 'absorbed variable' errors in 2-period panels
    because pre-demeaned variables (zero entity mean by construction) are flagged even
    with check_rank=False (absorption check fires before the rank check).  Explicit
    dummies in plain OLS are orthogonal to the already-zero-mean regressors, so no
    collinearity is introduced and estimation proceeds normally.
    """
    if len(df_clean) < len(exog_cols) + 3:
        logger.warning(f"_fit_ols_fe: only {len(df_clean)} obs for {len(exog_cols)} regressors")
        return None

    df_r = df_clean.reset_index(drop=True)
    y = df_r[dep_col].astype(float)
    X_core = df_r[exog_cols].astype(float)

    parts: list[pd.DataFrame] = [X_core]
    if include_entity_fe:
        c_dum = pd.get_dummies(df_r["country_iso3"], drop_first=True, dtype=float, prefix="C")
        parts.append(c_dum)
    if include_time_fe:
        t_dum = pd.get_dummies(df_r["period_start"], drop_first=True, dtype=float, prefix="T")
        parts.append(t_dum)

    X = pd.concat(parts, axis=1)
    X.insert(0, "const", 1.0)
    groups = df_r["country_iso3"]

    try:
        mod = sm.OLS(y, X)
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
        return res
    except Exception as exc:
        logger.error(f"OLS FE failed: {exc}")
        return None


def _extract(res, name: str) -> tuple[float, float, float]:
    """Return (coef, se, pval) from a statsmodels result; nan on missing."""
    if res is None:
        return float("nan"), float("nan"), float("nan")
    return (
        float(res.params.get(name, float("nan"))),
        float(res.bse.get(name, float("nan"))),
        float(res.pvalues.get(name, float("nan"))),
    )


def _r2(res) -> float:
    if res is None:
        return float("nan")
    return float(res.rsquared)


# ── Step 0: Load Data ─────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def load_data() -> tuple[pd.DataFrame, list[dict]]:
    logger.info(f"Loading data from {FULL_DATA}")
    raw = json.loads(FULL_DATA.read_text())
    examples = raw["datasets"][0]["examples"]
    rows = []
    for ex in examples:
        d = json.loads(ex["input"])
        d["v2x_libdem"] = float(ex["output"])
        d["country_name"] = ex["metadata_country_name"]
        rows.append(d)
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} rows, {df['country_iso3'].nunique()} countries")
    logger.info(f"Periods: {sorted(df['period_start'].unique())}")
    assert len(df) == 161, f"Expected 161 rows, got {len(df)}"
    return df, examples


# ── Step 1: OECD / High-Income Exclusion ─────────────────────────────────────
OECD_THRESHOLD = 15_000

_HARDCODED_HIGHINCOME = {
    "CZE", "EST", "HUN", "POL", "SVK", "SVN", "KOR", "CHL",
    "URY", "LTU", "LVA", "HRV", "GRC", "PRT", "ESP",
}


def fetch_wb_gdp(countries: list[str]) -> pd.DataFrame:
    import wbgapi
    logger.info("Fetching WB GDP PPP data (1990-2010) for OECD exclusion…")
    try:
        gdp_raw = wbgapi.data.DataFrame(
            "NY.GDP.PCAP.PP.KD",
            economy=countries,
            time=range(1990, 2011),
            skipBlanks=True,
        )
        # wbgapi returns wide format: index=economy, columns='YR1990','YR1991',…
        # reset_index() puts economy as a column; melt converts to long.
        gdp_r = gdp_raw.reset_index()
        id_col = "economy" if "economy" in gdp_r.columns else gdp_r.columns[0]
        yr_cols = [c for c in gdp_r.columns if c != id_col]
        gdp_long = gdp_r.melt(
            id_vars=id_col, value_vars=yr_cols,
            var_name="time", value_name="value",
        ).rename(columns={id_col: "economy"})

        # Strip 'YR' prefix so 'YR1995' → 1995 (numeric)
        gdp_long["time"] = (
            gdp_long["time"]
            .astype(str)
            .str.replace(r"^YR", "", regex=True)
        )
        gdp_long["time"] = pd.to_numeric(gdp_long["time"], errors="coerce")
        gdp_long["value"] = pd.to_numeric(gdp_long["value"], errors="coerce")
        gdp_long = gdp_long.dropna(subset=["value", "time"])
        logger.info(
            f"WB GDP: {len(gdp_long)} rows, {gdp_long['economy'].nunique()} countries"
        )
        return gdp_long
    except Exception as exc:
        logger.error(f"wbgapi failed: {exc}. Using hardcoded list.")
        return pd.DataFrame(columns=["economy", "time", "value"])


def _gdp_ref(iso3: str, gdp_df: pd.DataFrame) -> float:
    """Average GDP PPP in 1995-2005 — proxy for democratization-era wealth."""
    sub = gdp_df[
        (gdp_df["economy"] == iso3) &
        (gdp_df["time"] >= 1995) &
        (gdp_df["time"] <= 2005)
    ]
    if sub.empty:
        sub = gdp_df[gdp_df["economy"] == iso3]
    return float(sub["value"].mean()) if not sub.empty else float("nan")


def exclude_high_income(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    countries = df["country_iso3"].unique().tolist()
    gdp_df = fetch_wb_gdp(countries)

    df = df.copy()
    if gdp_df.empty:
        excluded = sorted(_HARDCODED_HIGHINCOME & set(countries))
        logger.warning(f"Hardcoded exclusion ({len(excluded)}): {excluded}")
        df["gdp_reference"] = float("nan")
    else:
        gdp_map = {iso3: _gdp_ref(iso3, gdp_df) for iso3 in countries}
        df["gdp_reference"] = df["country_iso3"].map(gdp_map)
        hi = {k: v for k, v in gdp_map.items() if not math.isnan(v) and v > 10_000}
        logger.info(
            "Countries GDP > $10K (1995-2005 avg): "
            + str(sorted(hi.items(), key=lambda x: -x[1])[:15])
        )
        high_mask = df["gdp_reference"] >= OECD_THRESHOLD
        excluded = sorted(df.loc[high_mask, "country_iso3"].unique().tolist())
        logger.info(f"Excluded {len(excluded)} high-income countries: {excluded}")

    df_restricted = df[~df["country_iso3"].isin(excluded)].copy()
    df_oecd = df[df["country_iso3"].isin(excluded)].copy()
    logger.info(
        f"Restricted sample: {len(df_restricted)} obs, "
        f"{df_restricted['country_iso3'].nunique()} countries"
    )
    return df_restricted, df_oecd, excluded


# ── Step 2: Acemoglu Null ─────────────────────────────────────────────────────
def run_acemoglu_null(df_full: pd.DataFrame) -> dict:
    """Replicates Acemoglu et al. null: lagged education ≠ predictor of LDem.

    After a 5-yr lag shift, each country has at most T=1 obs (the 2020 period,
    with education_2015 as the lag).  Entity FE can't be estimated with T=1 per
    entity; we use pooled OLS with a period dummy and cluster-robust SE instead.
    """
    df = df_full.sort_values(["country_iso3", "period_start"]).copy()
    df["education_lag1"] = df.groupby("country_iso3")["education"].shift(1)
    df_lag = df.dropna(subset=["education_lag1"]).copy()
    logger.info(
        f"Acemoglu null: {len(df_lag)} obs after lag "
        f"({df_lag['country_iso3'].nunique()} countries)"
    )

    if len(df_lag) < 5:
        return {
            "error": "insufficient obs after lagging",
            "n_obs": len(df_lag),
            "coef_education_lag1": float("nan"),
            "pval": float("nan"),
            "replicated": None,
        }

    # Period dummy + pooled OLS (no entity FE — T=1 after lag makes it inestimable)
    res = _fit_ols_fe(
        df_lag, "v2x_libdem", ["education_lag1"],
        include_entity_fe=False, include_time_fe=True,
    )
    coef, se, pval = _extract(res, "education_lag1")
    logger.info(f"Acemoglu null — education_lag1 coef={coef:.4f}, p={pval:.4f}")
    return {
        "coef_education_lag1": coef,
        "se": se,
        "pval": pval,
        "n_obs": len(df_lag),
        "n_countries": int(df_lag["country_iso3"].nunique()),
        "replicated": bool(pval > 0.10) if not math.isnan(pval) else None,
        "note": (
            "Pooled OLS with period FE (not entity FE): each country has T=1 after "
            "5-yr lag, entity FE inestimable.  OLS + period dummy + cluster SE."
        ),
    }


# ── Step 3: Recompute Within-Country Deviations ───────────────────────────────
def recompute_deviations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for var in ["education", "gini_disp", "socprot_coverage"]:
        m = df.groupby("country_iso3")[var].transform("mean")
        df[f"mean_{var}_new"] = m
        df[f"e_{var}_new"] = df[var] - m

    for var in ["education", "gini_disp", "socprot_coverage"]:
        max_diff = (df[f"e_{var}"] - df[f"e_{var}_new"]).abs().max()
        logger.info(f"Max deviation diff {var} (pre- vs post-exclusion): {max_diff:.6f}")
        logger.info(f"Within-country SD of {var}: {df[f'e_{var}_new'].std():.4f}")

    return df


# ── Step 4: Naïve FE-Product Estimator (baseline) ────────────────────────────
def run_naive_estimator(
    df: pd.DataFrame,
) -> tuple[dict, object | None, pd.DataFrame]:
    """Naïve: original-level triple interaction in OLS with country+period FE.

    Uses original Education/Gini/SocProt values (not within-deviations) so that
    entity FE absorbs country means normally — no pre-demeaning, so no absorption
    issue.  This is the conventional FE-product baseline.
    """
    df = df.copy()
    df["EG_naive"] = df["education"] * df["gini_disp"]
    df["ES_naive"] = df["education"] * df["socprot_coverage"]
    df["GS_naive"] = df["gini_disp"] * df["socprot_coverage"]
    df["EGS_naive"] = df["education"] * df["gini_disp"] * df["socprot_coverage"]

    exog_cols = [
        "education", "gini_disp", "socprot_coverage",
        "EG_naive", "ES_naive", "GS_naive", "EGS_naive",
    ]
    df_clean = df.dropna(subset=["v2x_libdem"] + exog_cols).copy()
    res = _fit_ols_fe(df_clean, "v2x_libdem", exog_cols)

    beta7, se7, pval7 = _extract(res, "EGS_naive")
    logger.info(f"Naïve: β7={beta7:.4f} ({se7:.4f}), p={pval7:.4f}")

    result: dict = {
        "beta7": beta7, "se7": se7, "pval7": pval7,
        "rsquared": _r2(res),
        "n_obs": int(res.nobs) if res is not None else 0,
    }
    if res is not None:
        result["full_table"] = {
            k: {
                "coef": float(v),
                "se": float(res.bse[k]),
                "pval": float(res.pvalues[k]),
            }
            for k, v in res.params.items()
            if k in exog_cols
        }
    return result, res, df_clean


# ── Step 5: DD Estimator ──────────────────────────────────────────────────────
_DD_EXOG = [
    "e_education_new", "e_gini_disp_new", "e_socprot_coverage_new",
    "dd_EG_dd", "dd_ES_dd", "dd_GS_dd", "dd_EGS_dd",
]


def build_dd_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    e_E = df["e_education_new"]
    e_G = df["e_gini_disp_new"]
    e_S = df["e_socprot_coverage_new"]

    df["dd_EG"] = e_E * e_G
    df["dd_ES"] = e_E * e_S
    df["dd_GS"] = e_G * e_S
    df["dd_EGS"] = e_E * e_G * e_S

    for prod_col in ["dd_EG", "dd_ES", "dd_GS", "dd_EGS"]:
        prod_mean = df.groupby("country_iso3")[prod_col].transform("mean")
        df[f"{prod_col}_dd"] = df[prod_col] - prod_mean

    for col in ["dd_EG_dd", "dd_ES_dd", "dd_GS_dd", "dd_EGS_dd"]:
        max_cm = df.groupby("country_iso3")[col].mean().abs().max()
        if max_cm > 1e-6:
            logger.warning(f"{col} demeaning residual: {max_cm:.2e}")
        else:
            logger.info(f"{col} demeaning OK (max country mean: {max_cm:.2e})")

    return df


def run_dd_estimator(
    df: pd.DataFrame,
) -> tuple[dict, object | None, pd.DataFrame]:
    """Giesselmann-Schmidt-Catran (2022) DD estimator.

    Pre-demeaned variables (e_education_new etc., dd_EGS_dd etc.) all have zero
    entity means by construction.  Using linearmodels PanelOLS with entity_effects
    fires the 'absorbed' check even with check_rank=False.  We use plain OLS with
    explicit country/period dummies instead — equivalent to TWFE, avoids the
    absorption detection entirely.
    """
    df = build_dd_columns(df)
    df_clean = df.dropna(subset=["v2x_libdem"] + _DD_EXOG).copy()
    res = _fit_ols_fe(df_clean, "v2x_libdem", _DD_EXOG)

    beta7, se7, pval7 = _extract(res, "dd_EGS_dd")
    logger.info(f"DD: β7={beta7:.4f} ({se7:.4f}), p={pval7:.4f}")

    # F4: identification check — SE >> 100× |β7| = not identified
    identified = True
    if not math.isnan(beta7) and not math.isnan(se7) and se7 > 0:
        if abs(beta7) > 1e-10 and se7 / abs(beta7) > 100:
            identified = False
            logger.warning(
                f"F4: SE ({se7:.4f}) >> 100×|β7| ({abs(beta7):.6f}) — "
                "tiny within-SD of education (0.025) limits DD identification"
            )
        elif abs(beta7) <= 1e-10:
            identified = False
            logger.warning("F4: β7 ≈ 0 — DD triple interaction not identified")
    elif math.isnan(beta7):
        identified = False

    result: dict = {
        "beta7": beta7,
        "se7": se7,
        "pval7": pval7,
        "ci95_lower": float(beta7 - 1.96 * se7) if not math.isnan(se7) else float("nan"),
        "ci95_upper": float(beta7 + 1.96 * se7) if not math.isnan(se7) else float("nan"),
        "identified": identified,
        "rsquared": _r2(res),
        "n_obs": int(res.nobs) if res is not None else 0,
    }
    if not identified:
        result["identification_note"] = (
            "F4 FALLBACK: within-SD of education (0.025 yrs) too small for DD "
            "triple-interaction to be identified.  Acemoglu null replication is the "
            "primary methodological result."
        )
    if res is not None:
        result["full_table"] = {
            k: {
                "coef": float(v),
                "se": float(res.bse[k]),
                "pval": float(res.pvalues[k]),
            }
            for k, v in res.params.items()
            if k in _DD_EXOG
        }
    return result, res, df_clean


# ── Step 6: Marginal Effects with Delta-Method CI ────────────────────────────
def compute_marginal_effects(
    df: pd.DataFrame,
    res_dd,
    df_full_restricted: pd.DataFrame,
) -> list[dict]:
    if res_dd is None:
        logger.warning("DD result unavailable; skipping marginal effects")
        return []

    b1 = float(res_dd.params.get("e_education_new", 0.0))
    b4 = float(res_dd.params.get("dd_EG_dd", 0.0))
    b5 = float(res_dd.params.get("dd_ES_dd", 0.0))
    b7 = float(res_dd.params.get("dd_EGS_dd", 0.0))

    param_names = list(res_dd.params.index)
    idx_map = {n: i for i, n in enumerate(param_names)}
    cov_vals = res_dd.cov_params().values

    mean_socprot = float(df_full_restricted["mean_socprot_coverage_new"].mean()
                         if "mean_socprot_coverage_new" in df_full_restricted.columns
                         else df_full_restricted["socprot_coverage"].mean())
    e_S_vals = df_full_restricted["e_socprot_coverage_new"]
    socprot_grid = np.linspace(
        float(e_S_vals.quantile(0.05)),
        float(e_S_vals.quantile(0.95)),
        100,
    )
    gini_p25 = float(df_full_restricted["e_gini_disp_new"].quantile(0.25))
    gini_p75 = float(df_full_restricted["e_gini_disp_new"].quantile(0.75))

    results_me = []
    for gini_level, gini_label in [(gini_p25, "p25"), (gini_p75, "p75")]:
        mes, lower95, upper95 = [], [], []
        for s in socprot_grid:
            me = b1 + b4 * gini_level + b5 * s + b7 * gini_level * s
            g_vec = np.zeros(len(param_names))
            for name, weight in [
                ("e_education_new", 1.0),
                ("dd_EG_dd", gini_level),
                ("dd_ES_dd", s),
                ("dd_EGS_dd", gini_level * s),
            ]:
                if name in idx_map:
                    g_vec[idx_map[name]] = weight
            var_me = float(g_vec @ cov_vals @ g_vec)
            se_me = math.sqrt(max(var_me, 0.0))
            mes.append(float(me))
            lower95.append(float(me - 1.96 * se_me))
            upper95.append(float(me + 1.96 * se_me))

        zero_cross = None
        for i in range(len(mes) - 1):
            if mes[i] * mes[i + 1] <= 0:
                x1, x2, y1, y2 = socprot_grid[i], socprot_grid[i + 1], mes[i], mes[i + 1]
                if abs(y2 - y1) > 1e-12:
                    zero_cross = float(x1 - y1 * (x2 - x1) / (y2 - y1))
                break

        zero_orig = (
            float(zero_cross + mean_socprot)
            if zero_cross is not None
            else None
        )
        results_me.append({
            "gini_label": gini_label,
            "gini_value": float(gini_level),
            "socprot_grid_within": socprot_grid.tolist(),
            "socprot_grid_original_scale": (socprot_grid + mean_socprot).tolist(),
            "marginal_effect": mes,
            "ci_lower": lower95,
            "ci_upper": upper95,
            "zero_crossing_socprot_within": zero_cross,
            "zero_crossing_socprot_original": zero_orig,
        })
        logger.info(
            f"ME Gini {gini_label}: range [{min(mes):.4f}, {max(mes):.4f}], "
            f"zero-crossing={'%.1f%%' % zero_orig if zero_orig is not None else 'none'}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, result in zip(axes, results_me):
        sp_orig = result["socprot_grid_original_scale"]
        ax.plot(sp_orig, result["marginal_effect"], "b-", lw=2, label="ME")
        ax.fill_between(sp_orig, result["ci_lower"], result["ci_upper"], alpha=0.2, color="b")
        ax.axhline(0, color="k", linestyle="--", lw=0.8)
        ax.set_title(f"ME(Education→LDem) Gini {result['gini_label']}")
        ax.set_xlabel("Social Protection Coverage (%)")
        ax.set_ylabel("Marginal Effect")
        if result["zero_crossing_socprot_original"] is not None:
            ax.axvline(
                result["zero_crossing_socprot_original"],
                color="r", linestyle=":", lw=1.5,
                label=f"Zero @ {result['zero_crossing_socprot_original']:.1f}%",
            )
            ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = WORKSPACE / "marginal_effects.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    logger.info(f"Saved marginal effects figure to {fig_path}")
    return results_me


# ── Step 7: Lagged Education Specification ────────────────────────────────────
def run_lagged_spec(df: pd.DataFrame) -> dict:
    note_10yr = (
        "ILO SDG 1.3.1 only available from 2015; "
        "t-2 period (2005-09) has no socprot_coverage — 10-yr lag is infeasible."
    )
    df = df.sort_values(["country_iso3", "period_start"]).copy()
    df["education_lag1"] = df.groupby("country_iso3")["education"].shift(1)
    df_lag = df.dropna(subset=["education_lag1"]).copy()

    if len(df_lag) < 10:
        return {
            "error": "insufficient obs",
            "n_obs": len(df_lag),
            "beta7": float("nan"),
            "lagged_10yr_feasible": False,
            "lagged_10yr_reason": note_10yr,
        }

    df_lag["e_educ_lag1_new"] = (
        df_lag["education_lag1"]
        - df_lag.groupby("country_iso3")["education_lag1"].transform("mean")
    )

    within_sd_lag = float(df_lag["e_educ_lag1_new"].std())
    logger.info(f"Lagged spec within-SD of education_lag1: {within_sd_lag:.6f}")

    # When each country appears only once after lagging (T=1), country mean ==
    # the observation itself, so e_educ_lag1_new = 0 for all rows.
    # All DD products are then identically zero → perfect collinearity → no ID.
    if within_sd_lag < 1e-8:
        logger.warning(
            "Lagged spec: within-SD of education_lag1 ≈ 0 (T=1 per country after lag). "
            "DD triple-interaction not identified. Returning data limitation report."
        )
        return {
            "beta7": float("nan"), "se7": float("nan"), "pval7": float("nan"),
            "n_obs": len(df_lag),
            "lagged_10yr_feasible": False,
            "lagged_10yr_reason": note_10yr,
            "lagged_5yr_feasible": False,
            "lagged_5yr_reason": (
                "With T=2 periods and ILO data from 2015, lagging by 1 period reduces "
                "each country to T=1 obs. Within-country deviation of education_lag1 "
                "collapses to 0, making the DD triple-interaction unidentified."
            ),
        }

    for prod_col, v1, v2 in [
        ("dd_ElagG", "e_educ_lag1_new", "e_gini_disp_new"),
        ("dd_ElagS", "e_educ_lag1_new", "e_socprot_coverage_new"),
    ]:
        df_lag[prod_col] = df_lag[v1] * df_lag[v2]
        m = df_lag.groupby("country_iso3")[prod_col].transform("mean")
        df_lag[f"{prod_col}_dd"] = df_lag[prod_col] - m

    df_lag["dd_ElagGS"] = (
        df_lag["e_educ_lag1_new"]
        * df_lag["e_gini_disp_new"]
        * df_lag["e_socprot_coverage_new"]
    )
    m = df_lag.groupby("country_iso3")["dd_ElagGS"].transform("mean")
    df_lag["dd_ElagGS_dd"] = df_lag["dd_ElagGS"] - m

    lag_exog = [
        "e_educ_lag1_new", "e_gini_disp_new", "e_socprot_coverage_new",
        "dd_ElagG_dd", "dd_ElagS_dd", "dd_ElagGS_dd",
    ]
    df_clean = df_lag.dropna(subset=["v2x_libdem"] + lag_exog).copy()
    # After lag, each country has T=1 obs → entity FE is inestimable (singular cov).
    # Use period FE only, same rationale as run_acemoglu_null.
    res = _fit_ols_fe(
        df_clean, "v2x_libdem", lag_exog,
        include_entity_fe=False, include_time_fe=True,
    )
    beta7, se7, pval7 = _extract(res, "dd_ElagGS_dd")
    logger.info(f"Lagged spec β7_DD={beta7:.4f} ({se7:.4f}), p={pval7:.4f}")
    return {
        "beta7": beta7, "se7": se7, "pval7": pval7,
        "n_obs": int(res.nobs) if res is not None else 0,
        "lagged_10yr_feasible": False,
        "lagged_10yr_reason": note_10yr,
    }


# ── Step 8: Sub-Index Mechanism Tests ────────────────────────────────────────
_SUB_DVS = [
    ("v2x_libdem", "Liberal Democracy Index (primary)"),
    ("v2x_jucon", "Judicial Constraints Aggregate"),
    ("v2jucomp", "Judicial Government Compliance"),
    ("v2cseeorgs", "CSO Entry/Exit Autonomy"),
    ("v2csprtcpt", "CSO Population Participation"),
    ("v2x_polyarchy", "Electoral Democracy Index"),
]


def run_sub_index_tests(df: pd.DataFrame) -> dict:
    sub_results: dict = {}
    for dv, label in _SUB_DVS:
        if dv not in df.columns:
            sub_results[dv] = {"label": label, "error": "not in dataset",
                                "beta7_DD": float("nan")}
            continue
        df_clean = df.dropna(subset=[dv] + _DD_EXOG).copy()
        if len(df_clean) < 10:
            sub_results[dv] = {"label": label, "error": f"only {len(df_clean)} obs",
                                "beta7_DD": float("nan")}
            continue
        res = _fit_ols_fe(df_clean, dv, _DD_EXOG)
        beta7, se7, pval7 = _extract(res, "dd_EGS_dd")
        sub_results[dv] = {
            "label": label,
            "beta7_DD": beta7, "se7_DD": se7, "pval7_DD": pval7,
            "ci95_lower": float(beta7 - 1.96 * se7),
            "ci95_upper": float(beta7 + 1.96 * se7),
            "n_obs": int(res.nobs) if res is not None else 0,
        }
        logger.info(f"Sub-index {dv}: β7={beta7:.4f}, p={pval7:.4f}")
    return sub_results


# ── Step 9: Oster (2019) Bounds ───────────────────────────────────────────────
def run_oster_bounds(
    df: pd.DataFrame,
    res_dd,
    beta7_naive: float,
    r2_naive: float,
) -> dict:
    if res_dd is None:
        return {"error": "DD estimator not available"}

    df_unc = df.dropna(subset=["v2x_libdem", "e_education_new"]).copy()
    res_unc = _fit_ols_fe(df_unc, "v2x_libdem", ["e_education_new"])
    if res_unc is None:
        return {"error": "uncontrolled model failed"}

    beta_unc = float(res_unc.params.get("e_education_new", float("nan")))
    r2_unc = _r2(res_unc)
    beta_con = float(res_dd.params.get("e_education_new", float("nan")))
    r2_con = _r2(res_dd)
    beta7_dd = float(res_dd.params.get("dd_EGS_dd", float("nan")))
    rmax = min(1.3 * r2_con, 1.0)

    if abs(r2_con - r2_unc) > 1e-6 and not math.isnan(beta_unc):
        oster_beta1_star = (
            beta_con * (rmax - r2_unc) - beta_unc * (rmax - r2_con)
        ) / (r2_con - r2_unc)
    else:
        oster_beta1_star = float("nan")

    if not math.isnan(r2_naive) and abs(r2_con - r2_naive) > 1e-6 and not math.isnan(beta7_naive):
        rmax7 = min(1.3 * r2_con, 1.0)
        oster_beta7_star = (
            beta7_dd * (rmax7 - r2_naive) - beta7_naive * (rmax7 - r2_con)
        ) / (r2_con - r2_naive)
    else:
        oster_beta7_star = float("nan")

    interp = (
        "β7* > 0 with δ=1 → robust to proportional selection on unobservables"
        if not math.isnan(oster_beta7_star) and oster_beta7_star > 0
        else (
            "β7* sign reversal or unavailable — confounding cannot be ruled out"
        )
    )
    logger.info(
        f"Oster: R2_unc={r2_unc:.4f}, R2_con={r2_con:.4f}, Rmax={rmax:.4f}, "
        f"β1*={oster_beta1_star:.4f}, β7*={oster_beta7_star:.4f}"
    )
    return {
        "rmax": float(rmax), "delta": 1.0,
        "R2_uncontrolled": float(r2_unc), "R2_controlled": float(r2_con),
        "beta1_uncontrolled": float(beta_unc), "beta1_controlled": float(beta_con),
        "beta1_oster_adjusted": float(oster_beta1_star),
        "beta7_naive": float(beta7_naive), "beta7_DD": float(beta7_dd),
        "beta7_oster_adjusted": float(oster_beta7_star),
        "interpretation": interp,
    }


# ── Step 10: OECD Falsification ───────────────────────────────────────────────
def run_oecd_falsification(df_oecd: pd.DataFrame) -> dict:
    interp_note = "Theory predicts near-zero or positive β7 in OECD sample"
    if len(df_oecd) < 10:
        return {
            "beta7_DD": None, "n_obs": len(df_oecd),
            "interpretation": interp_note, "skipped": True,
            "reason": f"only {len(df_oecd)} obs < 10",
        }

    df = df_oecd.copy()
    for var in ["education", "gini_disp", "socprot_coverage"]:
        df[f"e_{var}_new"] = df[var] - df.groupby("country_iso3")[var].transform("mean")
    df = build_dd_columns(df)

    df_clean = df.dropna(subset=["v2x_libdem"] + _DD_EXOG).copy()
    if df_clean["country_iso3"].nunique() < 3:
        return {
            "beta7_DD": None, "n_obs": len(df_clean),
            "interpretation": interp_note, "skipped": True,
            "reason": f"only {df_clean['country_iso3'].nunique()} countries",
        }

    res = _fit_ols_fe(df_clean, "v2x_libdem", _DD_EXOG)
    beta7, se7, pval7 = _extract(res, "dd_EGS_dd")
    logger.info(f"OECD falsification β7={beta7:.4f} ({se7:.4f}), p={pval7:.4f}")
    return {
        "beta7_DD": float(beta7), "se7_DD": float(se7), "pval7_DD": float(pval7),
        "n_obs": int(res.nobs) if res is not None else 0,
        "n_countries": int(df_clean["country_iso3"].nunique()),
        "interpretation": interp_note,
    }


# ── Step 11: Diagnostics ──────────────────────────────────────────────────────
def compute_diagnostics(df: pd.DataFrame) -> tuple[list, str]:
    period_counts = (
        df.groupby("period_start")
        .agg(
            n_countries=("country_iso3", "nunique"),
            n_obs=("v2x_libdem", "count"),
            pct_directly_reported=("directly_reported_any", "mean"),
            pct_education_imputed=("education_imputed", "mean"),
        )
        .reset_index()
    )
    for _, row in period_counts.iterrows():
        logger.info(
            f"  {int(row['period_start'])}: {int(row['n_countries'])} countries, "
            f"{row['pct_directly_reported']*100:.0f}% directly-reported ILO"
        )
    e_sd = df["e_education_new"].std()
    g_sd = df["e_gini_disp_new"].std()
    s_sd = df["e_socprot_coverage_new"].std()
    mde = (
        f"N_obs={len(df)}, N_countries={df['country_iso3'].nunique()}, "
        f"within-SD education={e_sd:.4f}, gini={g_sd:.2f}, socprot={s_sd:.2f}. "
        "Very small education SD implies MDE >> 1 SD — interpret results cautiously."
    )
    logger.warning(mde)
    return period_counts.to_dict(orient="records"), mde


# ── Fitted values for schema ──────────────────────────────────────────────────
def extract_fitted(
    res, df_clean: pd.DataFrame,
) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    if res is None:
        return out
    try:
        fv = res.fittedvalues
        df_r = df_clean.reset_index(drop=True)
        for i, val in enumerate(fv):
            key = (str(df_r.at[i, "country_iso3"]), int(df_r.at[i, "period_start"]))
            out[key] = str(round(float(val), 6))
    except Exception as exc:
        logger.warning(f"Could not extract fitted values: {exc}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main() -> None:
    # Step 0 — Load
    df_full, examples = load_data()

    # Step 1 — OECD exclusion
    df_restricted, df_oecd, excluded = exclude_high_income(df_full)

    # Step 2 — Acemoglu null (full pre-exclusion sample for max power)
    acemoglu_result = run_acemoglu_null(df_full)

    # Step 3 — Recompute deviations on restricted sample
    df_restricted = recompute_deviations(df_restricted)

    # Step 11 — Diagnostics
    period_counts, mde_note = compute_diagnostics(df_restricted)

    # Step 4 — Naïve estimator
    naive_result, res_naive, df_naive_clean = run_naive_estimator(df_restricted)
    beta7_naive = naive_result.get("beta7", float("nan"))
    r2_naive = naive_result.get("rsquared", float("nan"))

    # Step 5 — DD estimator
    dd_result, res_dd, df_dd_clean = run_dd_estimator(df_restricted)
    beta7_DD = dd_result.get("beta7", float("nan"))
    se7_DD = dd_result.get("se7", float("nan"))

    # Bias documentation
    if not math.isnan(beta7_naive) and not math.isnan(beta7_DD) and se7_DD > 0:
        bias = beta7_naive - beta7_DD
        bias_in_se = bias / se7_DD
    else:
        bias, bias_in_se = float("nan"), float("nan")
    logger.info(
        f"Bias: β7_naive={beta7_naive:.4f}, β7_DD={beta7_DD:.4f}, "
        f"bias={bias:.4f} ({bias_in_se:.2f} SEs)"
    )
    bias_doc = {
        "bias_absolute": float(bias),
        "bias_in_DD_SEs": float(bias_in_se),
        "interpretation": (
            "Substantial bias (>0.3 SE): naive mixes within- and between-country variation"
            if not math.isnan(bias_in_se) and abs(bias_in_se) > 0.3
            else (
                "Small bias: naive and DD agree closely"
                if not math.isnan(bias_in_se)
                else "Bias could not be computed (DD identification failure)"
            )
        ),
    }

    # Step 6 — Marginal effects (pass full restricted df for grid computation)
    marginal_effects = compute_marginal_effects(df_dd_clean, res_dd, df_restricted)

    # Step 7 — Lagged education
    lag_result = run_lagged_spec(df_restricted)

    # Step 8 — Sub-index tests (need DD columns — use df_dd_clean which already has them)
    sub_results = run_sub_index_tests(df_dd_clean)

    # Step 9 — Oster bounds
    oster_result = run_oster_bounds(df_restricted, res_dd, beta7_naive, r2_naive)

    # Step 10 — OECD falsification
    oecd_result = run_oecd_falsification(df_oecd)

    # Fitted values for schema
    fitted_naive = extract_fitted(res_naive, df_naive_clean)
    fitted_dd = extract_fitted(res_dd, df_dd_clean)

    # ── Assemble method_out.json ───────────────────────────────────────────────
    out_examples = []
    for ex in examples:
        d = json.loads(ex["input"])
        key = (str(d["country_iso3"]), int(d["period_start"]))
        new_ex = dict(ex)
        new_ex["predict_naive_fe"] = fitted_naive.get(key, "")
        new_ex["predict_dd_estimator"] = fitted_dd.get(key, "")
        out_examples.append(new_ex)

    method_out = {
        "metadata": {
            "method_name": "DD Triple-Interaction Estimator (Giesselmann-Schmidt-Catran 2022)",
            "description": (
                "Double-demeaning estimator for triple interactions (Education×Gini×SocProt) "
                "on post-1990 democratizer panel, OECD-excluded via WB GDP 1995-2005 avg. "
                "Baseline: naïve FE product model. Implementation via statsmodels OLS + "
                "explicit country/period dummies + cluster-robust SE (avoids linearmodels "
                "absorption detection issue with pre-demeaned variables in 2-period panels)."
            ),
            "n_obs_full_sample": len(df_full),
            "n_obs_restricted": len(df_restricted),
            "n_countries_restricted": int(df_restricted["country_iso3"].nunique()),
            "excluded_countries": excluded,
            "n_excluded": len(excluded),
            "within_sd": {
                "education": float(df_restricted["e_education_new"].std()),
                "gini_disp": float(df_restricted["e_gini_disp_new"].std()),
                "socprot_coverage": float(df_restricted["e_socprot_coverage_new"].std()),
            },
            "mde_warning": mde_note,
            "periods": period_counts,
            "pct_education_imputed": float(df_restricted["education_imputed"].mean()),
            "lagged_10yr_feasible": False,
            "lagged_10yr_reason": (
                "ILO SDG 1.3.1 only available from 2015; "
                "t-2 period (2005-09) has no socprot_coverage"
            ),
            "acemoglu_null": acemoglu_result,
            "naive_estimator": naive_result,
            "dd_estimator": dd_result,
            "bias_documentation": bias_doc,
            "lagged_education_5yr": lag_result,
            "sub_index_results": sub_results,
            "marginal_effects": marginal_effects,
            "oster_bounds": oster_result,
            "oecd_falsification": oecd_result,
            "figure_files": ["marginal_effects.png"],
        },
        "datasets": [
            {
                "dataset": "vdem_ilo_gini_edu_panel_complete",
                "examples": out_examples,
            }
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    def _clean(obj):
        """Recursively replace float NaN/Inf with None for valid JSON."""
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    out_path.write_text(json.dumps(_clean(method_out), indent=2))
    logger.info(f"Wrote method_out.json ({out_path.stat().st_size / 1024:.1f} KB)")

    # Soft validation
    with open(out_path) as f:
        loaded = json.load(f)
    assert "datasets" in loaded, "Missing datasets key"
    assert "metadata" in loaded, "Missing metadata key"
    meta = loaded["metadata"]
    assert "dd_estimator" in meta, "Missing dd_estimator"
    assert "beta7" in meta["dd_estimator"], "dd_estimator missing beta7"
    assert "bias_documentation" in meta, "Missing bias_documentation"
    assert "marginal_effects" in meta, "Missing marginal_effects"
    logger.info("method_out.json structure validation PASSED")

    beta7 = meta["dd_estimator"].get("beta7")
    pval7 = meta["dd_estimator"].get("pval7")
    identified = meta["dd_estimator"].get("identified", True)
    logger.info(f"FINAL: DD β7={beta7}, p={pval7}, identified={identified}")
    logger.info(
        f"FINAL: Acemoglu null p={meta['acemoglu_null'].get('pval')}, "
        f"replicated={meta['acemoglu_null'].get('replicated')}"
    )
    logger.info(f"FINAL: Bias={meta['bias_documentation'].get('bias_absolute')}")


if __name__ == "__main__":
    main()
