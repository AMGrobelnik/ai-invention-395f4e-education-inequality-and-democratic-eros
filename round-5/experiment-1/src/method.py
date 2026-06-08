#!/usr/bin/env python3
"""
Iter-5 Corrected DD: Standardized Variables, Per-Spec R², Oster Bounds, Sub-Index Merge.

Fixes five iter-4 errors:
(a) Standardize E/G/S before computing DD products → eliminates O(10^11) blow-up
(b) Each spec draws R² from its own fitted model
(c) Oster bounds for Spec C only, with ΔR² threshold check
(d) Sub-index merge from iter-1 dataset (v2jucomp, v2x_jucon)
(e) sigma_eps from Spec C residuals for MDE calculation
"""

import gc
import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy import stats

# ── Logging ────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
ITER1_DATASET = Path("/home/adrian/projects/ai-inventor/aii_data/users/adrian.marina.photos/runs/run_zXdSkAIIk5J3/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json")
ITER3_DATASET = Path("/home/adrian/projects/ai-inventor/aii_data/users/adrian.marina.photos/runs/run_zXdSkAIIk5J3/3_invention_loop/iter_3/gen_art/gen_art_dataset_1/full_data_out.json")

# ── Hardware & memory limits ────────────────────────────────────────────────
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

import psutil as _psutil

_avail_ram = _psutil.virtual_memory().available
RAM_BUDGET = min(int(_avail_ram * 0.6), 8 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f}GB")

# ── Constants ───────────────────────────────────────────────────────────────
PERIOD_LABELS = {1: "1990-94", 2: "1995-99", 3: "2000-04", 4: "2005-09",
                 5: "2010-14", 6: "2015-19", 7: "2020-22"}
GDP_THRESHOLD = 15_000
SUBIDX_THRESHOLD = 30
BONFERRONI_P = 0.025  # k=2 pre-registered sub-indices


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Load iter-3 panel
# ═══════════════════════════════════════════════════════════════════════════════
@logger.catch(reraise=True)
def load_iter3_panel() -> pd.DataFrame:
    logger.info(f"Loading iter-3 panel from {ITER3_DATASET}")
    with open(ITER3_DATASET) as f:
        raw3 = json.load(f)

    records = []
    for ex in raw3["datasets"][0]["examples"]:
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        out_raw = ex["output"]
        try:
            out_dict = json.loads(out_raw) if isinstance(out_raw, str) and out_raw.strip().startswith("{") else {}
        except (json.JSONDecodeError, AttributeError):
            out_dict = {}

        ldem_val = out_dict.get("v2x_libdem") or out_dict.get("ldem")
        if ldem_val is None:
            try:
                ldem_val = float(out_raw)
            except (ValueError, TypeError):
                ldem_val = None

        gdppc_raw = ex.get("metadata_gdppc_at_transition")
        try:
            gdppc_val = float(gdppc_raw) if gdppc_raw is not None else None
        except (ValueError, TypeError):
            gdppc_val = None

        records.append({
            "country": ex["metadata_country_code"],
            "period_id": int(ex["metadata_period_id"]),
            "ldem": ldem_val,
            "educ_mys": inp.get("educ_mys"),
            "educ_tertiary": inp.get("educ_tertiary"),
            "gini": inp.get("gini_disp"),
            "socprot": inp.get("socprot"),
            "gdppc": gdppc_val,
        })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} rows, {df['country'].nunique()} countries")
    assert len(df) == 425, f"Expected 425 obs from iter-3, got {len(df)}"
    assert df["country"].nunique() == 61, f"Expected 61 countries, got {df['country'].nunique()}"

    df["period_label"] = df["period_id"].map(PERIOD_LABELS)

    # GDP exclusion
    gdp_map = df.groupby("country")["gdppc"].first()
    gdp_exclusions = gdp_map[gdp_map > GDP_THRESHOLD].index.tolist()
    logger.info(f"Excluding {len(gdp_exclusions)} high-income countries: {gdp_exclusions}")
    df_main = df[~df["country"].isin(gdp_exclusions)].copy()

    del raw3
    gc.collect()
    return df_main


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Standardize within analysis sample
# ═══════════════════════════════════════════════════════════════════════════════
@logger.catch(reraise=True)
def build_analysis_sample(df_main: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_main["complete"] = df_main[["educ_tertiary", "gini", "socprot", "ldem"]].notna().all(axis=1)
    df_complete = df_main[df_main["complete"]].copy().reset_index(drop=True)
    N = len(df_complete)
    G = df_complete["country"].nunique()
    logger.info(f"Analysis sample (ter+gini+socprot+ldem all non-null): N={N}, G={G}")
    assert N >= 50, f"Too few complete obs: {N}"

    # FIX (a): standardize globally within analysis sample BEFORE DD products
    def std_col(series: pd.Series) -> tuple[pd.Series, float, float]:
        m, s = float(series.mean()), float(series.std())
        return (series - m) / s, m, s

    df_complete["E_ter_std"], mean_ter, sd_ter = std_col(df_complete["educ_tertiary"])
    df_complete["G_std"], mean_G, sd_G         = std_col(df_complete["gini"])
    df_complete["S_std"], mean_S, sd_S         = std_col(df_complete["socprot"])
    df_complete["Y"] = df_complete["ldem"]

    # MYS standardization (only where non-null)
    mys_mask = df_complete["educ_mys"].notna()
    mean_mys = float(df_complete.loc[mys_mask, "educ_mys"].mean())
    sd_mys   = float(df_complete.loc[mys_mask, "educ_mys"].std())
    df_complete["E_mys_std"] = (df_complete["educ_mys"] - mean_mys) / sd_mys
    N_mys = int(mys_mask.sum())
    logger.info(f"MYS complete obs: N={N_mys}")

    std_params = {
        "E_ter": {"mean": mean_ter, "sd": sd_ter},
        "E_mys": {"mean": mean_mys, "sd": sd_mys},
        "G":     {"mean": mean_G,   "sd": sd_G},
        "S":     {"mean": mean_S,   "sd": sd_S},
    }
    logger.info(f"Std params: sd_ter={sd_ter:.3f}, sd_mys={sd_mys:.4f}, sd_G={sd_G:.3f}, sd_S={sd_S:.3f}")

    # Verify standardization correctness
    assert abs(df_complete["E_ter_std"].mean()) < 1e-10, "E_ter_std not zero-mean"
    assert abs(df_complete["E_ter_std"].std() - 1.0) < 1e-6, "E_ter_std not unit-variance"
    assert abs(df_complete["G_std"].mean()) < 1e-10, "G_std not zero-mean"
    assert abs(df_complete["S_std"].mean()) < 1e-10, "S_std not zero-mean"
    logger.info("Standardization assertions PASSED")

    return df_complete, std_params


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Double-Demeaning on standardized variables
# ═══════════════════════════════════════════════════════════════════════════════
def dd_demean(df_in: pd.DataFrame, y_col: str, e_col: str, g_col: str, s_col: str,
              unit_col: str = "country") -> pd.DataFrame:
    """Giesselmann-Schmidt-Catran (2022) DD estimator on pre-standardized vars."""
    d = df_in.copy()

    # Step A: within-unit demeaning
    for v in [y_col, e_col, g_col, s_col]:
        group_mean = d.groupby(unit_col)[v].transform("mean")
        d[f"{v}_w"] = d[v] - group_mean

    # Step B: pairwise and triple products of within-unit residuals
    d["prod_EG"]  = d[f"{e_col}_w"] * d[f"{g_col}_w"]
    d["prod_ES"]  = d[f"{e_col}_w"] * d[f"{s_col}_w"]
    d["prod_GS"]  = d[f"{g_col}_w"] * d[f"{s_col}_w"]
    d["prod_EGS"] = d[f"{e_col}_w"] * d[f"{g_col}_w"] * d[f"{s_col}_w"]

    # Step C: demean product terms within unit (critical DD step)
    for v in ["prod_EG", "prod_ES", "prod_GS", "prod_EGS"]:
        d[f"dd_{v}"] = d[v] - d.groupby(unit_col)[v].transform("mean")

    return d


def verify_dd_zero_mean(df_dd: pd.DataFrame, unit_col: str = "country") -> None:
    for col in ["dd_prod_EG", "dd_prod_ES", "dd_prod_GS", "dd_prod_EGS"]:
        within_mean_max = df_dd.groupby(unit_col)[col].mean().abs().max()
        assert within_mean_max < 1e-8, f"DD zero-mean violated for {col}: max={within_mean_max:.2e}"
    logger.info("DD zero-mean invariant PASSED for all 4 product terms")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — OLS with entity+time FE and clustered SEs
# ═══════════════════════════════════════════════════════════════════════════════
def run_spec_ols(df_reg: pd.DataFrame, y_col: str, x_names: list[str],
                 unit_col: str = "country", period_col: str = "period_id") -> dict:
    """
    OLS with entity+time dummies and clustered SEs by unit.
    FIX (b): R² read from this model only.
    Returns dict with coefs, N, G, R2_within_own_model, residuals, fitted.
    """
    from statsmodels.stats.sandwich_covariance import cov_cluster

    coef_labels = ["beta_E", "beta_G", "beta_S", "beta_EG", "beta_ES", "beta_GS", "beta_EGS"]
    d = df_reg.dropna(subset=[y_col] + x_names).copy()

    country_dummies = pd.get_dummies(d[unit_col], prefix="c", drop_first=True).astype(float)
    period_dummies  = pd.get_dummies(d[period_col].astype(str), prefix="p", drop_first=True).astype(float)
    X_interact = d[x_names].astype(float)
    X = pd.concat([X_interact, country_dummies, period_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = X.loc[:, X.nunique() > 1]  # drop constant columns

    y = d[y_col].astype(float)
    model = sm.OLS(y, X).fit()

    try:
        cov = cov_cluster(model, group=d[unit_col].values)
    except Exception:
        logger.warning("Clustered SE computation failed; using HC3 robust SEs")
        cov = model.cov_HC3()

    coefs_out = {}
    for x_name, label in zip(x_names, coef_labels):
        if x_name not in model.params.index:
            coefs_out[label] = {"coef": None, "se": None, "p": None, "sig": "n/a"}
            continue
        idx = list(model.params.index).index(x_name)
        coef_val = float(model.params[x_name])
        se_val   = float(np.sqrt(max(cov[idx, idx], 0.0)))
        dof      = max(1, d[unit_col].nunique() - 1)
        t_stat   = coef_val / se_val if se_val > 1e-15 else 0.0
        p_val    = float(2 * stats.t.sf(abs(t_stat), df=dof))
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        coefs_out[label] = {
            "coef": round(coef_val, 8),
            "se":   round(se_val, 8),
            "p":    round(p_val, 4),
            "sig":  sig,
        }

    return {
        "N": int(len(d)),
        "G": int(d[unit_col].nunique()),
        "R2_within_own_model": round(float(model.rsquared), 8),
        "coefs": coefs_out,
        "residuals": model.resid.values.tolist(),
        "fitted": model.fittedvalues.tolist(),
        "design_rank": int(model.df_model),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Run the four specifications
# ═══════════════════════════════════════════════════════════════════════════════
@logger.catch(reraise=True)
def run_four_specs(df_complete: pd.DataFrame) -> dict:
    logger.info("Running four specifications...")

    # DD frames
    df_dd_ter = dd_demean(df_complete, "Y", "E_ter_std", "G_std", "S_std")
    df_dd_mys = dd_demean(
        df_complete.dropna(subset=["E_mys_std"]).copy(),
        "Y", "E_mys_std", "G_std", "S_std"
    )
    verify_dd_zero_mean(df_dd_ter)

    # Naive interaction terms (standardized products)
    df_complete["EG_naive_mys"]  = df_complete["E_mys_std"] * df_complete["G_std"]
    df_complete["ES_naive_mys"]  = df_complete["E_mys_std"] * df_complete["S_std"]
    df_complete["GS_naive"]      = df_complete["G_std"]     * df_complete["S_std"]
    df_complete["EGS_naive_mys"] = df_complete["E_mys_std"] * df_complete["G_std"] * df_complete["S_std"]
    df_complete["EG_naive_ter"]  = df_complete["E_ter_std"] * df_complete["G_std"]
    df_complete["ES_naive_ter"]  = df_complete["E_ter_std"] * df_complete["S_std"]
    df_complete["EGS_naive_ter"] = df_complete["E_ter_std"] * df_complete["G_std"] * df_complete["S_std"]

    # Spec A: Naive FE, MYS
    logger.info("Spec A: Naive FE, MYS standardized")
    spec_a = run_spec_ols(
        df_complete.dropna(subset=["E_mys_std"]),
        y_col="Y",
        x_names=["E_mys_std", "G_std", "S_std", "EG_naive_mys", "ES_naive_mys", "GS_naive", "EGS_naive_mys"],
    )
    logger.info(f"Spec A: N={spec_a['N']}, R2={spec_a['R2_within_own_model']:.6f}, beta_EGS={spec_a['coefs']['beta_EGS']['coef']}")

    # Spec B: DD, MYS
    logger.info("Spec B: DD, MYS standardized")
    spec_b = run_spec_ols(
        df_dd_mys,
        y_col="Y",
        x_names=["E_mys_std_w", "G_std_w", "S_std_w", "dd_prod_EG", "dd_prod_ES", "dd_prod_GS", "dd_prod_EGS"],
    )
    logger.info(f"Spec B: N={spec_b['N']}, R2={spec_b['R2_within_own_model']:.6f}, beta_EGS={spec_b['coefs']['beta_EGS']['coef']}")

    # Spec C: DD, WB Tertiary [PRIMARY]
    logger.info("Spec C: DD, WB Tertiary standardized [PRIMARY]")
    spec_c = run_spec_ols(
        df_dd_ter,
        y_col="Y",
        x_names=["E_ter_std_w", "G_std_w", "S_std_w", "dd_prod_EG", "dd_prod_ES", "dd_prod_GS", "dd_prod_EGS"],
    )
    logger.info(f"Spec C: N={spec_c['N']}, R2={spec_c['R2_within_own_model']:.6f}, beta_EGS={spec_c['coefs']['beta_EGS']['coef']}")

    # Validate Spec C — no blow-up
    for label, cd in spec_c["coefs"].items():
        if cd["se"] is not None:
            assert cd["se"] < 1e8, f"Spec C {label} SE blown up: {cd['se']}"
    assert abs(spec_c["coefs"]["beta_EGS"]["coef"]) < 100, "beta_EGS unreasonably large after standardization"
    logger.info("Spec C interaction SE bounds PASSED")

    # Spec D: Naive FE, WB Tertiary
    logger.info("Spec D: Naive FE, WB Tertiary standardized")
    spec_d = run_spec_ols(
        df_complete,
        y_col="Y",
        x_names=["E_ter_std", "G_std", "S_std", "EG_naive_ter", "ES_naive_ter", "GS_naive", "EGS_naive_ter"],
    )
    logger.info(f"Spec D: N={spec_d['N']}, R2={spec_d['R2_within_own_model']:.6f}, beta_EGS={spec_d['coefs']['beta_EGS']['coef']}")

    # Per-spec R² validation
    for name, spec in [("A", spec_a), ("B", spec_b), ("C", spec_c), ("D", spec_d)]:
        r2 = spec["R2_within_own_model"]
        assert 0.0 <= r2 <= 1.0, f"R2={r2} out of [0,1] for Spec {name}"
    logger.info(f"R2 values: A={spec_a['R2_within_own_model']:.4f}, B={spec_b['R2_within_own_model']:.4f}, C={spec_c['R2_within_own_model']:.4f}, D={spec_d['R2_within_own_model']:.4f}")

    return {
        "spec_a": spec_a,
        "spec_b": spec_b,
        "spec_c": spec_c,
        "spec_d": spec_d,
        "df_dd_ter": df_dd_ter,
        "df_dd_mys": df_dd_mys,
        "df_complete": df_complete,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Oster (2019) bounds for Spec C
# ═══════════════════════════════════════════════════════════════════════════════
@logger.catch(reraise=True)
def compute_oster_bounds(spec_c: dict, df_dd_ter: pd.DataFrame) -> dict:
    """FIX (c): Compute Oster delta for Spec C only."""
    logger.info("Computing Oster bounds for Spec C...")

    spec_c_partial = run_spec_ols(
        df_dd_ter,
        y_col="Y",
        x_names=["E_ter_std_w", "G_std_w", "S_std_w", "dd_prod_EG", "dd_prod_ES", "dd_prod_GS"],  # no EGS
    )
    R2_full    = spec_c["R2_within_own_model"]
    R2_partial = spec_c_partial["R2_within_own_model"]
    delta_R2   = R2_full - R2_partial
    OSTER_R2_MAX_MULT = 1.3
    R2_max = OSTER_R2_MAX_MULT * R2_full
    beta_EGS_full = spec_c["coefs"]["beta_EGS"]["coef"]

    logger.info(f"Oster: R2_full={R2_full:.6f}, R2_partial={R2_partial:.6f}, delta_R2={delta_R2:.2e}")

    if abs(delta_R2) < 1e-4:
        result = {
            "status": "numerically_undefined",
            "reason": f"delta_R2={delta_R2:.2e} < 1e-4 threshold; denominator near zero",
            "note": "Oster bounds numerically undefined for Spec C",
            "R2_full": R2_full,
            "R2_partial": R2_partial,
            "delta_R2": round(delta_R2, 8),
            "R2_max": round(R2_max, 6),
            "beta_EGS_full": beta_EGS_full,
            "interpretation": (
                "Near-zero delta_R2 implies triple interaction explains essentially no "
                "additional variance beyond two-way terms, making Oster comparison uninformative"
            ),
        }
    else:
        oster_delta = (R2_max - R2_full) / delta_R2
        result = {
            "status": "computed",
            "oster_delta": round(oster_delta, 4),
            "R2_full": R2_full,
            "R2_partial": R2_partial,
            "delta_R2": round(delta_R2, 8),
            "R2_max": round(R2_max, 6),
            "beta_EGS_full": beta_EGS_full,
            "interpretation": (
                f"delta={oster_delta:.2f}: selection on unobservables would need to be "
                f"{oster_delta:.1f}x as large as observed to nullify the estimate"
            ),
            "robust_to_omv": oster_delta > 1.0,
        }

    logger.info(f"Oster Spec C: {result['status']}, delta_R2={delta_R2:.2e}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Sub-index DD (v2jucomp, v2x_jucon) from iter-1
# ═══════════════════════════════════════════════════════════════════════════════
@logger.catch(reraise=True)
def run_subindex_analysis(df_dd_ter: pd.DataFrame) -> dict:
    """FIX (d): Load v2jucomp/v2x_jucon from iter-1, merge, run DD."""
    logger.info("Loading iter-1 sub-indices (v2jucomp, v2x_jucon)...")

    PERIOD_MAP = {
        "2015-19": 6, "2015-2019": 6,
        "2020-22": 7, "2020-2022": 7,
    }

    try:
        with open(ITER1_DATASET) as f:
            raw1 = json.load(f)

        sub_records = []
        for ex in raw1["datasets"][0]["examples"]:
            inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
            country_code = inp.get("country_iso3")
            period_str   = inp.get("period")
            if not country_code or not period_str:
                continue
            period_id = PERIOD_MAP.get(period_str)
            if period_id is None:
                continue
            sub_records.append({
                "country":   country_code,
                "period_id": period_id,
                "v2jucomp":  inp.get("v2jucomp"),
                "v2x_jucon": inp.get("v2x_jucon"),
            })

        del raw1
        gc.collect()

        sub_df = pd.DataFrame(sub_records).drop_duplicates(subset=["country", "period_id"])
        n_j  = sub_df["v2jucomp"].notna().sum()
        n_xj = sub_df["v2x_jucon"].notna().sum()
        logger.info(f"Iter-1 sub-index rows: v2jucomp={n_j}, v2x_jucon={n_xj}")

    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"Iter-1 dataset error: {e}")
        sub_df = pd.DataFrame(columns=["country", "period_id", "v2jucomp", "v2x_jucon"])
        n_j, n_xj = 0, 0

    # Merge onto DD ter frame
    df_sub = df_dd_ter.merge(
        sub_df[["country", "period_id", "v2jucomp", "v2x_jucon"]],
        on=["country", "period_id"],
        how="left",
    )

    subindex_results = {}
    for subidx in ["v2jucomp", "v2x_jucon"]:
        n_nonull = int(df_sub[subidx].notna().sum())
        logger.info(f"{subidx}: {n_nonull} non-null obs after merge onto DD frame")

        if n_nonull < SUBIDX_THRESHOLD:
            subindex_results[subidx] = {
                "status": "insufficient_data",
                "n_nonull": n_nonull,
                "n_required": SUBIDX_THRESHOLD,
                "note": f"Only {n_nonull} non-null obs after merge; min {SUBIDX_THRESHOLD} required",
                "provenance": "iter_1/gen_art_dataset_1 — vdem_ilo_gini_edu_panel_complete",
            }
            continue

        sub_spec = run_spec_ols(
            df_sub.dropna(subset=[subidx]),
            y_col=subidx,
            x_names=["E_ter_std_w", "G_std_w", "S_std_w", "dd_prod_EG", "dd_prod_ES", "dd_prod_GS", "dd_prod_EGS"],
        )
        p_egs = sub_spec["coefs"]["beta_EGS"]["p"]
        passes_bonferroni = bool(p_egs < BONFERRONI_P) if p_egs is not None else None
        subindex_results[subidx] = {
            "status": "computed",
            "N": sub_spec["N"],
            "G": sub_spec["G"],
            "R2_within": sub_spec["R2_within_own_model"],
            "coefs": sub_spec["coefs"],
            "bonferroni_threshold": BONFERRONI_P,
            "passes_bonferroni": passes_bonferroni,
            "source": "iter_1/gen_art_dataset_1 — vdem_ilo_gini_edu_panel_complete",
        }
        logger.info(f"{subidx}: beta_EGS={sub_spec['coefs']['beta_EGS']['coef']}, p={p_egs}, bonferroni={passes_bonferroni}")

    if not subindex_results:
        subindex_results = {
            "status": "unavailable",
            "source_attempted": "iter_1/gen_art_dataset_1",
            "reason": "v2jucomp/v2x_jucon absent; V-Dem sub-indices require V-Dem v16 bulk CSV",
            "pre_registered_indices": ["v2jucomp", "v2x_jucon"],
            "bonferroni_threshold": BONFERRONI_P,
        }

    return subindex_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Marginal effects grid (standardized, bounded)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_marginal_effects(spec_c: dict, df_complete: pd.DataFrame) -> dict:
    """dY/dE_std = beta_E + beta_EG*G_std + beta_ES*S_std + beta_EGS*G_std*S_std"""
    beta_E   = spec_c["coefs"]["beta_E"]["coef"]   or 0.0
    beta_EG  = spec_c["coefs"]["beta_EG"]["coef"]  or 0.0
    beta_ES  = spec_c["coefs"]["beta_ES"]["coef"]  or 0.0
    beta_EGS = spec_c["coefs"]["beta_EGS"]["coef"] or 0.0
    se_E     = spec_c["coefs"]["beta_E"]["se"]     or 0.0
    se_EG    = spec_c["coefs"]["beta_EG"]["se"]    or 0.0
    se_ES    = spec_c["coefs"]["beta_ES"]["se"]    or 0.0
    se_EGS   = spec_c["coefs"]["beta_EGS"]["se"]   or 0.0

    gini_q25_std = float(df_complete["G_std"].quantile(0.25))
    gini_q75_std = float(df_complete["G_std"].quantile(0.75))
    sp_q25_std   = float(df_complete["S_std"].quantile(0.25))
    sp_q75_std   = float(df_complete["S_std"].quantile(0.75))
    gini_q25_raw = float(df_complete["gini"].quantile(0.25))
    gini_q75_raw = float(df_complete["gini"].quantile(0.75))
    sp_q25_raw   = float(df_complete["socprot"].quantile(0.25))
    sp_q75_raw   = float(df_complete["socprot"].quantile(0.75))

    marginal_effects = {}
    for g_lbl, G_s, G_r in [("p25", gini_q25_std, gini_q25_raw), ("p75", gini_q75_std, gini_q75_raw)]:
        for s_lbl, S_s, S_r in [("p25", sp_q25_std, sp_q25_raw), ("p75", sp_q75_std, sp_q75_raw)]:
            me_val = beta_E + beta_EG * G_s + beta_ES * S_s + beta_EGS * G_s * S_s
            me_se  = float(np.sqrt(se_E**2 + (G_s*se_EG)**2 + (S_s*se_ES)**2 + (G_s*S_s*se_EGS)**2))
            cell = f"gini_{g_lbl}_sp_{s_lbl}"
            assert abs(me_val) < 50, f"ME={me_val:.4f} out of bounds at {cell}; standardization may have failed"
            marginal_effects[cell] = {
                "gini_quantile": g_lbl,
                "sp_quantile":   s_lbl,
                "gini_std":  round(G_s, 4),
                "sp_std":    round(S_s, 4),
                "gini_raw":  round(G_r, 2),
                "sp_raw":    round(S_r, 2),
                "me":       round(me_val, 8),
                "me_se":    round(me_se, 8),
                "me_95ci_lo": round(me_val - 1.96 * me_se, 8),
                "me_95ci_hi": round(me_val + 1.96 * me_se, 8),
                "sign": "+" if me_val > 0 else "-",
                "units": "LDem SD change per 1-SD change in E_ter_std",
            }

    logger.info("Marginal effects grid:")
    for cell, v in marginal_effects.items():
        logger.info(f"  {cell}: ME={v['me']:.6f}, SE={v['me_se']:.6f}, sign={v['sign']}")
    return marginal_effects


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Power / MDE from Spec C residuals
# ═══════════════════════════════════════════════════════════════════════════════
def compute_power_mde(spec_c: dict) -> dict:
    """FIX (e): sigma_eps from actual Spec C residuals."""
    residuals = np.array(spec_c["residuals"])
    sigma_eps = float(np.std(residuals))
    se_beta7  = spec_c["coefs"]["beta_EGS"]["se"]

    if se_beta7 is None or se_beta7 == 0:
        logger.warning("beta_EGS SE not available; returning partial power result")
        return {"status": "se_unavailable", "sigma_eps_from_spec_c_residuals": round(sigma_eps, 6)}

    Z_POWER, Z_SIG = 0.842, 1.960
    mde_80pct = (Z_POWER + Z_SIG) * se_beta7
    effect_over_mde = abs(spec_c["coefs"]["beta_EGS"]["coef"]) / mde_80pct if mde_80pct > 0 else 0.0

    result = {
        "sigma_eps_from_spec_c_residuals": round(sigma_eps, 6),
        "se_beta_EGS_spec_c": round(se_beta7, 8),
        "mde_80pct_5pct": round(mde_80pct, 8),
        "mde_units": "1 SD change in LDem per SD change in E_ter_std × (G_std × S_std)",
        "current_beta_EGS": spec_c["coefs"]["beta_EGS"]["coef"],
        "effect_over_mde_ratio": round(effect_over_mde, 4),
        "powered": bool(effect_over_mde >= 1.0),
        "G_doubly_observed": spec_c["G"],
        "interpretation": (
            f"MDE={mde_80pct:.5f} vs |beta_EGS|={abs(spec_c['coefs']['beta_EGS']['coef']):.6f}; "
            f"ratio={effect_over_mde:.2f}. "
            + ("Underpowered: estimated effect smaller than MDE at 80%/5%." if effect_over_mde < 1.0 else "Adequately powered.")
        ),
    }
    logger.info(f"Power/MDE: sigma={sigma_eps:.4f}, MDE={mde_80pct:.5f}, ratio={effect_over_mde:.3f}")
    assert sigma_eps > 0, "sigma_eps must be positive"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — DD sanity checks
# ═══════════════════════════════════════════════════════════════════════════════
def build_dd_sanity_checks(df_dd_ter: pd.DataFrame, spec_c: dict) -> list[dict]:
    dd_checks = []

    for col in ["dd_prod_EG", "dd_prod_ES", "dd_prod_GS", "dd_prod_EGS"]:
        within_mean_max = float(df_dd_ter.groupby("country")[col].mean().abs().max())
        dd_checks.append({
            "input": json.dumps({"check": "dd_zero_mean", "column": col}),
            "output": "PASSED" if within_mean_max < 1e-8 else "FAILED",
            "metadata_column": col,
            "metadata_max_within_mean": str(within_mean_max),
            "metadata_threshold": "1e-8",
        })

    for label, cd in spec_c["coefs"].items():
        if cd.get("se") is not None:
            se_bounded = abs(cd["se"]) < 1e6
            dd_checks.append({
                "input": json.dumps({"check": "se_bounded", "spec": "C", "coef": label}),
                "output": "PASSED" if se_bounded else "FAILED",
                "metadata_coef": label,
                "metadata_se": str(cd["se"]),
                "metadata_threshold": "1e6",
            })

    return dd_checks


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Assemble output JSON
# ═══════════════════════════════════════════════════════════════════════════════
def assemble_output(
    specs: dict,
    df_complete: pd.DataFrame,
    std_params: dict,
    oster_result: dict,
    subindex_results: dict,
    power_result: dict,
    marginal_effects: dict,
    dd_checks: list[dict],
) -> dict:
    spec_a = specs["spec_a"]
    spec_b = specs["spec_b"]
    spec_c = specs["spec_c"]
    spec_d = specs["spec_d"]

    # Dataset 1: Corrected Table 1 — all 4 specs
    reg_examples = []
    for spec_label, spec_res, edu_type, estimator in [
        ("A_naive_mys_std",        spec_a, "UNDP MYS (standardized)",        "Naive FE"),
        ("B_dd_mys_std",           spec_b, "UNDP MYS (standardized)",        "DD (Giesselmann-Schmidt-Catran 2022)"),
        ("C_dd_wb_tertiary_std",   spec_c, "WB SE.TER.ENRR (standardized)",  "DD (Giesselmann-Schmidt-Catran 2022)"),
        ("D_naive_wb_tertiary_std",spec_d, "WB SE.TER.ENRR (standardized)",  "Naive FE"),
    ]:
        for coef_name, coef_data in spec_res["coefs"].items():
            reg_examples.append({
                "input": json.dumps({"spec": spec_label, "coefficient": coef_name}),
                "output": str(coef_data.get("coef")) if coef_data.get("coef") is not None else "null",
                "metadata_spec": spec_label,
                "metadata_coef": coef_name,
                "metadata_se": str(coef_data.get("se")),
                "metadata_p": str(coef_data.get("p")),
                "metadata_sig": coef_data.get("sig", ""),
                "metadata_N": str(spec_res["N"]),
                "metadata_R2_within": str(spec_res["R2_within_own_model"]),
                "metadata_G_clusters": str(spec_res["G"]),
                "metadata_estimator": estimator,
                "metadata_education_proxy": edu_type,
                "metadata_standardized": "True",
            })

    # Dataset 2: Oster bounds
    oster_examples = [{
        "input": json.dumps({"analysis": "oster_bounds", "spec": "C_dd_wb_tertiary_std"}),
        "output": str(oster_result.get("oster_delta", oster_result.get("status", "undefined"))),
        **{f"metadata_{k}": str(v) for k, v in oster_result.items()},
    }]

    # Dataset 3: Sub-index DD results
    sub_examples = []
    if isinstance(subindex_results, dict) and "status" in subindex_results and subindex_results.get("status") == "unavailable":
        sub_examples.append({
            "input": json.dumps({"analysis": "sub_index_dd", "indices": "v2jucomp,v2x_jucon"}),
            "output": "unavailable",
            **{f"metadata_{k}": str(v) for k, v in subindex_results.items()},
        })
    else:
        for idx_name, idx_res in subindex_results.items():
            if isinstance(idx_res, dict):
                coef_egs = idx_res.get("coefs", {}).get("beta_EGS", {}) if idx_res.get("status") == "computed" else {}
                sub_examples.append({
                    "input": json.dumps({"sub_index": idx_name}),
                    "output": str(coef_egs.get("coef", idx_res.get("status", "null"))),
                    "metadata_index": idx_name,
                    "metadata_status": idx_res.get("status", ""),
                    "metadata_N": str(idx_res.get("N", "")),
                    "metadata_G": str(idx_res.get("G", "")),
                    "metadata_R2_within": str(idx_res.get("R2_within", "")),
                    "metadata_p_egs": str(coef_egs.get("p", "")),
                    "metadata_passes_bonferroni": str(idx_res.get("passes_bonferroni", "")),
                    "metadata_note": str(idx_res.get("note", "")),
                })

    # Dataset 4: Power/MDE
    power_examples = [{
        "input": json.dumps({"analysis": "power_mde", "spec": "C_primary"}),
        "output": str(power_result.get("mde_80pct_5pct", "unavailable")),
        **{f"metadata_{k}": str(v) for k, v in power_result.items()},
    }]

    # Dataset 5: Marginal effects grid
    me_examples = []
    for cell_key, cell_val in marginal_effects.items():
        me_examples.append({
            "input": json.dumps({"cell": cell_key}),
            "output": str(cell_val["me"]),
            **{f"metadata_{k}": str(v) for k, v in cell_val.items()},
        })

    # Dataset 6: DD sanity checks
    check_examples = dd_checks

    # Dataset 7: Provenance disclosure
    provenance = {
        "original_headline_estimate": {
            "beta_EGS": 2.398,
            "dataset": "2-period UNDP HDR25 MYS panel, post-1990 developing democratizers",
            "education_proxy": "UNDP HDR25 Mean Years of Schooling (MYS)",
            "within_SD_education_2period": 0.026,
            "within_SD_units": "years of schooling, within-country SD across 2 periods",
            "critical_flaw": "within-SD=0.026 is imputation noise (MYS frozen at 2017 in UNDP HDR25, linearly interpolated 2015-2022)",
        },
        "current_primary_estimate": {
            "spec": "C_dd_wb_tertiary_standardized",
            "education_proxy": "World Bank SE.TER.ENRR (gross tertiary enrollment rate, %)",
            "standardization": "E_std = (educ_tertiary - mean_ter) / sd_ter within N analysis sample",
            "standardization_params": {k: {m: round(v[m], 6) for m in v} for k, v in std_params.items()},
        },
    }
    assert provenance["original_headline_estimate"]["beta_EGS"] == 2.398
    assert provenance["original_headline_estimate"]["within_SD_education_2period"] == 0.026
    prov_examples = [{
        "input": json.dumps({"analysis": "provenance_disclosure"}),
        "output": str(provenance["original_headline_estimate"]["beta_EGS"]),
        **{f"metadata_original_{k}": str(v) for k, v in provenance["original_headline_estimate"].items()},
        **{f"metadata_current_{k}": str(v) for k, v in provenance["current_primary_estimate"].items()},
    }]

    # Dataset 8: Spec C per-country-period predictions
    pred_examples = []
    for i, row in df_complete.reset_index(drop=True).iterrows():
        if i < len(spec_c["fitted"]):
            pred_examples.append({
                "input": json.dumps({
                    "country_iso3": row["country"],
                    "period_id": int(row["period_id"]),
                    "E_ter_std": round(float(row["E_ter_std"]), 4),
                    "G_std": round(float(row["G_std"]), 4),
                    "S_std": round(float(row["S_std"]), 4),
                }),
                "output": str(round(float(row["Y"]), 6)),
                "metadata_country": str(row["country"]),
                "metadata_period_id": str(int(row["period_id"])),
                "metadata_period_label": str(row.get("period_label", "")),
                "predict_dd_corrected": str(round(float(spec_c["fitted"][i]), 6)),
                "predict_baseline": str(round(float(spec_d["fitted"][i]), 6)),
            })

    full_out = {
        "datasets": [
            {"dataset": "spec_c_primary_predictions_per_country_period",   "examples": pred_examples},
            {"dataset": "corrected_table1_all_4specs_standardized",       "examples": reg_examples},
            {"dataset": "oster_bounds_spec_c_only",                        "examples": oster_examples},
            {"dataset": "subindex_dd_pre_registered_k2_v2jucomp_v2x_jucon","examples": sub_examples},
            {"dataset": "power_mde_from_spec_c_residuals",                 "examples": power_examples},
            {"dataset": "marginal_effects_standardized_2x2_grid",          "examples": me_examples},
            {"dataset": "dd_sanity_checks_zero_mean_and_se_bounds",        "examples": check_examples},
            {"dataset": "provenance_disclosure_beta7_original_vs_corrected","examples": prov_examples},
        ]
    }
    return full_out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
@logger.catch(reraise=True)
def main() -> None:
    logger.info("=== ITER-5 CORRECTED DD ANALYSIS START ===")

    # Step 0: Load data
    df_main = load_iter3_panel()

    # Step 1: Build analysis sample and standardize
    df_complete, std_params = build_analysis_sample(df_main)
    del df_main
    gc.collect()

    # Steps 2–4: DD and four specs
    results = run_four_specs(df_complete)
    spec_c    = results["spec_c"]
    df_dd_ter = results["df_dd_ter"]
    df_complete = results["df_complete"]  # updated with naive interaction cols

    # Step 5: Oster bounds
    oster_result = compute_oster_bounds(spec_c, df_dd_ter)

    # Step 6: Sub-index analysis
    subindex_results = run_subindex_analysis(df_dd_ter)

    # Step 7: Marginal effects
    marginal_effects = compute_marginal_effects(spec_c, df_complete)

    # Step 8: Power/MDE
    power_result = compute_power_mde(spec_c)

    # Step 9: Sanity checks
    dd_checks = build_dd_sanity_checks(df_dd_ter, spec_c)

    # Step 10: Assemble and write output
    full_out = assemble_output(
        specs={k: v for k, v in results.items() if k.startswith("spec_")},
        df_complete=df_complete,
        std_params=std_params,
        oster_result=oster_result,
        subindex_results=subindex_results,
        power_result=power_result,
        marginal_effects=marginal_effects,
        dd_checks=dd_checks,
    )

    # Write output files
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(full_out, indent=2))
    logger.info(f"Wrote {out_path}")

    # Final summary
    beta_egs = spec_c["coefs"]["beta_EGS"]["coef"]
    se_egs   = spec_c["coefs"]["beta_EGS"]["se"]
    p_egs    = spec_c["coefs"]["beta_EGS"]["p"]
    sig_egs  = spec_c["coefs"]["beta_EGS"]["sig"]

    logger.info("=== ITER-5 COMPLETE ===")
    logger.info(f"Spec C (PRIMARY): beta_EGS={beta_egs:.6f}, SE={se_egs:.6f}, p={p_egs}, sig={sig_egs}")
    logger.info(f"Oster Spec C: {oster_result['status']}, delta_R2={oster_result['delta_R2']:.2e}")
    logger.info(f"Power/MDE: sigma={power_result.get('sigma_eps_from_spec_c_residuals')}, MDE={power_result.get('mde_80pct_5pct')}")
    logger.info(f"Sub-index results: {list(subindex_results.keys())}")
    logger.info(f"All Spec C interaction SEs bounded: {all(abs(cd['se']) < 1e6 for cd in spec_c['coefs'].values() if cd.get('se') is not None)}")
    logger.info(f"DD sanity checks: {sum(1 for c in dd_checks if c['output']=='PASSED')}/{len(dd_checks)} PASSED")


if __name__ == "__main__":
    main()
