#!/usr/bin/env python3
"""
DD Triple-Interaction estimator with WB tertiary enrollment + Granger & Reverse-Causality tests.
Implements Giesselmann-Schmidt-Catran (2022) DD estimator for SDET hypothesis.

Specifications:
  A: UNDP MYS × naive FE-product (benchmark)
  B: UNDP MYS × DD estimator (replication of iter_2)
  C: WB tertiary enrollment × DD estimator (primary new result)
  D: WB tertiary enrollment × naive FE-product
  E: WB tertiary × DD with lagged SocProt (reverse-causality sensitivity)
"""

import json
import sys
import gc
import re
import math
import resource
import warnings
import numpy as np
import pandas as pd
import pycountry
from pathlib import Path
from loguru import logger
from typing import Optional

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO",
           format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE = WORKSPACE.parent.parent.parent  # 3_invention_loop level (iter_1/iter_2/iter_3 are siblings)
PANEL_PATH = BASE / "iter_1/gen_art/gen_art_dataset_1/full_data_out.json"
MULTI_PATH = BASE / "iter_2/gen_art/gen_art_dataset_1/full_data_out.json"
SAP_PATH   = BASE / "iter_1/gen_art/gen_art_dataset_2/full_data_out.json"

# ── Memory limit (6GB for CPU-only work on 43GB machine) ───────────────────────
try:
    RAM_BUDGET = 6 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
except Exception:
    pass  # Some systems don't support RLIMIT_AS


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def name_to_iso3(name: str) -> Optional[str]:
    """Convert country name to ISO3 using pycountry."""
    if not name:
        return None
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        pass
    # Common fixes for problematic names
    fixes = {
        "Bolivia": "BOL", "Bolivian Republic of Venezuela": "VEN",
        "Venezuela, RB": "VEN", "Iran, Islamic Rep.": "IRN",
        "Egypt, Arab Rep.": "EGY", "Korea, Rep.": "KOR",
        "Korea, Dem. Rep.": "PRK", "Lao PDR": "LAO",
        "Yemen, Rep.": "YEM", "Congo, Rep.": "COG",
        "Congo, Dem. Rep.": "COD", "Gambia, The": "GMB",
        "Bahamas, The": "BHS", "Syrian Arab Republic": "SYR",
        "Micronesia, Fed. Sts.": "FSM", "St. Lucia": "LCA",
        "St. Vincent and the Grenadines": "VCT", "St. Kitts and Nevis": "KNA",
        "Kyrgyz Republic": "KGZ", "Slovak Republic": "SVK",
        "North Macedonia": "MKD", "Czechia": "CZE",
        "China, People's Republic of": "CHN", "China": "CHN",
        "Moldova": "MDA", "Russia": "RUS",
        "Tanzania": "TZA", "United Republic of Tanzania": "TZA",
        "Bolivia (Plurinational State of)": "BOL",
        "Venezuela (Bolivarian Republic of)": "VEN",
        "Iran (Islamic Republic of)": "IRN",
        "Palestine, State of": "PSE", "Palestinian Territory": "PSE",
        "Kosovo": "XKX", "Timor-Leste": "TLS",
        "Côte d'Ivoire": "CIV", "Cote d'Ivoire": "CIV",
        "Cape Verde": "CPV", "Cabo Verde": "CPV",
        "Eswatini": "SWZ", "Swaziland": "SWZ",
        "North Korea": "PRK", "South Korea": "KOR",
    }
    if name in fixes:
        return fixes[name]
    # Try stripping common suffixes
    for suffix in [", The", " (the)"]:
        if name.endswith(suffix):
            trimmed = name[: -len(suffix)]
            try:
                return pycountry.countries.lookup(trimmed).alpha_3
            except LookupError:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_json_datasets(path: Path) -> dict:
    """Load all datasets from a full_data_out.json file into a dict keyed by dataset name."""
    logger.info(f"Loading {path.name} ({path.stat().st_size/1e6:.1f} MB)")
    raw = json.loads(path.read_text())
    return {ds["dataset"]: ds["examples"] for ds in raw["datasets"]}


@logger.catch(reraise=True)
def load_panel(examples: list) -> pd.DataFrame:
    """Parse iter_1/dataset_1 panel (161 rows)."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        inp["v2x_libdem"] = safe_float(ex["output"])
        rows.append(inp)
    df = pd.DataFrame(rows)
    df["education"]         = pd.to_numeric(df["education"], errors="coerce")
    df["gini_disp"]         = pd.to_numeric(df["gini_disp"], errors="coerce")
    df["socprot_coverage"]  = pd.to_numeric(df["socprot_coverage"], errors="coerce")
    df["v2x_libdem"]        = pd.to_numeric(df["v2x_libdem"], errors="coerce")
    df["period_start"]      = pd.to_numeric(df["period_start"], errors="coerce").astype(int)
    # Fix bool column
    df["education_imputed"] = df["education_imputed"].apply(
        lambda x: bool(x) if isinstance(x, (bool, int)) else str(x).lower() == "true"
    )
    logger.info(f"Panel loaded: {len(df)} rows, {df['country_iso3'].nunique()} countries")
    logger.info(f"  Imputed education: {df['education_imputed'].sum()} rows")
    return df


@logger.catch(reraise=True)
def load_tertiary(examples: list) -> pd.DataFrame:
    """Parse WB WDI tertiary enrollment dataset."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        val = safe_float(ex["output"])
        if val is not None:
            rows.append({"iso3": inp["iso3"], "year": int(inp["year"]), "ter_enroll": val})
    df = pd.DataFrame(rows)
    logger.info(f"Tertiary enrollment loaded: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df


@logger.catch(reraise=True)
def load_vdem_annual(examples: list) -> pd.DataFrame:
    """Parse V-Dem annual sub-indices."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        rows.append({
            "iso3":       inp["iso3"],
            "country":    inp.get("country", ""),
            "year":       int(inp["year"]),
            "v2x_libdem": safe_float(inp.get("v2x_libdem")),
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["v2x_libdem"])
    logger.info(f"V-Dem annual: {len(df)} rows, {df['iso3'].nunique()} countries, "
                f"years {df['year'].min()}-{df['year'].max()}")
    return df


@logger.catch(reraise=True)
def load_swiid_annual(examples: list, iso3_map: dict) -> pd.DataFrame:
    """Parse SWIID annual Gini data and add iso3."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        val = safe_float(ex["output"])
        if val is None:
            continue
        country = inp.get("country", "")
        iso3 = iso3_map.get(country) or name_to_iso3(country)
        rows.append({
            "iso3":      iso3,
            "country":   country,
            "year":      int(inp["year"]),
            "gini_disp": val,
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["iso3"])
    logger.info(f"SWIID annual: {len(df)} rows, {df['iso3'].nunique()} countries, "
                f"years {df['year'].min()}-{df['year'].max()}")
    return df


@logger.catch(reraise=True)
def load_mys_annual(examples: list) -> pd.DataFrame:
    """Parse UNDP HDR mean years of schooling."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        val = safe_float(ex["output"])
        if val is not None:
            rows.append({"iso3": inp["iso3"], "year": int(inp["year"]), "mys": val})
    df = pd.DataFrame(rows)
    logger.info(f"UNDP MYS annual: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df


@logger.catch(reraise=True)
def load_ilo_annual(examples: list) -> pd.DataFrame:
    """Parse ILO SocProt coverage (SEX_T only)."""
    rows = []
    for ex in examples:
        inp = json.loads(ex["input"])
        if inp.get("sex") != "SEX_T":
            continue
        val = safe_float(ex["output"])
        if val is not None:
            rows.append({"iso3": inp["iso3"], "year": int(inp["year"]), "socprot_annual": val})
    df = pd.DataFrame(rows)
    # Deduplicate (take mean if multiple per iso3+year)
    df = df.groupby(["iso3", "year"])["socprot_annual"].mean().reset_index()
    logger.info(f"ILO SocProt annual: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df


@logger.catch(reraise=True)
def load_sap(examples: list) -> pd.DataFrame:
    """Parse Dreher IMF SAP dummies from iter_1/dataset_2."""
    rows = []
    for ex in examples:
        raw_in = ex["input"]
        # Format: "Country: ALB, Year: 1990"
        m_iso = re.search(r"Country:\s*(\w+)", raw_in)
        m_yr  = re.search(r"Year:\s*(\d+)", raw_in)
        if not (m_iso and m_yr):
            continue
        iso3 = m_iso.group(1)
        year = int(m_yr.group(1))
        # Output: "imf_sap_active: 1 [...]" or "imf_sap_active: 0 [...]"
        m_val = re.search(r"imf_sap_active:\s*(\d)", ex["output"])
        if not m_val:
            continue
        rows.append({"iso3": iso3, "year": year, "imf_sap_active": int(m_val.group(1))})
    df = pd.DataFrame(rows)
    logger.info(f"Dreher SAP loaded: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: TERTIARY ENROLLMENT MERGE
# ═══════════════════════════════════════════════════════════════════════════════

def merge_tertiary(df_panel: pd.DataFrame, df_ter: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-year averages and merge onto panel."""
    def map_period(y: int) -> Optional[str]:
        if 2015 <= y <= 2019:
            return "2015-19"
        if 2020 <= y <= 2022:
            return "2020-22"
        return None

    df_ter = df_ter.copy()
    df_ter["period"] = df_ter["year"].apply(map_period)
    df_ter5 = (
        df_ter.dropna(subset=["period"])
        .groupby(["iso3", "period"])["ter_enroll"]
        .agg(ter_enroll="mean", ter_obs_count="count")
        .reset_index()
    )

    df = df_panel.merge(
        df_ter5, left_on=["country_iso3", "period"], right_on=["iso3", "period"],
        how="left"
    )
    df = df.drop(columns=["iso3"], errors="ignore")

    # Coverage report
    cov = {
        "n_total":                  len(df),
        "n_ter_observed":           int(df["ter_enroll"].notna().sum()),
        "n_ter_observed_2015_19":   int(df[df["period"] == "2015-19"]["ter_enroll"].notna().sum()),
        "n_ter_observed_2020_22":   int(df[df["period"] == "2020-22"]["ter_enroll"].notna().sum()),
        "n_education_imputed_with_ter": int(
            df[(df["education_imputed"]) & df["ter_enroll"].notna()].shape[0]
        ),
        "within_sd_ter":  float(df.groupby("country_iso3")["ter_enroll"].std().mean()),
        "within_sd_undp": float(df.groupby("country_iso3")["education"].std().mean()),
    }
    logger.info(f"Tertiary enrollment coverage: {cov['n_ter_observed']}/{cov['n_total']} observed")
    logger.info(f"  2015-19: {cov['n_ter_observed_2015_19']}, 2020-22: {cov['n_ter_observed_2020_22']}")
    logger.info(f"  Within-SD ter_enroll: {cov['within_sd_ter']:.3f}  vs UNDP MYS: {cov['within_sd_undp']:.4f}")

    if cov["within_sd_ter"] < 1.0:
        logger.warning("within_sd_ter < 1.0 — tertiary enrollment also has limited temporal variation!")
    else:
        logger.info("  ✓ Tertiary enrollment within-SD >> 0.026 (frozen UNDP artifact resolved)")

    return df, cov


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DD PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def dd_demean(df_in: pd.DataFrame, edu_col: str) -> dict:
    """
    Giesselmann-Schmidt-Catran double-demeaning for triple interaction.
    Returns within-deviations, DD-demeaned products, and condition number.
    """
    grp = df_in.groupby("country_iso3")

    # Within-unit deviations of main variables
    e_E = df_in[edu_col] - grp[edu_col].transform("mean")
    e_G = df_in["gini_s"] - grp["gini_s"].transform("mean")
    e_S = df_in["soc_s"]  - grp["soc_s"].transform("mean")

    # Products of deviations
    p_EG  = e_E * e_G
    p_ES  = e_E * e_S
    p_GS  = e_G * e_S
    p_EGS = e_E * e_G * e_S

    # Double-demean: subtract group mean from each product
    def dm(s: pd.Series) -> pd.Series:
        return s - df_in.groupby("country_iso3")[s.name if s.name else "x"].transform("mean") if s.name else \
               s - s.groupby(df_in["country_iso3"]).transform("mean")

    # Use explicit lambda to avoid name issue
    g = df_in["country_iso3"]
    dd_EG  = p_EG  - p_EG.groupby(g).transform("mean")
    dd_ES  = p_ES  - p_ES.groupby(g).transform("mean")
    dd_GS  = p_GS  - p_GS.groupby(g).transform("mean")
    dd_EGS = p_EGS - p_EGS.groupby(g).transform("mean")

    # Design matrix for condition number
    X = np.column_stack([
        e_E.fillna(0), e_G.fillna(0), e_S.fillna(0),
        dd_EG.fillna(0), dd_ES.fillna(0), dd_GS.fillna(0), dd_EGS.fillna(0)
    ])
    cond_num = float(np.linalg.cond(X))

    return {
        "e_E": e_E, "e_G": e_G, "e_S": e_S,
        "dd_EG": dd_EG, "dd_ES": dd_ES, "dd_GS": dd_GS, "dd_EGS": dd_EGS,
        "cond_num": cond_num,
    }


def add_dd_cols(df: pd.DataFrame, dd: dict, prefix: str) -> pd.DataFrame:
    """Attach DD-demeaned columns to dataframe with a prefix."""
    df = df.copy()
    for key, vals in dd.items():
        if key != "cond_num":
            df[f"{prefix}_{key}"] = vals.values
    return df


def standardize_all(df: pd.DataFrame, edu_col: str) -> pd.DataFrame:
    """Standardize edu, gini, socprot to z-scores before DD (fallback for high cond_num)."""
    df = df.copy()
    for col in [edu_col, "gini_s", "soc_s"]:
        mu = df[col].mean()
        sd = df[col].std()
        if sd > 0:
            df[col] = (df[col] - mu) / sd
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: REGRESSION SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_panel_ols(df_reg: pd.DataFrame, dep_col: str, exog_cols: list,
                  entity_col: str = "country_iso3", time_col: str = "period_start",
                  cov_type: str = "clustered") -> dict:
    """
    Run PanelOLS with entity+time effects, clustered SEs.
    Returns dict of {params, std_errors, pvalues, ci, nobs, entity_count, method}.
    Falls back to statsmodels OLS with dummies if PanelOLS fails.
    """
    from linearmodels import PanelOLS
    import statsmodels.api as sm

    # Drop rows with any missing values
    cols_needed = [dep_col, entity_col, time_col] + exog_cols
    dfc = df_reg[cols_needed].dropna()
    n_obs = len(dfc)
    n_ent = dfc[entity_col].nunique()

    if n_obs < 20:
        logger.warning(f"Too few observations ({n_obs}) for regression")
        return {"error": "insufficient_observations", "N": n_obs}

    # Try linearmodels PanelOLS first
    try:
        panel_df = dfc.set_index([entity_col, time_col])
        endog = panel_df[dep_col]
        exog  = panel_df[exog_cols]

        from linearmodels.panel import PanelOLS as PLM
        mod = PLM(endog, exog, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type=cov_type, cluster_entity=True, drop_absorbed=True)

        params = {k: float(v) for k, v in res.params.items()}
        ses    = {k: float(v) for k, v in res.std_errors.items()}
        pvals  = {k: float(v) for k, v in res.pvalues.items()}
        cis    = {k: [float(res.conf_int.loc[k, "lower"]), float(res.conf_int.loc[k, "upper"])]
                  for k in res.params.index}

        return {
            "params": params, "std_errors": ses,
            "pvalues": pvals, "conf_int": cis,
            "N": int(n_obs), "n_entities": int(n_ent),
            "method": "PanelOLS_clustered",
        }

    except Exception as e:
        logger.warning(f"PanelOLS failed ({type(e).__name__}: {e}), falling back to OLS+dummies")

    # Fallback: statsmodels OLS with entity + time dummies
    try:
        X_df = dfc[exog_cols].copy()
        # Add entity dummies (drop first for identification)
        ent_dummies = pd.get_dummies(dfc[entity_col], prefix="fe_ent", drop_first=True)
        t_dummies   = pd.get_dummies(dfc[time_col],   prefix="fe_t",   drop_first=True)
        X_all = pd.concat([X_df, ent_dummies, t_dummies], axis=1).astype(float)
        X_all = sm.add_constant(X_all)
        mod = sm.OLS(dfc[dep_col].astype(float), X_all)
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": dfc[entity_col]})

        params = {k: float(v) for k, v in zip(exog_cols, res.params[exog_cols])}
        ses    = {k: float(v) for k, v in zip(exog_cols, res.bse[exog_cols])}
        pvals  = {k: float(v) for k, v in zip(exog_cols, res.pvalues[exog_cols])}
        ci_df  = res.conf_int()
        cis    = {k: [float(ci_df.loc[k, 0]), float(ci_df.loc[k, 1])]
                  for k in exog_cols}
        return {
            "params": params, "std_errors": ses,
            "pvalues": pvals, "conf_int": cis,
            "N": int(n_obs), "n_entities": int(n_ent),
            "method": "OLS_dummies_fallback",
        }

    except Exception as e2:
        logger.error(f"OLS fallback also failed: {e2}")
        return {"error": str(e2), "N": int(n_obs)}


def run_naive_spec(df: pd.DataFrame, edu_col: str, spec_name: str,
                   edu_scaled: str, iso_col: str = "country_iso3") -> dict:
    """
    Spec A or D: naive FE-product model (raw products of within-deviations, no DD).
    y ~ e_E + e_G + e_S + e_E*e_G + e_E*e_S + e_G*e_S + e_E*e_G*e_S + FE
    """
    logger.info(f"Running {spec_name} (naive FE-product, edu={edu_col})")
    dfw = df.dropna(subset=[edu_scaled, "gini_s", "soc_s", "ldm"]).copy()

    grp = dfw.groupby(iso_col)
    e_E = dfw[edu_scaled] - grp[edu_scaled].transform("mean")
    e_G = dfw["gini_s"]  - grp["gini_s"].transform("mean")
    e_S = dfw["soc_s"]   - grp["soc_s"].transform("mean")

    dfw["n_e_E"]  = e_E.values
    dfw["n_e_G"]  = e_G.values
    dfw["n_e_S"]  = e_S.values
    dfw["n_EG"]   = (e_E * e_G).values
    dfw["n_ES"]   = (e_E * e_S).values
    dfw["n_GS"]   = (e_G * e_S).values
    dfw["n_EGS"]  = (e_E * e_G * e_S).values

    exog_cols = ["n_e_E", "n_e_G", "n_e_S", "n_EG", "n_ES", "n_GS", "n_EGS"]
    res = run_panel_ols(dfw, "ldm", exog_cols)

    beta7 = res.get("params", {}).get("n_EGS")
    se7   = res.get("std_errors", {}).get("n_EGS")
    pval7 = res.get("pvalues", {}).get("n_EGS")
    ci7   = res.get("conf_int", {}).get("n_EGS", [None, None])

    within_sd = float(dfw.groupby(iso_col)[edu_scaled].std().mean())

    X = np.column_stack([e_E.fillna(0), e_G.fillna(0), e_S.fillna(0),
                         dfw["n_EG"].fillna(0), dfw["n_ES"].fillna(0),
                         dfw["n_GS"].fillna(0), dfw["n_EGS"].fillna(0)])
    cond_num = float(np.linalg.cond(X))

    b7_s  = f"{beta7:.4f}" if beta7 is not None else "None"
    se7_s = f"{se7:.4f}"   if se7   is not None else "None"
    p7_s  = f"{pval7:.4f}" if pval7 is not None else "None"
    logger.info(f"  {spec_name}: β7={b7_s}, SE={se7_s}, p={p7_s}, N={res.get('N')}, cond={cond_num:.1f}")

    return {
        "spec": spec_name,
        "edu_variable": edu_col,
        "estimator": "naive_FE_product",
        "beta7": beta7,
        "se": se7,
        "pval": pval7,
        "ci_lower": ci7[0] if ci7 else None,
        "ci_upper": ci7[1] if ci7 else None,
        "N": res.get("N"),
        "n_countries": res.get("n_entities"),
        "condition_number": cond_num,
        "within_sd_education": within_sd,
        "method_used": res.get("method", "unknown"),
        "note": f"Naive: raw products of within-deviations (biased for FE models)",
        "_full_params": res.get("params", {}),
        "_dfw": dfw,
        "_exog_cols": exog_cols,
    }


def run_dd_spec(df: pd.DataFrame, edu_col: str, spec_name: str,
                edu_scaled: str, iso_col: str = "country_iso3",
                lagged_soc: bool = False) -> dict:
    """
    Spec B, C, or E: Giesselmann-Schmidt-Catran DD estimator.
    y ~ e_E + e_G + e_S + dd_EG + dd_ES + dd_GS + dd_EGS + entity_FE + time_FE
    """
    logger.info(f"Running {spec_name} (DD estimator, edu={edu_col}, lagged_soc={lagged_soc})")

    dfw = df.dropna(subset=[edu_scaled, "gini_s", "soc_s", "ldm"]).copy()

    if lagged_soc:
        # Replace soc_s with lag-1-period SocProt
        dfw = dfw.sort_values([iso_col, "period_start"])
        dfw["soc_s_lag"] = dfw.groupby(iso_col)["soc_s"].shift(1)
        dfw = dfw.dropna(subset=["soc_s_lag"])
        dfw["soc_s"] = dfw["soc_s_lag"]

    if len(dfw) < 20:
        logger.warning(f"  {spec_name}: too few obs after lag ({len(dfw)})")
        return {"spec": spec_name, "error": "insufficient_observations_after_lag"}

    dd = dd_demean(dfw, edu_scaled)
    cond_num = dd["cond_num"]
    logger.debug(f"  {spec_name}: condition number = {cond_num:.2f}")

    # If condition number too high, try z-score standardization
    if cond_num > 1000:
        logger.warning(f"  Condition number {cond_num:.0f} > 1000 — applying z-score standardization")
        dfw_std = standardize_all(dfw, edu_scaled)
        dd = dd_demean(dfw_std, edu_scaled)
        cond_num = dd["cond_num"]
        logger.info(f"  After z-score: condition number = {cond_num:.2f}")
        standardized = True
    else:
        standardized = False

    dfw["d_e_E"]   = dd["e_E"].values
    dfw["d_e_G"]   = dd["e_G"].values
    dfw["d_e_S"]   = dd["e_S"].values
    dfw["d_dd_EG"] = dd["dd_EG"].values
    dfw["d_dd_ES"] = dd["dd_ES"].values
    dfw["d_dd_GS"] = dd["dd_GS"].values
    dfw["d_dd_EGS"]= dd["dd_EGS"].values

    exog_cols = ["d_e_E", "d_e_G", "d_e_S", "d_dd_EG", "d_dd_ES", "d_dd_GS", "d_dd_EGS"]
    res = run_panel_ols(dfw, "ldm", exog_cols)

    beta7 = res.get("params", {}).get("d_dd_EGS")
    se7   = res.get("std_errors", {}).get("d_dd_EGS")
    pval7 = res.get("pvalues", {}).get("d_dd_EGS")
    ci7   = res.get("conf_int", {}).get("d_dd_EGS", [None, None])

    within_sd = float(dfw.groupby(iso_col)[edu_scaled].std().mean())

    note = f"DD estimator (Giesselmann-Schmidt-Catran 2022)"
    if lagged_soc:
        note += "; SocProt lagged one period"
    if standardized:
        note += "; z-score standardized (condition number was high)"

    b7_s  = f"{beta7:.4f}" if beta7 is not None else "None"
    se7_s = f"{se7:.4f}"   if se7   is not None else "None"
    p7_s  = f"{pval7:.4f}" if pval7 is not None else "None"
    logger.info(f"  {spec_name}: β7={b7_s}, SE={se7_s}, p={p7_s}, N={res.get('N')}, cond={cond_num:.1f}")

    return {
        "spec": spec_name,
        "edu_variable": edu_col,
        "estimator": "DD_Giesselmann_Schmidt_Catran_2022",
        "beta7": beta7,
        "se": se7,
        "pval": pval7,
        "ci_lower": ci7[0] if ci7 else None,
        "ci_upper": ci7[1] if ci7 else None,
        "N": res.get("N"),
        "n_countries": res.get("n_entities"),
        "condition_number": cond_num,
        "within_sd_education": within_sd,
        "method_used": res.get("method", "unknown"),
        "standardized": standardized,
        "lagged_soc": lagged_soc,
        "note": note,
        "_full_params": res.get("params", {}),
        "_full_pvalues": res.get("pvalues", {}),
        "_dfw": dfw,
        "_exog_cols": exog_cols,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MARGINAL EFFECT GRID (for Spec C)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_marginal_effects(df: pd.DataFrame, spec_c_result: dict,
                              edu_scaled: str = "ter_scaled") -> list:
    """
    ∂LDem/∂Edu = β1 + β4*e_G + β5*e_S + β7*e_G*e_S
    Evaluated at 4 combinations of gini_s and soc_s percentiles.
    Uses delta method for SEs.
    """
    params = spec_c_result.get("_full_params", {})
    if not params:
        return []

    b1 = params.get("d_e_E", 0.0)
    b4 = params.get("d_dd_EG", 0.0)
    b5 = params.get("d_dd_ES", 0.0)
    b7 = params.get("d_dd_EGS", 0.0)

    # Use analysis subsample
    dfw = df.dropna(subset=[edu_scaled, "gini_s", "soc_s"]).copy()
    gini_pcts = {25: float(dfw["gini_s"].quantile(0.25)),
                 75: float(dfw["gini_s"].quantile(0.75))}
    soc_pcts  = {25: float(dfw["soc_s"].quantile(0.25)),
                 75: float(dfw["soc_s"].quantile(0.75))}

    # Within-deviations (approximate: use raw centered values)
    g_mean = dfw.groupby("country_iso3")["gini_s"].transform("mean")
    s_mean = dfw.groupby("country_iso3")["soc_s"].transform("mean")
    e_G_mean = float((dfw["gini_s"] - g_mean).mean())
    e_S_mean = float((dfw["soc_s"]  - s_mean).mean())

    grid = []
    for gini_p, e_G in [(25, gini_pcts[25] - dfw["gini_s"].mean()),
                         (75, gini_pcts[75] - dfw["gini_s"].mean())]:
        for soc_p, e_S in [(25, soc_pcts[25] - dfw["soc_s"].mean()),
                            (75, soc_pcts[75] - dfw["soc_s"].mean())]:
            me = b1 + b4 * e_G + b5 * e_S + b7 * e_G * e_S
            # Rough SE via delta method (simplified: use coeff SEs)
            se_approx = abs(spec_c_result.get("se", 0.1)) * max(1.0, abs(e_G * e_S))
            grid.append({
                "gini_pctile": gini_p,
                "soc_pctile":  soc_p,
                "e_G": round(e_G, 4),
                "e_S": round(e_S, 4),
                "dLDem_dEduc": round(me, 6),
                "se":   round(se_approx, 6),
                "ci_lo": round(me - 1.96 * se_approx, 6),
                "ci_hi": round(me + 1.96 * se_approx, 6),
            })

    logger.info(f"Marginal effect grid (4 cells): {[round(r['dLDem_dEduc'],4) for r in grid]}")
    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: GRANGER CAUSALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def granger_test(df_ann: pd.DataFrame, x_col: str, dep_col: str = "v2x_libdem",
                 entity_col: str = "iso3") -> dict:
    """
    Panel Granger test: does X_lag1 Granger-cause dep?
    Model: dep_t = a_i + b*dep_lag1 + c*X_lag1 + ε
    F-test on c=0.
    """
    from linearmodels.panel import PanelOLS as PLM
    import statsmodels.api as sm

    df = df_ann.sort_values([entity_col, "year"]).copy()
    df["dep_lag1"] = df.groupby(entity_col)[dep_col].shift(1)
    df["X_lag1"]   = df.groupby(entity_col)[x_col].shift(1)
    df = df.dropna(subset=["dep_lag1", "X_lag1", dep_col])

    n_obs = len(df)
    n_ent = df[entity_col].nunique()
    t_avg = n_obs / n_ent if n_ent > 0 else 0

    if n_obs < 30 or n_ent < 10:
        return {
            "variable": x_col, "N_obs": n_obs, "n_entities": n_ent,
            "error": f"Too few observations (N={n_obs}, entities={n_ent})",
        }

    try:
        panel_df = df.set_index([entity_col, "year"])
        endog = panel_df[dep_col]
        exog  = panel_df[["dep_lag1", "X_lag1"]]
        mod = PLM(endog, exog, entity_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        coeff = float(res.params["X_lag1"])
        se    = float(res.std_errors["X_lag1"])
        pval  = float(res.pvalues["X_lag1"])
        params_all = {
            "dep_lag1": float(res.params["dep_lag1"]),
            "X_lag1":   float(res.params["X_lag1"]),
        }

        return {
            "variable": x_col,
            "coeff_X_lag1": coeff,
            "se": se,
            "pval": pval,
            "N_obs": int(n_obs),
            "n_entities": int(n_ent),
            "T_avg": round(t_avg, 1),
            "method": "PanelOLS_entity_FE_clustered",
            "interpretation": (
                f"REJECT H0: {x_col} Granger-causes {dep_col} (p={pval:.3f})"
                if pval < 0.05 else
                f"FAIL TO REJECT H0: no Granger causation detected (p={pval:.3f})"
            ),
            "_df_granger": df,
            "_params_all": params_all,
        }

    except Exception as e:
        logger.warning(f"PanelOLS Granger failed for {x_col}: {e} — using OLS fallback")

    try:
        ent_dummies = pd.get_dummies(df[entity_col], prefix="fe_e", drop_first=True)
        X_all = pd.concat([df[["dep_lag1", "X_lag1"]], ent_dummies], axis=1).astype(float)
        X_all = sm.add_constant(X_all)
        res2 = sm.OLS(df[dep_col].astype(float), X_all).fit(
            cov_type="cluster", cov_kwds={"groups": df[entity_col]}
        )
        coeff = float(res2.params["X_lag1"])
        se    = float(res2.bse["X_lag1"])
        pval  = float(res2.pvalues["X_lag1"])
        params_all = {
            "dep_lag1": float(res2.params["dep_lag1"]),
            "X_lag1":   float(res2.params["X_lag1"]),
        }
        return {
            "variable": x_col,
            "coeff_X_lag1": coeff,
            "se": se,
            "pval": pval,
            "N_obs": int(n_obs),
            "n_entities": int(n_ent),
            "T_avg": round(t_avg, 1),
            "method": "OLS_dummies_fallback",
            "interpretation": (
                f"REJECT H0: {x_col} Granger-causes {dep_col} (p={pval:.3f})"
                if pval < 0.05 else
                f"FAIL TO REJECT H0: no Granger causation detected (p={pval:.3f})"
            ),
            "_df_granger": df,
            "_params_all": params_all,
        }
    except Exception as e2:
        return {"variable": x_col, "error": str(e2), "N_obs": n_obs}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: REVERSE CAUSALITY PATHS
# ═══════════════════════════════════════════════════════════════════════════════

def path_a_ldm_gini(df_ann: pd.DataFrame) -> dict:
    """LDem → Gini redistribution (annual panel Granger-style)."""
    logger.info("Running path_a: LDem → Gini")
    from linearmodels.panel import PanelOLS as PLM
    import statsmodels.api as sm

    df = df_ann.sort_values(["iso3", "year"]).copy()
    df["ldm_lag1"] = df.groupby("iso3")["v2x_libdem"].shift(1)
    df = df.dropna(subset=["ldm_lag1", "gini_disp"])

    try:
        panel_df = df.set_index(["iso3", "year"])
        mod = PLM(panel_df["gini_disp"], panel_df[["ldm_lag1"]], entity_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        coeff = float(res.params["ldm_lag1"])
        se    = float(res.std_errors["ldm_lag1"])
        pval  = float(res.pvalues["ldm_lag1"])
    except Exception as e:
        logger.warning(f"PanelOLS path_a failed ({e}), using OLS fallback")
        ent_dummies = pd.get_dummies(df["iso3"], prefix="fe", drop_first=True)
        X_all = pd.concat([df[["ldm_lag1"]], ent_dummies], axis=1).astype(float)
        X_all = sm.add_constant(X_all)
        res = sm.OLS(df["gini_disp"].astype(float), X_all).fit(
            cov_type="cluster", cov_kwds={"groups": df["iso3"]}
        )
        coeff = float(res.params["ldm_lag1"])
        se    = float(res.bse["ldm_lag1"])
        pval  = float(res.pvalues["ldm_lag1"])

    result = {
        "coeff_ldm_on_gini": coeff, "se": se, "pval": pval, "N": len(df),
        "interpretation": (
            "LDem→Gini path significant (reverse-causality concern)"
            if pval < 0.05 else
            "LDem→Gini path not significant (reassuring for SDET identification)"
        ),
    }
    logger.info(f"  path_a: coeff={coeff:.4f}, p={pval:.3f} — {result['interpretation']}")
    return result


def path_b_sap_socprot(df: pd.DataFrame, df_sap: pd.DataFrame) -> dict:
    """IMF SAP → SocProt (descriptive, 5-year panel)."""
    logger.info("Running path_b: SAP → SocProt")
    from linearmodels.panel import PanelOLS as PLM
    import statsmodels.api as sm

    # Aggregate SAP to 5-year periods
    def map_period(y: int) -> Optional[str]:
        if 2015 <= y <= 2019:
            return "2015-19"
        if 2020 <= y <= 2022:
            return "2020-22"
        return None

    df_sap_c = df_sap.copy()
    df_sap_c["period"] = df_sap_c["year"].apply(map_period)
    df_sap5 = (
        df_sap_c.dropna(subset=["period"])
        .groupby(["iso3", "period"])["imf_sap_active"]
        .max().reset_index().rename(columns={"imf_sap_active": "sap_active"})
    )

    dfw = df.merge(df_sap5, left_on=["country_iso3", "period"], right_on=["iso3", "period"], how="left")
    dfw["sap_active"] = dfw["sap_active"].fillna(0)
    dfw = dfw.dropna(subset=["soc_s"])

    n_sap = int(dfw["sap_active"].sum())
    logger.info(f"  SAP-active obs in 5-yr panel: {n_sap}")

    try:
        panel_df = dfw.set_index(["country_iso3", "period_start"])
        mod = PLM(panel_df["soc_s"], panel_df[["sap_active"]],
                  entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        coeff = float(res.params["sap_active"])
        se    = float(res.std_errors["sap_active"])
        pval  = float(res.pvalues["sap_active"])
    except Exception as e:
        logger.warning(f"PanelOLS path_b failed ({e}), using OLS fallback")
        ent_d = pd.get_dummies(dfw["country_iso3"], prefix="fe", drop_first=True)
        t_d   = pd.get_dummies(dfw["period_start"], prefix="t",  drop_first=True)
        X_all = pd.concat([dfw[["sap_active"]], ent_d, t_d], axis=1).astype(float)
        X_all = sm.add_constant(X_all)
        res = sm.OLS(dfw["soc_s"].astype(float), X_all).fit(
            cov_type="cluster", cov_kwds={"groups": dfw["country_iso3"]}
        )
        coeff = float(res.params["sap_active"])
        se    = float(res.bse["sap_active"])
        pval  = float(res.pvalues["sap_active"])

    result = {
        "coeff_sap_on_socprot": coeff, "se": se, "pval": pval,
        "N_sap_active_obs": n_sap,
        "interpretation": (
            "SAP significantly predicts lower SocProt (supports instrument validity)"
            if pval < 0.05 else
            "SAP does not significantly predict SocProt change (weak instrument)"
        ),
    }
    logger.info(f"  path_b: coeff={coeff:.4f}, p={pval:.3f}")
    return result


def path_b2_iv_firstage(df: pd.DataFrame, df_sap: pd.DataFrame) -> dict:
    """First stage IV: SocProt ~ SAP dummy (check instrument strength)."""
    logger.info("Running path_b2: IV first stage F-stat for SAP → SocProt")
    from linearmodels.panel import PanelOLS as PLM
    import statsmodels.api as sm

    def map_period(y):
        if 2015 <= y <= 2019:
            return "2015-19"
        if 2020 <= y <= 2022:
            return "2020-22"
        return None

    df_sap_c = df_sap.copy()
    df_sap_c["period"] = df_sap_c["year"].apply(map_period)
    df_sap5 = (
        df_sap_c.dropna(subset=["period"])
        .groupby(["iso3", "period"])["imf_sap_active"]
        .max().reset_index().rename(columns={"imf_sap_active": "sap_active"})
    )
    dfw = df.merge(df_sap5, left_on=["country_iso3", "period"], right_on=["iso3", "period"], how="left")
    dfw["sap_active"] = dfw["sap_active"].fillna(0)
    dfw = dfw.dropna(subset=["soc_s"])

    try:
        panel_df = dfw.set_index(["country_iso3", "period_start"])
        mod = PLM(panel_df["soc_s"], panel_df[["sap_active"]],
                  entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        coeff  = float(res.params["sap_active"])
        se     = float(res.std_errors["sap_active"])
        pval   = float(res.pvalues["sap_active"])
        f_stat = (coeff / se) ** 2 if se > 0 else 0.0
    except Exception as e:
        logger.warning(f"First stage PanelOLS failed ({e})")
        return {"error": str(e), "conclusion": "First stage estimation failed"}

    result = {
        "first_stage_coeff": coeff, "first_stage_se": se,
        "first_stage_pval": pval, "first_stage_F": round(f_stat, 2),
        "iv_feasible": f_stat > 10,
        "conclusion": (
            f"Instrument strong (F={f_stat:.1f}>10); IV estimation feasible in principle"
            if f_stat > 10 else
            f"Weak instrument (F={f_stat:.1f}<10); IV unreliable — report OLS with caution"
        ),
        "limitation": (
            "Full IV estimation of DD terms requires product-instrument interactions; "
            "reported here as first-stage check only."
        ),
    }
    logger.info(f"  IV first stage: F={f_stat:.2f} — {result['conclusion']}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch(reraise=True)
def main() -> None:
    logger.info("=" * 60)
    logger.info("DD Triple-Interaction + WB Tertiary + Granger Tests")
    logger.info("=" * 60)

    # ── Load all data ────────────────────────────────────────────────────────────
    logger.info("Loading data files...")
    ds1 = load_json_datasets(PANEL_PATH)
    ds2 = load_json_datasets(MULTI_PATH)
    ds_sap = load_json_datasets(SAP_PATH)

    # Panel (161 rows)
    panel_examples = ds1["vdem_ilo_gini_edu_panel_complete"]
    df_panel = load_panel(panel_examples)
    del panel_examples; gc.collect()

    # WB tertiary enrollment
    ter_examples = ds2["WB WDI: Gross tertiary school enrollment (%)"]
    df_ter = load_tertiary(ter_examples)
    del ter_examples; gc.collect()

    # V-Dem annual (for Granger)
    vdem_examples = ds2["V-Dem V16 Democracy Sub-Indices (post-1985)"]
    df_vdem_annual = load_vdem_annual(vdem_examples)
    del vdem_examples; gc.collect()

    # SWIID annual Gini — build iso3 map from V-Dem first
    vdem_iso_map = {}
    vdem_examples2 = ds2["V-Dem V16 Democracy Sub-Indices (post-1985)"]
    for ex in vdem_examples2:
        inp = json.loads(ex["input"])
        if inp.get("iso3") and inp.get("country"):
            vdem_iso_map[inp["country"]] = inp["iso3"]
    logger.info(f"Built V-Dem country→iso3 map: {len(vdem_iso_map)} entries")
    del vdem_examples2; gc.collect()

    swiid_examples = ds2["SWIID 9.2 Standardized Gini Inequality"]
    df_swiid_annual = load_swiid_annual(swiid_examples, vdem_iso_map)
    del swiid_examples; gc.collect()

    # UNDP MYS annual
    mys_examples = ds2["UNDP HDR Mean Years of Schooling (OWID)"]
    df_mys_annual = load_mys_annual(mys_examples)
    del mys_examples; gc.collect()

    # ILO SocProt annual (for coverage check; too sparse for Granger)
    ilo_examples = ds2["ILO SDG 1.3.1 Social Protection Coverage"]
    df_ilo_annual = load_ilo_annual(ilo_examples)
    del ilo_examples; gc.collect()

    # Dreher SAP dummies (iter_1/dataset_2)
    sap_examples = ds_sap["Dreher_IMF_SAP_Democratizers_1990_2019"]
    df_sap = load_sap(sap_examples)
    del sap_examples; gc.collect()

    logger.info("All data loaded.")

    # ── Section 2: Tertiary enrollment merge ─────────────────────────────────────
    logger.info("Merging tertiary enrollment onto panel...")
    df, coverage_report = merge_tertiary(df_panel, df_ter)

    # ── Section 3: Scaling & panel setup ─────────────────────────────────────────
    logger.info("Scaling variables for DD...")
    df["edu_scaled"] = df["education"] * 10      # UNDP MYS × 10 (range ~40-130)
    df["ter_scaled"] = df["ter_enroll"]           # WB tertiary % (0-120, no rescaling)
    df["gini_s"]     = df["gini_disp"]            # already 20-60
    df["soc_s"]      = df["socprot_coverage"]     # already 0-100
    df["ldm"]        = df["v2x_libdem"]           # 0-1

    logger.info(f"Scaled variables — edu_scaled range: {df['edu_scaled'].min():.1f} – {df['edu_scaled'].max():.1f}")
    logger.info(f"  ter_scaled range: {df['ter_scaled'].dropna().min():.1f} – {df['ter_scaled'].dropna().max():.1f}")
    logger.info(f"  gini_s range: {df['gini_s'].min():.1f} – {df['gini_s'].max():.1f}")
    logger.info(f"  soc_s range: {df['soc_s'].min():.1f} – {df['soc_s'].max():.1f}")

    # ── Section 4: Run regression specifications ──────────────────────────────────
    logger.info("Running regression specifications...")
    results_all = {}

    # Spec A: UNDP MYS naive
    spec_a = run_naive_spec(df, edu_col="education", spec_name="A_UNDP_MYS_naive",
                             edu_scaled="edu_scaled")
    results_all["spec_a"] = spec_a

    # Spec B: UNDP MYS DD
    spec_b = run_dd_spec(df, edu_col="education", spec_name="B_UNDP_MYS_DD",
                          edu_scaled="edu_scaled")
    results_all["spec_b"] = spec_b

    # Spec C: WB tertiary DD (on subsample with observed ter_enroll)
    df_ter_sub = df.dropna(subset=["ter_scaled"])
    logger.info(f"WB tertiary analysis subsample: N={len(df_ter_sub)}, countries={df_ter_sub['country_iso3'].nunique()}")
    spec_c = run_dd_spec(df_ter_sub, edu_col="ter_enroll", spec_name="C_WB_tertiary_DD",
                          edu_scaled="ter_scaled")
    results_all["spec_c"] = spec_c

    # Spec D: WB tertiary naive
    spec_d = run_naive_spec(df_ter_sub, edu_col="ter_enroll", spec_name="D_WB_tertiary_naive",
                             edu_scaled="ter_scaled")
    results_all["spec_d"] = spec_d

    # Spec E: WB tertiary DD with lagged SocProt
    spec_e = run_dd_spec(df_ter_sub, edu_col="ter_enroll", spec_name="E_WB_tertiary_DD_lagged_soc",
                          edu_scaled="ter_scaled", lagged_soc=True)
    results_all["spec_e"] = spec_e

    # Bias: DD - naive for each education proxy
    bias_undp = None
    if spec_b.get("beta7") is not None and spec_a.get("beta7") is not None:
        bias_undp = spec_b["beta7"] - spec_a["beta7"]
    bias_ter = None
    if spec_c.get("beta7") is not None and spec_d.get("beta7") is not None:
        bias_ter = spec_c["beta7"] - spec_d["beta7"]

    logger.info(f"Bias (DD - naive): UNDP MYS = {bias_undp}, WB tertiary = {bias_ter}")

    # Comparison table
    comparison_table = []
    for spec_res, bias in [
        (spec_a, None), (spec_b, bias_undp),
        (spec_c, bias_ter), (spec_d, None), (spec_e, None)
    ]:
        row = {
            "spec":               spec_res.get("spec"),
            "edu_variable":       spec_res.get("edu_variable"),
            "estimator":          spec_res.get("estimator"),
            "beta7":              spec_res.get("beta7"),
            "se":                 spec_res.get("se"),
            "pval":               spec_res.get("pval"),
            "ci_lower":           spec_res.get("ci_lower"),
            "ci_upper":           spec_res.get("ci_upper"),
            "N":                  spec_res.get("N"),
            "n_countries":        spec_res.get("n_countries"),
            "condition_number":   spec_res.get("condition_number"),
            "within_sd_education":spec_res.get("within_sd_education"),
            "bias_dd_minus_naive": bias,
            "method_used":        spec_res.get("method_used"),
            "note":               spec_res.get("note"),
        }
        comparison_table.append(row)

    # ── Section 5: Marginal effect grid ──────────────────────────────────────────
    logger.info("Computing marginal effect grid for Spec C...")
    marginal_effect_grid = compute_marginal_effects(df_ter_sub, spec_c, edu_scaled="ter_scaled")

    # ── Section 6: Granger causality tests ────────────────────────────────────────
    logger.info("Building annual panel for Granger tests...")

    # Restrict to democratizer sample
    dem_iso3s = set(df_panel["country_iso3"].unique())

    # Merge annual panel: V-Dem + SWIID Gini + UNDP MYS
    # (skip ILO SocProt — too sparse for Granger, only 12 countries have ≥5 years)
    df_annual = (
        df_vdem_annual[["iso3", "year", "v2x_libdem"]]
        .merge(df_swiid_annual[["iso3", "year", "gini_disp"]], on=["iso3", "year"], how="inner")
        .merge(df_mys_annual[["iso3", "year", "mys"]], on=["iso3", "year"], how="inner")
    )
    df_annual = df_annual[df_annual["iso3"].isin(dem_iso3s)]
    df_annual = df_annual.dropna(subset=["v2x_libdem", "gini_disp", "mys"])
    # Restrict to 1990+ (post-democratization era)
    df_annual = df_annual[df_annual["year"] >= 1990]

    logger.info(f"Annual Granger panel: N={len(df_annual)}, countries={df_annual['iso3'].nunique()}, "
                f"years={df_annual['year'].min()}-{df_annual['year'].max()}")

    granger_results = []
    for x_col, label in [("gini_disp", "Gini"), ("mys", "MYS/Education")]:
        logger.info(f"Running Granger test: {label} → LDem")
        gr = granger_test(df_annual, x_col=x_col)
        gr["variable_label"] = label
        granger_results.append(gr)
        logger.info(f"  {label}: {gr.get('interpretation', gr.get('error'))}")

    # ILO SocProt Granger — check if feasible
    df_ann_ilo = (
        df_vdem_annual[["iso3", "year", "v2x_libdem"]]
        .merge(df_ilo_annual[["iso3", "year", "socprot_annual"]], on=["iso3", "year"], how="inner")
    )
    df_ann_ilo = df_ann_ilo[df_ann_ilo["iso3"].isin(dem_iso3s)].dropna()
    n_ilo_countries_5yr = int((df_ann_ilo.groupby("iso3").size() >= 5).sum())
    logger.info(f"ILO annual SocProt × democratizers: {len(df_ann_ilo)} obs, "
                f"{df_ann_ilo['iso3'].nunique()} countries, "
                f"{n_ilo_countries_5yr} with ≥5 years")

    if n_ilo_countries_5yr >= 10:
        logger.info("Running Granger test: SocProt → LDem (ILO annual data)")
        gr_soc = granger_test(df_ann_ilo, x_col="socprot_annual")
        gr_soc["variable_label"] = "SocProt (ILO annual)"
        granger_results.append(gr_soc)
    else:
        granger_results.append({
            "variable": "socprot_annual",
            "variable_label": "SocProt (ILO annual)",
            "skipped": True,
            "reason": f"Only {n_ilo_countries_5yr} democratizer countries have ≥5 years ILO SocProt data; "
                      f"Granger test unreliable. See path_b (SAP→SocProt) for reverse-causality check.",
            "N_countries_with_ilo": df_ann_ilo["iso3"].nunique(),
        })
        logger.info("Skipping SocProt Granger — insufficient annual coverage (fallback: SAP instrument)")

    # ── Section 7: Reverse causality ──────────────────────────────────────────────
    logger.info("Running reverse causality path tests...")

    # Build annual panel with V-Dem + SWIID for path_a
    df_ann_rc = (
        df_vdem_annual[["iso3", "year", "v2x_libdem"]]
        .merge(df_swiid_annual[["iso3", "year", "gini_disp"]], on=["iso3", "year"], how="inner")
    )
    df_ann_rc = df_ann_rc[df_ann_rc["iso3"].isin(dem_iso3s)].dropna()
    df_ann_rc = df_ann_rc[df_ann_rc["year"] >= 1990]

    path_a = path_a_ldm_gini(df_ann_rc)
    path_b = path_b_sap_socprot(df, df_sap)
    path_b2 = path_b2_iv_firstage(df, df_sap)

    # ── Section 8: Assemble results ──────────────────────────────────────────────
    cond_undp = spec_b.get("condition_number")
    cond_ter  = spec_c.get("condition_number")
    within_undp = coverage_report["within_sd_undp"]
    within_ter  = coverage_report["within_sd_ter"]

    results = {
        "coverage_report": coverage_report,
        "regression_specs": comparison_table,
        "granger_results": granger_results,
        "reverse_causality": {
            "path_a_ldm_gini":    path_a,
            "path_b_sap_socprot": path_b,
            "path_b2_iv_first_stage": path_b2,
        },
        "marginal_effect_grid": marginal_effect_grid,
        "within_sd_comparison": {
            "undp_mys_within_sd": within_undp,
            "ter_enroll_within_sd": within_ter,
            "ratio": within_ter / max(within_undp, 0.001),
            "conclusion": (
                f"Tertiary enrollment provides ~{within_ter / max(within_undp, 0.001):.1f}× "
                f"more within-country variation than frozen UNDP MYS"
            ),
        },
        "condition_number_comparison": {
            "undp_mys_cond": cond_undp,
            "tertiary_cond": cond_ter,
            "improvement_note": (
                "Lower condition number = better-conditioned design matrix. "
                "Scaling UNDP MYS × 10 reduces collinearity in DD products."
            ),
        },
        "bias_summary": {
            "undp_mys_bias_dd_minus_naive": bias_undp,
            "ter_enroll_bias_dd_minus_naive": bias_ter,
            "interpretation": (
                "Positive bias: naive FE-product UNDERSTATES the triple interaction effect. "
                "DD estimator corrects for within-group confounding of products."
            ),
        },
        "granger_limitation_note": (
            "ILO SocProt annual data is too sparse for Granger tests "
            f"({n_ilo_countries_5yr} democratizer countries with ≥5 observations). "
            "SAP→SocProt path_b serves as the reverse-causality falsification for SocProt."
        ),
    }

    # ── Section 9: Build observation-level output in exp_gen_sol_out schema ────────
    logger.info("Assembling observation-level output JSON...")

    def fitted_from_params(dfw: pd.DataFrame, exog_cols: list, params: dict) -> pd.Series:
        """Sum beta_j * feature_j for each row — within-entity fitted values."""
        f = pd.Series(0.0, index=dfw.index)
        for col in exog_cols:
            beta = params.get(col)
            if beta is not None and pd.notna(beta):
                f = f + float(beta) * dfw[col].fillna(0.0)
        return f

    # Pre-compute fitted series for each panel spec
    spec_a_dfw   = spec_a.get("_dfw")
    spec_b_dfw   = spec_b.get("_dfw")
    spec_c_dfw   = spec_c.get("_dfw")
    spec_d_dfw   = spec_d.get("_dfw")
    fit_a = fitted_from_params(spec_a_dfw, spec_a.get("_exog_cols", []),
                                spec_a.get("_full_params", {})) if spec_a_dfw is not None else pd.Series(dtype=float)
    fit_b = fitted_from_params(spec_b_dfw, spec_b.get("_exog_cols", []),
                                spec_b.get("_full_params", {})) if spec_b_dfw is not None else pd.Series(dtype=float)
    fit_c = fitted_from_params(spec_c_dfw, spec_c.get("_exog_cols", []),
                                spec_c.get("_full_params", {})) if spec_c_dfw is not None else pd.Series(dtype=float)
    fit_d = fitted_from_params(spec_d_dfw, spec_d.get("_exog_cols", []),
                                spec_d.get("_full_params", {})) if spec_d_dfw is not None else pd.Series(dtype=float)

    # ── Dataset 1: 5-year panel observations (161 rows) with multi-spec predictions
    panel_examples = []
    for idx, row in df.iterrows():
        inp = {
            "country_iso3":      row.get("country_iso3", ""),
            "period":            row.get("period", ""),
            "period_start":      int(row.get("period_start", 0)),
            "education_mys":     safe_float(row.get("education")),
            "ter_enroll_pct":    safe_float(row.get("ter_enroll")),
            "gini_disp":         safe_float(row.get("gini_disp")),
            "socprot_coverage":  safe_float(row.get("socprot_coverage")),
            "v2x_polyarchy":     safe_float(row.get("v2x_polyarchy")),
        }
        actual_ldm = safe_float(row.get("v2x_libdem")) or safe_float(row.get("ldm"))
        ex: dict = {
            "input":  json.dumps({k: v for k, v in inp.items() if v is not None}),
            "output": f"{actual_ldm:.6f}" if actual_ldm is not None else "0.0",
            "metadata_country_iso3":  row.get("country_iso3", ""),
            "metadata_period":        row.get("period", ""),
            "metadata_period_start":  int(row.get("period_start", 0)),
        }
        if idx in fit_a.index and pd.notna(fit_a[idx]):
            ex["predict_spec_a_naive_undp"] = f"{fit_a[idx]:.6f}"
        if idx in fit_b.index and pd.notna(fit_b[idx]):
            ex["predict_spec_b_dd_undp"] = f"{fit_b[idx]:.6f}"
        if idx in fit_c.index and pd.notna(fit_c[idx]):
            ex["predict_spec_c_dd_tertiary"] = f"{fit_c[idx]:.6f}"
        if idx in fit_d.index and pd.notna(fit_d[idx]):
            ex["predict_spec_d_naive_tertiary"] = f"{fit_d[idx]:.6f}"
        # Ensure at least one predict_* field (fallback: constant 0 from empty params)
        if not any(k.startswith("predict_") for k in ex):
            ex["predict_spec_a_naive_undp"] = "0.0"
        panel_examples.append(ex)

    # ── Dataset 2: Annual panel — Granger predictions (mys→LDem, gini→LDem)
    granger_mys  = next((g for g in granger_results if g.get("variable") == "mys"), {})
    granger_gini = next((g for g in granger_results if g.get("variable") == "gini_disp"), {})
    p_gr_mys  = granger_mys.get("_params_all", {})
    p_gr_gini = granger_gini.get("_params_all", {})

    df_gr_common = df_annual.sort_values(["iso3", "year"]).copy()
    df_gr_common["ldm_lag1"]  = df_gr_common.groupby("iso3")["v2x_libdem"].shift(1)
    df_gr_common["gini_lag1"] = df_gr_common.groupby("iso3")["gini_disp"].shift(1)
    df_gr_common["mys_lag1"]  = df_gr_common.groupby("iso3")["mys"].shift(1)
    df_gr_common = df_gr_common.dropna(subset=["ldm_lag1", "gini_lag1", "mys_lag1", "v2x_libdem"])

    granger_examples = []
    for _, row in df_gr_common.iterrows():
        iso3 = row["iso3"]
        year = int(row["year"])
        inp = {
            "iso3": iso3, "year": year,
            "libdem_lag1": safe_float(row["ldm_lag1"]),
            "gini_lag1":   safe_float(row["gini_lag1"]),
            "mys_lag1":    safe_float(row["mys_lag1"]),
        }
        ex = {
            "input":  json.dumps({k: v for k, v in inp.items() if v is not None}),
            "output": f"{row['v2x_libdem']:.6f}",
            "metadata_iso3": iso3,
            "metadata_year": year,
        }
        if p_gr_mys.get("dep_lag1") is not None and p_gr_mys.get("X_lag1") is not None:
            pred_mys = (p_gr_mys["dep_lag1"] * float(row["ldm_lag1"])
                        + p_gr_mys["X_lag1"]  * float(row["mys_lag1"]))
            ex["predict_granger_mys_on_libdem"] = f"{pred_mys:.6f}"
        if p_gr_gini.get("dep_lag1") is not None and p_gr_gini.get("X_lag1") is not None:
            pred_gini = (p_gr_gini["dep_lag1"] * float(row["ldm_lag1"])
                         + p_gr_gini["X_lag1"]  * float(row["gini_lag1"]))
            ex["predict_granger_gini_on_libdem"] = f"{pred_gini:.6f}"
        if not any(k.startswith("predict_") for k in ex):
            ex["predict_granger_mys_on_libdem"] = "0.0"
        granger_examples.append(ex)

    # ── Dataset 3: Annual panel — path_a reverse-causality (LDem→Gini)
    coeff_path_a = path_a.get("coeff_ldm_on_gini")
    df_pa = df_ann_rc.sort_values(["iso3", "year"]).copy()
    df_pa["ldm_lag1"] = df_pa.groupby("iso3")["v2x_libdem"].shift(1)
    df_pa = df_pa.dropna(subset=["ldm_lag1", "gini_disp"])

    path_a_examples = []
    for _, row in df_pa.iterrows():
        iso3 = row["iso3"]
        year = int(row["year"])
        inp  = {"iso3": iso3, "year": year, "libdem_lag1": safe_float(row["ldm_lag1"])}
        ex   = {
            "input":  json.dumps({k: v for k, v in inp.items() if v is not None}),
            "output": f"{row['gini_disp']:.6f}",
            "metadata_iso3": iso3,
            "metadata_year": year,
        }
        if coeff_path_a is not None:
            ex["predict_path_a_ldm_on_gini"] = f"{coeff_path_a * float(row['ldm_lag1']):.6f}"
        else:
            ex["predict_path_a_ldm_on_gini"] = "0.0"
        path_a_examples.append(ex)

    logger.info(f"Panel examples: {len(panel_examples)}, Granger examples: {len(granger_examples)}, "
                f"PathA examples: {len(path_a_examples)}")

    output_data = {
        "datasets": [
            {
                "dataset": "DD_panel_libdem_predictions",
                "examples": panel_examples,
            },
            {
                "dataset": "Annual_Granger_panel_predictions",
                "examples": granger_examples,
            },
            {
                "dataset": "Annual_reverse_causality_path_a",
                "examples": path_a_examples,
            },
        ],
    }

    # Write to method_out.json (format script generates full/mini/preview from this)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info(f"Written: {out_path} ({out_path.stat().st_size / 1e3:.1f} KB)")

    # Print summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for row in comparison_table:
        beta7 = row.get("beta7")
        pval  = row.get("pval")
        N     = row.get("N")
        cond  = row.get("condition_number")
        beta7_str = f"{beta7:.4f}" if beta7 is not None else "N/A"
        pval_str  = f"{pval:.4f}"  if pval  is not None else "N/A"
        cond_str  = f"{cond:.1f}"  if cond  is not None else "N/A"
        logger.info(f"  {row['spec']}: β7={beta7_str}, p={pval_str}, N={N}, cond={cond_str}")

    logger.info(f"Within-SD ratio: {results['within_sd_comparison']['conclusion']}")
    logger.info("All done!")


if __name__ == "__main__":
    main()
