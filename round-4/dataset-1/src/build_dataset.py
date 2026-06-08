#!/usr/bin/env python3
"""Build SDET DD panel dataset and generate exp_sel_data_out JSON.

Sources:
  - Polity5 p5v2018.xls: democracy/polity2, democratizer identification
  - SWIID swiid_summary.csv: Gini inequality
  - OWID V-Dem: regime_row_owid classification
  - OWID WB Education: school_enrollment__tertiary__pct_gross
  - OWID ILOSTAT: sdg_1_3_1_population_covered_by_social_protection
"""

import sys
import json
import math
import resource
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add("logs/build_dataset.log", rotation="30 MB", level="DEBUG")

# ── RAM guard ────────────────────────────────────────────────────────────────
RAM_BUDGET = 20 * 1024**3  # 20 GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

WORKSPACE = Path(__file__).parent
DATA_DIR = WORKSPACE / "temp" / "datasets"
OWID_DIR = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables")
OUT_DIR = WORKSPACE

PERIODS = [
    ("1990-94", 1990, 1994),
    ("1995-99", 1995, 1999),
    ("2000-04", 2000, 2004),
    ("2005-09", 2005, 2009),
    ("2010-14", 2010, 2014),
    ("2015-19", 2015, 2019),
    ("2020-22", 2020, 2022),
]


def assign_period(year: int) -> str | None:
    for label, start, end in PERIODS:
        if start <= year <= end:
            return label
    return None


def _round(v: Any, n: int = 3) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, float):
        return round(v, n)
    return v


# ── Load Polity5 ─────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def load_polity5() -> pd.DataFrame:
    path = DATA_DIR / "polity5_p5v2018.xls"
    logger.info(f"Loading Polity5 from {path}")
    df = pd.read_excel(path, engine="xlrd")
    df = df[df["year"] >= 1985].copy()
    df["polity2"] = pd.to_numeric(df["polity2"], errors="coerce")
    df["democ"] = pd.to_numeric(df["democ"], errors="coerce")
    df["autoc"] = pd.to_numeric(df["autoc"], errors="coerce")
    # Keep only needed cols
    df = df[["country", "scode", "year", "polity2", "democ", "autoc", "durable"]].copy()
    df["country_polity"] = df["country"].str.strip()
    logger.info(f"Polity5: {len(df)} rows, {df['country'].nunique()} countries (1985+)")
    return df


# ── Load SWIID ───────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def load_swiid() -> pd.DataFrame:
    path = DATA_DIR / "swiid_summary.csv"
    logger.info(f"Loading SWIID from {path}")
    df = pd.read_csv(path)
    df = df[df["year"] >= 1985].copy()
    df["country_swiid"] = df["country"].str.strip()
    logger.info(f"SWIID: {len(df)} rows, {df['country'].nunique()} countries")
    return df


# ── Load OWID V-Dem ──────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def load_vdem() -> pd.DataFrame:
    path = OWID_DIR / "full_garden_democracy_2025-03-17_vdem_vdem_uni_without_regions.json"
    logger.info(f"Loading OWID V-Dem from {path} ({path.stat().st_size / 1e6:.1f}MB)")
    df = pd.read_json(path)
    df = df[df["year"] >= 1985].copy()
    df = df[["country", "year", "regime_row_owid"]].copy()
    df["country_vdem"] = df["country"].str.strip()
    logger.info(f"OWID V-Dem: {len(df)} rows, {df['country'].nunique()} countries")
    return df


# ── Load OWID WB Education ───────────────────────────────────────────────────

@logger.catch(reraise=True)
def load_education() -> pd.DataFrame:
    path = OWID_DIR / "full_garden_wb_2023-07-10_education_education.json"
    logger.info(f"Loading OWID WB Education from {path} ({path.stat().st_size / 1e6:.1f}MB)")
    df = pd.read_json(path)
    df = df[df["year"] >= 1985].copy()
    df = df[["country", "year", "school_enrollment__tertiary__pct_gross"]].copy()
    df = df.rename(columns={"school_enrollment__tertiary__pct_gross": "tertiary_enroll"})
    df["tertiary_enroll"] = pd.to_numeric(df["tertiary_enroll"], errors="coerce")
    df["country_educ"] = df["country"].str.strip()
    logger.info(f"OWID Education: {len(df)} rows, {df['country'].nunique()} countries")
    return df


# ── Load OWID ILOSTAT (social protection) ────────────────────────────────────

@logger.catch(reraise=True)
def load_ilostat() -> pd.DataFrame:
    path = OWID_DIR / "full_garden_un_2025-08-12_ilostat_ilostat.json"
    logger.info(f"Loading OWID ILOSTAT from {path} ({path.stat().st_size / 1e6:.1f}MB)")
    # Read only needed columns to save memory
    df = pd.read_json(path)
    col = "sdg_1_3_1_population_covered_by_social_protection"
    df = df[df["sex"] == "Total"].copy()
    df = df[df[col].notna()].copy()
    df = df[["country", "year", col, "classif1"]].copy()
    df = df.rename(columns={col: "socprot_rate"})
    # Prefer "Age (youth, adults): 15+" or take first non-null per country-year
    df["country_ilo"] = df["country"].str.strip()
    # Group by country-year to get single rate (mean across classif1 if multiple)
    df = df.groupby(["country_ilo", "year"])["socprot_rate"].mean().reset_index()
    logger.info(f"ILOSTAT social protection: {len(df)} rows, {df['country_ilo'].nunique()} countries")
    return df


# ── Build country name → ISO3 mapping ────────────────────────────────────────

def build_name_map() -> dict[str, str]:
    """Manual mapping of common name variants to ISO3."""
    import pycountry

    manual: dict[str, str] = {
        "United States": "USA", "United Kingdom": "GBR", "Russia": "RUS",
        "South Korea": "KOR", "North Korea": "PRK", "Czech Republic": "CZE",
        "Czechia": "CZE", "Slovakia": "SVK", "Bosnia": "BIH",
        "Bosnia and Herzegovina": "BIH", "Macedonia": "MKD",
        "North Macedonia": "MKD", "Serbia": "SRB", "Kosovo": "XKX",
        "Congo": "COD", "DR Congo": "COD",
        "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG",
        "Democratic Republic of Congo": "COD",
        "Timor-Leste": "TLS", "East Timor": "TLS",
        "Gambia": "GMB", "The Gambia": "GMB",
        "Moldova": "MDA", "Iran": "IRN",
        "Syria": "SYR", "Venezuela": "VEN",
        "Tanzania": "TZA", "Bolivia": "BOL",
        "Vietnam": "VNM", "Laos": "LAO",
        "Ivory Coast": "CIV", "Cote d'Ivoire": "CIV",
        "Cape Verde": "CPV", "Cabo Verde": "CPV",
        "Palestine": "PSE", "West Bank and Gaza": "PSE",
        "Hong Kong": "HKG", "Macao": "MAC",
        "Kyrgyzstan": "KGZ", "Kyrgyz Republic": "KGZ",
        "Eswatini": "SWZ", "Swaziland": "SWZ",
        "Trinidad and Tobago": "TTO",
        "Slovak Republic": "SVK",
        "Egypt, Arab Rep.": "EGY", "Egypt": "EGY",
        "Yemen, Rep.": "YEM", "Yemen": "YEM",
        "Korea, Rep.": "KOR", "Korea, Dem. Rep.": "PRK",
        "Guinea-Bissau": "GNB",
        "Equatorial Guinea": "GNQ",
        "Burkina Faso": "BFA",
        "Central African Republic": "CAF",
        "Dominican Republic": "DOM",
    }

    def lookup(name: str) -> str | None:
        name = name.strip()
        if name in manual:
            return manual[name]
        try:
            c = pycountry.countries.lookup(name)
            return c.alpha_3
        except LookupError:
            return None

    return lookup


# ── Identify post-1990 democratizers via Polity5 ─────────────────────────────

def identify_democratizers(polity: pd.DataFrame) -> dict[str, int]:
    """Return {scode: transition_year} for post-1990 democratizers.

    Criterion: polity2 ≤ 0 in some year in [1985,1994] AND polity2 ≥ 6 in
    some year in [1990,2010], with the transition happening after 1989.
    """
    results: dict[str, int] = {}
    for scode, grp in polity.groupby("scode"):
        grp = grp.sort_values("year")
        # Was autocratic before 1995?
        pre = grp[grp["year"] <= 1994]
        if pre.empty or pre["polity2"].dropna().empty:
            continue
        min_pre = pre["polity2"].dropna().min()
        if min_pre > 0:
            continue  # Never truly autocratic

        # Did it democratize by 2010?
        post = grp[(grp["year"] >= 1990) & (grp["year"] <= 2010)]
        if post.empty or post["polity2"].dropna().empty:
            continue
        dem_years = post[post["polity2"] >= 6]["year"]
        if dem_years.empty:
            continue

        # Transition year: first year polity2 crossed from ≤0 to ≥6
        transition_year = None
        prev_val = None
        for _, row in grp[grp["year"] >= 1988].sort_values("year").iterrows():
            cur = row["polity2"]
            if pd.isna(cur):
                continue
            if prev_val is not None and prev_val <= 0 and cur >= 6:
                transition_year = int(row["year"])
                break
            prev_val = cur

        if transition_year is None:
            transition_year = int(dem_years.min())

        if transition_year >= 1990:
            results[str(scode)] = transition_year
        elif transition_year < 1990 and int(dem_years.min()) >= 1990:
            results[str(scode)] = int(dem_years.min())

    logger.info(f"Identified {len(results)} post-1990 democratizers via Polity5")
    return results


# ── Aggregate to 5-year periods ───────────────────────────────────────────────

def period_mean(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    df = df.copy()
    df["period"] = df["year"].apply(assign_period)
    df = df[df["period"].notna()].copy()
    return (
        df.groupby(group_cols + ["period"])[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: f"{value_col}_mean"})
    )


# ── Build merged panel ────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def build_panel(
    polity: pd.DataFrame,
    swiid: pd.DataFrame,
    vdem: pd.DataFrame,
    educ: pd.DataFrame,
    ilo: pd.DataFrame,
    democratizers: dict[str, int],
) -> pd.DataFrame:
    lookup = build_name_map()

    # --- Polity5 period averages for democratizer sample ---
    dem_scodes = set(democratizers.keys())
    pol_dem = polity[polity["scode"].isin(dem_scodes)].copy()

    pol_agg = period_mean(pol_dem, ["country_polity", "scode"], "polity2")
    pol_agg["iso3"] = pol_agg["scode"].map(
        lambda s: _polity_scode_to_iso3(s, pol_agg[pol_agg["scode"] == s]["country_polity"].iloc[0] if len(pol_agg[pol_agg["scode"] == s]) > 0 else "", lookup)
        if len(pol_agg[pol_agg["scode"] == s]) > 0 else None
    )
    # Better: map via country name
    scode_to_iso: dict[str, str] = {}
    for scode, grp in polity[polity["scode"].isin(dem_scodes)].groupby("scode"):
        name = grp["country_polity"].iloc[0]
        iso = lookup(name)
        if iso:
            scode_to_iso[str(scode)] = iso
    pol_agg["iso3"] = pol_agg["scode"].map(scode_to_iso)
    pol_agg = pol_agg[pol_agg["iso3"].notna()].copy()

    # Add transition year
    def _trans_year(scode: str) -> int | None:
        return democratizers.get(str(scode))
    pol_agg["transition_year"] = pol_agg["scode"].apply(_trans_year)
    pol_agg["country_name"] = pol_agg["country_polity"]

    # Keep per period: iso3, period, polity2_mean, transition_year, country_name
    base = pol_agg[["iso3", "country_name", "period", "polity2_mean", "transition_year"]].copy()
    logger.info(f"Base panel: {len(base)} country-period rows, {base['iso3'].nunique()} countries")

    # --- Merge SWIID Gini ---
    swiid["iso3"] = swiid["country_swiid"].apply(lookup)
    gini_agg = period_mean(swiid[swiid["iso3"].notna()], ["iso3"], "gini_disp")
    # Also aggregate SE for uncertainty
    gini_se = swiid[swiid["iso3"].notna()].copy()
    gini_se["period"] = gini_se["year"].apply(assign_period)
    gini_se = gini_se[gini_se["period"].notna()]
    gini_se_agg = (
        gini_se.groupby(["iso3", "period"])["gini_disp_se"]
        .apply(lambda x: float(np.sqrt(np.mean(x.dropna() ** 2))) if x.dropna().shape[0] > 0 else None)
        .reset_index()
        .rename(columns={"gini_disp_se": "gini_se_mean"})
    )
    base = base.merge(gini_agg, on=["iso3", "period"], how="left")
    base = base.merge(gini_se_agg, on=["iso3", "period"], how="left")
    logger.info(f"After SWIID merge: {base['gini_disp_mean'].notna().sum()} Gini observations")

    # --- Merge OWID V-Dem regime ---
    vdem["iso3"] = vdem["country_vdem"].apply(lookup)
    vdem_agg = period_mean(vdem[vdem["iso3"].notna()], ["iso3"], "regime_row_owid")
    base = base.merge(vdem_agg, on=["iso3", "period"], how="left")
    logger.info(f"After V-Dem merge: {base['regime_row_owid_mean'].notna().sum()} regime observations")

    # --- Merge OWID Education ---
    educ["iso3"] = educ["country_educ"].apply(lookup)
    educ_agg = period_mean(educ[educ["iso3"].notna()], ["iso3"], "tertiary_enroll")
    # Forward fill for 2020-22 (UNESCO lag)
    educ_all = educ[educ["iso3"].notna()].copy()
    educ_all["period"] = educ_all["year"].apply(assign_period)
    last_educ = (
        educ_all[educ_all["year"] >= 2016]
        .groupby("iso3")["tertiary_enroll"]
        .last()
        .reset_index()
        .rename(columns={"tertiary_enroll": "tertiary_enroll_fwd"})
    )
    educ_agg_2022 = educ_agg[educ_agg["period"] == "2020-22"].merge(last_educ, on="iso3", how="left")
    educ_agg_2022["tertiary_enroll_mean"] = educ_agg_2022["tertiary_enroll_mean"].fillna(
        educ_agg_2022["tertiary_enroll_fwd"]
    )
    educ_agg_2022["educ_forward_filled"] = educ_agg_2022["tertiary_enroll_fwd"].notna() & educ_agg_2022["tertiary_enroll_mean"].notna()
    educ_agg = educ_agg.merge(
        educ_agg_2022[["iso3", "period", "tertiary_enroll_mean", "educ_forward_filled"]]
        .rename(columns={"tertiary_enroll_mean": "tertiary_enroll_mean_2022"}),
        on=["iso3", "period"], how="left"
    )
    educ_agg["tertiary_enroll_mean"] = educ_agg.apply(
        lambda r: r["tertiary_enroll_mean_2022"] if pd.notna(r.get("tertiary_enroll_mean_2022")) else r["tertiary_enroll_mean"],
        axis=1,
    )
    educ_agg["educ_forward_filled"] = educ_agg.get("educ_forward_filled", False).fillna(False)
    base = base.merge(
        educ_agg[["iso3", "period", "tertiary_enroll_mean", "educ_forward_filled"]],
        on=["iso3", "period"], how="left"
    )
    logger.info(f"After Education merge: {base['tertiary_enroll_mean'].notna().sum()} Educ observations")

    # --- Merge ILO social protection ---
    ilo["iso3"] = ilo["country_ilo"].apply(lookup)
    ilo_agg = ilo[ilo["iso3"].notna()].copy()
    ilo_agg["period"] = ilo_agg["year"].apply(assign_period)
    ilo_agg = ilo_agg[ilo_agg["period"].notna()].copy()
    ilo_period = (
        ilo_agg.groupby(["iso3", "period"])["socprot_rate"]
        .mean()
        .reset_index()
        .rename(columns={"socprot_rate": "socprot_ilo_mean"})
    )
    base = base.merge(ilo_period, on=["iso3", "period"], how="left")
    # Only post-2015 ILO is reliable
    base["socprot_continuous"] = base.apply(
        lambda r: r["socprot_ilo_mean"] if r["period"] in ("2015-19", "2020-22") else None,
        axis=1,
    )
    base["socprot_binary"] = base["socprot_continuous"].apply(
        lambda v: 1.0 if pd.notna(v) and v > 0 else (0.0 if pd.notna(v) else None)
    )
    base["pre2015"] = base["period"].apply(lambda p: p not in ("2015-19", "2020-22"))
    base["harmonization_source"] = base.apply(
        lambda r: "ILO" if pd.notna(r["socprot_continuous"]) else "none",
        axis=1,
    )
    logger.info(f"After ILO merge: {base['socprot_continuous'].notna().sum()} SocProt observations")

    return base


def _polity_scode_to_iso3(scode: str, name: str, lookup) -> str | None:
    return lookup(name)


# ── Build data audit ──────────────────────────────────────────────────────────

def build_audit(panel: pd.DataFrame) -> dict:
    def coverage(col: str) -> dict:
        valid = panel[col].dropna()
        within_sd = float(
            panel.groupby("iso3")[col].std().dropna().mean()
        ) if panel[col].notna().any() else 0.0
        return {
            "n_obs": int(valid.shape[0]),
            "n_countries": int(panel[panel[col].notna()]["iso3"].nunique()),
            "within_sd": round(within_sd, 4),
        }

    socprot_bin = panel["socprot_binary"]
    doubly_obs = int(
        panel[panel["socprot_binary"].notna()]
        .groupby("iso3")["period"]
        .count()
        .gt(1)
        .sum()
    )
    singleton = int(
        panel[panel["socprot_binary"].notna()]
        .groupby("iso3")["period"]
        .count()
        .eq(1)
        .sum()
    )

    return {
        "n_countries_total": int(panel["iso3"].nunique()),
        "n_country_periods_total": int(len(panel)),
        "variable_coverage": {
            "Polity2": coverage("polity2_mean"),
            "Educ": coverage("tertiary_enroll_mean"),
            "Gini": coverage("gini_disp_mean"),
            "SocProt_continuous": coverage("socprot_continuous"),
            "SocProt_binary": coverage("socprot_binary"),
            "Regime_OWID": coverage("regime_row_owid_mean"),
        },
        "dd_identification_base": {
            "binary_strategy": {
                "doubly_observed_countries": doubly_obs,
                "singleton_countries": singleton,
                "effective_country_periods": int(panel["socprot_binary"].notna().sum()),
            }
        },
        "data_availability_notes": {
            "VDem_subindices": "UNAVAILABLE - v2jucomp, v2x_jucon, v2cseeorgs, v2csprtcpt, v2mecenefm require V-Dem full download (form/CAPTCHA). Substituted with polity2 from Polity5.",
            "NSTP_binary": "UNAVAILABLE - GESIS page requires authentication despite CC BY license. SocProt_binary pre-2015 is null.",
            "ILO_socprot": "Available 2015+ via OWID ILOSTAT SDG 1.3.1",
            "democracy_substitute": "Using Polity5 polity2 (-10 to +10) as democracy intensity proxy",
        },
        "educ_quality": {
            "SE_TER_ENRR_within_sd": coverage("tertiary_enroll_mean")["within_sd"],
            "n_forward_filled_periods": int(panel.get("educ_forward_filled", pd.Series(False)).fillna(False).sum()),
        },
    }


# ── Generate input/output pairs ───────────────────────────────────────────────

def generate_examples(panel: pd.DataFrame) -> list[dict]:
    examples = []
    for _, row in panel.iterrows():
        country = row["country_name"]
        period = row["period"]
        iso3 = row["iso3"]

        polity = _round(row.get("polity2_mean"))
        gini = _round(row.get("gini_disp_mean"))
        gini_se = _round(row.get("gini_se_mean"))
        educ = _round(row.get("tertiary_enroll_mean"))
        regime = _round(row.get("regime_row_owid_mean"))
        socprot_cont = _round(row.get("socprot_continuous"))
        socprot_bin = _round(row.get("socprot_binary"))
        pre2015 = bool(row.get("pre2015", True))
        trans_yr = row.get("transition_year")
        harm_src = row.get("harmonization_source", "none")
        educ_fwd = bool(row.get("educ_forward_filled", False))

        regime_label = {0: "closed autocracy", 1: "electoral autocracy",
                        2: "electoral democracy", 3: "liberal democracy"}.get(
            int(round(regime)) if regime is not None else -1, "unknown"
        )

        inp = (
            f"Retrieve SDET difference-in-differences panel variables for {country} ({iso3}) "
            f"during the {period} period. Include: democracy quality (Polity2 index), "
            f"tertiary education enrollment (gross %), Gini inequality (disposable income), "
            f"and social protection coverage (ILO SDG 1.3.1 if available). "
            f"Context: post-1990 democratizer sample for triple-interaction SDET DD experiment."
        )

        out_parts = [
            f"Country: {country} ({iso3})",
            f"Period: {period}",
            f"Transition year: {trans_yr if trans_yr else 'N/A'}",
            f"Democracy (Polity2 5yr avg): {polity if polity is not None else 'N/A'} "
            f"(scale -10 to +10; ≥6 = democracy)",
            f"Regime classification (OWID V-Dem): {regime_label} "
            f"(avg={regime})",
            f"Education (Tertiary Gross Enrollment %): "
            f"{educ if educ is not None else 'N/A'}"
            + (" [forward-filled from last obs]" if educ_fwd else ""),
            f"Gini (disposable income): "
            f"{gini if gini is not None else 'N/A'}"
            + (f" ± {gini_se}" if gini_se is not None else ""),
            f"Social Protection Coverage (ILO SDG 1.3.1): "
            + (f"{socprot_cont}% (binary={socprot_bin}; source={harm_src})"
               if socprot_cont is not None else "N/A (pre-2015 NSTP data unavailable)"),
            f"pre-2015 period: {pre2015}",
        ]
        out = "\n".join(out_parts)

        examples.append({"input": inp, "output": out})
    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main() -> None:
    Path("logs").mkdir(exist_ok=True)

    logger.info("=== Step 1: Load source datasets ===")
    polity = load_polity5()
    swiid = load_swiid()
    vdem = load_vdem()
    educ = load_education()
    ilo = load_ilostat()

    logger.info("=== Step 2: Identify democratizers ===")
    democratizers = identify_democratizers(polity)

    logger.info("=== Step 3: Build merged panel ===")
    panel = build_panel(polity, swiid, vdem, educ, ilo, democratizers)
    logger.info(f"Panel: {len(panel)} rows, {panel['iso3'].nunique()} countries")

    # Quality assertions
    educ_within_sd = panel.groupby("iso3")["tertiary_enroll_mean"].std().dropna().mean()
    logger.info(f"Educ within-SD: {educ_within_sd:.3f} (target ≥2.0)")
    n_ldem = panel["polity2_mean"].notna().sum()
    logger.info(f"N Polity2 obs: {n_ldem} (target ≥200)")

    logger.info("=== Step 4: Generate input/output examples ===")
    examples = generate_examples(panel)
    logger.info(f"Generated {len(examples)} input/output examples")

    logger.info("=== Step 5: Build audit ===")
    audit = build_audit(panel)
    logger.info(f"Audit: {json.dumps({k: v for k, v in audit.items() if k != 'variable_coverage'}, indent=2)}")

    # Build exp_sel_data_out structure
    output_obj = {
        "datasets": [
            {
                "dataset": "SDET_DD_Panel_post1990_democratizers",
                "examples": examples,
            }
        ]
    }

    logger.info("=== Step 6: Write outputs ===")
    full_path = OUT_DIR / "full_data_out.json"
    full_path.write_text(json.dumps(output_obj, indent=2, ensure_ascii=False))
    logger.info(f"full_data_out.json: {full_path.stat().st_size / 1e6:.2f} MB")

    audit_path = OUT_DIR / "data_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    logger.info(f"data_audit.json written")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
