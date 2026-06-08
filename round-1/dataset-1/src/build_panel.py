#!/usr/bin/env python3
"""Build merged panel dataset: V-Dem × ILO × Education × SWIID for post-1990 democratizers."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pycountry
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WS = Path(__file__).parent
DATASETS = WS / "temp" / "datasets"

ISO3_EXCEPTIONS: dict[str, str] = {
    "Kosovo": "XKX",
    "Micronesia": "FSM",
    "South Korea": "KOR",
    "North Korea": "PRK",
    "Republic of Korea": "KOR",
    "Democratic Republic of Congo": "COD",
    "Congo, Dem. Rep.": "COD",
    "Congo, Rep.": "COG",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Slovakia": "SVK",
    "Taiwan": "TWN",
    "Bolivia": "BOL",
    "Iran": "IRN",
    "Syria": "SYR",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "Laos": "LAO",
    "Tanzania": "TZA",
    "Moldova": "MDA",
    "Palestine": "PSE",
    "West Bank and Gaza": "PSE",
    "Timor-Leste": "TLS",
    "Eswatini": "SWZ",
    "Swaziland": "SWZ",
    "Cape Verde": "CPV",
    "Cabo Verde": "CPV",
    "Cote d'Ivoire": "CIV",
    "Ivory Coast": "CIV",
    "Macedonia": "MKD",
    "North Macedonia": "MKD",
    "Kyrgyz Republic": "KGZ",
    "Kyrgyzstan": "KGZ",
    "Russia": "RUS",
    "Germany": "DEU",
    "Yemen": "YEM",
    "Serbia and Montenegro": "SCG",
    "Kosovo Republic": "XKX",
}


def country_to_iso3(name: str) -> str | None:
    if name in ISO3_EXCEPTIONS:
        return ISO3_EXCEPTIONS[name]
    try:
        return pycountry.countries.search_fuzzy(name)[0].alpha_3
    except Exception:
        return None


def assign_period(year: int) -> str | None:
    if 1990 <= year <= 1994:
        return "1990-94"
    elif 1995 <= year <= 1999:
        return "1995-99"
    elif 2000 <= year <= 2004:
        return "2000-04"
    elif 2005 <= year <= 2009:
        return "2005-09"
    elif 2010 <= year <= 2014:
        return "2010-14"
    elif 2015 <= year <= 2019:
        return "2015-19"
    elif 2020 <= year <= 2022:
        return "2020-22"
    return None


PERIOD_START = {
    "1990-94": 1990, "1995-99": 1995, "2000-04": 2000,
    "2005-09": 2005, "2010-14": 2010, "2015-19": 2015, "2020-22": 2020,
}
PERIOD_END = {
    "1990-94": 1994, "1995-99": 1999, "2000-04": 2004,
    "2005-09": 2009, "2010-14": 2014, "2015-19": 2019, "2020-22": 2022,
}


@logger.catch(reraise=True)
def load_vdem() -> tuple[pd.DataFrame, set[str]]:
    """Load V-Dem, identify democratizers, return period-averaged panel."""
    logger.info("Loading V-Dem CSV (using usecols for memory efficiency)...")
    vdem_path = DATASETS / "vdem_v15" / "V-Dem-CY-Full+Others-v15.csv"
    cols = [
        "country_name", "country_text_id", "COWcode", "year",
        "v2x_libdem", "v2x_jucon", "v2jucomp", "v2cseeorgs", "v2csprtcpt",
        "v2x_civlib", "v2xcl_rol", "v2x_polyarchy", "v2x_regime",
    ]
    df = pd.read_csv(vdem_path, usecols=cols, low_memory=False)
    logger.info(f"V-Dem loaded: {len(df):,} rows, {df['country_name'].nunique()} countries")

    dem_countries = set()
    for iso3, grp in df.groupby("country_text_id"):
        pre = grp[grp["year"] <= 1989]["v2x_regime"].dropna()
        post = grp[grp["year"] >= 1990]["v2x_regime"].dropna()
        if len(pre) == 0:
            pre = grp.nsmallest(5, "year")["v2x_regime"].dropna()
        if len(pre) > 0 and len(post) > 0 and pre.min() <= 1 and post.max() >= 2:
            dem_countries.add(iso3)

    logger.info(f"Post-1990 democratizers identified: {len(dem_countries)}")

    df = df[(df["year"] >= 1990) & (df["year"] <= 2022)].copy()
    df["period"] = df["year"].apply(assign_period)
    df = df.dropna(subset=["period"])
    df["is_democratizer"] = df["country_text_id"].isin(dem_countries)

    vdem_cols = ["v2x_libdem", "v2x_jucon", "v2jucomp", "v2cseeorgs",
                 "v2csprtcpt", "v2x_civlib", "v2xcl_rol", "v2x_polyarchy"]
    agg = (
        df.groupby(["country_name", "country_text_id", "is_democratizer", "period"])
        [vdem_cols]
        .mean()
        .reset_index()
    )
    logger.info(f"V-Dem period panel: {len(agg):,} country-period rows")
    return agg, dem_countries


@logger.catch(reraise=True)
def load_ilo() -> pd.DataFrame:
    """Load ILO SDG 1.3.1, filter to total coverage, period-average.

    IMPORTANT: ILO country-level data (ISO codes) only available from 2015.
    Earlier years (2010-14) exist only as X* regional aggregates — these are excluded.
    obs_status='' means directly reported from ILO Social Security Inquiry Database.
    obs_status='U' means unreliable/unknown.
    """
    logger.info("Loading ILO social protection data...")
    df = pd.read_csv(DATASETS / "ilo_sdg_1_3_1.csv", encoding="utf-8-sig", low_memory=False)

    df = df[(df["sex"] == "SEX_T") & (df["classif1"] == "SOC_CONTIG_TOTAL")].copy()
    df = df.rename(columns={"ref_area": "country_iso3_ilo", "time": "year", "obs_value": "socprot_coverage"})
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2022)]
    df["period"] = df["year"].apply(assign_period)
    df = df.dropna(subset=["period", "socprot_coverage"])

    # Exclude ILO regional aggregate codes (X01, X02, ... = ILO regional groupings)
    # These are not individual countries and inflate the country count for 2010-14
    n_before = len(df)
    df = df[~df["country_iso3_ilo"].str.startswith("X")]
    n_regional = n_before - len(df)
    logger.info(f"Excluded {n_regional} ILO regional aggregate rows (X* codes)")

    df["directly_reported"] = df["obs_status"].isna() | (df["obs_status"] == "")

    def agg_ilo(grp: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "socprot_coverage": grp["socprot_coverage"].mean(),
            "directly_reported_any": bool(grp["directly_reported"].any()),
            "directly_reported_all": bool(grp["directly_reported"].all()),
            "n_ilo_years": len(grp),
        })

    agg = df.groupby(["country_iso3_ilo", "period"]).apply(agg_ilo, include_groups=False).reset_index()
    period_counts = agg.groupby("period").size().to_dict()
    logger.info(f"ILO period panel: {len(agg):,} country-period rows — {period_counts}")
    return agg


@logger.catch(reraise=True)
def load_swiid() -> pd.DataFrame:
    """Load SWIID Gini data, period-average."""
    logger.info("Loading SWIID Gini data...")
    df = pd.read_csv(DATASETS / "swiid_summary.csv")
    logger.info(f"SWIID columns: {list(df.columns[:8])}")

    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if "gini_disp" in cl and "se" not in cl:
            col_map[c] = "gini_disp"
        elif "gini_mkt" in cl and "se" not in cl:
            col_map[c] = "gini_mkt"
        elif "gini_disp" in cl and "se" in cl:
            col_map[c] = "gini_disp_se"
    if col_map:
        df = df.rename(columns=col_map)

    df = df[(df["year"] >= 1990) & (df["year"] <= 2022)].copy()
    df["period"] = df["year"].apply(assign_period)
    df = df.dropna(subset=["period"])

    gini_cols = [c for c in ["gini_disp", "gini_mkt", "gini_disp_se"] if c in df.columns]
    agg = df.groupby(["country", "period"])[gini_cols].mean().reset_index()
    agg["country_iso3_swiid"] = agg["country"].apply(country_to_iso3)
    agg = agg.dropna(subset=["country_iso3_swiid"])
    logger.info(f"SWIID period panel: {len(agg):,} rows, {agg['country_iso3_swiid'].nunique()} countries")
    return agg


@logger.catch(reraise=True)
def load_education() -> pd.DataFrame:
    """Load OWID Lee-Barro education data.

    Source data ends at 2017. The 2020-22 period is forward-filled from 2017
    (last available year) because education changes slowly over 3-5 years.
    Forward-filled observations are flagged via education_imputed=True.
    """
    logger.info("Loading education data (OWID Lee-Barro)...")
    edu_path = DATASETS / "education_lee_barro.json"
    records = json.loads(edu_path.read_text())
    df = pd.DataFrame(records)
    logger.info(f"Education raw: {len(df):,} rows")

    val_col = next((c for c in df.columns if "schooling" in c.lower() or "years" in c.lower()), None)
    if val_col is None:
        id_cols = {"entity_name", "entity_id", "entity_code", "year"}
        val_col = next(c for c in df.columns if c not in id_cols)
    logger.info(f"Education value column: {val_col}")

    df = df.rename(columns={"entity_name": "country", "entity_code": "country_iso3_edu", val_col: "education"})
    df = df[(df["year"] >= 1985) & (df["year"] <= 2022)].copy()
    df["education"] = pd.to_numeric(df["education"], errors="coerce")
    df = df.dropna(subset=["education", "country_iso3_edu"])
    df["education_imputed"] = False

    # Forward-fill 2020-22 period: education source ends at 2017
    # For each country missing 2020-22 data, copy their last 2015-2019 value
    patches = []
    for iso3, grp in df.groupby("country_iso3_edu"):
        has_2020 = any(2020 <= y <= 2022 for y in grp["year"])
        if not has_2020:
            recent = grp[(grp["year"] >= 2015) & (grp["year"] <= 2019)].sort_values("year")
            if len(recent) > 0:
                patches.append({
                    "country": recent.iloc[-1]["country"],
                    "country_iso3_edu": iso3,
                    "year": 2020,
                    "education": recent.iloc[-1]["education"],
                    "education_imputed": True,
                })

    if patches:
        df = pd.concat([df, pd.DataFrame(patches)], ignore_index=True)
        logger.info(f"Forward-filled education into 2020-22 for {len(patches)} countries")

    df["period"] = df["year"].apply(assign_period)
    df = df.dropna(subset=["period", "country_iso3_edu"])

    agg = (
        df.groupby(["country_iso3_edu", "period"])
        .agg(education=("education", "mean"), education_imputed=("education_imputed", "any"))
        .reset_index()
    )
    logger.info(f"Education period panel: {len(agg):,} rows, {agg['country_iso3_edu'].nunique()} countries")
    return agg


@logger.catch(reraise=True)
def build_panel() -> tuple[list[dict], dict]:
    vdem, dem_countries = load_vdem()
    ilo = load_ilo()
    swiid = load_swiid()
    edu = load_education()

    vdem_dem = vdem[vdem["is_democratizer"]].copy()
    logger.info(f"V-Dem democratizer country-periods: {len(vdem_dem):,}")

    merged = vdem_dem.merge(
        ilo.rename(columns={"country_iso3_ilo": "country_text_id"}),
        on=["country_text_id", "period"],
        how="left",
    )
    merged = merged.merge(
        swiid.rename(columns={"country_iso3_swiid": "country_text_id"}),
        on=["country_text_id", "period"],
        how="left",
    )
    merged = merged.merge(
        edu.rename(columns={"country_iso3_edu": "country_text_id"}),
        on=["country_text_id", "period"],
        how="left",
    )

    logger.info(f"Merged panel: {len(merged):,} country-period rows")

    core = ["v2x_libdem", "socprot_coverage", "education", "gini_disp"]
    complete = merged.dropna(subset=core)
    logger.info(f"Complete rows (all 4 core vars): {len(complete):,} "
                f"({complete['country_text_id'].nunique()} countries)")
    logger.info(f"Complete by period: {complete.groupby('period').size().to_dict()}")

    # Within-country deviations computed on all rows (not just complete)
    for var in ["education", "gini_disp", "socprot_coverage"]:
        merged[f"mean_{var}"] = merged.groupby("country_text_id")[var].transform("mean")
        merged[f"e_{var}"] = merged[var] - merged[f"mean_{var}"]

    merged["period_start"] = merged["period"].map(PERIOD_START)
    merged["period_end"] = merged["period"].map(PERIOD_END)
    merged = merged.sort_values(["country_text_id", "period_start"])

    float_cols = [
        "v2x_libdem", "v2x_jucon", "v2jucomp", "v2cseeorgs", "v2csprtcpt",
        "v2x_civlib", "v2xcl_rol", "v2x_polyarchy",
        "education", "gini_disp", "gini_mkt", "gini_disp_se", "socprot_coverage",
        "mean_education", "mean_gini_disp", "mean_socprot_coverage",
        "e_education", "e_gini_disp", "e_socprot_coverage",
    ]

    def safe_float(v) -> float | None:
        if pd.isna(v):
            return None
        return round(float(v), 6)

    def safe_bool(v) -> bool | None:
        if pd.isna(v):
            return None
        return bool(v)

    records = []
    for _, row in merged.iterrows():
        rec: dict = {
            "country_name": str(row["country_name"]),
            "country_iso3": str(row["country_text_id"]),
            "period": str(row["period"]),
            "period_start": int(row["period_start"]),
            "period_end": int(row["period_end"]),
            "is_democratizer": bool(row["is_democratizer"]),
        }
        for col in float_cols:
            rec[col] = safe_float(row.get(col))
        rec["directly_reported_any"] = safe_bool(row.get("directly_reported_any"))
        rec["directly_reported_all"] = safe_bool(row.get("directly_reported_all"))
        rec["education_imputed"] = safe_bool(row.get("education_imputed"))
        records.append(rec)

    audit = build_audit(merged, complete, dem_countries)
    return records, audit


def build_audit(merged: pd.DataFrame, complete: pd.DataFrame, dem_countries: set) -> dict:
    n_dem = len(dem_countries)
    n_all = len(complete)

    # Qualifying: ≥2 complete periods (max achievable given ILO only has 2015-19 and 2020-22)
    periods_per_country = complete.groupby("country_text_id")["period"].nunique()
    qualifying = periods_per_country[periods_per_country >= 2]
    countries_qualifying = []
    for iso3, n_p in qualifying.items():
        name = merged[merged["country_text_id"] == iso3]["country_name"].iloc[0]
        countries_qualifying.append({"country": name, "iso3": iso3, "n_complete_periods": int(n_p)})
    countries_qualifying.sort(key=lambda x: -x["n_complete_periods"])

    dr_col = complete["directly_reported_any"].fillna(False)
    n_directly = int(dr_col.sum())
    frac_dr = round(n_directly / n_all, 4) if n_all > 0 else 0.0

    dr_rows = complete[complete["directly_reported_any"].fillna(False)]
    socprot_med = float(dr_rows["socprot_coverage"].median()) if len(dr_rows) > 0 else float(complete["socprot_coverage"].median())
    gini_med = float(dr_rows["gini_disp"].median()) if len(dr_rows) > 0 else float(complete["gini_disp"].median())

    def quadrant_stats(socprot_high: bool, gini_high: bool) -> dict:
        sp_mask = complete["socprot_coverage"] >= socprot_med if socprot_high else complete["socprot_coverage"] < socprot_med
        g_mask = complete["gini_disp"] >= gini_med if gini_high else complete["gini_disp"] < gini_med
        q = complete[sp_mask & g_mask]
        n = len(q)
        n_dr = int(q["directly_reported_any"].fillna(False).sum())
        return {"n_obs": n, "n_directly_reported": n_dr, "frac_directly_reported": round(n_dr / n, 4) if n > 0 else 0.0}

    def within_sd(var: str) -> float:
        grp = complete.groupby("country_text_id")[var].std(ddof=1)
        return round(float(grp.mean(skipna=True)), 6)

    sd_edu = within_sd("education")
    sd_gini = within_sd("gini_disp")
    sd_sp = within_sd("socprot_coverage")

    n_eff = max(n_all // 4, 1)
    sd_ldem = float(complete["v2x_libdem"].std(ddof=1)) if len(complete) > 1 else 0.1
    denom = sd_edu * sd_gini * sd_sp * math.sqrt(n_eff) if (sd_edu > 0 and sd_gini > 0 and sd_sp > 0) else None
    se_b7 = sd_ldem / denom if denom else None
    mde = round(2.8 * se_b7, 6) if se_b7 else None

    flags = []
    sp_low = complete["socprot_coverage"] < socprot_med
    g_high = complete["gini_disp"] >= gini_med
    low_sp_hi_g = complete[sp_low & g_high]
    if len(low_sp_hi_g) > 0:
        q_frac_dr = float(low_sp_hi_g["directly_reported_any"].fillna(False).mean())
        if q_frac_dr < 0.4:
            flags.append(f"LOW_SOCPROT_HIGH_GINI quadrant: only {q_frac_dr:.1%} directly reported (threshold: 40%)")
    if len(countries_qualifying) < 30:
        flags.append(f"Only {len(countries_qualifying)} countries qualify with ≥2 complete periods (target: ≥30)")
    for var, sd, label in [("education", sd_edu, "Education"), ("gini_disp", sd_gini, "Gini"), ("socprot_coverage", sd_sp, "SocProt")]:
        if sd < 0.05:
            flags.append(f"INSUFFICIENT_VARIATION: within-SD of {label} ({sd:.4f}) < 0.05 threshold")
    flags.append(
        "ILO country-level (ISO) data only available from 2015 (rplumber.ilo.org API); "
        "2010-14 data exists only as X* regional aggregates (excluded). "
        "Pre-2015 periods (1990-94, 1995-99, 2000-04, 2005-09, 2010-14) have no socprot_coverage."
    )
    flags.append(
        "Education source (Lee-Lee/Barro-Lee/UNDP) ends at 2017; "
        "2020-22 period education values are forward-filled from 2017 (education_imputed=True). "
        "Education changes slowly so this is a minor bias, but treat 2020-22 education with caution."
    )
    n_imputed = int(complete.get("education_imputed", pd.Series([False]*n_all)).fillna(False).sum())
    if n_imputed > 0:
        flags.append(f"{n_imputed} complete observations use forward-filled (imputed) education values for 2020-22")

    return {
        "n_democratizer_countries": n_dem,
        "n_country_periods_all_vars": n_all,
        "n_country_periods_directly_reported": n_directly,
        "fraction_directly_reported": frac_dr,
        "countries_qualifying": countries_qualifying,
        "qualifying_threshold": "n_complete_periods >= 2",
        "quadrant_table": {
            "low_socprot_low_gini": quadrant_stats(False, False),
            "low_socprot_high_gini": quadrant_stats(False, True),
            "high_socprot_low_gini": quadrant_stats(True, False),
            "high_socprot_high_gini": quadrant_stats(True, True),
        },
        "quadrant_thresholds": {
            "socprot_median": round(socprot_med, 4),
            "gini_median": round(gini_med, 4),
        },
        "within_country_sds": {
            "education_sd": sd_edu,
            "gini_disp_sd": sd_gini,
            "socprot_coverage_sd": sd_sp,
        },
        "mde_estimate": {
            "n_effective": n_eff,
            "assumed_r2": 0.5,
            "alpha": 0.05,
            "power": 0.8,
            "mde_std_units": mde,
            "interpretation": (
                f"Approximate MDE for DD triple-interaction coefficient β7 ≈ {mde:.4f} "
                f"SD-units of v2x_libdem given N_eff={n_eff} country-period obs. "
                "Formula: MDE = 2.8 × SD_ldem / (SD_educ × SD_gini × SD_sp × √N_eff). "
                "Rough lower bound; actual power depends on FE structure and clustering."
            ) if mde else "Insufficient data to compute MDE",
        },
        "data_quality_flags": flags,
        "ilo_obs_status_note": (
            "ILO data: rplumber.ilo.org/data/indicator/?id=SDG_0131_SEX_SOC_RT_A (SEX_T, SOC_CONTIG_TOTAL). "
            "obs_status='' → directly reported from ILO Social Security Inquiry Database. "
            "obs_status='U' → unreliable/unknown. "
            "X* codes (ILO regional groupings) excluded — country-level ISO codes only used."
        ),
    }


@logger.catch(reraise=True)
def main() -> None:
    records, audit = build_panel()

    data_out_path = WS / "data_out.json"
    data_audit_path = WS / "data_audit.json"

    data_out_path.write_text(json.dumps(records, indent=2))
    data_audit_path.write_text(json.dumps(audit, indent=2))

    logger.info(f"Wrote data_out.json: {len(records):,} records, {data_out_path.stat().st_size/1024:.1f}KB")
    logger.info(f"Wrote data_audit.json: {data_audit_path.stat().st_size/1024:.1f}KB")
    logger.info(f"Qualifying countries (≥2 complete periods): {len(audit['countries_qualifying'])}")
    logger.info(f"Total complete country-periods: {audit['n_country_periods_all_vars']}")
    logger.info(f"MDE estimate: {audit['mde_estimate']['mde_std_units']}")
    if audit["data_quality_flags"]:
        logger.warning(f"Quality flags ({len(audit['data_quality_flags'])}):")
        for f in audit["data_quality_flags"]:
            logger.warning(f"  • {f}")


if __name__ == "__main__":
    main()
