#!/usr/bin/env python3
"""Build 7-period 5-year panel for Education Trap study (1990-2022).

Selects post-1985 developing democratizers and merges:
  V-Dem V16, SWIID 9.92, UNDP HDR schooling, ILO SDG 1.3.1,
  World Bank WDI, Dreher 2006 IMF programs.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

WORKSPACE = Path(__file__).parent
DATA = WORKSPACE / "temp" / "datasets"
OUT_DIR = WORKSPACE / "data" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
fmt = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=fmt)
logger.add(str(LOG_DIR / "build_panel.log"), rotation="30 MB", level="DEBUG")

# 5-year periods + partial last period
PERIODS = {
    "P1": (1990, 1994),
    "P2": (1995, 1999),
    "P3": (2000, 2004),
    "P4": (2005, 2009),
    "P5": (2010, 2014),
    "P6": (2015, 2019),
    "P7": (2020, 2022),
}

# OECD founding + early members to exclude (not developing)
OECD_EARLY = {
    "AUS","AUT","BEL","CAN","DNK","FIN","FRA","DEU","GRC","ISL",
    "IRL","ITA","JPN","LUX","NLD","NZL","NOR","PRT","ESP","SWE",
    "CHE","TUR","GBR","USA",
}


def load_vdem() -> pd.DataFrame:
    path = DATA / "vdem_core_v16_slim.csv"
    df = pd.read_csv(path)
    df = df[df["year"].between(1985, 2023)].copy()
    logger.info(f"V-Dem: {len(df)} rows, {df['country_text_id'].nunique()} countries")
    return df


def identify_democratizers(vdem: pd.DataFrame) -> set[str]:
    """Post-1985 developing democratizers: crossed electoral-democracy threshold after 1985."""
    # v2x_regime: 0=Closed Autocracy, 1=Electoral Autocracy, 2=Electoral Democracy, 3=Liberal Democracy
    # Democratizer = reached regime>=2 for the first time after 1985 AND not already there pre-1985

    results = []
    for ccode, grp in vdem.groupby("country_text_id"):
        grp = grp.sort_values("year")
        pre85 = grp[grp["year"] < 1985]["v2x_regime"]
        post85 = grp[grp["year"].between(1985, 2000)]["v2x_regime"]

        if pre85.empty and post85.empty:
            continue

        # Was already established democracy before 1985?
        was_dem_pre85 = (not pre85.empty) and (pre85 >= 2).any()
        # Reached democracy threshold at some point 1985-2000?
        reached_dem = (not post85.empty) and (post85 >= 2).any()

        if reached_dem and not was_dem_pre85 and ccode not in OECD_EARLY:
            results.append(ccode)

    logger.info(f"Post-1985 democratizers identified: {len(results)}")
    return set(results)


def period_avg(df: pd.DataFrame, year_col: str, value_cols: list[str],
               id_cols: list[str]) -> pd.DataFrame:
    """Compute 5-year period averages for value_cols, labeled P1..P7."""
    rows = []
    for period, (y0, y1) in PERIODS.items():
        sub = df[df[year_col].between(y0, y1)].copy()
        if sub.empty:
            continue
        agg = sub.groupby(id_cols)[value_cols].mean().reset_index()
        agg["period"] = period
        agg["period_start"] = y0
        agg["period_end"] = y1
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_swiid() -> pd.DataFrame:
    df = pd.read_csv(DATA / "swiid_summary.csv")
    logger.info(f"SWIID: {len(df)} rows, {df['country'].nunique()} countries, {df['year'].min()}-{df['year'].max()}")
    return df


def load_schooling() -> pd.DataFrame:
    df = pd.read_csv(DATA / "owid_schooling_undp.csv")
    df = df.rename(columns={"Code": "iso3", "Year": "year", "Both genders": "mys"})
    df = df.dropna(subset=["iso3", "mys"])
    logger.info(f"Schooling: {len(df)} rows, {df['iso3'].nunique()} countries, {df['year'].min()}-{df['year'].max()}")
    return df[["iso3", "year", "mys"]]


def load_ilo_sdg0131() -> pd.DataFrame:
    path = DATA / "ilo_sdg0131_total_allsex.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={
        "REF_AREA": "iso3",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "soc_prot_coverage",
        "OBS_STATUS": "obs_status",
    })
    df = df[["iso3", "year", "soc_prot_coverage", "obs_status"]].dropna(subset=["soc_prot_coverage"])
    logger.info(f"ILO SDG1.3.1: {len(df)} rows, {df['iso3'].nunique()} areas, {df['year'].min()}-{df['year'].max()}")
    return df


def load_wb(indicator: str, col_name: str) -> pd.DataFrame:
    fname = f"wb_{indicator.replace('.', '_')}.csv"
    df = pd.read_csv(DATA / fname)
    df["year"] = df["year"].astype(str).str.replace("YR", "").astype(int)
    df = df.rename(columns={"country_code": "iso3", "value": col_name})
    return df[["iso3", "year", col_name]].dropna(subset=[col_name])


def load_dreher_imf() -> pd.DataFrame:
    """Load Dreher 2006 IMF programs, combine SBA+EFF+PRGF into any-program binary."""
    program_sheets = ["IMF SBA", "IMF EFF", "IMF PRGF", "IMF SAF"]
    dfs = []
    for sheet in program_sheets:
        try:
            df = pd.read_excel(DATA / "dreher_imf_wb.xls", sheet_name=sheet, engine="xlrd")
            # Columns: Country Code, Country Name, then years 1970..2019
            if "Country Code" not in df.columns:
                continue
            year_cols = [c for c in df.columns if isinstance(c, int)]
            melted = df.melt(id_vars=["Country Code"], value_vars=year_cols,
                             var_name="year", value_name="program")
            melted = melted.rename(columns={"Country Code": "iso3"})
            melted["program"] = pd.to_numeric(melted["program"], errors="coerce").fillna(0)
            dfs.append(melted)
        except Exception as e:
            logger.warning(f"Dreher sheet {sheet}: {e}")

    if not dfs:
        logger.warning("No Dreher IMF data loaded")
        return pd.DataFrame(columns=["iso3", "year", "imf_program"])

    combined = pd.concat(dfs).groupby(["iso3", "year"])["program"].max().reset_index()
    combined = combined.rename(columns={"program": "imf_program"})
    combined["imf_program"] = (combined["imf_program"] > 0).astype(int)
    logger.info(f"Dreher IMF: {len(combined)} rows, {combined['iso3'].nunique()} countries")
    return combined


@logger.catch(reraise=True)
def main() -> None:
    logger.info("=== Building Education Trap Panel Dataset ===")

    # --- Load raw sources ---
    vdem = load_vdem()
    democratizers = identify_democratizers(vdem)

    # Filter V-Dem to 1990+ and democratizers
    vdem_panel = vdem[
        vdem["country_text_id"].isin(democratizers) & vdem["year"].between(1990, 2022)
    ].copy()
    logger.info(f"V-Dem (democratizers 1990-2022): {len(vdem_panel)} rows, {vdem_panel['country_text_id'].nunique()} countries")

    vdem_cols = ["v2x_libdem", "v2x_polyarchy", "v2x_regime", "v2juhcind",
                 "v2cseeorgs", "v2xcs_ccsi", "v2x_corr"]
    vdem_agg = period_avg(
        vdem_panel, "year", vdem_cols,
        id_cols=["country_text_id", "country_name"],
    )
    vdem_agg = vdem_agg.rename(columns={"country_text_id": "iso3"})

    # Schooling
    schooling = load_schooling()
    schooling_agg = period_avg(schooling, "year", ["mys"], id_cols=["iso3"])

    # SWIID
    swiid = load_swiid()
    swiid_agg = period_avg(swiid, "year", ["gini_disp", "gini_disp_se"], id_cols=["country"])
    swiid_agg = swiid_agg.rename(columns={"country": "swiid_country"})

    # ILO SDG 1.3.1
    ilo = load_ilo_sdg0131()
    ilo_agg = period_avg(ilo, "year", ["soc_prot_coverage"], id_cols=["iso3"])

    # WB indicators
    gdp_pc = load_wb("NY.GDP.PCAP.PP.KD", "gdp_pc_ppp")
    gdp_agg = period_avg(gdp_pc, "year", ["gdp_pc_ppp"], id_cols=["iso3"])

    tertiary = load_wb("SE.TER.ENRR", "tert_enrol")
    tert_agg = period_avg(tertiary, "year", ["tert_enrol"], id_cols=["iso3"])

    socprot = load_wb("per_allsp.cov.pop.tot", "wb_socprot_cov")
    # file uses . notation → already loaded as wb_per_allsp_cov_pop_tot.csv
    socprot_path = DATA / "wb_per_allsp_cov_pop_tot.csv"
    sp_df = pd.read_csv(socprot_path)
    sp_df["year"] = sp_df["year"].astype(str).str.replace("YR", "").astype(int)
    sp_df = sp_df.rename(columns={"country_code": "iso3", "value": "wb_socprot_cov"})
    sp_df = sp_df[["iso3", "year", "wb_socprot_cov"]].dropna()
    sp_agg = period_avg(sp_df, "year", ["wb_socprot_cov"], id_cols=["iso3"])

    gdp_growth = load_wb("NY.GDP.MKTP.KD.ZG", "gdp_growth")
    growth_agg = period_avg(gdp_growth, "year", ["gdp_growth"], id_cols=["iso3"])

    # Dreher IMF
    dreher = load_dreher_imf()
    dreher_agg = period_avg(dreher, "year", ["imf_program"], id_cols=["iso3"])

    # --- Merge on iso3 + period ---
    panel = vdem_agg.copy()
    panel = panel.merge(schooling_agg[["iso3", "period", "mys"]], on=["iso3", "period"], how="left")
    panel = panel.merge(ilo_agg[["iso3", "period", "soc_prot_coverage"]], on=["iso3", "period"], how="left")
    panel = panel.merge(gdp_agg[["iso3", "period", "gdp_pc_ppp"]], on=["iso3", "period"], how="left")
    panel = panel.merge(tert_agg[["iso3", "period", "tert_enrol"]], on=["iso3", "period"], how="left")
    panel = panel.merge(sp_agg[["iso3", "period", "wb_socprot_cov"]], on=["iso3", "period"], how="left")
    panel = panel.merge(growth_agg[["iso3", "period", "gdp_growth"]], on=["iso3", "period"], how="left")
    panel = panel.merge(dreher_agg[["iso3", "period", "imf_program"]], on=["iso3", "period"], how="left")

    # SWIID: match by country_name → swiid_country (name-based, imperfect)
    # Use a best-effort name match via country_name column
    swiid_lookup = swiid_agg[["swiid_country", "period", "gini_disp", "gini_disp_se"]].copy()
    panel = panel.merge(
        swiid_lookup.rename(columns={"swiid_country": "country_name"}),
        on=["country_name", "period"],
        how="left",
    )

    n_countries = panel["iso3"].nunique()
    n_rows = len(panel)
    logger.info(f"Merged panel: {n_rows} rows, {n_countries} countries, {panel['period'].nunique()} periods")

    # Coverage stats
    for col in ["mys", "gini_disp", "soc_prot_coverage", "gdp_pc_ppp", "v2x_libdem", "imf_program"]:
        pct = panel[col].notna().mean() * 100
        logger.info(f"  {col}: {pct:.1f}% coverage")

    # --- Save CSV ---
    panel_path = OUT_DIR / "panel_education_trap.csv"
    panel.to_csv(panel_path, index=False)
    logger.info(f"Panel saved: {panel_path} ({panel_path.stat().st_size / 1e3:.0f} KB)")

    # --- Build exp_sel_data_out JSON ---
    examples = []
    for _, row in panel.iterrows():
        # Input: predictors and identifiers
        inp = {
            "iso3": row.get("iso3"),
            "country_name": row.get("country_name"),
            "period": row.get("period"),
            "period_start": int(row.get("period_start")) if pd.notna(row.get("period_start")) else None,
            "period_end": int(row.get("period_end")) if pd.notna(row.get("period_end")) else None,
            "v2x_libdem": round(float(row["v2x_libdem"]), 4) if pd.notna(row.get("v2x_libdem")) else None,
            "v2x_polyarchy": round(float(row["v2x_polyarchy"]), 4) if pd.notna(row.get("v2x_polyarchy")) else None,
            "v2x_regime": round(float(row["v2x_regime"]), 4) if pd.notna(row.get("v2x_regime")) else None,
            "v2juhcind": round(float(row["v2juhcind"]), 4) if pd.notna(row.get("v2juhcind")) else None,
            "v2cseeorgs": round(float(row["v2cseeorgs"]), 4) if pd.notna(row.get("v2cseeorgs")) else None,
            "v2xcs_ccsi": round(float(row["v2xcs_ccsi"]), 4) if pd.notna(row.get("v2xcs_ccsi")) else None,
            "v2x_corr": round(float(row["v2x_corr"]), 4) if pd.notna(row.get("v2x_corr")) else None,
            "gini_disp": round(float(row["gini_disp"]), 4) if pd.notna(row.get("gini_disp")) else None,
            "gini_disp_se": round(float(row["gini_disp_se"]), 4) if pd.notna(row.get("gini_disp_se")) else None,
            "soc_prot_coverage": round(float(row["soc_prot_coverage"]), 2) if pd.notna(row.get("soc_prot_coverage")) else None,
            "wb_socprot_cov": round(float(row["wb_socprot_cov"]), 2) if pd.notna(row.get("wb_socprot_cov")) else None,
            "gdp_pc_ppp": round(float(row["gdp_pc_ppp"]), 1) if pd.notna(row.get("gdp_pc_ppp")) else None,
            "gdp_growth": round(float(row["gdp_growth"]), 4) if pd.notna(row.get("gdp_growth")) else None,
            "tert_enrol": round(float(row["tert_enrol"]), 2) if pd.notna(row.get("tert_enrol")) else None,
            "imf_program": int(row["imf_program"]) if pd.notna(row.get("imf_program")) else None,
        }
        # Output: education outcome (mean years schooling)
        out = {
            "mys": round(float(row["mys"]), 4) if pd.notna(row.get("mys")) else None,
            "mys_available": pd.notna(row.get("mys")),
        }
        examples.append({
            "input": json.dumps(inp, ensure_ascii=False),
            "output": json.dumps(out, ensure_ascii=False),
            "metadata_iso3": str(row.get("iso3", "")),
            "metadata_period": str(row.get("period", "")),
            "metadata_country": str(row.get("country_name", "")),
        })

    data_out = {
        "metadata": {
            "study": "State-Dependent Education Traps in Post-1985 Developing Democratizers",
            "panel_structure": "7 five-year periods, 1990-2022",
            "n_countries": n_countries,
            "n_observations": n_rows,
            "n_observations_with_mys": int(panel["mys"].notna().sum()),
            "democratizer_criterion": "v2x_regime reached >=2 (Electoral Democracy) after 1985, was <2 pre-1985, not OECD founding member",
            "sources": {
                "democracy": "V-Dem V16 (v2x_libdem, v2x_regime, v2juhcind, v2cseeorgs)",
                "inequality": "SWIID 9.92 (gini_disp, gini_disp_se)",
                "education": "UNDP HDR via OWID (mean years schooling, 1990-2023)",
                "social_protection": "ILO SDG 1.3.1 (coverage rate) + WB ASPIRE (wb_socprot_cov)",
                "economic": "World Bank WDI (GDP/cap PPP, tertiary enrollment, GDP growth)",
                "imf_programs": "Dreher 2006 (binary any-program, 1990-2019)",
            },
            "data_quality_notes": {
                "ilo_obs_status": "ILO API response lacks OBS_STATUS; cannot filter directly-reported vs modelled estimates. Coverage 2009-2023 only; pre-2009 observations will be NaN.",
                "swiid_ssa": "Sub-Saharan Africa SWIID values are heavily imputed; use gini_disp_se>3 as unreliability threshold.",
                "vdem_regime": "v2x_regime rounding: 0=Closed Autocracy, 1=Electoral Autocracy, 2=Electoral Democracy, 3=Liberal Democracy (period averages are continuous).",
                "dreher_imf": "Dreher 2006 covers through 2019; 2020-2022 IMF data not included.",
                "wb_year_format": "WB WDI year column originally 'YR{year}'; converted to integer.",
            },
        },
        "datasets": [
            {
                "dataset": "Education Trap Panel — Post-1985 Developing Democratizers (1990-2022)",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(data_out, indent=2, ensure_ascii=False))
    logger.info(f"Output saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"=== Done: {n_rows} panel obs, {n_countries} countries ===")


if __name__ == "__main__":
    main()
