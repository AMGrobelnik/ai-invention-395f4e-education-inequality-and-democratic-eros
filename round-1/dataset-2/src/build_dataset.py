#!/usr/bin/env python3
"""Build merged panel: ILO public sector employment + IMF SAP dummy → data_out.json."""

import json
import sys
from pathlib import Path

import country_converter as coco
import pandas as pd
import xlrd
from dbnomics import fetch_series
from loguru import logger

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "temp" / "datasets"
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "build.log"), rotation="30 MB", level="DEBUG")

# Post-1990 democratizers (~80 countries, V-Dem-based, ISO3)
POST90_DEMOCRATIZERS = [
    "ALB", "ARM", "AZE", "BIH", "BGR", "HRV", "CZE", "EST", "GEO", "HUN",
    "KAZ", "KGZ", "LVA", "LTU", "MDA", "MNE", "MKD", "POL", "ROU", "RUS",
    "SRB", "SVK", "SVN", "TJK", "TKM", "UKR", "UZB",
    "BEN", "BWA", "CPV", "GHA", "KEN", "LSO", "MWI", "MLI", "MOZ", "NAM",
    "NER", "NGA", "SEN", "SLE", "TZA", "ZMB", "ZWE",
    "ECU", "MEX", "PRY", "PER", "SLV", "GTM", "HND", "NIC", "DOM",
    "BOL", "CHL", "ARG", "BRA", "COL", "URY", "VEN",
    "MNG", "IDN", "PHL", "THA", "TWN", "KOR",
    "TUN", "MAR",
]

DREHER_PATH = OUTPUT_DIR / "Dreher_IMF_WB.xls"
YEARS = list(range(1990, 2020))  # Dreher covers 1970-2019


def fetch_pse_data() -> pd.DataFrame:
    """Fetch ILO public sector employment (thousands) via DBnomics."""
    logger.info("Fetching ILO PSE_TPSE_GOV_NB from DBnomics...")
    df = fetch_series("ILO", "PSE_TPSE_GOV_NB", max_nb_series=5000)
    logger.info(f"Raw PSE: {len(df)} rows, cols: {df.columns.tolist()[:8]}")

    df = df[df["classif1"] == "GOV_LVL_PSE"].copy()
    df["year"] = pd.to_datetime(df["period"]).dt.year
    df = df[df["year"].between(1990, 2022)]
    df = df[["ref_area", "year", "value"]].rename(columns={"ref_area": "iso2", "value": "pse_thousands"})
    df = df.dropna(subset=["pse_thousands"])

    cc = coco.CountryConverter()
    df["iso3"] = cc.pandas_convert(df["iso2"], to="ISO3", not_found=None)
    df = df.dropna(subset=["iso3"])
    # Average across multiple survey sources for the same country-year
    df = df.groupby(["iso3", "year"])["pse_thousands"].mean().reset_index()
    logger.info(f"PSE after filtering: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df[["iso3", "year", "pse_thousands"]]


def fetch_emp_data() -> pd.DataFrame:
    """Fetch ILO total employment (thousands) via DBnomics."""
    logger.info("Fetching ILO EMP_TEMP_SEX_ECO_NB (total economy, both sexes) from DBnomics...")
    try:
        df = fetch_series(
            "ILO", "EMP_TEMP_SEX_ECO_NB",
            dimensions={"classif1": ["ECO_AGGREGATE_TOTAL"], "sex": ["SEX_T"]},
            max_nb_series=5000,
        )
    except Exception as e:
        logger.warning(f"Dimension filter failed ({e}), fetching unfiltered")
        df = fetch_series("ILO", "EMP_TEMP_SEX_ECO_NB", max_nb_series=10000)
        df = df[(df["classif1"] == "ECO_AGGREGATE_TOTAL") & (df["sex"] == "SEX_T")]

    logger.info(f"Raw EMP: {len(df)} rows")
    df["year"] = pd.to_datetime(df["period"]).dt.year
    df = df[df["year"].between(1990, 2022)]
    df = df[["ref_area", "year", "value"]].rename(columns={"ref_area": "iso2", "value": "emp_thousands"})
    df = df.dropna(subset=["emp_thousands"])

    cc = coco.CountryConverter()
    df["iso3"] = cc.pandas_convert(df["iso2"], to="ISO3", not_found=None)
    df = df.dropna(subset=["iso3"])
    # Average across multiple survey sources for the same country-year
    df = df.groupby(["iso3", "year"])["emp_thousands"].mean().reset_index()
    logger.info(f"EMP after filtering: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df[["iso3", "year", "emp_thousands"]]


def build_sap_dummy() -> pd.DataFrame:
    """Build IMF SAP binary dummy from Dreher (2006) XLS — SBA + EFF programs."""
    logger.info(f"Loading Dreher XLS from {DREHER_PATH}")
    wb = xlrd.open_workbook(str(DREHER_PATH))

    frames = []
    for sheet_name in ["IMF SBA", "IMF EFF"]:
        sh = wb.sheet_by_name(sheet_name)
        headers = sh.row_values(0)
        year_cols = {i: int(h) for i, h in enumerate(headers) if isinstance(h, float)}

        rows = []
        for r in range(1, sh.nrows):
            row = sh.row_values(r)
            iso3 = str(row[0]).strip()
            for col_idx, year in year_cols.items():
                if year in YEARS:
                    val = row[col_idx]
                    try:
                        prog = int(float(val)) if val and val != "." else 0
                    except (ValueError, TypeError):
                        prog = 0
                    rows.append({"iso3": iso3, "year": year, "program": prog})
        frames.append(pd.DataFrame(rows))
        logger.info(f"  {sheet_name}: {len(rows)} rows loaded")

    combined = pd.concat(frames, ignore_index=True)
    # SAP dummy = 1 if either SBA or EFF active in that country-year
    sap = (
        combined.groupby(["iso3", "year"])["program"]
        .max()
        .reset_index()
        .rename(columns={"program": "sap_active"})
    )
    logger.info(f"SAP dummy: {len(sap)} country-year rows, {sap['iso3'].nunique()} countries")
    return sap


def build_panel(pse: pd.DataFrame, emp: pd.DataFrame, sap: pd.DataFrame) -> pd.DataFrame:
    """Merge PSE, EMP, SAP on country-year; filter to democratizer list."""
    # Full democratizer × year grid
    grid = pd.MultiIndex.from_product(
        [POST90_DEMOCRATIZERS, YEARS], names=["iso3", "year"]
    ).to_frame(index=False)

    panel = grid.merge(pse, on=["iso3", "year"], how="left")
    panel = panel.merge(emp, on=["iso3", "year"], how="left")
    panel = panel.merge(sap, on=["iso3", "year"], how="left")

    # Compute PSE share where both numerator and denominator available
    mask = panel["pse_thousands"].notna() & panel["emp_thousands"].notna() & (panel["emp_thousands"] > 0)
    panel["pse_share_pct"] = None
    panel.loc[mask, "pse_share_pct"] = (
        panel.loc[mask, "pse_thousands"] / panel.loc[mask, "emp_thousands"] * 100
    ).round(4)

    panel["sap_active"] = panel["sap_active"].fillna(0).astype(int)
    return panel


def compute_coverage(panel: pd.DataFrame) -> dict:
    """Compute coverage statistics."""
    n_obs = len(panel)
    pse_cov = panel["pse_thousands"].notna().sum()
    share_cov = panel["pse_share_pct"].notna().sum()
    sap_cov = (panel["sap_active"] >= 0).sum()
    sap_active = (panel["sap_active"] == 1).sum()

    by_country = (
        panel.groupby("iso3")
        .agg(
            pse_years=("pse_thousands", lambda x: x.notna().sum()),
            sap_years=("sap_active", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )

    return {
        "total_country_year_obs": n_obs,
        "n_countries": len(POST90_DEMOCRATIZERS),
        "n_years": len(YEARS),
        "year_range": f"{min(YEARS)}-{max(YEARS)}",
        "pse_coverage_obs": int(pse_cov),
        "pse_coverage_pct": round(pse_cov / n_obs * 100, 1),
        "pse_share_coverage_obs": int(share_cov),
        "pse_share_coverage_pct": round(share_cov / n_obs * 100, 1),
        "sap_active_obs": int(sap_active),
        "sap_active_pct_of_total": round(sap_active / n_obs * 100, 1),
        "countries_with_any_pse": int((by_country["pse_years"] > 0).sum()),
        "countries_with_any_sap": int((by_country["sap_years"] > 0).sum()),
    }


@logger.catch(reraise=True)
def main() -> None:
    logger.info("=== Building merged panel dataset ===")

    pse = fetch_pse_data()
    emp = fetch_emp_data()
    sap = build_sap_dummy()
    panel = build_panel(pse, emp, sap)
    coverage = compute_coverage(panel)

    logger.info("Coverage stats:")
    for k, v in coverage.items():
        logger.info(f"  {k}: {v}")

    # Build exp_sel_data_out format
    examples = []
    for _, row in panel.iterrows():
        pse_val = f"{row['pse_thousands']:.1f}" if pd.notna(row["pse_thousands"]) else "NA"
        emp_val = f"{row['emp_thousands']:.1f}" if pd.notna(row["emp_thousands"]) else "NA"
        share_val = f"{row['pse_share_pct']:.2f}" if pd.notna(row["pse_share_pct"]) else "NA"
        examples.append({
            "input": f"Country: {row['iso3']}, Year: {int(row['year'])}",
            "output": (
                f"pse_thousands: {pse_val}, "
                f"total_emp_thousands: {emp_val}, "
                f"pse_share_pct: {share_val}, "
                f"imf_sap_active: {int(row['sap_active'])}"
            ),
            "metadata_iso3": row["iso3"],
            "metadata_year": int(row["year"]),
        })

    output = {
        "metadata": {
            "description": (
                "Panel dataset: ILO public sector employment (thousands) and IMF SAP "
                "participation dummy for post-1990 democratizers, 1990-2019. "
                "Sources: ILO ILOSTAT via DBnomics, Dreher (2006) IMF programs XLS."
            ),
            "sources": [
                "ILO ILOSTAT PSE_TPSE_GOV_NB via DBnomics (public sector employment, thousands)",
                "ILO ILOSTAT EMP_TEMP_SEX_ECO_NB via DBnomics (total employment, thousands)",
                "Dreher (2006) IMF SBA + EFF program participation dummies (1990-2019)",
            ],
            "coverage": coverage,
            "variables": {
                "pse_thousands": "ILO public sector employment, thousands of workers",
                "total_emp_thousands": "ILO total employment, thousands of workers",
                "pse_share_pct": "Public sector employment as % of total employment",
                "imf_sap_active": "1 if IMF SBA or EFF program active in that year (Dreher 2006)",
            },
            "note_sap_gap": "Dreher data ends 2019; no MONA supplement for 2020-2022.",
            "democratizer_list": POST90_DEMOCRATIZERS,
        },
        "datasets": [
            {
                "dataset": "ILO_PSE_IMF_SAP_Panel_1990_2019",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {len(examples)} rows to {out_path}")
    logger.info(f"File size: {out_path.stat().st_size / 1024:.1f} KB")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
