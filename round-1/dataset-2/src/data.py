#!/usr/bin/env python3
"""Standardize raw datasets into exp_sel_data_out format → full_data_out.json."""

import json
import sys
from pathlib import Path

import country_converter as coco
import pandas as pd
import xlrd
from dbnomics import fetch_series
from loguru import logger

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "data.log"), rotation="30 MB", level="DEBUG")

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

YEARS = list(range(1990, 2020))
DREHER_PATH = DATASETS_DIR / "Dreher_IMF_WB.xls"
PSE_CACHE = DATASETS_DIR / "ILO_PSE_TPSE_GOV_NB.csv"
EMP_CACHE = DATASETS_DIR / "ILO_EMP_TEMP_SEX_ECO_NB.csv"


def load_pse() -> pd.DataFrame:
    """Load ILO public sector employment; fetch from DBnomics and cache if needed."""
    if PSE_CACHE.exists():
        logger.info(f"Loading PSE from cache: {PSE_CACHE}")
        return pd.read_csv(PSE_CACHE)

    logger.info("Fetching ILO PSE_TPSE_GOV_NB from DBnomics (will cache)...")
    df = fetch_series("ILO", "PSE_TPSE_GOV_NB", max_nb_series=5000)
    df = df[df["classif1"] == "GOV_LVL_PSE"].copy()
    df["year"] = pd.to_datetime(df["period"]).dt.year
    df = df[df["year"].between(1990, 2022)]
    df = df[["ref_area", "year", "value"]].rename(columns={"ref_area": "iso2", "value": "pse_thousands"})
    df = df.dropna(subset=["pse_thousands"])
    cc = coco.CountryConverter()
    df["iso3"] = cc.pandas_convert(df["iso2"], to="ISO3", not_found=None)
    df = df.dropna(subset=["iso3"])
    df = df.groupby(["iso3", "year"])["pse_thousands"].mean().reset_index()
    df[["iso3", "year", "pse_thousands"]].to_csv(str(PSE_CACHE), index=False)
    logger.info(f"Cached PSE: {len(df)} rows → {PSE_CACHE}")
    return df[["iso3", "year", "pse_thousands"]]


def load_emp() -> pd.DataFrame:
    """Load ILO total employment; fetch from DBnomics and cache if needed."""
    if EMP_CACHE.exists():
        logger.info(f"Loading EMP from cache: {EMP_CACHE}")
        return pd.read_csv(EMP_CACHE)

    logger.info("Fetching ILO EMP_TEMP_SEX_ECO_NB from DBnomics (will cache)...")
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
    df["year"] = pd.to_datetime(df["period"]).dt.year
    df = df[df["year"].between(1990, 2022)]
    df = df[["ref_area", "year", "value"]].rename(columns={"ref_area": "iso2", "value": "emp_thousands"})
    df = df.dropna(subset=["emp_thousands"])
    cc = coco.CountryConverter()
    df["iso3"] = cc.pandas_convert(df["iso2"], to="ISO3", not_found=None)
    df = df.dropna(subset=["iso3"])
    df = df.groupby(["iso3", "year"])["emp_thousands"].mean().reset_index()
    df[["iso3", "year", "emp_thousands"]].to_csv(str(EMP_CACHE), index=False)
    logger.info(f"Cached EMP: {len(df)} rows → {EMP_CACHE}")
    return df[["iso3", "year", "emp_thousands"]]


def load_dreher_sap() -> pd.DataFrame:
    """Load Dreher (2006) IMF SAP dummies (SBA + EFF) from XLS."""
    logger.info(f"Loading Dreher XLS: {DREHER_PATH}")
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
        logger.info(f"  {sheet_name}: {len(rows)} rows")
    combined = pd.concat(frames, ignore_index=True)
    sap = (
        combined.groupby(["iso3", "year"])["program"]
        .max()
        .reset_index()
        .rename(columns={"program": "sap_active"})
    )
    logger.info(f"SAP dummy: {len(sap)} rows, {sap['iso3'].nunique()} countries")
    return sap


def build_pse_dataset(pse: pd.DataFrame, emp: pd.DataFrame) -> list[dict]:
    """Build ILO PSE dataset examples — rows where PSE data is available for democratizers."""
    merged = pse.merge(emp, on=["iso3", "year"], how="left")
    demo = merged[merged["iso3"].isin(POST90_DEMOCRATIZERS)].copy()
    demo = demo[demo["year"].isin(YEARS)]
    demo = demo.dropna(subset=["pse_thousands"])

    mask = demo["emp_thousands"].notna() & (demo["emp_thousands"] > 0)
    demo["pse_share_pct"] = None
    demo.loc[mask, "pse_share_pct"] = (
        demo.loc[mask, "pse_thousands"] / demo.loc[mask, "emp_thousands"] * 100
    ).round(4)

    examples = []
    for _, row in demo.iterrows():
        share_val = f"{row['pse_share_pct']:.2f}%" if pd.notna(row["pse_share_pct"]) else "NA"
        examples.append({
            "input": f"Country: {row['iso3']}, Year: {int(row['year'])}",
            "output": (
                f"public_sector_employment: {row['pse_thousands']:.1f} thousand workers, "
                f"pse_share_of_total_employment: {share_val}"
            ),
            "metadata_iso3": row["iso3"],
            "metadata_year": int(row["year"]),
            "metadata_source": "ILO ILOSTAT PSE_TPSE_GOV_NB via DBnomics",
        })
    logger.info(f"ILO PSE dataset: {len(examples)} examples, {demo['iso3'].nunique()} countries")
    return examples


def build_sap_dataset(sap: pd.DataFrame) -> list[dict]:
    """Build Dreher IMF SAP dataset examples — full democratizer grid 1990-2019."""
    grid = pd.MultiIndex.from_product(
        [POST90_DEMOCRATIZERS, YEARS], names=["iso3", "year"]
    ).to_frame(index=False)
    panel = grid.merge(sap, on=["iso3", "year"], how="left")
    panel["sap_active"] = panel["sap_active"].fillna(0).astype(int)

    examples = []
    for _, row in panel.iterrows():
        status = "active" if row["sap_active"] == 1 else "not active"
        examples.append({
            "input": f"Country: {row['iso3']}, Year: {int(row['year'])}",
            "output": (
                f"imf_sap_active: {int(row['sap_active'])} "
                f"[IMF Structural Adjustment Program {status} — SBA or EFF, source: Dreher 2006]"
            ),
            "metadata_iso3": row["iso3"],
            "metadata_year": int(row["year"]),
            "metadata_source": "Dreher (2006) IMF SBA + EFF program dummies",
        })
    sap_count = panel["sap_active"].sum()
    logger.info(f"Dreher SAP dataset: {len(examples)} examples, {sap_count} SAP-active obs ({sap_count/len(examples)*100:.1f}%)")
    return examples


@logger.catch(reraise=True)
def main() -> None:
    logger.info("=== Standardizing datasets → full_data_out.json ===")

    pse = load_pse()
    emp = load_emp()
    sap = load_dreher_sap()

    pse_examples = build_pse_dataset(pse, emp)
    sap_examples = build_sap_dataset(sap)

    output = {
        "datasets": [
            {
                "dataset": "ILO_PSE_ILOSTAT_Democratizers_1990_2019",
                "examples": pse_examples,
            },
            {
                "dataset": "Dreher_IMF_SAP_Democratizers_1990_2019",
                "examples": sap_examples,
            },
        ]
    }

    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {len(pse_examples) + len(sap_examples)} total examples to {out_path}")
    logger.info(f"  Dataset 1 (ILO PSE): {len(pse_examples)} examples")
    logger.info(f"  Dataset 2 (Dreher SAP): {len(sap_examples)} examples")
    logger.info(f"File size: {out_path.stat().st_size / 1024:.1f} KB")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
