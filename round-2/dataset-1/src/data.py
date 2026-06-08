#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
#   "openpyxl>=3.1",
#   "xlrd>=2.0",
#   "loguru>=0.7",
# ]
# ///
"""
Produce exp_sel_data_out examples from 9 source datasets for the Education Trap study.

Each source dataset becomes one group; each non-null row becomes one example.
  input  = JSON string of identifying/contextual features for that row
  output = string representation of the primary target variable
"""

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

WORKSPACE = Path(__file__).parent
DATA = WORKSPACE / "temp" / "datasets"
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
fmt = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=fmt)
logger.add(str(LOG_DIR / "data_py.log"), rotation="30 MB", level="DEBUG")

YEAR_MIN, YEAR_MAX = 1985, 2023

# WB non-country aggregate codes to exclude
WB_AGGREGATES = {
    "WLD","HIC","LIC","LMC","UMC","MIC","EAP","ECA","LAC","MNA","NAC","SAS",
    "SSA","SSF","EAS","ECS","LCN","MEA","OED","ARB","CEB","EMU","EUU","FCS",
    "HPC","IBD","IBT","IDA","IDB","IDX","OSS","LTE","LDC","CSS","PSS","PST",
    "INX","PRE","TSS","TEC","XZN","ZXQ","ZQS","ZTJ","ZJ","ZT","ZG","ZB","ZA",
}


def _fmt(v) -> str:
    """Format a numeric value as a clean string."""
    if pd.isna(v):
        return "null"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _clean_year_wb(y) -> int | None:
    """Convert WB year string 'YR1985' → 1985."""
    try:
        return int(str(y).replace("YR", ""))
    except (ValueError, TypeError):
        return None


def _is_country_iso3(code) -> bool:
    if pd.isna(code):
        return False
    c = str(code).strip().upper()
    return len(c) == 3 and c.isalpha() and c not in WB_AGGREGATES


# ---------------------------------------------------------------------------
# 1. V-Dem V16 slim
# ---------------------------------------------------------------------------
def load_vdem() -> list[dict]:
    path = DATA / "vdem_core_v16_slim.csv"
    df = pd.read_csv(path)
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].dropna(subset=["v2x_libdem"])
    examples = []
    for _, r in df.iterrows():
        regime_raw = r.get("v2x_regime")
        if pd.isna(regime_raw):
            continue
        regime_int = int(round(float(regime_raw)))
        regime_label = {0: "Closed Autocracy", 1: "Electoral Autocracy",
                        2: "Electoral Democracy", 3: "Liberal Democracy"}.get(regime_int, str(regime_int))
        inp = {
            "iso3": str(r["country_text_id"]),
            "country": str(r["country_name"]),
            "year": int(r["year"]),
            "v2x_libdem": _fmt(r.get("v2x_libdem")),
            "v2x_polyarchy": _fmt(r.get("v2x_polyarchy")),
            "v2juhcind": _fmt(r.get("v2juhcind")),
            "v2cseeorgs": _fmt(r.get("v2cseeorgs")),
            "v2xcs_ccsi": _fmt(r.get("v2xcs_ccsi")),
            "v2x_corr": _fmt(r.get("v2x_corr")),
            "v2csreprss": _fmt(r.get("v2csreprss")),
            "v2csprtcpt": _fmt(r.get("v2csprtcpt")),
        }
        examples.append({
            "input": json.dumps(inp, ensure_ascii=False),
            "output": f"{regime_int} ({regime_label})",
            "metadata_iso3": str(r["country_text_id"]),
            "metadata_year": int(r["year"]),
            "metadata_country": str(r["country_name"]),
            "metadata_task_type": "classification",
            "metadata_target": "v2x_regime",
            "metadata_n_classes": 4,
        })
    logger.info(f"V-Dem: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# 2. SWIID 9.2 — Gini inequality
# ---------------------------------------------------------------------------
def load_swiid() -> list[dict]:
    df = pd.read_csv(DATA / "swiid_summary.csv")
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].dropna(subset=["gini_disp"])
    examples = []
    for _, r in df.iterrows():
        inp = {
            "country": str(r["country"]),
            "year": int(r["year"]),
            "gini_disp_se": _fmt(r.get("gini_disp_se")),
            "gini_mkt": _fmt(r.get("gini_mkt")),
        }
        examples.append({
            "input": json.dumps(inp, ensure_ascii=False),
            "output": _fmt(r["gini_disp"]),
            "metadata_country": str(r["country"]),
            "metadata_year": int(r["year"]),
            "metadata_task_type": "regression",
            "metadata_target": "gini_disp",
        })
    logger.info(f"SWIID: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# 3. OWID/UNDP Mean Years of Schooling
# ---------------------------------------------------------------------------
def load_owid_mys_undp() -> list[dict]:
    df = pd.read_csv(DATA / "owid_schooling_undp.csv")
    # Column with MYS is 'Both genders'
    mys_col = "Both genders"
    df = df.rename(columns={"Entity": "entity", "Code": "iso3", "Year": "year"})
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]
    df = df[df["iso3"].apply(_is_country_iso3)].dropna(subset=[mys_col])
    examples = []
    for _, r in df.iterrows():
        inp = {
            "iso3": str(r["iso3"]),
            "country": str(r["entity"]),
            "year": int(r["year"]),
        }
        examples.append({
            "input": json.dumps(inp, ensure_ascii=False),
            "output": _fmt(r[mys_col]),
            "metadata_iso3": str(r["iso3"]),
            "metadata_year": int(r["year"]),
            "metadata_country": str(r["entity"]),
            "metadata_task_type": "regression",
            "metadata_target": "mean_years_schooling",
            "metadata_source": "UNDP_HDR",
        })
    logger.info(f"OWID Schooling UNDP: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# 4. ILO SDG 1.3.1 — Social Protection Coverage
# ---------------------------------------------------------------------------
def load_ilo_sdg0131() -> list[dict]:
    df = pd.read_csv(DATA / "ilo_sdg0131_total_allsex.csv")
    df = df.rename(columns={
        "REF_AREA": "iso3",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "soc_prot_coverage",
        "OBS_STATUS": "obs_status",
        "SEX": "sex",
        "SOC": "soc_group",
    })
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]
    df = df[df["iso3"].apply(_is_country_iso3)].dropna(subset=["soc_prot_coverage"])
    # Directly-reported flag: obs_status in A, DR, NE
    direct_codes = {"A", "DR", "NE"}
    df["ilo_directly_reported"] = df["obs_status"].apply(
        lambda x: str(x).strip() in direct_codes if pd.notna(x) else False
    )
    examples = []
    for _, r in df.iterrows():
        inp = {
            "iso3": str(r["iso3"]),
            "year": int(r["year"]),
            "obs_status": str(r.get("obs_status", "")),
            "sex": str(r.get("sex", "")),
            "soc_group": str(r.get("soc_group", "")),
            "ilo_directly_reported": bool(r["ilo_directly_reported"]),
        }
        examples.append({
            "input": json.dumps(inp, ensure_ascii=False),
            "output": _fmt(r["soc_prot_coverage"]),
            "metadata_iso3": str(r["iso3"]),
            "metadata_year": int(r["year"]),
            "metadata_obs_status": str(r.get("obs_status", "")),
            "metadata_directly_reported": bool(r["ilo_directly_reported"]),
            "metadata_task_type": "regression",
            "metadata_target": "social_protection_coverage_pct",
        })
    logger.info(f"ILO SDG 1.3.1: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# 5. Dreher 2006 IMF Programs
# ---------------------------------------------------------------------------
def load_dreher_imf() -> list[dict]:
    program_sheets = ["IMF SBA", "IMF EFF", "IMF PRGF", "IMF SAF"]
    program_dfs = []
    for sheet in program_sheets:
        try:
            df = pd.read_excel(DATA / "dreher_imf_wb.xls", sheet_name=sheet, engine="xlrd")
            if "Country Code" not in df.columns:
                continue
            year_cols = [c for c in df.columns if isinstance(c, int)]
            melted = df.melt(
                id_vars=["Country Code", "Country Name"],
                value_vars=year_cols,
                var_name="year",
                value_name="program",
            )
            melted = melted.rename(columns={"Country Code": "iso3", "Country Name": "country_name"})
            melted["program"] = pd.to_numeric(melted["program"], errors="coerce").fillna(0)
            program_dfs.append(melted)
        except Exception as e:
            logger.warning(f"Dreher sheet {sheet}: {e}")

    if not program_dfs:
        logger.warning("No Dreher IMF data loaded")
        return []

    combined = pd.concat(program_dfs)
    combined = combined.groupby(["iso3", "country_name", "year"])["program"].max().reset_index()
    combined = combined[combined["year"].between(YEAR_MIN, 2021)]
    combined = combined[combined["iso3"].apply(_is_country_iso3)]
    # Only keep rows where program is defined (0 or 1 — not just NaN from missing)
    combined["imf_program"] = (combined["program"] > 0).astype(int)

    examples = []
    for _, r in combined.iterrows():
        inp = {
            "iso3": str(r["iso3"]),
            "country": str(r["country_name"]),
            "year": int(r["year"]),
        }
        examples.append({
            "input": json.dumps(inp, ensure_ascii=False),
            "output": str(int(r["imf_program"])),
            "metadata_iso3": str(r["iso3"]),
            "metadata_year": int(r["year"]),
            "metadata_country": str(r["country_name"]),
            "metadata_task_type": "classification",
            "metadata_target": "imf_program_active",
            "metadata_n_classes": 2,
        })
    logger.info(f"Dreher IMF: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# 6-9. World Bank WDI indicators (generic loader)
# ---------------------------------------------------------------------------
_WB_DATASETS = [
    ("wb_NY_GDP_PCAP_PP_KD.csv", "gdp_pc_ppp_2017usd",
     "WB WDI: GDP per capita PPP (constant 2017 int$)", "regression"),
    ("wb_SE_TER_ENRR.csv", "gross_tertiary_enrolment_pct",
     "WB WDI: Gross tertiary school enrollment (%)", "regression"),
    ("wb_NY_GDP_MKTP_KD_ZG.csv", "gdp_growth_pct",
     "WB WDI: GDP growth (annual %)", "regression"),
    ("wb_SL_TLF_CACT_ZS.csv", "labour_force_participation_pct",
     "WB WDI: Labour force participation rate (%)", "regression"),
]


def load_wb_indicator(
    filename: str,
    target_col: str,
    description: str,
    task_type: str,
) -> list[dict]:
    df = pd.read_csv(DATA / filename)
    df["year"] = df["year"].apply(_clean_year_wb)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df.rename(columns={"country_code": "iso3"})
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]
    df = df[df["iso3"].apply(_is_country_iso3)].dropna(subset=["value"])

    examples = []
    for _, r in df.iterrows():
        inp = {
            "iso3": str(r["iso3"]),
            "year": int(r["year"]),
        }
        examples.append({
            "input": json.dumps(inp, ensure_ascii=False),
            "output": _fmt(r["value"]),
            "metadata_iso3": str(r["iso3"]),
            "metadata_year": int(r["year"]),
            "metadata_task_type": task_type,
            "metadata_target": target_col,
        })
    logger.info(f"{filename}: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Building exp_sel_data_out from 9 selected datasets ===")

    datasets = [
        {
            "dataset": "V-Dem V16 Democracy Sub-Indices (post-1985)",
            "examples": load_vdem(),
        },
        {
            "dataset": "SWIID 9.2 Standardized Gini Inequality",
            "examples": load_swiid(),
        },
        {
            "dataset": "UNDP HDR Mean Years of Schooling (OWID)",
            "examples": load_owid_mys_undp(),
        },
        {
            "dataset": "ILO SDG 1.3.1 Social Protection Coverage",
            "examples": load_ilo_sdg0131(),
        },
        {
            "dataset": "Dreher 2006 IMF Program Binary Dummies",
            "examples": load_dreher_imf(),
        },
    ]

    # Add all WB indicator datasets
    for filename, target_col, description, task_type in _WB_DATASETS:
        datasets.append({
            "dataset": description,
            "examples": load_wb_indicator(
                filename=filename,
                target_col=target_col,
                description=description,
                task_type=task_type,
            ),
        })

    # Drop datasets that produced zero examples
    datasets = [d for d in datasets if d["examples"]]

    total_examples = sum(len(d["examples"]) for d in datasets)
    logger.info(f"Total: {len(datasets)} datasets, {total_examples} examples")

    out = {"datasets": datasets}
    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved full_data_out.json ({size_mb:.1f} MB)")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
