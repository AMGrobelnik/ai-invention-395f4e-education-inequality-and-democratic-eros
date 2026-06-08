#!/usr/bin/env python3
"""Build SDET DD panel dataset: Polity5 post-1990 democratizers.

Each data row = one example.
- Polity5: input = component scores JSON, output = polity2 score (string)

Dataset chosen: polity5_democratizers
Reason: Direct democracy quality outcome (polity2), 12 consistent features,
highest within-country variation (SD=3.11), 39 countries 1990-2018.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
FMT = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=FMT)
logger.add("logs/data_run.log", rotation="10 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"

# 5-year period bins used in the SDET DD design
PERIODS = [
    ("1990-94", 1990, 1994),
    ("1995-99", 1995, 1999),
    ("2000-04", 2000, 2004),
    ("2005-09", 2005, 2009),
    ("2010-14", 2010, 2014),
    ("2015-19", 2015, 2019),
    ("2020-22", 2020, 2022),
]


def _period_label(year: int) -> str | None:
    for label, lo, hi in PERIODS:
        if lo <= year <= hi:
            return label
    return None


def _fix_nan(obj: object) -> object:
    """Recursively replace float NaN with None (JSON-safe)."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _fix_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_nan(v) for v in obj]
    return obj


def identify_democratizer_scodes(p5: pd.DataFrame) -> set[str]:
    """Return Polity5 country codes for post-1990 democratizers.

    Criterion: polity2 <= 0 in any year 1985-1994 AND polity2 >= 6 in 1990-2010,
    with transition year > 1989.
    """
    # Exclude special codes (-66 interrupted, -77 interregnum, -88 transition)
    valid = p5[p5["polity2"] >= -10].copy()

    pre_auth = (
        valid[(valid["year"] >= 1985) & (valid["year"] <= 1994)]
        .groupby("scode")["polity2"]
        .min()
    )
    post_dem = (
        valid[(valid["year"] >= 1990) & (valid["year"] <= 2010)]
        .groupby("scode")["polity2"]
        .max()
    )

    pre_auth_scodes = set(pre_auth[pre_auth <= 0].index)
    post_dem_scodes = set(post_dem[post_dem >= 6].index)

    candidates = pre_auth_scodes & post_dem_scodes

    # Require transition (first year polity2 >= 6) after 1989
    democratizers = set()
    for scode in candidates:
        country_data = valid[valid["scode"] == scode].sort_values("year")
        dem_years = country_data[country_data["polity2"] >= 6]["year"]
        if len(dem_years) > 0 and dem_years.min() > 1989:
            democratizers.add(scode)

    logger.info(f"Identified {len(democratizers)} post-1990 democratizers")
    return democratizers


@logger.catch(reraise=True)
def load_polity5() -> tuple[pd.DataFrame, set[str]]:
    path = DATASETS_DIR / "polity5_p5v2018.xls"
    logger.info(f"Loading Polity5 from {path}")
    p5 = pd.read_excel(path, engine="xlrd")
    p5 = p5[p5["year"] >= 1985].copy()
    logger.info(f"Polity5: {len(p5)} rows, {p5['scode'].nunique()} countries (1985+)")

    democratizer_scodes = identify_democratizer_scodes(p5)

    # Filter to democratizers, 1990+
    p5_dem = p5[
        (p5["scode"].isin(democratizer_scodes)) & (p5["year"] >= 1990)
    ].copy()
    logger.info(
        f"Polity5 democratizer sample: {len(p5_dem)} rows, "
        f"{p5_dem['scode'].nunique()} countries"
    )
    return p5_dem, democratizer_scodes


def build_polity5_examples(p5: pd.DataFrame) -> list[dict]:
    """One example per country-year row; input = component scores, output = polity2."""
    # Component columns that actually form polity2
    feature_cols = ["democ", "autoc", "xrreg", "xrcomp", "xropen", "xconst",
                    "parreg", "parcomp", "exrec", "exconst", "polcomp", "durable"]

    # Keep rows where polity2 is a valid number
    valid = p5[p5["polity2"].between(-10, 10)].copy()
    logger.info(f"Polity5 valid rows (polity2 in [-10,10]): {len(valid)}")

    examples = []
    for _, row in valid.iterrows():
        features = {}
        for col in feature_cols:
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                features[col] = int(v) if isinstance(v, float) and v == int(v) else v

        period = _period_label(int(row["year"]))
        inp = json.dumps(features)
        out = str(int(row["polity2"])) if row["polity2"] == int(row["polity2"]) else str(round(float(row["polity2"]), 2))

        examples.append({
            "input": inp,
            "output": out,
            "metadata_country": str(row["country"]),
            "metadata_scode": str(row["scode"]),
            "metadata_year": int(row["year"]),
            "metadata_period": period,
            "metadata_ccode": int(row["ccode"]) if pd.notna(row.get("ccode")) else None,
        })

    logger.info(f"Built {len(examples)} Polity5 examples")
    return examples


@logger.catch(reraise=True)
def main() -> None:
    logger.info("=== Step 1: Load Polity5 ===")
    p5_dem, _dem_scodes = load_polity5()

    logger.info("=== Step 2: Build examples ===")
    polity5_examples = build_polity5_examples(p5_dem)

    output_obj = {
        "datasets": [
            {
                "dataset": "polity5_democratizers",
                "examples": polity5_examples,
            },
        ]
    }

    output_obj = _fix_nan(output_obj)

    total = len(polity5_examples)
    logger.info(f"Total examples: {total}")

    logger.info("=== Step 3: Write full_data_out.json ===")
    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output_obj, indent=2))
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"full_data_out.json: {size_mb:.2f} MB, {total} examples")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
