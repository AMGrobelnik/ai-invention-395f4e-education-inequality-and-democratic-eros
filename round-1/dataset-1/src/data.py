#!/usr/bin/env python3
"""Convert merged panel data to exp_sel_data_out format.

Two dataset variants:
  1. vdem_ilo_gini_edu_panel_full   — all 790 democratizer country-period rows
  2. vdem_ilo_gini_edu_panel_complete — 161 rows with all 4 core vars non-null
"""
import json
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WS = Path(__file__).parent

# Feature columns fed into 'input'; target is v2x_libdem
INPUT_FEATURES = [
    "country_iso3",
    "period",
    "period_start",
    "period_end",
    "v2x_jucon",
    "v2jucomp",
    "v2cseeorgs",
    "v2csprtcpt",
    "v2x_civlib",
    "v2xcl_rol",
    "v2x_polyarchy",
    "education",
    "gini_disp",
    "gini_mkt",
    "gini_disp_se",
    "socprot_coverage",
    "directly_reported_any",
    "directly_reported_all",
    "mean_education",
    "mean_gini_disp",
    "mean_socprot_coverage",
    "e_education",
    "e_gini_disp",
    "e_socprot_coverage",
    "education_imputed",
]
TARGET = "v2x_libdem"
CORE_VARS = ["v2x_libdem", "socprot_coverage", "education", "gini_disp"]


def make_examples(records: list[dict], dataset_name: str) -> dict:
    examples = []
    for i, row in enumerate(records):
        target_val = row.get(TARGET)
        output_str = str(round(target_val, 6)) if target_val is not None else "null"
        features = {k: row.get(k) for k in INPUT_FEATURES}
        has_all_core = all(row.get(v) is not None for v in CORE_VARS)
        examples.append({
            "input": json.dumps(features, separators=(",", ":")),
            "output": output_str,
            "metadata_row_index": i,
            "metadata_country_name": row.get("country_name"),
            "metadata_task_type": "regression",
            "metadata_target_variable": TARGET,
            "metadata_is_democratizer": row.get("is_democratizer"),
            "metadata_has_all_core_vars": has_all_core,
        })
    logger.info(f"Dataset '{dataset_name}': {len(examples)} examples")
    return {"dataset": dataset_name, "examples": examples}


def main() -> None:
    data_path = WS / "data_out.json"
    logger.info(f"Loading {data_path}")
    records = json.loads(data_path.read_text())
    logger.info(f"Loaded {len(records)} records")

    ds_full = make_examples(records, "vdem_ilo_gini_edu_panel_full")

    complete = [r for r in records if all(r.get(v) is not None for v in CORE_VARS)]
    logger.info(f"Complete-case records: {len(complete)}")
    ds_complete = make_examples(complete, "vdem_ilo_gini_edu_panel_complete")

    # Best dataset: complete analysis sample — all 4 core vars present in every row,
    # directly usable for the DD triple-interaction estimator.
    # The full panel (ds_full) has most rows missing socprot_coverage (pre-2015, ILO constraint).
    output = {"datasets": [ds_complete]}
    out_path = WS / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(
        f"Wrote full_data_out.json: {out_path.stat().st_size / 1024:.1f}KB, "
        f"{len(ds_complete['examples'])} examples (analysis sample only)"
    )


if __name__ == "__main__":
    main()
