#!/usr/bin/env python3
"""Fetch World Bank WDI indicators via wbgapi and save to temp/datasets/."""

import sys
import json
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{function}| {message}")
logger.add("logs/fetch_wb.log", rotation="10 MB", level="DEBUG")

import wbgapi as wb


@logger.catch(reraise=True)
def main() -> None:
    out_dir = Path("temp/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    indicators = {
        "SE.TER.ENRR": "tertiary_enroll_gross",
        "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp_const2017",
        "SP.SOC.PROT.FL.ZS": "socprot_coverage_wb",  # fallback for ILO
    }

    results = {}
    for code, name in indicators.items():
        logger.info(f"Fetching {code} ({name})...")
        try:
            df = wb.data.DataFrame(
                series=code,
                time=range(1990, 2023),
                labels=True,
                numericTimeKeys=True,
            )
            df = df.reset_index()
            records = df.to_dict(orient="records")
            results[code] = {"name": name, "rows": len(records), "sample": records[:3]}
            out_path = out_dir / f"wb_{name}.json"
            out_path.write_text(json.dumps(records, indent=2))
            logger.info(f"  → {len(records)} rows saved to {out_path}")
        except Exception as e:
            logger.error(f"  Failed {code}: {e}")

    summary = {k: {"name": v["name"], "rows": v["rows"], "sample": v["sample"]} for k, v in results.items()}
    Path("temp/datasets/wb_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Done. Summary written.")


if __name__ == "__main__":
    main()
