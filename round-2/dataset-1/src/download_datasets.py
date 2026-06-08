#!/usr/bin/env python3
"""Download all candidate datasets for the Education Trap panel study."""

import asyncio
import gzip
import io
import json
import os
import sys
from pathlib import Path

import aiohttp
import pandas as pd
from loguru import logger

WORKSPACE = Path(__file__).parent
OUT = WORKSPACE / "temp" / "datasets"
OUT.mkdir(parents=True, exist_ok=True)

LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{function}| {message}")
logger.add(str(LOG_DIR / "download.log"), rotation="30 MB", level="DEBUG")

DIRECT_DOWNLOADS = [
    {
        "id": "swiid_summary",
        "url": "https://raw.githubusercontent.com/fsolt/swiid/master/data/swiid_summary.csv",
        "filename": "swiid_summary.csv",
        "description": "SWIID 9.92 summary: gini_disp for 199 countries 1960-present",
    },
    {
        "id": "owid_schooling_undp",
        "url": "https://ourworldindata.org/grapher/years-of-schooling.csv?v=1&csvType=full&useColumnShortNames=false",
        "filename": "owid_schooling_undp.csv",
        "description": "OWID UNDP HDR 2025: mean years of schooling, 193 countries annual",
    },
    {
        "id": "owid_schooling_barro_lee",
        "url": "https://ourworldindata.org/grapher/mean-years-of-schooling-long-run.csv?v=1&csvType=full&useColumnShortNames=false",
        "filename": "owid_schooling_barro_lee.csv",
        "description": "OWID Barro-Lee long-run: mean years schooling, historical 1870-2022",
    },
    {
        "id": "undp_hdr_xlsx",
        "url": "https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Statistical_Annex_HDI_Table.xlsx",
        "filename": "undp_hdr_2023_annex.xlsx",
        "description": "UNDP HDR 2023-24 Statistical Annex: HDI, mean years schooling, 193 countries",
    },
    {
        "id": "dreher_imf_xls",
        "url": "https://axel-dreher.de/wp-content/uploads/Dreher%20IMF%20and%20WB.xls",
        "filename": "dreher_imf_wb.xls",
        "description": "Dreher 2006: IMF program binary dummies, 160 countries 1970-2019",
    },
]

ILO_DOWNLOADS = [
    {
        "id": "ilo_sdg0131",
        "code": "SDG_0131_SEX_SOC_RT",
        "filename": "ilo_sdg0131_social_protection.csv.gz",
        "description": "ILO SDG 1.3.1: social protection coverage rate, 249 areas annual",
    },
    {
        "id": "ilo_sps_gexp",
        "code": "SPS_GEXP_GDP",
        "filename": "ilo_sps_gexp_gdp.csv.gz",
        "description": "ILO: public social protection expenditure % GDP",
    },
    {
        "id": "ilo_emp_temp",
        "code": "EMP_TEMP_SEX_INS_NB",
        "filename": "ilo_emp_temp_public.csv.gz",
        "description": "ILO: employment by institutional sector (public) - OECD-centric",
    },
]

ILO_BASE = "https://www.ilo.org/ilostat-files/WEB_bulk_download/indicator/"

WB_INDICATORS = {
    "NY.GDP.PCAP.PP.KD": "GDP per capita PPP (constant 2017 intl $)",
    "SE.TER.ENRR": "Gross tertiary school enrollment (%)",
    "SE.PRM.ENRR": "Gross primary school enrollment (%)",
    "SE.SEC.ENRR": "Gross secondary school enrollment (%)",
    "SI.POV.GINI": "GINI index (World Bank estimate)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
}


async def download_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    output_path: Path,
    label: str,
    max_retries: int = 3,
) -> bool:
    headers = {"User-Agent": "Mozilla/5.0 (research dataset downloader)"}
    for attempt in range(max_retries):
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    size_mb = len(content) / 1e6
                    output_path.write_bytes(content)
                    logger.info(f"[{label}] Downloaded {size_mb:.1f}MB → {output_path.name}")
                    return True
                else:
                    logger.warning(f"[{label}] HTTP {resp.status} (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"[{label}] Error attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    logger.error(f"[{label}] All {max_retries} attempts failed")
    return False


async def download_direct(session: aiohttp.ClientSession, item: dict) -> dict:
    out_path = OUT / item["filename"]
    if out_path.exists() and out_path.stat().st_size > 1000:
        logger.info(f"[{item['id']}] Already exists, skipping")
        return {"id": item["id"], "status": "cached", "path": str(out_path)}

    ok = await download_with_retry(session, item["url"], out_path, item["id"])
    return {
        "id": item["id"],
        "status": "ok" if ok else "failed",
        "path": str(out_path) if ok else None,
    }


async def download_ilo(session: aiohttp.ClientSession, item: dict) -> dict:
    out_path = OUT / item["filename"]
    if out_path.exists() and out_path.stat().st_size > 10000:
        logger.info(f"[{item['id']}] Already exists, skipping")
        return {"id": item["id"], "status": "cached", "path": str(out_path)}

    url = f"{ILO_BASE}{item['code']}_A.csv.gz"
    ok = await download_with_retry(session, url, out_path, item["id"])
    if not ok:
        # Try alternate URL pattern
        url2 = f"https://rplumber.ilo.org/files/website/bulk/indicator/{item['code']}_A.csv.gz"
        ok = await download_with_retry(session, url2, out_path, item["id"])

    return {
        "id": item["id"],
        "status": "ok" if ok else "failed",
        "path": str(out_path) if ok else None,
    }


def download_worldbank() -> dict:
    """Download World Bank indicators via wbgapi."""
    import wbgapi as wb

    results = {}
    for code, label in WB_INDICATORS.items():
        out_path = OUT / f"wb_{code.replace('.', '_')}.csv"
        if out_path.exists() and out_path.stat().st_size > 1000:
            logger.info(f"[WB:{code}] Already exists")
            results[code] = {"status": "cached", "path": str(out_path)}
            continue
        try:
            logger.info(f"[WB:{code}] Downloading: {label}")
            df = wb.data.DataFrame(
                code,
                time=range(1985, 2025),
                skipAggs=True,
            )
            # Stack from wide (time as cols) to long
            df = df.stack(level=0).reset_index()
            df.columns = ["country_code", "year", "value"]
            df["indicator"] = code
            df["indicator_label"] = label
            df.to_csv(out_path, index=False)
            logger.info(f"[WB:{code}] Saved {len(df)} rows → {out_path.name}")
            results[code] = {"status": "ok", "path": str(out_path)}
        except Exception as e:
            logger.error(f"[WB:{code}] Failed: {e}")
            results[code] = {"status": "failed", "error": str(e)}

    return results


def preview_csv(path: Path, label: str, nrows: int = 5) -> dict:
    """Read first N rows and return shape + sample for audit."""
    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                df = pd.read_csv(f, nrows=nrows)
        elif str(path).endswith(".xlsx"):
            df = pd.read_excel(path, nrows=nrows)
        elif str(path).endswith(".xls"):
            df = pd.read_excel(path, nrows=nrows, engine="xlrd")
        else:
            df = pd.read_csv(path, nrows=nrows, encoding="utf-8", errors="replace")

        return {
            "label": label,
            "columns": list(df.columns),
            "sample_rows": df.head(3).to_dict(orient="records"),
            "status": "ok",
        }
    except Exception as e:
        return {"label": label, "status": "error", "error": str(e)}


def count_rows_csv(path: Path) -> int:
    """Count total rows without loading full file into memory."""
    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f) - 1  # subtract header
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f) - 1
    except Exception:
        return -1


async def main() -> None:
    logger.info("=== Starting parallel dataset downloads ===")
    logger.info(f"Output dir: {OUT}")

    results: dict = {}

    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for item in DIRECT_DOWNLOADS:
            tasks.append(download_direct(session, item))
        for item in ILO_DOWNLOADS:
            tasks.append(download_ilo(session, item))

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in all_results:
        if isinstance(r, Exception):
            logger.error(f"Task exception: {r}")
        else:
            results[r["id"]] = r

    logger.info("--- World Bank downloads ---")
    wb_results = download_worldbank()
    for code, r in wb_results.items():
        results[f"wb_{code}"] = r

    # Preview each downloaded file
    logger.info("=== Previewing downloaded datasets ===")
    previews: dict = {}
    all_items = DIRECT_DOWNLOADS + ILO_DOWNLOADS
    for item in all_items:
        out_path = OUT / item["filename"]
        if out_path.exists():
            size_mb = out_path.stat().st_size / 1e6
            preview = preview_csv(out_path, item["description"])
            nrows = count_rows_csv(out_path)
            previews[item["id"]] = {
                "path": str(out_path),
                "size_mb": round(size_mb, 2),
                "nrows": nrows,
                "description": item["description"],
                **preview,
            }
            logger.info(f"[{item['id']}] {size_mb:.1f}MB, {nrows} rows, cols={preview.get('columns', [])[:5]}")

    for code in WB_INDICATORS:
        out_path = OUT / f"wb_{code.replace('.', '_')}.csv"
        if out_path.exists():
            size_mb = out_path.stat().st_size / 1e6
            preview = preview_csv(out_path, WB_INDICATORS[code])
            nrows = count_rows_csv(out_path)
            previews[f"wb_{code}"] = {
                "path": str(out_path),
                "size_mb": round(size_mb, 2),
                "nrows": nrows,
                "description": WB_INDICATORS[code],
                **preview,
            }
            logger.info(f"[WB:{code}] {size_mb:.1f}MB, {nrows} rows")

    # Save download manifest
    manifest = {
        "download_results": results,
        "previews": previews,
        "total_downloaded": sum(1 for r in results.values() if r.get("status") in ("ok", "cached")),
        "total_failed": sum(1 for r in results.values() if r.get("status") == "failed"),
    }
    manifest_path = OUT / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info(f"=== Manifest saved: {manifest_path} ===")
    logger.info(f"Downloaded: {manifest['total_downloaded']}, Failed: {manifest['total_failed']}")


if __name__ == "__main__":
    asyncio.run(main())
