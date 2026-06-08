#!/usr/bin/env python3
"""Download ILO public sector employment and IMF SAP datasets for post-1990 democratizer panel."""

import gzip
import io
import json
import sys
import time
from pathlib import Path

import requests
from loguru import logger

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "download.log"), rotation="30 MB", level="DEBUG")


# Post-1990 democratizers (V-Dem-based, ~80 countries that transitioned 1990-2010)
POST90_DEMOCRATIZERS = [
    "ALB","ARM","AZE","BIH","BGR","HRV","CZE","EST","GEO","HUN",
    "KAZ","KGZ","LVA","LTU","MDA","MNE","MKD","POL","ROU","RUS",
    "SRB","SVK","SVN","TJK","TKM","UKR","UZB",
    "BEN","BWA","CPV","GHA","KEN","LSO","MWI","MLI","MOZ","NAM",
    "NER","NGA","SEN","SLE","TZA","ZMB","ZWE",
    "ECU","MEX","PRY","PER","SLV","GTM","HND","NIC","DOM",
    "BOL","CHL","ARG","BRA","COL","URY","VEN",
    "MNG","IDN","PHL","THA","TWN","KOR",
    "TUN","MAR",
]

ILO_URL_PATTERN = "https://rplumber.ilo.org/files/website/bulk/indicator/{indicator}_A.csv.gz"
DREHER_URL = "https://axel-dreher.de/wp-content/uploads/Dreher%20IMF%20and%20WB.xls"
MONA_URL = "https://www.imf.org/external/np/pdr/mona/Arrangements.aspx"


def download_ilo_bulk(indicator: str, output_path: Path) -> bool:
    """Download ILO ILOSTAT bulk CSV.gz file. Returns True on success."""
    url = ILO_URL_PATTERN.format(indicator=indicator)
    logger.info(f"Downloading ILO {indicator} from {url}")
    try:
        resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 Research"})
        resp.raise_for_status()
        import gzip, io
        df_bytes = gzip.decompress(resp.content)
        output_path.write_bytes(df_bytes)
        import pandas as pd
        df = pd.read_csv(io.BytesIO(df_bytes))
        logger.info(f"ILO {indicator}: {len(df)} rows, {df.columns.tolist()[:6]}")
        return True
    except Exception as e:
        logger.warning(f"ILO bulk download failed for {indicator}: {e}")
        return False


def download_ilo_via_dbnomics(indicator: str, output_path: Path) -> bool:
    """Fallback: download ILO data via DBnomics Python client."""
    logger.info(f"Trying DBnomics fallback for ILO/{indicator}")
    try:
        import dbnomics
        import pandas as pd
        df = dbnomics.fetch_dataset("ILO", indicator)
        if df is None or len(df) == 0:
            logger.warning(f"DBnomics returned empty for ILO/{indicator}")
            return False
        df.to_csv(str(output_path), index=False)
        logger.info(f"DBnomics ILO/{indicator}: {len(df)} rows, cols: {df.columns.tolist()[:8]}")
        return True
    except Exception as e:
        logger.warning(f"DBnomics download failed for {indicator}: {e}")
        return False


def download_dreher(output_path: Path) -> bool:
    """Download Dreher (2006) IMF/WB programs dataset."""
    logger.info(f"Downloading Dreher dataset from {DREHER_URL}")
    try:
        resp = requests.get(DREHER_URL, timeout=120, headers={"User-Agent": "Mozilla/5.0 Research"})
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        logger.info(f"Dreher dataset: {len(resp.content)} bytes saved to {output_path}")
        return True
    except Exception as e:
        logger.warning(f"Dreher download failed: {e}")
        return False


def download_mona(output_path: Path) -> bool:
    """Download IMF MONA arrangements page (tab-delimited or HTML)."""
    logger.info(f"Fetching IMF MONA arrangements from {MONA_URL}")
    try:
        resp = requests.get(
            MONA_URL, timeout=120,
            headers={"User-Agent": "Mozilla/5.0 Research", "Accept": "text/html,text/plain,*/*"}
        )
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        logger.info(f"MONA page: {len(resp.content)} bytes, content-type: {resp.headers.get('content-type','?')}")
        return True
    except Exception as e:
        logger.warning(f"MONA download failed: {e}")
        return False


def download_wwbi_via_wbgapi(output_path: Path) -> bool:
    """Download World Bank WWBI public employment share via wbgapi."""
    logger.info("Downloading WWBI public sector employment via wbgapi")
    try:
        import wbgapi as wb
        import pandas as pd
        # BI.EMP.TOTL.PB.ZS = public sector employment % total employment (WWBI source=31)
        df = wb.data.DataFrame("BI.EMP.TOTL.PB.ZS", time=range(1990, 2023), db=31)
        df.to_csv(str(output_path))
        logger.info(f"WWBI: {df.shape} saved")
        return True
    except Exception as e:
        logger.warning(f"WWBI wbgapi download failed: {e}")
        return False


@logger.catch(reraise=True)
def main() -> None:
    logger.info("=== Dataset download starting ===")
    results: dict[str, bool] = {}

    # --- 1. ILO PSE_TPSE_GOV_NB (public sector employment in thousands) ---
    pse_csv = OUTPUT_DIR / "ILO_PSE_TPSE_GOV_NB_A.csv"
    pse_gz = OUTPUT_DIR / "ILO_PSE_TPSE_GOV_NB_A.csv.gz"
    ok = download_ilo_bulk("PSE_TPSE_GOV_NB", pse_csv)
    if not ok:
        ok = download_ilo_via_dbnomics("PSE_TPSE_GOV_NB", pse_csv)
    results["ILO_PSE_TPSE_GOV_NB"] = ok

    # --- 2. ILO EMP_TEMP_SEX_ECO_NB (total employment by sector) ---
    emp_csv = OUTPUT_DIR / "ILO_EMP_TEMP_SEX_ECO_NB_A.csv"
    ok = download_ilo_bulk("EMP_TEMP_SEX_ECO_NB", emp_csv)
    if not ok:
        ok = download_ilo_via_dbnomics("EMP_TEMP_SEX_ECO_NB", emp_csv)
    results["ILO_EMP_TEMP_SEX_ECO_NB"] = ok

    # --- 3. Dreher (2006) IMF programs Excel ---
    dreher_xls = OUTPUT_DIR / "Dreher_IMF_WB.xls"
    ok = download_dreher(dreher_xls)
    results["DREHER_IMF_WB"] = ok

    # --- 4. IMF MONA arrangements ---
    mona_html = OUTPUT_DIR / "MONA_Arrangements.html"
    ok = download_mona(mona_html)
    results["IMF_MONA"] = ok

    # --- 5. WWBI fallback (if PSE download failed) ---
    if not results["ILO_PSE_TPSE_GOV_NB"]:
        wwbi_csv = OUTPUT_DIR / "WWBI_public_emp_pct.csv"
        ok = download_wwbi_via_wbgapi(wwbi_csv)
        results["WWBI_fallback"] = ok

    logger.info("=== Download summary ===")
    for ds, status in results.items():
        logger.info(f"  {ds}: {'OK' if status else 'FAILED'}")

    # Save status
    status_path = OUTPUT_DIR / "download_status.json"
    status_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Status saved to {status_path}")


if __name__ == "__main__":
    main()
