#!/usr/bin/env python3
"""
Build a 7-period (1990-2022) country-period panel for post-1990 developing democratizers.
Sources: OWID V-Dem, ILO SDG 1.3.1, SWIID, UNDP HDR25, World Bank (ASPIRE+GDP+tertiary).
"""

import asyncio
import gc
import gzip
import io
import json
import math
import os
import resource
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger

# ─── Logging ─────────────────────────────────────────────────────────────────
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
_fmt = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=_fmt)
logger.add("logs/build_panel.log", rotation="30 MB", level="DEBUG")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
RAW = BASE / "workspace" / "raw"
PROC = BASE / "workspace" / "processed"
TEMP = BASE / "temp" / "datasets"
for d in [RAW/"ilo", RAW/"vdem", RAW/"swiid", RAW/"undp", RAW/"wb", RAW/"nstp", PROC, TEMP]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Hardware / RAM guard ────────────────────────────────────────────────────
def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    import psutil
    return psutil.virtual_memory().total / 1e9

RAM_GB = _container_ram_gb()
RAM_BUDGET = int(min(RAM_GB * 0.7, 20) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f} GB")

# ─── Seven periods ───────────────────────────────────────────────────────────
PERIODS = {
    1: (1990, 1994), 2: (1995, 1999), 3: (2000, 2004),
    4: (2005, 2009), 5: (2010, 2014), 6: (2015, 2019), 7: (2020, 2022),
}
PERIOD_LABELS = {k: f"{v[0]}-{str(v[1])[2:]}" for k, v in PERIODS.items()}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 research-dataset-download"})

# ─── Download helpers ────────────────────────────────────────────────────────

def download_or_cache(url: str, dest: Path, *, chunk_size: int = 1 << 20) -> Path:
    if dest.exists() and dest.stat().st_size > 10_000:
        logger.info(f"Cache hit: {dest.name}")
        return dest
    logger.info(f"Downloading {url}")
    r = SESSION.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    logger.info(f"Saved {dest.stat().st_size/1e6:.1f} MB → {dest}")
    return dest


def download_wb_zip(url: str, dest_dir: Path, indicator_code: str) -> Path:
    zip_path = dest_dir / f"{indicator_code}.zip"
    if not zip_path.exists():
        download_or_cache(url, zip_path)
    # Find the main data CSV inside the ZIP
    with zipfile.ZipFile(zip_path) as z:
        csvs = [n for n in z.namelist() if n.endswith(".csv") and "metadata" not in n.lower() and "country" not in n.lower().split("/")[-1][:8]]
        if not csvs:
            csvs = [n for n in z.namelist() if n.endswith(".csv")]
        logger.info(f"ZIP contents: {z.namelist()}")
        # pick the one with the indicator code in the name
        data_csv = next((c for c in csvs if indicator_code.replace(".", "_") in c or indicator_code in c), csvs[0])
        out = dest_dir / Path(data_csv).name
        if not out.exists():
            with z.open(data_csv) as f:
                out.write_bytes(f.read())
    return out


# ─── Step 1: ILO SDG 1.3.1 ──────────────────────────────────────────────────

@logger.catch(reraise=True)
def load_ilo() -> pd.DataFrame:
    # ILO SDMX API: SEX_T + SOC_CONTIG_TOTAL (at least one benefit), 1990-2023
    url = (
        "https://sdmx.ilo.org/rest/data/ILO,SDG_0131_SEX_SOC_RT,1.0/"
        ".A.SDG_0131_RT.SEX_T.SOC_CONTIG_TOTAL."
        "?format=csv&startPeriod=1990&endPeriod=2023"
    )
    dest = RAW / "ilo" / "SDG_0131_SEX_T_TOTAL.csv"
    download_or_cache(url, dest)
    df = pd.read_csv(dest, low_memory=False)
    logger.info(f"ILO raw shape: {df.shape}, columns: {list(df.columns)}")

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"ref_area": "iso3", "time_period": "year", "obs_value": "socprot_ilo"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["socprot_ilo"] = pd.to_numeric(df["socprot_ilo"], errors="coerce")
    df = df.dropna(subset=["iso3", "year", "socprot_ilo"])
    df["year"] = df["year"].astype(int)
    if "obs_status" in df.columns:
        df["ilo_direct"] = df["obs_status"].fillna("").astype(str).str.upper().isin(["A", "P"])
    else:
        df["ilo_direct"] = False
    df = df.groupby(["iso3", "year"], as_index=False).agg(
        socprot_ilo=("socprot_ilo", "mean"),
        ilo_direct=("ilo_direct", "max"),
    )
    logger.info(f"ILO: {df.shape[0]} rows, {df['iso3'].nunique()} countries, years {df['year'].min()}-{df['year'].max()}")
    return df


# ─── Step 2: World Bank ASPIRE (pre-2015 SocProt fallback) ──────────────────

@logger.catch(reraise=True)
def load_aspire() -> pd.DataFrame:
    """
    Load WB ASPIRE social protection coverage (SL.COV.TOTL.ZS).
    Returns empty DataFrame if unavailable — SocProt will rely on ILO only.
    ASPIRE data is in WB database 73, which doesn't support the standard JSON API;
    the downloadformat=csv endpoint returns an error XML for this indicator.
    """
    cache_csv = RAW / "wb" / "aspire_long.csv"
    if cache_csv.exists() and cache_csv.stat().st_size > 1000:
        df = pd.read_csv(cache_csv)
        logger.info(f"ASPIRE cache: {len(df)} rows")
        return df

    try:
        import wbgapi as wb
        # SL.COV.TOTL.ZS is in WDI (db 2) even though it's ASPIRE-sourced
        df_raw = wb.data.DataFrame("SL.COV.TOTL.ZS", time=range(1985, 2025)).reset_index()
        df_raw.to_csv(RAW / "wb" / "aspire_wbgapi_wide.csv", index=False)
        df = _wb_wide_to_long(df_raw, value_name="socprot_aspire")
        df.to_csv(cache_csv, index=False)
        logger.info(f"ASPIRE via wbgapi: {len(df)} rows, {df['iso3'].nunique()} countries")
        return df
    except Exception as e:
        logger.warning(f"ASPIRE unavailable ({e}) — pre-ILO socprot will be NaN; ILO covers 2009+")
        return pd.DataFrame(columns=["iso3", "year", "socprot_aspire"])


def _wb_wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert World Bank wide-format CSV to long format."""
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    def _parse_year(col: str) -> int | None:
        s = col.lstrip("YR").lstrip("yr")
        try:
            y = int(s)
            return y if 1985 <= y <= 2025 else None
        except ValueError:
            return None

    year_cols = [c for c in df.columns if _parse_year(c) is not None]
    iso_col = next((c for c in df.columns if c in ("Country Code", "economy", "ISO3")), None)
    if iso_col is None:
        iso_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    id_vars = [iso_col]
    long = df[id_vars + year_cols].melt(id_vars=id_vars, var_name="year", value_name=value_name)
    long.columns = ["iso3", "year", value_name]
    long["year"] = long["year"].apply(lambda c: _parse_year(str(c)) or int(str(c).lstrip("YRyr")))
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long.dropna(subset=[value_name])


# ─── Step 3: V-Dem via OWID (regime + libdem + polyarchy) ────────────────────

@logger.catch(reraise=True)
def load_vdem_owid() -> pd.DataFrame:
    sources = {
        "regime":    "https://ourworldindata.org/grapher/political-regime.csv",
        "libdem":    "https://ourworldindata.org/grapher/liberal-democracy.csv",
        "polyarchy": "https://ourworldindata.org/grapher/electoral-democracy.csv",
    }
    dfs = []
    for key, url in sources.items():
        dest = RAW / "vdem" / f"owid_{key}.csv"
        try:
            download_or_cache(url, dest)
            df = pd.read_csv(dest)
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            iso_col = next(c for c in df.columns if c in ("code",))
            year_col = "year"
            val_col = [c for c in df.columns if c not in ("entity", "code", "year", "world_region_according_to_owid")][0]
            df = df[[iso_col, year_col, val_col]].rename(columns={iso_col: "iso3", val_col: key})
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df[key] = pd.to_numeric(df[key], errors="coerce")
            df = df.dropna(subset=["iso3", "year", key])
            dfs.append(df)
            logger.info(f"OWID V-Dem {key}: {len(df)} rows, {df['iso3'].nunique()} countries")
        except Exception as e:
            logger.error(f"Failed to download OWID {key}: {e}")

    if not dfs:
        raise RuntimeError("All OWID V-Dem downloads failed")

    vdem = dfs[0]
    for d in dfs[1:]:
        vdem = pd.merge(vdem, d, on=["iso3", "year"], how="outer")

    # Convert regime to int for filter (0=closed autocracy, 1=electoral autocracy,
    # 2=electoral democracy, 3=liberal democracy)
    if "regime" in vdem.columns:
        vdem["regime"] = pd.to_numeric(vdem["regime"], errors="coerce")

    logger.info(f"V-Dem OWID merged: {vdem.shape}")
    return vdem


# ─── Step 4: Attempt V-Dem full dataset (sub-indices) ─────────────────────────

def load_vdem_subindices() -> Optional[pd.DataFrame]:
    """Try to get V-Dem sub-indices from Demscore or direct form submission."""
    # First try Demscore (https://demscore.se/ aggregates V-Dem + others)
    try:
        url = "https://demscore.se/wp-content/uploads/2023/09/demscore_2023_09.csv"
        dest = RAW / "vdem" / "demscore.csv"
        download_or_cache(url, dest)
        df = pd.read_csv(dest, low_memory=False)
        cols = [c for c in df.columns if any(x in c for x in ["v2jucomp", "v2x_jucon", "v2cseeorgs", "v2csprtcpt"])]
        if cols and len(cols) >= 2:
            logger.info(f"Demscore sub-indices found: {cols}")
            id_cols = [c for c in df.columns if c in ("iso3c", "iso3", "ccode", "country_id")]
            year_cols = [c for c in df.columns if c in ("year",)]
            if id_cols and year_cols:
                keep = id_cols[:1] + year_cols[:1] + cols
                df = df[keep].copy()
                df.columns = ["iso3", "year"] + cols
                return df
    except Exception as e:
        logger.warning(f"Demscore attempt failed: {e}")

    # Try Harvard Dataverse V-Dem CSV (archive)
    try:
        url = "https://dataverse.harvard.edu/api/access/datafile/7440490"
        dest = RAW / "vdem" / "vdem_harvard.csv"
        download_or_cache(url, dest)
        df = pd.read_csv(dest, low_memory=False)
        needed = ["country_text_id", "year", "v2jucomp", "v2x_jucon", "v2cseeorgs", "v2csprtcpt"]
        avail = [c for c in needed if c in df.columns]
        if len(avail) >= 4:
            df = df[avail].rename(columns={"country_text_id": "iso3"})
            return df
    except Exception as e:
        logger.warning(f"Harvard Dataverse V-Dem attempt failed: {e}")

    logger.warning("V-Dem sub-indices unavailable — will set to null in panel")
    return None


# ─── Step 5: SWIID v9.92 ────────────────────────────────────────────────────

@logger.catch(reraise=True)
def load_swiid() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/fsolt/swiid/master/data/swiid_summary.csv"
    dest = RAW / "swiid" / "swiid_summary.csv"
    download_or_cache(url, dest)
    df = pd.read_csv(dest)
    logger.info(f"SWIID columns: {list(df.columns)}")
    # Identify iso3, year, gini_disp, gini_disp_se
    iso_col = next((c for c in df.columns if "country" in c.lower()), df.columns[0])
    # SWIID uses country names — need to map to ISO3
    year_col = next((c for c in df.columns if "year" in c.lower()), None)
    gini_col = next((c for c in df.columns if "gini_disp" in c.lower() and "se" not in c.lower()), None)
    se_col = next((c for c in df.columns if "gini_disp" in c.lower() and "se" in c.lower()), None)

    if gini_col is None:
        # Fall back: find any column with 'gini' that isn't SE
        gini_col = next((c for c in df.columns if "gini" in c.lower() and "se" not in c.lower()), None)

    logger.info(f"SWIID: iso_col={iso_col} year_col={year_col} gini_col={gini_col} se_col={se_col}")
    keep = [iso_col, year_col, gini_col] + ([se_col] if se_col else [])
    df = df[keep].copy()
    new_cols = ["country_name", "year", "gini_disp"] + (["gini_disp_se"] if se_col else [])
    df.columns = new_cols
    if "gini_disp_se" not in df.columns:
        df["gini_disp_se"] = float("nan")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["gini_disp"] = pd.to_numeric(df["gini_disp"], errors="coerce")
    df["gini_disp_se"] = pd.to_numeric(df["gini_disp_se"], errors="coerce")
    df = df.dropna(subset=["country_name", "year", "gini_disp"])

    # Map country names → ISO3 using pycountry or a manual table
    df["iso3"] = df["country_name"].map(_build_swiid_iso3_map(df["country_name"].unique()))
    df = df.dropna(subset=["iso3"])
    logger.info(f"SWIID after ISO3 map: {len(df)} rows, {df['iso3'].nunique()} countries")
    return df[["iso3", "year", "gini_disp", "gini_disp_se"]]


def _build_swiid_iso3_map(names) -> dict:
    """Map SWIID country names to ISO3 codes via pycountry."""
    try:
        import pycountry
    except ImportError:
        logger.warning("pycountry not installed, using manual map only")
        pycountry = None

    manual = {
        "Bolivia (Plurinational State of)": "BOL", "Bolivia": "BOL",
        "Côte d'Ivoire": "CIV", "Cote d'Ivoire": "CIV", "Ivory Coast": "CIV",
        "Democratic Republic of the Congo": "COD", "DR Congo": "COD",
        "Republic of the Congo": "COG",
        "Iran (Islamic Republic of)": "IRN", "Iran": "IRN",
        "Korea, Republic of": "KOR", "South Korea": "KOR",
        "Korea, Dem. People's Rep.": "PRK", "North Korea": "PRK",
        "Lao People's Democratic Republic": "LAO", "Laos": "LAO",
        "Libyan Arab Jamahiriya": "LBY", "Libya": "LBY",
        "Macedonia": "MKD", "North Macedonia": "MKD",
        "Micronesia": "FSM", "Moldova": "MDA",
        "Palestinian Territory": "PSE", "Palestine": "PSE",
        "Russian Federation": "RUS", "Russia": "RUS",
        "Syria": "SYR", "Syrian Arab Republic": "SYR",
        "Taiwan": "TWN", "Tanzania": "TZA",
        "United Republic of Tanzania": "TZA",
        "United States": "USA", "United States of America": "USA",
        "Venezuela (Bolivarian Republic of)": "VEN", "Venezuela": "VEN",
        "Vietnam": "VNM", "Viet Nam": "VNM",
        "West Bank and Gaza": "PSE",
        "Kosovo": "XKX",
        "Czech Republic": "CZE", "Czechia": "CZE",
        "Slovakia": "SVK",
        "Turkey": "TUR", "Türkiye": "TUR",
        "Egypt, Arab Rep.": "EGY", "Egypt": "EGY",
        "Yemen, Rep.": "YEM", "Yemen": "YEM",
        "Gambia, The": "GMB", "Gambia": "GMB",
        "Kyrgyz Republic": "KGZ", "Kyrgyzstan": "KGZ",
        "Hong Kong": "HKG", "Hong Kong SAR, China": "HKG",
        "Macao": "MAC",
        "Reunion": "REU",
        "Serbia and Montenegro": "SCG",
        "Netherlands Antilles": "ANT",
    }
    result = {}
    for name in names:
        if name in manual:
            result[name] = manual[name]
            continue
        if pycountry:
            try:
                c = pycountry.countries.search_fuzzy(name)
                if c:
                    result[name] = c[0].alpha_3
                    continue
            except Exception:
                pass
        result[name] = None
    return result


# ─── Step 6: UNDP HDR25 (mean years of schooling) ───────────────────────────

@logger.catch(reraise=True)
def load_undp_mys() -> pd.DataFrame:
    url = "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv"
    dest = RAW / "undp" / "HDR25_Composite_indices_complete_time_series.csv"
    try:
        download_or_cache(url, dest)
    except Exception as e:
        logger.warning(f"UNDP primary URL failed: {e}, trying alt")
        alt = "https://hdr.undp.org/sites/default/files/2024_HDR/HDR24_Composite_indices_complete_time_series.csv"
        dest = RAW / "undp" / "HDR24_Composite_indices_complete_time_series.csv"
        download_or_cache(alt, dest)

    try:
        df = pd.read_csv(dest, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(dest, encoding="latin-1", low_memory=False)
    logger.info(f"UNDP raw: {df.shape}, cols sample: {list(df.columns[:10])}")

    iso_col = next((c for c in df.columns if c.lower() in ("iso3", "iso_code3")), None)
    if iso_col is None:
        iso_col = df.columns[0]

    # MYS columns: named like mys_1990, mys_2000, ... or mys (single year index)
    mys_cols = [c for c in df.columns if c.lower().startswith("mys")]
    if not mys_cols:
        # Try: columns with year suffix, filter for reasonable range
        mys_cols = [c for c in df.columns if "schooling" in c.lower() and "mean" in c.lower()]
    logger.info(f"MYS columns found: {mys_cols[:10]}")

    if not mys_cols:
        logger.error("No MYS columns found in UNDP CSV")
        return pd.DataFrame(columns=["iso3", "year", "educ_mys"])

    keep = [iso_col] + mys_cols
    df = df[keep].copy()
    df.columns = ["iso3"] + mys_cols

    # Melt year-wide columns to long format
    if len(mys_cols) > 1:
        # Year-tagged columns: mys_1990, mys_2000, etc.
        df_long = df.melt(id_vars=["iso3"], value_vars=mys_cols, var_name="year_str", value_name="educ_mys")
        df_long["year"] = df_long["year_str"].str.extract(r"(\d{4})").astype(float).astype("Int64")
        df_long = df_long.dropna(subset=["year"])
    else:
        # Single column — has year column separately?
        yr_col = next((c for c in df.columns if c.lower() == "year"), None)
        if yr_col:
            df_long = df[["iso3", yr_col, mys_cols[0]]].copy()
            df_long.columns = ["iso3", "year", "educ_mys"]
        else:
            logger.error("Cannot determine year structure for UNDP MYS")
            return pd.DataFrame(columns=["iso3", "year", "educ_mys"])

    df_long["educ_mys"] = pd.to_numeric(df_long["educ_mys"], errors="coerce")
    df_long = df_long.dropna(subset=["iso3", "year", "educ_mys"])
    df_long["year"] = df_long["year"].astype(int)

    # Interpolate within each country to fill gaps
    df_long = df_long.sort_values(["iso3", "year"])
    df_long = df_long.groupby(["iso3", "year"], as_index=False)["educ_mys"].mean()
    df_filled = []
    for iso3, grp in df_long.groupby("iso3"):
        grp = grp.set_index("year")
        full_range = pd.RangeIndex(grp.index.min(), grp.index.max() + 1)
        grp = grp.reindex(full_range)
        grp["educ_mys"] = grp["educ_mys"].interpolate(method="linear")
        grp["iso3"] = iso3
        grp.index.name = "year"
        df_filled.append(grp.reset_index()[["iso3", "year", "educ_mys"]])
    df_long = pd.concat(df_filled, ignore_index=True)

    logger.info(f"UNDP MYS: {len(df_long)} rows, {df_long['iso3'].nunique()} countries, years {df_long['year'].min()}-{df_long['year'].max()}")
    return df_long


# ─── Step 7: World Bank GDP PPP + Tertiary Enrollment ───────────────────────

@logger.catch(reraise=True)
def load_wb_indicators() -> pd.DataFrame:
    """Load GDP PPP and tertiary enrollment from WDI via wbgapi."""
    import wbgapi as wb
    indicators = {
        "NY.GDP.PCAP.PP.KD": "gdppc_ppp",
        "SE.TER.ENRR": "educ_tertiary",
    }
    dfs = []
    for code, varname in indicators.items():
        cache_csv = RAW / "wb" / f"{varname}_long.csv"
        if cache_csv.exists() and cache_csv.stat().st_size > 1000:
            df = pd.read_csv(cache_csv)
            logger.info(f"WB {code} cache: {len(df)} rows")
            dfs.append(df)
            continue
        try:
            df_wide = wb.data.DataFrame(code, time=range(1985, 2025)).reset_index()
            df_wide.to_csv(RAW / "wb" / f"{varname}_wide.csv", index=False)
            df = _wb_wide_to_long(df_wide, value_name=varname)
            df.to_csv(cache_csv, index=False)
            dfs.append(df)
            logger.info(f"WB {code}: {len(df)} rows, {df['iso3'].nunique()} countries")
        except Exception as e:
            logger.error(f"wbgapi failed for {code}: {e}")

    if not dfs:
        raise RuntimeError("All World Bank downloads failed")
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on=["iso3", "year"], how="outer")
    return merged


# ─── Step 8: Democratizer filter ────────────────────────────────────────────

def identify_democratizers(vdem: pd.DataFrame, wb: pd.DataFrame) -> pd.DataFrame:
    """
    Post-1985 democratizers: V-Dem v2x_regime transition from ≤1 to ≥2 after 1985,
    sustained ≥3 consecutive years. GDP PPP < $15,000 at transition year.
    """
    if "regime" not in vdem.columns:
        logger.warning("No regime column — skipping democratizer filter, using all countries")
        return pd.DataFrame({"iso3": vdem["iso3"].unique(), "transition_year": pd.NA, "gdppc_at_transition": pd.NA})

    results = []
    for iso3, grp in vdem.groupby("iso3"):
        grp = grp.sort_values("year").dropna(subset=["regime"])
        if len(grp) < 3:
            continue
        grp["regime_int"] = grp["regime"].round().astype(int)
        # Find first year where regime ≥ 2, preceded by regime ≤ 1
        for i in range(1, len(grp)):
            row = grp.iloc[i]
            prev = grp.iloc[i - 1]
            if row["year"] <= 1985:
                continue
            if prev["regime_int"] <= 1 and row["regime_int"] >= 2:
                # Check sustained ≥ 3 years
                subsequent = grp[grp["year"] >= row["year"]]["regime_int"]
                if (subsequent >= 2).sum() >= 3:
                    results.append({"iso3": iso3, "transition_year": int(row["year"])})
                    break

    dem_df = pd.DataFrame(results)
    if dem_df.empty:
        logger.warning("No democratizers found — check V-Dem regime data")
        return dem_df

    # Merge GDP PPP at transition year
    gdp = wb[["iso3", "year", "gdppc_ppp"]].dropna() if "gdppc_ppp" in wb.columns else None
    if gdp is not None:
        dem_df = pd.merge(dem_df, gdp.rename(columns={"year": "transition_year", "gdppc_ppp": "gdppc_at_transition"}),
                          on=["iso3", "transition_year"], how="left")
        # Fill with ±3 year range if exact year missing
        missing = dem_df["gdppc_at_transition"].isna()
        for idx in dem_df[missing].index:
            iso3 = dem_df.at[idx, "iso3"]
            ty = dem_df.at[idx, "transition_year"]
            nearby = gdp[(gdp["iso3"] == iso3) & (gdp["year"].between(ty - 3, ty + 3))]
            if not nearby.empty:
                dem_df.at[idx, "gdppc_at_transition"] = nearby.iloc[(nearby["year"] - ty).abs().argsort().iloc[0]]["gdppc_ppp"]
    else:
        dem_df["gdppc_at_transition"] = float("nan")

    # Apply GDP < $15,000 filter
    n_before = len(dem_df)
    gdp_mask = dem_df["gdppc_at_transition"].isna() | (dem_df["gdppc_at_transition"] < 15000)
    dem_df = dem_df[gdp_mask]
    logger.info(f"Democratizers: {n_before} before GDP filter → {len(dem_df)} after")

    return dem_df


# ─── Step 9: Period aggregation ─────────────────────────────────────────────

def year_to_period(year: int) -> Optional[int]:
    for pid, (lo, hi) in PERIODS.items():
        if lo <= year <= hi:
            return pid
    return None


def aggregate_to_periods(df: pd.DataFrame, value_cols: list, *, count_cols: Optional[list] = None) -> pd.DataFrame:
    """Collapse annual data to 7 periods by simple mean."""
    df = df.copy()
    df["period_id"] = df["year"].map(year_to_period)
    df = df.dropna(subset=["period_id"])
    df["period_id"] = df["period_id"].astype(int)

    named_agg: dict = {c: pd.NamedAgg(column=c, aggfunc="mean") for c in value_cols}
    if count_cols:
        for c in count_cols:
            named_agg[f"n_{c}"] = pd.NamedAgg(column=c, aggfunc="count")
    result = df.groupby(["iso3", "period_id"]).agg(**named_agg).reset_index()

    # Count of annual obs per period for each value col
    for c in value_cols:
        count_ser = df.groupby(["iso3", "period_id"])[c].count().reset_index(name=f"n_{c}")
        result = pd.merge(result, count_ser, on=["iso3", "period_id"], how="left")

    result["period"] = result["period_id"].map(PERIOD_LABELS)
    return result


# ─── Step 10: Build panel ────────────────────────────────────────────────────

@logger.catch(reraise=True)
def build_panel(ilo, aspire, vdem, wb, swiid, undp, sub_indices) -> pd.DataFrame:
    logger.info("=== Building panel ===")

    # --- Identify democratizers ---
    dem = identify_democratizers(vdem, wb)
    qualified = set(dem["iso3"].tolist())
    logger.info(f"Qualified democratizers: {len(qualified)}")

    # --- Filter all sources to 1985-2022 ---
    def _filter_years(df):
        return df[df["year"].between(1985, 2022)] if "year" in df.columns else df

    ilo = _filter_years(ilo)
    aspire = _filter_years(aspire)
    vdem = _filter_years(vdem)
    wb = _filter_years(wb)
    swiid = _filter_years(swiid)
    undp = _filter_years(undp)

    # --- Splice SocProt: ASPIRE (pre-2015) + ILO (2015+) ---
    socprot_pre = aspire[aspire["year"] < 2015][["iso3", "year", "socprot_aspire"]].rename(
        columns={"socprot_aspire": "socprot_raw"}
    ).copy()
    socprot_pre["socprot_source"] = "aspire_proxy"
    socprot_pre["ilo_direct"] = False

    socprot_post = ilo[ilo["year"] >= 2015][["iso3", "year", "socprot_ilo", "ilo_direct"]].rename(
        columns={"socprot_ilo": "socprot_raw"}
    ).copy()
    socprot_post["socprot_source"] = "ilo_sdg131"

    # Overlap validation: 2015 only
    overlap_2015_ilo = ilo[ilo["year"] == 2015][["iso3", "socprot_ilo"]].rename(columns={"socprot_ilo": "ilo_2015"})
    overlap_2015_asp = aspire[aspire["year"] == 2015][["iso3", "socprot_aspire"]].rename(columns={"socprot_aspire": "asp_2015"})
    overlap = pd.merge(overlap_2015_ilo, overlap_2015_asp, on="iso3")
    if len(overlap) > 5:
        corr = overlap[["ilo_2015", "asp_2015"]].corr().iloc[0, 1]
        mdiff = (overlap["ilo_2015"] - overlap["asp_2015"]).mean()
        logger.info(f"Overlap 2015 ILO vs ASPIRE: r={corr:.3f}, mean_diff={mdiff:.2f} (n={len(overlap)})")
        overlap_stats = {"n": len(overlap), "r": round(float(corr), 4), "mean_diff": round(float(mdiff), 4)}
    else:
        overlap_stats = {"n": len(overlap), "r": None, "mean_diff": None}

    socprot_long = pd.concat([socprot_pre, socprot_post], ignore_index=True)

    # --- Splice Education: UNDP MYS (≤2022) primary; WB tertiary as supplement ---
    educ_mys = undp[["iso3", "year", "educ_mys"]].copy()
    educ_ter = wb[wb["educ_tertiary"].notna()][["iso3", "year", "educ_tertiary"]].copy() if "educ_tertiary" in wb.columns else pd.DataFrame(columns=["iso3", "year", "educ_tertiary"])

    # Merge education sources
    educ_long = pd.merge(educ_mys, educ_ter, on=["iso3", "year"], how="outer")

    def _educ_source(row):
        if pd.notna(row.get("educ_mys")):
            return "mys"
        elif pd.notna(row.get("educ_tertiary")):
            return "tertiary"
        return None

    educ_long["educ_source"] = educ_long.apply(_educ_source, axis=1)
    educ_long["education"] = educ_long["educ_mys"].fillna(float("nan"))

    # --- Period aggregation for each source ---
    socprot_p = aggregate_to_periods(
        socprot_long[socprot_long["socprot_raw"].notna()],
        ["socprot_raw"]
    ).rename(columns={"socprot_raw": "socprot", "n_socprot_raw": "n_obs_socprot"})
    # Keep dominant source per period
    src_mode = socprot_long.groupby(["iso3", socprot_long["year"].map(year_to_period).rename("period_id")])["socprot_source"].agg(lambda x: x.mode()[0] if len(x) > 0 else None).reset_index()
    socprot_p = pd.merge(socprot_p, src_mode, on=["iso3", "period_id"], how="left")
    ilo_direct_p = socprot_long.groupby(["iso3", socprot_long["year"].map(year_to_period).rename("period_id")])["ilo_direct"].any().reset_index()
    socprot_p = pd.merge(socprot_p, ilo_direct_p, on=["iso3", "period_id"], how="left")

    vdem_p = aggregate_to_periods(
        vdem,
        [c for c in ["regime", "libdem", "polyarchy"] if c in vdem.columns]
    )

    swiid_p = aggregate_to_periods(swiid, ["gini_disp", "gini_disp_se"])
    educ_p = aggregate_to_periods(
        educ_long.dropna(subset=["iso3", "year"]),
        [c for c in ["education", "educ_mys", "educ_tertiary"] if c in educ_long.columns]
    )
    # Educ source per period
    educ_src_p = educ_long.dropna(subset=["educ_source"]).groupby(
        ["iso3", educ_long["year"].dropna().map(year_to_period).rename("period_id")]
    )["educ_source"].agg(lambda x: x.mode()[0] if len(x) > 0 else None).reset_index()
    educ_p = pd.merge(educ_p, educ_src_p, on=["iso3", "period_id"], how="left")

    gdp_p = aggregate_to_periods(
        wb[wb["gdppc_ppp"].notna()][["iso3", "year", "gdppc_ppp"]].copy(),
        ["gdppc_ppp"]
    ) if "gdppc_ppp" in wb.columns else pd.DataFrame(columns=["iso3", "period_id", "gdppc_ppp"])

    # --- Merge all period frames ---
    panel = socprot_p
    for df_src in [vdem_p, swiid_p, educ_p, gdp_p]:
        if len(df_src) > 0 and "period_id" in df_src.columns:
            merge_cols = [c for c in df_src.columns if c not in panel.columns or c in ["iso3", "period_id"]]
            panel = pd.merge(panel, df_src[list(set(["iso3", "period_id"] + [c for c in df_src.columns if c not in ["iso3", "period_id", "period"]]))], on=["iso3", "period_id"], how="outer")

    if "period" not in panel.columns or panel["period"].isna().all():
        panel["period"] = panel["period_id"].map(PERIOD_LABELS)

    # --- Restrict to qualified democratizers ---
    if qualified:
        panel = panel[panel["iso3"].isin(qualified)].copy()
        logger.info(f"Panel after democratizer filter: {len(panel)} rows, {panel['iso3'].nunique()} countries")
    else:
        logger.warning("No democratizer filter applied — keeping all countries")

    # --- Add transition info ---
    if len(dem) > 0:
        panel = pd.merge(panel, dem[["iso3", "transition_year", "gdppc_at_transition"]].drop_duplicates("iso3"),
                         on="iso3", how="left")
    else:
        panel["transition_year"] = pd.NA
        panel["gdppc_at_transition"] = pd.NA

    # --- Country names ---
    try:
        import pycountry
        def _cname(iso3):
            try:
                return pycountry.countries.get(alpha_3=iso3).name
            except Exception:
                return iso3
        panel["country_name"] = panel["iso3"].map(_cname)
    except ImportError:
        panel["country_name"] = panel["iso3"]

    # --- Sub-indices merge ---
    if sub_indices is not None:
        sub_p = aggregate_to_periods(sub_indices, [c for c in ["v2jucomp", "v2x_jucon", "v2cseeorgs", "v2csprtcpt"] if c in sub_indices.columns])
        panel = pd.merge(panel, sub_p.drop(columns=["period"], errors="ignore"), on=["iso3", "period_id"], how="left")
    else:
        for c in ["v2jucomp", "v2x_jucon", "v2cseeorgs", "v2csprtcpt"]:
            panel[c] = float("nan")

    # --- Within-country deviations ---
    for var_col, abbrev in [("socprot", "e_S"), ("gini_disp", "e_G"), ("education", "e_E")]:
        if var_col in panel.columns:
            cmean = panel.groupby("iso3")[var_col].transform("mean")
            panel[abbrev] = panel[var_col] - cmean
        else:
            panel[abbrev] = float("nan")

    # --- Keep only ≥1 non-missing core variable rows ---
    core = [c for c in ["socprot", "libdem", "gini_disp", "education"] if c in panel.columns]
    panel = panel.dropna(subset=core, how="all")

    logger.info(f"Final panel: {len(panel)} rows, {panel['iso3'].nunique()} countries")
    return panel, overlap_stats


# ─── Step 11: Schema-compliant JSON output ───────────────────────────────────

def panel_to_schema(panel: pd.DataFrame) -> dict:
    """Convert panel DataFrame to exp_sel_data_out schema."""
    examples = []
    for _, row in panel.iterrows():
        def _v(col, default=None):
            v = row.get(col, default)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            if hasattr(v, "item"):  # numpy scalar
                return v.item()
            return v

        input_obj = {
            "socprot": _v("socprot"),
            "socprot_source": _v("socprot_source"),
            "ilo_direct": _v("ilo_direct"),
            "gini_disp": _v("gini_disp"),
            "gini_disp_se": _v("gini_disp_se"),
            "education": _v("education"),
            "educ_mys": _v("educ_mys"),
            "educ_tertiary": _v("educ_tertiary"),
            "educ_source": _v("educ_source"),
            "v2jucomp": _v("v2jucomp"),
            "v2x_jucon": _v("v2x_jucon"),
            "v2cseeorgs": _v("v2cseeorgs"),
            "v2csprtcpt": _v("v2csprtcpt"),
            "v2x_polyarchy": _v("polyarchy"),
            "gdppc_ppp": _v("gdppc_ppp"),
            "e_E": _v("e_E"),
            "e_G": _v("e_G"),
            "e_S": _v("e_S"),
        }
        output_obj = {
            "ldem": _v("libdem"),
            "v2x_libdem": _v("libdem"),
            "v2x_polyarchy": _v("polyarchy"),
        }
        examples.append({
            "input": json.dumps(input_obj),
            "output": json.dumps(output_obj),
            "metadata_country_code": _v("iso3", ""),
            "metadata_country_name": _v("country_name", ""),
            "metadata_period": _v("period", ""),
            "metadata_period_id": int(_v("period_id") or 0),
            "metadata_transition_year": _v("transition_year"),
            "metadata_gdppc_at_transition": _v("gdppc_at_transition"),
            "metadata_n_obs_socprot": _v("n_obs_socprot"),
            "metadata_n_gini_disp": _v("n_gini_disp"),
            "metadata_n_libdem": _v("n_libdem"),
        })

    return {"datasets": [{"dataset": "extended_7period_democratizers_panel", "examples": examples}]}


# ─── Step 12: Data audit ────────────────────────────────────────────────────

def compute_audit(panel: pd.DataFrame, overlap_stats: dict) -> dict:
    n_countries = int(panel["iso3"].nunique())
    n_rows = int(len(panel))
    n_complete = int(panel.dropna(subset=[c for c in ["socprot", "libdem", "gini_disp", "education"] if c in panel.columns]).shape[0])
    n_3plus = int((panel.groupby("iso3")["period_id"].count() >= 3).sum())

    coverage_by_period = {}
    for pid, label in PERIOD_LABELS.items():
        sub = panel[panel["period_id"] == pid]
        coverage_by_period[label] = int(len(sub))

    ilo_frac = float(panel["ilo_direct"].mean()) if "ilo_direct" in panel.columns else None

    # Within-country SD of education
    if "education" in panel.columns:
        wcsd_ext = float(panel.groupby("iso3")["education"].std().mean())
        sub7 = panel[panel["period_id"].isin([6, 7])]
        wcsd_2p = float(sub7.groupby("iso3")["education"].std().mean()) if len(sub7) > 0 else None
    else:
        wcsd_ext = None
        wcsd_2p = None

    n_ilo_source = int((panel.get("socprot_source", pd.Series()) == "ilo_sdg131").sum()) if "socprot_source" in panel.columns else None
    n_aspire_source = int((panel.get("socprot_source", pd.Series()) == "aspire_proxy").sum()) if "socprot_source" in panel.columns else None

    cases = {}
    for country, iso3 in [("Hungary", "HUN"), ("Turkey", "TUR"), ("Thailand", "THA"), ("Venezuela", "VEN")]:
        cases[country] = iso3 in panel["iso3"].values

    ratio = round(wcsd_ext / wcsd_2p, 2) if wcsd_ext and wcsd_2p and wcsd_2p > 0 else None

    return {
        "n_countries_total": n_countries,
        "n_country_periods_total": n_rows,
        "n_country_periods_complete": n_complete,
        "n_countries_3plus_periods": n_3plus,
        "coverage_by_period": coverage_by_period,
        "nstp_countries": 0,
        "nstp_fallback_countries": n_aspire_source,
        "ilo_sdg131_periods": n_ilo_source,
        "ilo_directly_reported_fraction": round(ilo_frac, 4) if ilo_frac is not None else None,
        "within_sd_education_extended": round(wcsd_ext, 4) if wcsd_ext else None,
        "within_sd_education_twoperiod": round(wcsd_2p, 4) if wcsd_2p else None,
        "within_sd_education_ratio": ratio,
        "overlap_2015_ilo_vs_aspire": overlap_stats,
        "case_countries_included": cases,
        "vdem_subindices_available": any(
            panel[c].notna().any() for c in ["v2jucomp", "v2x_jucon", "v2cseeorgs", "v2csprtcpt"]
            if c in panel.columns
        ),
        "mde_estimate_note": (
            f"N={n_countries} countries, {n_3plus} with ≥3 periods. "
            f"Within-SD education={wcsd_ext:.3f} vs 2-period={wcsd_2p:.3f}. "
            f"Ratio={ratio}x larger in extended panel."
        ) if wcsd_ext and wcsd_2p else "Education within-SD not computed",
        "data_sources": {
            "socprot_pre2015": "World Bank ASPIRE (SL.COV.TOTL.ZS) — fallback (NSTP requires GESIS login)",
            "socprot_post2015": "ILO SDG 1.3.1 bulk download (SDG_0131_SEX_SOC_RT_A.csv.gz)",
            "vdem_main": "Our World in Data V-Dem CSV (political-regime, liberal-democracy, electoral-democracy)",
            "vdem_subindices": "Attempted Demscore/Harvard Dataverse — see vdem_subindices_available",
            "swiid": "SWIID v9.92 (April 2026) GitHub raw CSV",
            "education_primary": "UNDP HDR25 mean years of schooling (mys)",
            "education_supplement": "World Bank SE.TER.ENRR gross tertiary enrollment",
            "gdp": "World Bank NY.GDP.PCAP.PP.KD (GDP PPP constant 2017 international $)",
        }
    }


# ─── Main ────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main():
    logger.info("=== Panel build start ===")

    # Download all sources (sequential — each step depends on previous context)
    logger.info("--- ILO SDG 1.3.1 ---")
    ilo = load_ilo()
    ilo.to_parquet(PROC / "ilo.parquet", index=False)

    logger.info("--- World Bank ASPIRE ---")
    aspire = load_aspire()
    aspire.to_parquet(PROC / "aspire.parquet", index=False)

    logger.info("--- V-Dem OWID ---")
    vdem = load_vdem_owid()
    vdem.to_parquet(PROC / "vdem_owid.parquet", index=False)

    logger.info("--- V-Dem sub-indices (best-effort) ---")
    sub_indices = load_vdem_subindices()

    logger.info("--- SWIID ---")
    swiid = load_swiid()
    swiid.to_parquet(PROC / "swiid.parquet", index=False)

    logger.info("--- UNDP MYS ---")
    undp = load_undp_mys()
    undp.to_parquet(PROC / "undp_mys.parquet", index=False)

    logger.info("--- World Bank GDP + tertiary ---")
    wb = load_wb_indicators()
    wb.to_parquet(PROC / "wb.parquet", index=False)

    gc.collect()

    # Build panel
    panel, overlap_stats = build_panel(ilo, aspire, vdem, wb, swiid, undp, sub_indices)
    panel.to_parquet(PROC / "panel.parquet", index=False)
    logger.info(f"Panel rows: {len(panel)}, countries: {panel['iso3'].nunique()}")

    # Convert to schema
    schema_data = panel_to_schema(panel)
    n_examples = len(schema_data["datasets"][0]["examples"])
    logger.info(f"Schema examples: {n_examples}")

    # Write outputs
    out_dir = BASE
    full_path = out_dir / "full_data_out.json"
    full_path.write_text(json.dumps(schema_data, indent=2, ensure_ascii=False))
    logger.info(f"Written full_data_out.json ({full_path.stat().st_size/1e6:.2f} MB)")

    # Audit
    audit = compute_audit(panel, overlap_stats)
    (out_dir / "data_audit.json").write_text(json.dumps(audit, indent=2))
    logger.info(f"Written data_audit.json")

    logger.info("=== Panel build COMPLETE ===")
    logger.info(f"Summary: {audit['n_countries_total']} countries, {audit['n_country_periods_total']} country-periods, {audit['n_countries_3plus_periods']} with ≥3 periods")
    logger.info(f"Education within-SD: extended={audit['within_sd_education_extended']}, 2-period={audit['within_sd_education_twoperiod']}, ratio={audit['within_sd_education_ratio']}x")


if __name__ == "__main__":
    main()
