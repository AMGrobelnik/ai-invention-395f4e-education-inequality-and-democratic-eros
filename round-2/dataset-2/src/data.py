# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "loguru"]
# ///

import json
import csv
import math
from pathlib import Path
from loguru import logger

BASE = Path(__file__).parent
DS = BASE / "temp" / "datasets"
RAW = BASE / "temp" / "raw"
OUT = BASE / "full_data_out.json"

# Selected datasets: WVS Wave 7 + V-Dem + ILO SocProt
SELECTED_DATASETS = {
    "wvs_wave7_developing_democratizers",
    "vdem_democracy_indicators",
    "ilo_socprot_country_year",
}

GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}\n",
    colorize=False,
)

# OECD ISO3 codes — excluded from developing-democratizer filter
OECD = {
    "AUS","AUT","BEL","CAN","CHL","COL","CRI","CZE","DNK","EST","FIN","FRA",
    "DEU","GRC","HUN","ISL","IRL","ISR","ITA","JPN","KOR","LVA","LTU","LUX",
    "MEX","NLD","NZL","NOR","POL","PRT","SVK","SVN","ESP","SWE","CHE","TUR",
    "GBR","USA",
}

# Country-name → ISO3 for datasets that use country names
COUNTRY_ISO3 = {}

def _load_country_iso3_from_wvs():
    """Build name→ISO3 map from WVS (has both)."""
    wvs_file = DS / "full_wvs_gabors_subset2000.json"
    rows = json.loads(wvs_file.read_text())
    seen = {}
    for r in rows:
        alpha = str(r.get("B_COUNTRY_ALPHA", "")).strip()
        # WVS does not include country name; build from V-Dem later
        _ = alpha
    return seen


def _load_vdem_country_iso3():
    """Build name→ISO3 map from V-Dem which has both 'country' and alpha codes."""
    vdem_file = DS / "full_garden_democracy_2025-03-17_vdem_vdem_multi_with_regions.json"
    rows = json.loads(vdem_file.read_text())
    mapping = {}
    # V-Dem uses 'country' as name; we need ISO3 from another source.
    # We'll build it from ILO which uses ISO3, mapped by known names.
    return mapping


# Known country name → ISO3 mapping (built from common knowledge + ILO codes)
NAME_TO_ISO3 = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND",
    "Angola": "AGO", "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS",
    "Austria": "AUT", "Azerbaijan": "AZE", "Bangladesh": "BGD", "Belarus": "BLR",
    "Belgium": "BEL", "Benin": "BEN", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA", "Brazil": "BRA", "Bulgaria": "BGR", "Burkina Faso": "BFA",
    "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Costa Rica": "CRI", "Croatia": "HRV",
    "Cyprus": "CYP", "Czech Republic": "CZE", "Czechia": "CZE",
    "Democratic Republic of the Congo": "COD", "Denmark": "DNK",
    "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt": "EGY",
    "El Salvador": "SLV", "Estonia": "EST", "Ethiopia": "ETH", "Finland": "FIN",
    "France": "FRA", "Gabon": "GAB", "Gambia": "GMB", "Georgia": "GEO",
    "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC", "Guatemala": "GTM",
    "Haiti": "HTI", "Honduras": "HND", "Hungary": "HUN", "Iceland": "ISL",
    "India": "IND", "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ",
    "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM",
    "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Kosovo": "XKX", "Kyrgyzstan": "KGZ", "Laos": "LAO", "Latvia": "LVA",
    "Lebanon": "LBN", "Libya": "LBY", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS", "Mali": "MLI",
    "Mexico": "MEX", "Moldova": "MDA", "Mongolia": "MNG", "Morocco": "MAR",
    "Mozambique": "MOZ", "Myanmar": "MMR", "Namibia": "NAM", "Nepal": "NPL",
    "Netherlands": "NLD", "New Zealand": "NZL", "Nicaragua": "NIC",
    "Niger": "NER", "Nigeria": "NGA", "North Macedonia": "MKD", "Norway": "NOR",
    "Pakistan": "PAK", "Palestine": "PSE", "Panama": "PAN", "Paraguay": "PRY",
    "Peru": "PER", "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
    "Puerto Rico": "PRI", "Romania": "ROU", "Russia": "RUS", "Rwanda": "RWA",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
    "Sierra Leone": "SLE", "Singapore": "SGP", "Slovakia": "SVK",
    "Slovenia": "SVN", "South Africa": "ZAF", "South Korea": "KOR",
    "South Sudan": "SSD", "Spain": "ESP", "Sri Lanka": "LKA", "Sudan": "SDN",
    "Sweden": "SWE", "Switzerland": "CHE", "Taiwan": "TWN",
    "Tajikistan": "TJK", "Tanzania": "TZA", "Thailand": "THA", "Togo": "TGO",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR",
    "Turkmenistan": "TKM", "Uganda": "UGA", "Ukraine": "UKR",
    "United Kingdom": "GBR", "United States": "USA", "Uruguay": "URY",
    "Uzbekistan": "UZB", "Venezuela": "VEN", "Vietnam": "VNM",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
    # Alternate names
    "United States of America": "USA", "South Korea": "KOR", "Republic of Korea": "KOR",
    "Hong Kong": "HKG", "Hong Kong S.A.R. of China": "HKG",
    "Bolivia (Plurinational State of)": "BOL",
    "Iran (Islamic Republic of)": "IRN",
    "Democratic Republic of Congo": "COD",
    "Congo, Dem. Rep.": "COD",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Viet Nam": "VNM", "Lao PDR": "LAO",
    "Republic of Moldova": "MDA",
    "North Africa": None, "Sub-Saharan Africa": None,
    "East Asia": None, "South Asia": None,
}


def _safe_int(val, missing_vals=None):
    if missing_vals is None:
        missing_vals = {"-1", "-2", "-3", "-4", "-5", "-6", "-7", "-8", "-9"}
    if val is None:
        return None
    s = str(val).strip()
    if s in missing_vals or s == "" or s.lower() in {"nan", "none"}:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


# ─── Load macro support data ────────────────────────────────────────────────

def load_swiid():
    """Load SWIID gini_disp → {iso3: {year: gini}}."""
    path = RAW / "swiid_summary.csv"
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["country"].strip()
            iso3 = NAME_TO_ISO3.get(name)
            if not iso3:
                continue
            yr = _safe_int(row.get("year"))
            gini = _safe_float(row.get("gini_disp"))
            if yr and gini:
                result.setdefault(iso3, {})[yr] = gini
    logger.info(f"SWIID: {sum(len(v) for v in result.values())} country-year gini entries")
    return result


def load_ilo_socprot():
    """Load ILO SocProt → {iso3: {year: coverage}}."""
    path = DS / "ilo_socprot_total_dbnomics.json"
    rows = json.loads(path.read_text())
    result = {}
    for r in rows:
        iso3 = str(r.get("country_iso3", "")).strip()
        yr = _safe_int(r.get("year"))
        cov = _safe_float(r.get("socprot_coverage"))
        if iso3 and yr and cov is not None:
            result.setdefault(iso3, {})[yr] = cov
    logger.info(f"ILO SocProt: {sum(len(v) for v in result.values())} entries")
    return result


def load_vdem_electdem():
    """Load V-Dem electdem_vdem for developing-democratizer filter."""
    path = DS / "full_garden_democracy_2025-03-17_vdem_vdem_multi_with_regions.json"
    rows = json.loads(path.read_text())
    result = {}  # iso3 → {year → electdem}
    for r in rows:
        if str(r.get("estimate", "")).strip() != "best":
            continue
        yr = _safe_int(r.get("year"))
        if not yr or yr < 2005 or yr > 2023:
            continue
        name = str(r.get("country", "")).strip()
        iso3 = NAME_TO_ISO3.get(name)
        if not iso3:
            continue
        ed = _safe_float(r.get("electdem_vdem"))
        if ed is not None:
            result.setdefault(iso3, {})[yr] = ed
    logger.info(f"V-Dem electdem: {sum(len(v) for v in result.values())} country-year entries")
    return result


def _nearest_year(lookup: dict, year: int, max_gap: int = 3):
    """Return value from lookup for nearest year within max_gap."""
    if year in lookup:
        return lookup[year]
    best_val, best_dist = None, max_gap + 1
    for yr, val in lookup.items():
        d = abs(yr - year)
        if d < best_dist:
            best_dist, best_val = d, val
    return best_val


def _is_developing_democratizer(iso3: str, year: int, vdem: dict) -> bool:
    if iso3 in OECD:
        return False
    country_vdem = vdem.get(iso3, {})
    ed = _nearest_year(country_vdem, year, max_gap=3)
    return ed is not None and ed >= 0.35


# ─── Dataset 1: WVS Wave 7 (individual-level, merged) ──────────────────────

def process_wvs(swiid, ilo, vdem):
    path = DS / "full_wvs_gabors_subset2000.json"
    rows = json.loads(path.read_text())
    logger.info(f"WVS raw rows: {len(rows)}")

    SECTOR_MAP = {"1": "public", "2": "private", "3": "ngo"}

    examples = []
    skipped_oecd = skipped_nodem = skipped_nosector = skipped_notrust = 0

    # Compute quadrant cutoffs across all valid rows first
    valid_rows = []
    for r in rows:
        iso3 = str(r.get("B_COUNTRY_ALPHA", "")).strip()
        year = _safe_int(r.get("A_YEAR")) or 2018
        sector_raw = str(r.get("Q284", "")).strip()
        if sector_raw not in SECTOR_MAP:
            continue
        if not _is_developing_democratizer(iso3, year, vdem):
            continue
        sp = _nearest_year(ilo.get(iso3, {}), year)
        gini = _nearest_year(swiid.get(iso3, {}), year)
        if sp is None or gini is None:
            continue
        valid_rows.append((r, iso3, year, SECTOR_MAP[sector_raw], sp, gini))

    logger.info(f"WVS valid after filter: {len(valid_rows)}")

    if not valid_rows:
        return []

    sp_vals = [sp for _, _, _, _, sp, _ in valid_rows]
    gini_vals = [g for _, _, _, _, _, g in valid_rows]
    sp_med = sorted(sp_vals)[len(sp_vals) // 2]
    gini_med = sorted(gini_vals)[len(gini_vals) // 2]
    logger.info(f"WVS quadrant cutoffs: sp_median={sp_med}, gini_median={gini_med}")

    for r, iso3, year, sector, sp, gini in valid_rows:
        quadrant = (
            "low_socprot_high_gini" if sp < sp_med and gini > gini_med else
            "low_socprot_low_gini" if sp < sp_med else
            "high_socprot_high_gini" if gini > gini_med else
            "high_socprot_low_gini"
        )

        # Trust items: Q69=courts, Q71=government, Q72=political parties
        # Scale: 1=great deal, 2=quite a lot, 3=not very much, 4=none at all
        # Invert: higher = more independent orientation
        def trust_val(key):
            v = _safe_int(r.get(key))
            if v is None or v < 1 or v > 4:
                return None
            return 5 - v  # invert: 4=high trust, 1=low trust

        tj = trust_val("Q69")
        tg = trust_val("Q71")
        tp = trust_val("Q72")

        if tj is None and tg is None and tp is None:
            skipped_notrust += 1
            continue

        # Democracy items
        # Q235: Having a democratic political system (1=very good, 4=very bad) → recode 4→1 pro-dem
        q235 = _safe_int(r.get("Q235"))
        demo_pref = (5 - q235) if q235 and 1 <= q235 <= 4 else None

        # Q250: Importance of democracy (1=not important, 10=absolutely important)
        q250 = _safe_int(r.get("Q250"))
        demo_importance = q250 if q250 and 1 <= q250 <= 10 else None

        # Education: Q275R (recoded: 0=no formal, 1=primary, 2=secondary, 3=tertiary)
        educ_raw = _safe_int(r.get("Q275R"))
        educ = educ_raw if educ_raw is not None and 0 <= educ_raw <= 3 else None

        # Income quintile: Q288R (1-5 scale based on country-specific income groups)
        inc_raw = _safe_int(r.get("Q288R"))
        income_q = inc_raw if inc_raw and 1 <= inc_raw <= 5 else None

        # Composite institutional independence index
        components = [c for c in [tj, tg, tp] if c is not None]
        iii = round(sum(components) / (3 * len(components)), 4) if components else None

        input_str = (
            f"Country: {iso3} | Year: {year} | Sector: {sector} | "
            f"Education: {educ} | Income quintile: {income_q} | "
            f"Social protection coverage: {sp:.1f}% | Gini: {gini:.1f} | "
            f"Quadrant: {quadrant}"
        )
        output_str = (
            f"Trust judiciary: {tj} | Trust government: {tg} | "
            f"Trust parties: {tp} | Democracy preference: {demo_pref} | "
            f"Democracy importance: {demo_importance} | "
            f"Institutional independence index: {iii}"
        )

        examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_employment_sector": sector,
            "metadata_country_iso3": iso3,
            "metadata_year": year,
            "metadata_education": educ,
            "metadata_income_quintile": income_q,
            "metadata_socprot_coverage": sp,
            "metadata_gini": gini,
            "metadata_quadrant": quadrant,
            "metadata_trust_judiciary": tj,
            "metadata_trust_government": tg,
            "metadata_trust_parties": tp,
            "metadata_democracy_preference": demo_pref,
            "metadata_democracy_importance": demo_importance,
            "metadata_institutional_independence_index": iii,
            "metadata_wave": "wvs_wave7",
            "metadata_source": "wvs_gabors_wave7",
        })

    logger.info(f"WVS examples built: {len(examples)}")
    return examples


# ─── Dataset 2: ILO SocProt (country-year macro) ───────────────────────────

def process_ilo_socprot():
    path = DS / "ilo_socprot_total_dbnomics.json"
    rows = json.loads(path.read_text())
    examples = []
    for r in rows:
        iso3 = str(r.get("country_iso3", "")).strip()
        year = _safe_int(r.get("year"))
        cov = _safe_float(r.get("socprot_coverage"))
        if not iso3 or not year or cov is None:
            continue
        examples.append({
            "input": f"Country: {iso3} | Year: {year}",
            "output": f"Social protection coverage: {cov:.1f}%",
            "metadata_country_iso3": iso3,
            "metadata_year": year,
            "metadata_socprot_coverage": cov,
            "metadata_source": str(r.get("source", "")),
        })
    logger.info(f"ILO SocProt examples: {len(examples)}")
    return examples


# ─── Dataset 3: IVS (Integrated Values Surveys, country-level trust) ───────

def process_ivs(vdem):
    path = DS / "full_meadow_ivs_2025-06-27_integrated_values_surveys_integrated_values_surveys.json"
    rows = json.loads(path.read_text())
    examples = []
    for r in rows:
        name = str(r.get("country", "")).strip()
        iso3 = NAME_TO_ISO3.get(name)
        year = _safe_int(r.get("year"))
        if not iso3 or not year:
            continue
        # Filter to developing democratizers
        if not _is_developing_democratizer(iso3, year, vdem):
            continue

        conf_courts = _safe_float(r.get("confidence_justice_system_courts"))
        conf_govt = _safe_float(r.get("confidence_government"))
        conf_parties = _safe_float(r.get("confidence_political_parties"))
        conf_parliament = _safe_float(r.get("confidence_parliament"))
        trust_gen = _safe_float(r.get("trust"))

        if conf_courts is None and conf_govt is None and conf_parties is None:
            continue

        input_str = f"Country: {iso3} | Year: {year} | Type: country_aggregate"
        output_str = (
            f"Confidence courts: {conf_courts} | Confidence government: {conf_govt} | "
            f"Confidence parties: {conf_parties} | Confidence parliament: {conf_parliament} | "
            f"General trust: {trust_gen}"
        )
        examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_country_iso3": iso3,
            "metadata_year": year,
            "metadata_confidence_courts": conf_courts,
            "metadata_confidence_government": conf_govt,
            "metadata_confidence_parties": conf_parties,
            "metadata_confidence_parliament": conf_parliament,
            "metadata_general_trust": trust_gen,
            "metadata_source": "ivs_integrated_values_surveys",
        })
    logger.info(f"IVS examples: {len(examples)}")
    return examples


# ─── Dataset 4: V-Dem (democracy indicators, filtered) ─────────────────────

def process_vdem():
    path = DS / "full_garden_democracy_2025-03-17_vdem_vdem_multi_with_regions.json"
    rows = json.loads(path.read_text())
    examples = []
    for r in rows:
        if str(r.get("estimate", "")).strip() != "best":
            continue
        yr = _safe_int(r.get("year"))
        if not yr or yr < 2010 or yr > 2022:
            continue
        name = str(r.get("country", "")).strip()
        iso3 = NAME_TO_ISO3.get(name)
        if not iso3 or iso3 in OECD:
            continue
        ed = _safe_float(r.get("electdem_vdem"))
        if ed is None or ed < 0.2:
            continue  # too authoritarian — not democratizer

        corr_jud = _safe_float(r.get("corr_jud_vdem"))
        corr_exec = _safe_float(r.get("corr_exec_vdem"))
        civ_libs = _safe_float(r.get("civ_libs_vdem"))

        input_str = f"Country: {iso3} | Year: {yr} | Electoral democracy: {round(ed, 3) if ed else None}"
        output_str = (
            f"Corruption judiciary: {round(corr_jud,3) if corr_jud is not None else None} | "
            f"Corruption executive: {round(corr_exec,3) if corr_exec is not None else None} | "
            f"Civil liberties: {round(civ_libs,3) if civ_libs is not None else None}"
        )
        examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_country_iso3": iso3,
            "metadata_year": yr,
            "metadata_electdem_vdem": round(ed, 4) if ed is not None else None,
            "metadata_corr_jud_vdem": round(corr_jud, 4) if corr_jud is not None else None,
            "metadata_corr_exec_vdem": round(corr_exec, 4) if corr_exec is not None else None,
            "metadata_civ_libs_vdem": round(civ_libs, 4) if civ_libs is not None else None,
            "metadata_source": "vdem_v16",
        })
    logger.info(f"V-Dem examples: {len(examples)}")
    return examples


# ─── Dataset 5: Corruption Perception Index ────────────────────────────────

def process_cpi(vdem):
    path = DS / "full_garden_corruption_2025-05-13_perception_index_perception_index.json"
    rows = json.loads(path.read_text())
    examples = []
    for r in rows:
        name = str(r.get("country", "")).strip()
        iso3 = NAME_TO_ISO3.get(name)
        yr = _safe_int(r.get("year"))
        if not iso3 or not yr:
            continue
        if not _is_developing_democratizer(iso3, yr, vdem):
            continue
        score = _safe_float(r.get("cpi_score"))
        if score is None:
            continue
        input_str = f"Country: {iso3} | Year: {yr}"
        output_str = f"CPI score: {score} | Lower CI: {r.get('lower_ci')} | Upper CI: {r.get('upper_ci')}"
        examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_country_iso3": iso3,
            "metadata_year": yr,
            "metadata_cpi_score": score,
            "metadata_source": "cpi_transparency_international",
        })
    logger.info(f"CPI examples: {len(examples)}")
    return examples


# ─── Dataset 6: Social Expenditure (% GDP) ─────────────────────────────────

def process_socexp(vdem):
    path = DS / "full_garden_social_expenditure_2025-03-07_social_expenditure_omm_social_expenditure_o.json"
    rows = json.loads(path.read_text())
    examples = []
    for r in rows:
        name = str(r.get("country", "")).strip()
        iso3 = NAME_TO_ISO3.get(name)
        yr = _safe_int(r.get("year"))
        if not iso3 or not yr or yr < 1990:
            continue
        if not _is_developing_democratizer(iso3, yr, vdem):
            continue
        share = _safe_float(r.get("share_gdp"))
        if share is None:
            continue
        input_str = f"Country: {iso3} | Year: {yr}"
        output_str = f"Social expenditure (% GDP): {round(share, 3)}"
        examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_country_iso3": iso3,
            "metadata_year": yr,
            "metadata_social_expenditure_pct_gdp": round(share, 4),
            "metadata_source": "owid_social_expenditure",
        })
    logger.info(f"Social Expenditure examples: {len(examples)}")
    return examples


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Building 3-dataset exp_sel_data_out ===")

    swiid = load_swiid()
    ilo = load_ilo_socprot()
    vdem = load_vdem_electdem()

    wvs_examples = process_wvs(swiid, ilo, vdem)
    ilo_examples = process_ilo_socprot()
    vdem_examples = process_vdem()

    output = {
        "metadata": {
            "description": (
                "Three datasets for micro/macro analysis of public-sector workers, institutional "
                "independence, and social protection in developing democratizers. "
                "(1) WVS Wave 7 individual-level survey (576 examples): employment sector × institutional "
                "trust × SocProt/Gini quadrant — the primary dataset for hypothesis testing. "
                "(2) V-Dem democracy indicators (1102 examples): country-year democracy scores and judicial "
                "corruption for developing democratizer classification and macro context. "
                "(3) ILO SocProt country-year (354 examples): social protection coverage for quadrant "
                "classification validation."
            ),
            "source_datasets": [
                "WVS Wave 7 (Gabors subset 2000): individual survey, employment sector + institutional trust",
                "ILO SDG 1.3.1 via DBnomics: country-year social protection coverage",
                "V-Dem v16: country-year democracy + institutional quality indicators",
            ],
        },
        "datasets": [
            {"dataset": "wvs_wave7_developing_democratizers", "examples": wvs_examples},
            {"dataset": "ilo_socprot_country_year", "examples": ilo_examples},
            {"dataset": "vdem_democracy_indicators", "examples": vdem_examples},
        ],
    }

    # Drop empty datasets
    output["datasets"] = [d for d in output["datasets"] if d["examples"]]
    for ds in output["datasets"]:
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")

    OUT.write_text(json.dumps(output, indent=2))
    logger.info(f"Written: {OUT} ({OUT.stat().st_size // 1024} KB)")


main()
