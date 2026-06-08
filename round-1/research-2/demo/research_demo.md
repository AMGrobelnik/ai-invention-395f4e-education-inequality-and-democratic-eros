# Data Availability Audit: Education Trap & Democratization Panel Study Sources

## Summary

Systematic audit of six data sources required for a panel study of state-dependent education traps and democratization in post-1990 democratizing countries (N~50-80, 1990-2022). (1) ILO SDG 1.3.1 (SDG_0131_SEX_SOC_RT): 249 reference areas, annual through 2021, bulk CSV.gz download; obs_status field distinguishes directly-reported (DR/A) from ILO modelled (ME) values — many developing-country observations are modelled. (2) V-Dem V16 (March 2026): 202 countries 1789-2023; v2juhcind confirmed as correct variable for 'judicial compliance under government pressure'; NO variable exists for 'professional NGO staff independence' — best proxy is v2cseeorgs (CSO entry/exit freedom); v2csprtcpt measures population participation level, not staff independence. (3) Mean Years of Schooling: UNDP HDR (193 countries, annual 1990-2023 observed) is preferred over Barro-Lee (146 countries, 5-year intervals; post-2015 are projections only). (4) SWIID 9.2 (April 2026): gini_disp for 199 countries, 1960-present; Sub-Saharan Africa values heavily imputed — use gini_disp_se for quality assessment. (5) ILO EMP_TEMP_SEX_INS_NB: critical developing-country coverage gap; OECD-centric dataset cannot support a 50-80 country developing-democratizer panel; data as old as 2009 for some countries. (6) IMF MONA (2002-present, tab-delimited) plus Dreher 2006 supplement (160 countries, 1970-2019, Excel) for 1990-2001 gap.

## Research Findings

## 1. ILO ILOSTAT SDG 1.3.1 — Social Protection Coverage

**Variable:** `SDG_0131_SEX_SOC_RT`
**Download:** `https://ilostat.ilo.org/data/bulk/` → file `SDG_0131_SEX_SOC_RT_A.csv.gz` (annual, updated weekly)
**Coverage:** 249 reference areas, annual data through 2021

### obs_status Data Quality Flag

The `obs_status` column is essential for distinguishing data quality levels:

| Code | Meaning |
|------|--------|
| A    | Normal / directly reported |
| DR   | Directly reported |
| NE   | National estimate |
| ME   | ILO modelled estimate |
| BE   | Break in series |
| E    | Estimated |
| P    | Provisional |
| U    | Unreliable |
| D    | Definition differs |

Dictionary source: `[dic]/obs_status_en.csv` in bulk download; or R: `rilostat::get_ilostat_dic('obs_status')`.

Many developing-country observations carry `obs_status=ME` (ILO modelled estimate). Studies requiring directly-reported data should filter to `obs_status IN ('A', 'DR', 'NE')`, which reduces the usable sample in low-income country groups substantially.

**Verdict: AVAILABLE with caveats.** No 2022 data. Filter obs_status before analysis.

---

## 2. V-Dem Sub-Indices

**Version:** V16 (March 2026), 202 countries, 1789–2023
**Download:** `https://v-dem.net/data/the-v-dem-dataset/` → `V-Dem-CY-Full+Others-v16.csv.zip`
**Formats:** CSV, Stata, R, SPSS

### Confirmed Variables

**Democracy:**
- `v2x_libdem` — Liberal Democracy Index (0–1)
- `v2x_regime` — Regimes of the World (0=Closed Autocracy, 3=Liberal Democracy)

**Judicial (confirmed correct):**
- `v2juhcind` — High Court Independence: 'How often does the court make decisions reflecting government wishes regardless of sincere legal view?' Low value = bows to government. **This is the correct variable for judicial compliance under government pressure.**
- `v2juhccomp` — Government compliance WITH court rulings (opposite direction — do not confuse with v2juhcind)
- `v2x_jucon` — Judicial constraints index (aggregate including v2juhcind, v2juhccomp, v2jucomp, v2juncind)

**Civil society:**
- `v2cseeorgs` — CSO entry and exit: freedom for organizations to form and operate. **Best proxy for NGO operational autonomy.**
- `v2csreprss` — CSO repression by government
- `v2csprtcpt` — CSO participatory environment: population involvement level (0=state-sponsored, 3=highly participatory). **Measures how many people participate, NOT professional staff independence.**
- `v2xcs_ccsi` — Core Civil Society Index: aggregate of v2cseeorgs, v2csreprss, v2csprtcpt (0–1)

### Critical Variable Identification Problem

The hypothesis references 'professional NGO staff independence' as a V-Dem sub-indicator. **This variable does not exist in V-Dem V14/V16.** Codebook grep confirms no variable uses this label or concept. `v2csstruc` measures large vs. small CSO dominance; `v2csprtcpt` measures population participation level. Neither captures staff professionalism or independence.

**Required correction:** Substitute `v2cseeorgs` (CSO entry/exit freedom) as the primary proxy for NGO operational autonomy. Document the substitution in the methods section. If a composite is preferred, use `v2xcs_ccsi`.

**Verdict: AVAILABLE.** V16 covers 202 countries 1789-2023. Variable identification must be corrected before data assembly.

---

## 3. Mean Years of Schooling

### Source Comparison

| Feature | UNDP HDR | Barro-Lee |
|---------|----------|-----------|
| Countries | 193 | 146 |
| Frequency | Annual | 5-year intervals |
| Latest observed | 2023 | 2015 |
| 2015–2022 data | Observed | **Projections** |
| Download | CSV | CSV |

**UNDP HDR download:** `https://hdr.undp.org/data-center/documentation-and-downloads` → `HDR_STATS_2023.csv`
**Barro-Lee download:** `http://www.barrolee.com/` → `BL2013_MF1599_v2.2.csv`

Barro-Lee's 2015–2040 values are **projections**, not observed data. For a 1990–2022 panel, using Barro-Lee for the post-2015 window is methodologically inappropriate. UNDP HDR provides annual directly-observed data for 193 countries through 2023.

**Verdict: AVAILABLE.** Use UNDP HDR as primary source. Barro-Lee only for pre-1990 5-year interval analyses.

---

## 4. SWIID Standardized World Income Inequality Database

**Version:** 9.2 (April 2026)
**Download:** `https://fsolt.org/swiid/` or Harvard Dataverse `doi:10.7910/DVN/LM4OWF`
**Formats:** Stata (.dta), R (.rds), summary CSV (swiid_summary.csv)
**Coverage:** 199 countries, 1960–present

**Key variables:**
- `gini_disp` — Disposable income Gini (post-tax, post-transfer) — PRIMARY variable
- `gini_mkt` — Market income Gini (pre-tax, pre-transfer)
- `gini_disp_se` — Standard error of gini_disp (from multiple imputation)
- `gini_mkt_se` — Standard error of gini_mkt

SWIID fills missing country-years via multiple imputation from LIS microdata and secondary sources. Sub-Saharan Africa has a major source data gap; values are heavily imputed with high SE. Southeast Asia also has sparse post-1990 survey data. Filter or weight by `gini_disp_se`; values with SE > 3 are unreliable approximations. SWIID and WID.world Gini values can diverge by 5+ points for the same country (e.g., China) due to methodological differences.

**Verdict: AVAILABLE.** `gini_disp` is the correct variable. Sub-Saharan Africa coverage is imputed and approximate.

---

## 5. ILO ILOSTAT Public Sector Employment (EMP_TEMP_SEX_INS_NB)

**Variable:** `EMP_TEMP_SEX_INS_NB`
**Download:** `https://ilostat.ilo.org/data/bulk/` → `EMP_TEMP_SEX_INS_NB_A.csv.gz`

This indicator tracks employment by institutional sector (public vs. private) from Labour Force Surveys (LFS) where respondents self-identify their sector. Coverage is heavily skewed toward OECD/EU:

- OECD countries: strong coverage for ~38 members
- Subnational breakdown: only ~28 OECD countries have this (per OECD 2024 report)
- Developing countries: limited, patchy, often outdated; some countries' most recent data is 2009 or earlier
- Self-reported sector classification is unreliable in low-income countries lacking regular LFS programs

The OECD 2024 report *Size and Composition of Public Employment: Data Sources, Methods and Gaps* is the most authoritative audit of this dataset and confirms that developing-country coverage is insufficient for broad cross-national panels.

**A 50-80 country panel of post-1990 developing democratizers cannot be assembled from this source alone.**

Alternative sources to investigate: IMF Government Finance Statistics (general government employment by function); World Bank WDI government employment proxies; Cross-National Time-Series Data Archive (CNTS).

**Verdict: CRITICAL GAP for developing-country sample.** Strong for OECD only.

---

## 6. IMF MONA + Dreher (2006) Supplement

**MONA download:** `https://www.imf.org/external/np/pdr/mona/Arrangements.aspx` (tab-delimited)
**MONA coverage:** 2002–present

MONA does not include a pre-coded binary 'program active' column. Researchers must derive this from arrangement approval date and end date for each country-year.

**Critical gap:** MONA starts in 2002. A 1990–2022 panel requires supplementing 1990–2001 from Dreher (2006).

**Dreher (2006) supplement:**
- Download: `https://axel-dreher.de/wp-content/uploads/Dreher%20IMF%20and%20WB.xls`
- Format: Excel (.xls)
- Coverage: 160 countries, 1970–2019
- Variable: pre-coded binary yearly dummy (1 = IMF program active in country-year)
- This is the standard supplement used across the political economy of IMF conditionality literature

Combined MONA + Dreher: full 1990–2022 binary coverage for ~160 countries. Program type heterogeneity (SBA vs. EFF vs. PRGT) is conflated in a binary dummy — use MONA arrangement-type field if conditionality depth matters.

**Verdict: AVAILABLE with supplement.** Dreher (2006) fills the 1990–2001 gap. Together covers the full study window.

---

## Summary Table

| Source | Variable | Countries | Years | Status |
|--------|----------|-----------|-------|--------|
| ILO SDG 1.3.1 | SDG_0131_SEX_SOC_RT | 249 | 1990-2021 | Available (filter obs_status) |
| V-Dem V16 | v2juhcind, v2cseeorgs, v2x_regime | 202 | 1789-2023 | Available (fix variable ID) |
| UNDP HDR | Mean years schooling | 193 | 1990-2023 | Available (preferred over BL) |
| SWIID 9.2 | gini_disp | 199 | 1960-present | Available (SSA imputed) |
| ILO EMP_TEMP | EMP_TEMP_SEX_INS_NB | OECD only | Varies | CRITICAL GAP for dev. countries |
| IMF MONA | Program active (derived) | 160+ | 2002-2022 | Available (use Dreher pre-2002) |
| Dreher 2006 | IMF binary | 160 | 1970-2019 | Required supplement |

## Priority Actions

1. **Fix V-Dem variable:** replace 'professional NGO staff independence' with `v2cseeorgs`
2. **Assess EMP_TEMP_SEX_INS_NB gap:** query ILOSTAT for count of available developing-country observations; pursue alternatives if < 50% coverage
3. **Filter SDG 1.3.1 by obs_status:** count ME vs. DR/A by country income group
4. **Switch to UNDP HDR** for 2015-2022 schooling data — Barro-Lee post-2015 are projections
5. **Download Dreher 2006 XLS** to bridge the 1990-2001 MONA gap

## Sources

[1] [ILOSTAT Bulk Download — SDG_0131_SEX_SOC_RT and EMP_TEMP_SEX_INS_NB](https://ilostat.ilo.org/data/bulk/) — Official ILO bulk data portal. SDG 1.3.1 indicator (SDG_0131_SEX_SOC_RT_A.csv.gz): 249 reference areas, annual through 2021. Public sector employment (EMP_TEMP_SEX_INS_NB_A.csv.gz): LFS-based institutional sector employment, primarily OECD coverage. obs_status dictionary at [dic]/obs_status_en.csv distinguishes directly-reported (DR/A) from modelled (ME) values.

[2] [V-Dem Dataset V16 (March 2026) — Varieties of Democracy](https://v-dem.net/data/the-v-dem-dataset/) — V-Dem V16: 202 countries, 1789-2023. Key variables confirmed: v2juhcind (High Court Independence = judicial compliance under government pressure), v2cseeorgs (CSO entry/exit = best NGO autonomy proxy), v2csprtcpt (CSO participatory environment — population participation level, NOT professional staff independence), v2xcs_ccsi (Core Civil Society Index composite). No variable for 'professional NGO staff independence' exists in V14/V16 codebook.

[3] [UNDP Human Development Reports Data — Mean Years of Schooling](https://hdr.undp.org/data-center/documentation-and-downloads) — Annual mean years of schooling for 193 countries, 1990-2023, directly observed. CSV download (HDR_STATS_2023.csv). Preferred over Barro-Lee for the 2015-2022 study window because Barro-Lee post-2015 values are projections, not observed data.

[4] [SWIID 9.2 — Standardized World Income Inequality Database (April 2026)](https://fsolt.org/swiid/) — 199 countries, 1960-present. Variables: gini_disp (disposable income Gini), gini_mkt (market income Gini), gini_disp_se, gini_mkt_se. Multiple imputation from LIS + secondary sources. Also at Harvard Dataverse doi:10.7910/DVN/LM4OWF. Sub-Saharan Africa values heavily imputed; use gini_disp_se > 3 as unreliability threshold.

[5] [OECD (2024) — Size and Composition of Public Employment: Data Sources, Methods and Gaps](https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/12/size-and-composition-of-public-employment-data-sources-methods-and-gaps_f6c2babd/32c747be-en.pdf) — Most authoritative audit of ILOSTAT public sector employment data. Confirms coverage is primarily OECD-centric; only ~28 OECD countries have subnational breakdown. Developing-country data can be as old as 2009. Self-reported LFS sector classification is a key limitation in low-income countries. Developing democratizer panel of 50-80 countries cannot be built from EMP_TEMP_SEX_INS_NB alone.

[6] [Dreher (2006) — IMF and World Bank Dataset: 160 countries, 1970-2019](https://axel-dreher.de/wp-content/uploads/Dreher%20IMF%20and%20WB.xls) — Excel dataset providing binary yearly dummies for IMF program presence. Standard supplement in the political economy of IMF conditionality literature. Required to fill the 1990-2001 gap in IMF MONA (which starts 2002). Download at axel-dreher.de. 160 countries, annual, 1970-2019.

## Follow-up Questions

- For the EMP_TEMP_SEX_INS_NB gap: which specific developing-country democratizers in the target sample (e.g., Indonesia, Ghana, Senegal, Bolivia, Mongolia) have ILOSTAT public sector employment data available, and for what years?
- Is there a single cross-national dataset that combines public sector employment for both OECD and developing countries for 1990-2022, such as the IMF Government Finance Statistics or Cross-National Time-Series Data Archive?
- For V-Dem's v2cseeorgs as proxy for 'professional NGO staff independence': how strongly does v2cseeorgs correlate with direct measures of NGO professionalization from other sources (e.g., Johns Hopkins Comparative Nonprofit Sector Project), and is the substitution methodologically defensible?
- Given that SDG 1.3.1 data only run through 2021, is there a way to obtain provisional 2022 social protection coverage estimates from ILO's World Social Protection Report or administrative data — and would those be methodologically comparable to the ILOSTAT series?
- How should the panel handle countries where gini_disp_se > 3 for multiple consecutive years — exclude those observations, impute from WDI Gini, or include with inverse-variance weighting?

---
*Generated by AI Inventor Pipeline*
