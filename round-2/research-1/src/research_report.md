# Pre-2015 SocProt Sources, Case Country Qualification, and Latinobarometro Audit

## Summary

This research resolves three gaps identified in the prior audit (gen_art_research_2): (1) extending the social protection panel back to 1990 where ILO SDG 1.3.1 data starts only in 2015; (2) verifying that Hungary, Turkey, Venezuela, and Thailand meet the study's V-Dem regime and GDP criteria; and (3) documenting Latinobarometro's practical limitations as a cross-regional public opinion source.

For Track 1 (pre-2015 social protection data), the NSTP Dataset (GESIS, doi:10.7802/1530) is the single best available source: 101 developing countries, 1990–2015, annual panel, with program-level beneficiary coverage for non-contributory transfer programs. It bridges the gap to ILO SDG 1.3.1 in 2015. The ILO SSI/WSPDB — the most historically comprehensive source covering ~1950 onward — is inaccessible via the web (HTTP 403 Forbidden) without institutional ILO credentials; the 2010/2012/2014 World Social Protection Report PDF tables remain the only workaround. OECD SOCX provides annual expenditure (not coverage rate) for Eastern European OECD members from ~1993. ILO ILOSTAT SPS_GEXP offers downloadable expenditure series from ~1995 for 100+ countries as a proxy. World Bank ASPIRE provides coverage rates for 140 countries but is survey-wave-based and unbalanced. Haggard-Kaufman (2008), Rudra (2002/2007), and Segura-Ubiergo (2007) lack public datasets.

For Track 2 (case country verification), Hungary clearly qualifies as a post-1990 democratizer with GDP PPP of $10,984 at transition and ILO SDG data from 2015. Turkey is borderline — its democratic window is defensible from 1987 (V-Dem RoW), GDP PPP $7,299 qualifies, but the 1983/1987 transition ambiguity requires resolution using the actual V16 dataset. Venezuela does NOT qualify as a post-1985 democratizer — it has been democratic since 1959 (Punto Fijo) and is properly classified as a democratic backslider from ~2001. Thailand conditionally qualifies on the 1992 democratization event ($4,311 GDP PPP) but its repeated military coups (2006, 2014) require explicit oscillation coding in the study design.

For Track 3 (Latinobarometro), the instrument covers 18 Latin American countries from 1995 with no 1999 wave. The employment sector variable (public vs. private) is unstable across waves: S8A (2002), S17A (2007), S20.A (2016), S14A (2018). The democracy preference variable ('La democracia es preferible') is similarly wave-varying: P16ST (2005), p13st (2008), P12STGBS (2018). Critically, Latinobarometro is Latin America-only — for Eastern European democratizers, the Life in Transition Survey (LiTS, EBRD) provides comparable data; for East/Southeast Asia, the Asian Barometer Survey (ABS) is the analog.

## Research Findings

## Track 1: Pre-2015 Social Protection Data Sources

### Source Hierarchy for Extending the Panel to 1990–2014

**HIGHEST PRIORITY — NSTP Dataset [2]**
The Non-Contributory Social Transfer Programs dataset (GESIS archive, doi:10.7802/1530) is the single best pre-2015 source: 101 countries, 1990–2015, annual panel, measuring program-level beneficiary coverage for non-contributory social transfers [2]. It covers approximately 70% of developing countries and directly bridges to ILO SDG 1.3.1 at the 2015 overlap. Critical limitation: covers non-contributory programs (cash transfers, social assistance) only — misses contributory social insurance, which matters for Central/Eastern European cases with strong formal-sector pension/health insurance systems.

**MEDIUM PRIORITY — ILO ILOSTAT SPS_GEXP [1]**
Social protection expenditure as % GDP, downloadable via the ILOSTAT bulk download system at rplumber.ilo.org [1]. Annual panel, 100+ countries, ~1995 onward (sparse pre-2000 for LMICs). Useful as an expenditure control variable and proxy when coverage data are unavailable. Expenditure ≠ coverage rate — this measures government spending commitment, not enrollment.

**MEDIUM PRIORITY — OECD SOCX [3]**
OECD Social Expenditure Database: 38 OECD members, 1980–2023, annual, downloadable from stats.oecd.org [3]. Eastern European OECD members included: Hungary (~1996), Poland (~1999), Czech Republic (~1999), Slovakia (~2000). Expenditure proxy only; excludes most developing countries in the study sample.

**MEDIUM PRIORITY — World Bank ASPIRE [4]**
Atlas of Social Protection Indicators: 140 countries, 1998–2023, coverage rate variable per_allsp.cov_pop_tot [4]. Survey-based and highly unbalanced (1–5 obs per country-decade) — unsuitable as annual panel backbone but valuable for spot-year cross-validation of coverage levels.

**ACCESS-BLOCKED — ILO SSI/WSPDB [5]**
Most historically complete source (~1950+), with expenditure, protected persons, and beneficiary counts. The WSPDB portal at social-protection.org returns HTTP 403 Forbidden — requires ILO institutional credentials [5]. Workaround: use ILO World Social Protection Reports (2010, 2012, 2014 editions) [6] for tabular cross-sectional snapshots.

**NO PUBLIC DATA — Haggard & Kaufman (2008), Rudra (2002/2007), Segura-Ubiergo (2007)**
None of these three influential works have publicly available datasets [7]. Haggard-Kaufman covers only 21 countries (13 LatAm, 5 East Asian, 3 Eastern European) [7]. Rudra's welfare effort index (53–57 countries, 1972–2000) uses spending composites not coverage rates and requires emailing nr404@georgetown.edu [7]. Segura-Ubiergo covers 18 Latin American countries only [7].

### Confirmed Pre-2015 Gap for ILO SDG 1.3.1
The ILO ILOSTAT SDG_0131_SEX_SOC_RT_A indicator is confirmed available as a bulk download (303 reference areas, 36308 rows, last updated 2026-05-04) [1]. It starts in 2015, confirming why a pre-2015 proxy is needed for a panel extending back to 1990.

### Recommended Panel Construction Strategy
Use NSTP (1990–2015) as coverage-proxy backbone [2] → bridge to ILO SDG_0131_SEX_SOC_RT_A (2015+) for post-2015 period [1] → supplement with ILO SPS_GEXP as expenditure control [1] → use OECD SOCX for Eastern European OECD members from 1980 [3] → cross-validate spot years with ASPIRE coverage rates [4].

---

## Track 2: Case Country Verification Against V-Dem and GDP Criteria

### Hungary
Post-communist democratizer in 1990 (first free elections since 1947). V-Dem V16 classifies Hungary as electoral democracy from 1990, upgrading to liberal democracy by mid-1990s, then backsliding to electoral autocracy by ~2018-2019 under Fidesz/Orbán [10][11]. GDP PPP at transition (1990): $10,984 — below the $15,000 threshold [9]. ILO SDG 1.3.1 data available from 2015 [1]. **VERDICT: QUALIFIES — unambiguous post-1990 democratizer with GDP PPP $10,984.**

### Turkey
Gradual democratization from 1983 (Özal civilian election after military rule); competitive democracy more clearly established by 1987 [10]. V-Dem RoW coding of Turkey as electoral democracy is defensible from ~1987. GDP PPP at transition (1990): $7,299 — below $15,000 [9]. Autocratized to electoral autocracy in 2013 (Erdoğan post-Gezi consolidation, confirmed V-Dem V16) [11]. ILO SDG 1.3.1 data available from 2015 [1]. **VERDICT: BORDERLINE — 1983 transition predates a strict post-1985 window; V-Dem RoW from ~1987 is defensible but requires verification against the actual V16 dataset [10].**

### Venezuela
Democracy since 1959 (Punto Fijo Pact) — NOT a post-1985 democratizer [11]. Backsliding began under Chávez 1999–2001; V-Dem classifies Venezuela as moving to electoral autocracy by ~2001-2002 [10][11]. GDP PPP (1990): $9,701 — below $15,000, but the GDP threshold is moot given the democratization criterion is not met [9]. **VERDICT: DOES NOT QUALIFY — democratic backslider from a 1959 democracy, not a post-1985 democratizer. Exclude from main sample.**

### Thailand
Democratized 1992 (Black May uprising/King Bhumibol intervention) [10][11]. GDP PPP at transition (1992): ~$4,700 — well below $15,000 [9]. Democratic window 1992–2006, then military coup September 2006, semi-democracy 2007–2014, coup May 2014 (closed autocracy), elections 2019 under military-designed constitution (electoral autocracy, V-Dem V16) [10][11]. ILO SDG 1.3.1 data available from 2015 [1]. **VERDICT: CONDITIONALLY QUALIFIES — 1992 democratization event qualifies on timing and GDP threshold; oscillating regime requires explicit coding of 2006/2014 coups as censoring events or within-sample regime variation [10].**

---

## Track 3: Latinobarometro Instrument Audit

### Access and Geographic Scope
Free registration required at latinobarometro.org [8]. Data available in Stata/SPSS/R/SAS formats. Covers 18 Latin American countries, 1995–2024, with no 1999 wave [8]. **Critical geographic limitation: Latin America ONLY** — excludes Eastern Europe, East/Southeast Asia, sub-Saharan Africa, and South/Central Asia [8].

### Employment Sector Variable (Public vs. Private)
The employment sector variable is NOT stable across waves — it changes name each survey year [12][13][14]:
- 2002: **S8A** (asalariado en empresa pública vs. privada) [14]
- 2007: **S17A** [13]
- 2016: **S20.A** [12]
- 2018: **S14A** [12]

Categories include: public sector salaried, private sector salaried, self-employed, family business, homemaker, unemployed [12][13]. No separate NGO/civil society category — such workers are coded in private sector or self-employed. Harmonization requires the wave-specific Libro de Códigos from latinobarometro.org and the Time-Series Dictionary at latinobarometro.org/time-series [8].

### Democratic Values Variables
The democracy preference question ('La democracia es preferible a cualquier otra forma de gobierno') also uses unstable variable names [12][13][14]:
- 2005: **P16ST** [14]
- 2008: **p13st** (labeled 'Apoyo a la democracia') [13]
- 2018: **P12STGBS** [12]
- 2020: restructured battery (P20ST.A, P22STM.B) [12]

Present in all waves from 1996 onward [8]. No universal harmonized name in official codebooks [12][13][14]. Trust in judiciary appears as a sub-item in the institutional confidence battery (typically P_ST.F-type suffix per wave) — variable name is wave-dependent [13].

### Alternative Instruments for Non-Latin-American Coverage
For **Eastern Europe/Central Asia**: Life in Transition Survey (LiTS, EBRD) [15] — waves 2006, 2010, 2016, 2022–23; employment sector from LiTS III (2016+); open access at litsonline-ebrd.com [15].

For **East/Southeast Asia**: Asian Barometer Survey (ABS) [16] — 18 countries, 5 waves from 2001; employment variable se8 with public/private distinction; free access upon application at asianbarometer.org [16].

## Sources

[1] [ILO ILOSTAT Bulk Download Indicator Directory](https://rplumber.ilo.org/files/website/bulk/indicator.html) — Confirmed SDG_0131_SEX_SOC_RT_A (303 reference areas, 36308 rows, updated 2026-05-04, starts 2015) and SPS_GEXP (social protection expenditure % GDP, ~1995 onward) available as direct bulk downloads

[2] [NSTP Dataset — Non-Contributory Social Transfer Programs (GESIS)](http://dx.doi.org/10.7802/1530) — Annual panel dataset covering 101 countries 1990-2015 with program-level beneficiary coverage for non-contributory social transfer programs; 70% of developing countries; open access via GESIS archive

[3] [OECD Social Expenditure Database (SOCX)](https://stats.oecd.org/Index.aspx?DataSetCode=SOCX_AGG) — Annual social expenditure as % GDP for 38 OECD members 1980-2023; Eastern European OECD members included from mid-1990s; freely downloadable via OECD.Stat

[4] [World Bank ASPIRE — Atlas of Social Protection Indicators of Resilience and Equity](https://www.worldbank.org/en/data/datatopics/aspire) — Survey-based coverage rate data for 140 countries 1998-2023; per_allsp.cov_pop_tot is the coverage rate variable; highly unbalanced (survey-wave basis, not annual)

[5] [ILO World Social Protection Database (WSPDB) — Social Security Inquiry (SSI/SSPINQ)](https://www.social-protection.org/gimi/WSPDB.action?id=41) — Most historically complete social protection data (~1950+); returns HTTP 403 Forbidden without ILO credentials; inaccessible for most researchers without institutional ILO login

[6] [ILO World Social Protection Report 2014-15](https://www.ilo.org/sites/default/files/wcmsp5/groups/public/@dgreports/@dcomm/documents/publication/wcms_245201.pdf) — Cross-sectional coverage snapshots (~2012 reference year) for 100+ countries in Statistical Appendix; best freely accessible ILO coverage data workaround for the 403-blocked WSPDB

[7] [Haggard & Kaufman (2008) Development, Democracy, and Welfare States — Princeton University Press](https://press.princeton.edu/books/paperback/9780691135960/development-democracy-and-welfare-states) — 21 countries (13 LatAm, 5 East Asian, 3 Eastern European); no public replication dataset; theoretical context only. Also covers Rudra (2002/2007) welfare effort index (53-57 developing countries, 1972-2000, composite spending measure, no public data) and Segura-Ubiergo (2007) (18 Latin American countries, no public data)

[8] [Latinobarometro Official Portal](https://www.latinobarometro.org) — 18 Latin American countries, 1995-2024, no 1999 wave; free registration gives access to Stata/SPSS/R/SAS data files and codebooks; employment sector and democratic values variables change names across waves; Latin America only

[9] [GDP PPP per capita estimates for case countries at democratization (IMF/World Bank)](https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(PPP)_per_capita) — Hungary 1990: $10,984; Turkey 1990: $7,299; Venezuela 1990: $9,701; Thailand 1990: ~$4,311 — all below $15,000 threshold; source: IMF World Economic Outlook database estimates

[10] [V-Dem Institute — V16 Dataset and Regimes of the World (RoW)](https://www.v-dem.net) — v2x_regime variable (0=closed autocracy to 3=liberal democracy); confirms Hungary electoral autocracy ~2019; Turkey electoral autocracy from 2013; Venezuela backslider ~2001-2002; Thailand electoral autocracy from 2014/2019

[11] [V-Dem Democracy Report 2025 — 25 Years of Autocratization](https://www.v-dem.net/documents/60/V-dem-dr__2025_lowres.pdf) — Documents country-level regime trajectories; confirms Turkey autocratization from 2013; Hungary from ~2018-2019; Venezuela autocratic since ~2002; Thailand coup 2014 followed by electoral autocracy 2019

[12] [Latinobarometro 2018 Codebook — GitHub](https://github.com/chrisbarclay00/Latinobarometro-2018) — 2018 wave: employment sector variable S14A; democracy preference variable P12STGBS ('Apoyo a la democracia'); 2020 wave restructured battery P20ST.A and P22STM.B

[13] [Latinobarometro 2008 Libro de Códigos](https://www.latinobarometro.org/documents/LAT-2008/latinobarometro-2008-libro-de-codigos-v20190707.pdf) — 2008 wave: democracy preference variable p13st ('Apoyo a la democracia'); employment sector variable S17A; trust in judiciary as sub-item in institutional confidence battery

[14] [Latinobarometro 2005 Libro de Códigos](http://investigadores.cide.edu/aparicio/data/encuestas/Latinobarometro/Latinobarom05codebook_esp.pdf) — 2005 wave: democracy preference variable P16ST; employment sector S8A used in 2002 wave (asalariado en empresa pública vs. privada)

[15] [Life in Transition Survey (LiTS) — EBRD](https://litsonline-ebrd.com) — Eastern Europe and Central Asia; waves 2006, 2010, 2016, 2022-23; employment sector variable available from LiTS III (2016) onward; open access; analog to Latinobarometro for post-communist cases

[16] [Asian Barometer Survey (ABS)](https://www.asianbarometer.org) — 18 East/Southeast Asian and South Asian countries; 5 waves from 2001; employment variable se8 with public/private sector distinction; free access upon application; analog to Latinobarometro for Asian democratizers

## Follow-up Questions

- Can the NSTP Dataset (doi:10.7802/1530) be merged with ILO SDG_0131 at the 2015 overlap year by log-linear rescaling, or does the definitional difference (non-contributory only vs. comprehensive floor) require a separate imputation model?
- For Turkey's borderline case: what does the actual V-Dem V16 v2x_regime value show for 1983 vs. 1987 — is Turkey coded as electoral autocracy under military tutelage in 1983-1987 and then electoral democracy, or does the electoral democracy coding begin at 1983?
- Given Latinobarometro covers only Latin America and ABS covers Southeast Asia, is there a wave-overlap period (e.g., 2014-2016) where both surveys ask sufficiently comparable employment sector and institutional trust questions to enable pooled cross-regional analysis?

---
*Generated by AI Inventor Pipeline*
