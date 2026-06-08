# NSTP Verification, SE.TER.ENRR Gap Audit, and Iteration-4 Pre-registration Plan

## Summary

This research resolves three concrete data infrastructure questions for the iteration-4 DD triple interaction test.

Track 1 — NSTP Dataset: The dataset (DOI 10.7802/1530, 186 programs, 101 countries, 1990–2015) is freely downloadable without registration under CC BY 4.0 from GESIS. A critical structural finding is that coverage is stored as raw beneficiary counts (individuals or households), not as a normalized population share, and coverage data exists for only 110/186 programs at the individual level. The literature's standard aggregation is a binary dummy (1 if at least one program exists in the country-year) rather than a continuous rate. NSTP is also a strict subset of ILO SDG 1.3.1 — covering only non-contributory transfers, while ILO SDG 1.3.1 includes both contributory social insurance and non-contributory social assistance programs plus active social insurance contributors.

Track 2 — SE.TER.ENRR: World Bank tertiary gross enrollment (SE.TER.ENRR) has systematic 2020–2022 gaps for developing countries due to (a) a structural 2–4 year UNESCO UIS reporting lag and (b) COVID-19 disruption to national statistical offices. No clearly superior alternative exists; UNESCO UIS Mean Years of Schooling faces similar or worse constraints (biennial household surveys, ~103 country coverage vs 211 for SE.TER.ENRR). SE.TER.ENRR remains the recommended proxy with explicit forward-fill documentation and sensitivity analysis.

Track 3 — Pre-registration: The OSF Secondary Data Preregistration template (osf.io/v4z3x/) with the van den Akker et al. methodology is the correct platform. The Bonferroni-corrected threshold for two pre-registered sub-index tests (v2jucomp, v2x_jucon) is p < 0.025. The primary hypothesis specifying the DD triple interaction β₇ coefficient direction should be registered before constructing the NSTP-extended SocProt panel. The complete estimator specification follows Giesselmann & Schmidt-Catran (2022): double-demean the product of already-demeaned predictors to obtain a true within-unit estimator eliminating between-unit contamination.

## Research Findings

This research resolves three infrastructure questions for the iteration-4 DD triple interaction test.

**Track 1 — NSTP Dataset (DOI 10.7802/1530)**
The NSTP dataset by Dodlova, Giolbas & Lay is freely downloadable without registration (CC BY 4.0 license) from GESIS [1, 2]. Files include NSTP_Data_V1.1.dta (Stata, 869 KB), NSTP_Data_V1.1.xlsx (Excel, 244 KB), NSTP_Codebook_V1.1.pdf (328 KB), and NSTP_Text_Format_V1.1.pdf (3.09 MB) [2]. The dataset covers 186 programs from 101 countries in an annual panel through 2015, with both country-year and program-period panel structures [3, 4].

CRITICAL STRUCTURAL FINDING: Coverage in NSTP is reported as raw beneficiary counts only — individuals for 110 programs, households for 55 programs — with no normalized population-share rate [3]. Exact Stata variable names for coverage are not documented in publicly accessible sources without downloading the codebook. The literature's standard aggregation for country-year analysis is a binary dummy equal to 1 if at least one social transfer program exists [5], not a coverage rate. This binary approach is used in both Dodlova & Giolbas (2017) WIDER WP [5] and Dodlova et al. (2017) European Journal of Political Economy [6].

SCOPE MISMATCH WITH ILO SDG 1.3.1: Confirmed and critical. NSTP covers only non-contributory transfers (social pensions, CCTs, family benefits, public works). ILO SDG 1.3.1 measures the proportion of population receiving at least one contributory OR non-contributory benefit, or actively contributing to a social security scheme [7, 8]. The ILO indicator explicitly encompasses both contributory social insurance and non-contributory social assistance. Therefore NSTP is a strict subset of what ILO SDG 1.3.1 measures. Any bridging requires acknowledging that NSTP-derived social protection values will be lower-bound estimates of the full ILO SDG 1.3.1 concept.

Post-1990 democratizer overlap: The PMC paper confirms 91% of Europe/Central Asia and 90% of Latin America are in the NSTP 101-country set [3]. No published paper has explicitly counted V-Dem post-1990 democratizers among the 101; a rough estimate based on regional coverage suggests ~60–75 of the 101 NSTP countries overlap with post-1990 developing democratizers per V-Dem criteria.

**Track 2 — SE.TER.ENRR Availability for 2020–22**
SE.TER.ENRR (World Bank, sourced from UNESCO UIS) covers 1970–2025 in aggregate, but country-level data for 2020–2022 is systematically incomplete for developing countries [9, 10]. Two compounding factors: (a) a structural 2–4 year reporting lag for UNESCO UIS submissions from low/middle-income countries, and (b) COVID-19 pandemic disruptions to national data collection systems in 2020–2021 [10, 11]. The World Bank solely relies on the UIS pipeline with no gap-filling imputation, so UIS gaps flow directly into WDI [9]. Large-scale international assessment programs also postponed data collection for 2020–2022 [11].

The UNESCO UIS MYS (Mean Years of Schooling, OPRI indicator) faces similar or worse availability constraints: it depends on biennial household surveys and population censuses, covering only ~103 countries; the 2018 UIS release covered only through the 2017 school year [12]. As of the September 2025 UIS database release, the LITEA survey engaged 100+ countries for new data, but the reference years and developing country coverage for 2020–2022 remain limited [12]. No clearly superior alternative to SE.TER.ENRR exists for 2020–2022 — both SE.TER.ENRR and MYS require forward-filling for many developing countries, but SE.TER.ENRR (enrollment-based, administrative source) is collected more frequently than MYS (household survey-based).

**Track 3 — OSF Pre-registration Specification**
The OSF Secondary Data Preregistration template (osf.io/v4z3x/) and the PsyArXiv tutorial by van den Akker et al. [13] are the appropriate platform. The template requires: (1) explicit hypothesis statement with directional prediction, (2) full statistical model specification including all interaction terms (any unspecified analysis must be labeled exploratory), (3) multiple comparison correction methodology — the template explicitly uses Bonferroni as the example (alpha = 0.05/k where k = number of planned tests) [14], (4) power analysis with sample size, effect size, and software documentation. For two pre-registered sub-index tests (v2jucomp, v2x_jucon), Bonferroni threshold = 0.05/2 = 0.025.

Giesselmann & Schmidt-Catran (2022) [15] confirm that the double-demeaning procedure (demean product of already-demeaned predictors) yields a true within-unit estimator for interactions, eliminating between-unit contamination. The efficiency cost is real: the estimator requires meaningful within-unit variation and loses power with small T. Monte Carlo simulations indicate T ≥ 10 per unit is recommended. For a triple interaction in a ~70–100 country × 6-period panel (~420–600 observations), effective within-country degrees of freedom are materially reduced after absorbing unit and period FEs plus all demeaned product terms.

Primary hypothesis (exact pre-registration wording): In post-1985 developing democratizers — countries experiencing a V-Dem Political Regime (v2x_regime) transition from closed autocracy (0) or electoral autocracy (1) to electoral democracy (2) or liberal democracy (3) after 1985 and before 2010, with GDP per capita <$15,000 PPP (constant 2017 international $) at year of democratic transition — the DD estimator coefficient β₇ on the triple interaction term (demeaned Education × demeaned Gini × demeaned SocProt) is positive and statistically significant at p < 0.05 (two-tailed), with country fixed effects and period fixed effects and cluster-robust standard errors at the country level.

Recommended pre-registration timing: Register before constructing the NSTP-extended SocProt panel, i.e., before merging NSTP with ILO SDG 1.3.1 and running the first DD regression, making the registration genuinely prospective on the merged dataset [16, 17, 18].

## Sources

[1] [NSTP Dataset — GESIS Landing Page (DOI 10.7802/1530)](https://search.gesis.org/research_data/SDN-10.7802-1530?doi=10.7802/1530) — Confirmed DOI 10.7802/1530 resolves to GESIS Data Archive. Free access without registration (CC BY 4.0). 186 programs, 101 countries, 29 variables.

[2] [da-ra Registry: NSTP Dataset (DOI 10.7802/1530)](https://www.da-ra.de/dara/study/web_show?res_id=604354&lang=&mdlang=en&detail=true) — Confirmed file list: NSTP_Data_V1.1.dta (869 KB), NSTP_Data_V1.1.xlsx (244 KB), NSTP_Codebook_V1.1.pdf (328 KB), NSTP_Text_Format_V1.1.pdf (3.09 MB). Free access without registration.

[3] [Dodlova, Giolbas, Lay (2017): Non-contributory social transfer programs — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5709306/) — Primary data paper. Confirms: coverage is raw beneficiary count (individuals for 110 programs, households for 55), not a population share. Both country-year and program-period panel structures. No methodology for normalized coverage rate. 91% Europe/Central Asia and 90% Latin America included.

[4] [GIGA Hamburg: NSTP Dataset Research Data Page](https://www.giga-hamburg.de/en/publications/research-data/non-contributory-social-transfer-programmes-nstp-developing-countries-data-set) — Dataset landing page. Confirms file formats, DOI, free access, 186 programs in 101 countries up to 2015.

[5] [Dodlova & Giolbas (2017): Regime Type, Inequality, and Redistributive Transfers — WIDER WP 2017/30](https://www.wider.unu.edu/publication/regime-type-inequality-and-redistributive-transfers-developing-countries) — Confirms that country-level social protection variable from NSTP is a BINARY DUMMY = 1 if at least one transfer program exists. Uses 1990–2015 panel. Democracies more likely to have a transfer program.

[6] [Dodlova, Giolbas, Lay (2017): Social Transfers and Conditionalities under Different Regime Types — EJPE](https://econpapers.repec.org/article/eeepoleco/v_3a50_3ay_3a2017_3ai_3ac_3ap_3a141-156.htm) — Published article using NSTP data in country-level panel. Examines regime type effect on conditionality and transfer type (CCT vs unconditional).

[7] [UN Statistics: SDG Indicator 1.3.1 Official Metadata (2025)](https://unstats.un.org/sdgs/metadata/files/Metadata-01-03-01a.pdf) — Confirms ILO SDG 1.3.1 includes BOTH contributory and non-contributory programs. Effective coverage = proportion receiving any contributory or non-contributory benefit OR actively contributing to social insurance. Scope is broader than NSTP.

[8] [UN Statistics eHandbook: SDG Indicator 1.3.1](https://unstats.un.org/wiki/spaces/SDGeHandbook/pages/35291384/Indicator+1.3.1) — Confirms scope definition: ratio of population receiving cash benefits under at least one contingency (contributory or non-contributory) or actively contributing to social security scheme.

[9] [World Bank: SE.TER.ENRR — School Enrollment, Tertiary (% Gross)](https://data.worldbank.org/indicator/SE.TER.ENRR) — Series covers 1970–2025. Sourced from UNESCO UIS. Published February 2026, data accessed March 2026 from UNESCO UIS bulk CSV. No individual country imputation for gaps.

[10] [World Bank Metadata: SE.TER.ENRR](https://databank.worldbank.org/metadataglossary/world-development-indicators/series/SE.TER.ENRR) — Confirms sole dependence on UNESCO UIS API pipeline. 2–4 year reporting lag documented for developing countries. Sub-Saharan Africa data sourced from UNESCO UIS API, some regional aggregates lag.

[11] [World Bank IEG: Chapter 3 — Education Finance Watch 2022](https://ieg.worldbankgroup.org/evaluations/confronting-learning-crisis/chapter-3-world-banks-approach-basic-education-and-learning-outcomes) — Large-scale international assessments postponed for 2020–2022 due to COVID-19, creating data gaps. Bilateral aid to education fell post-pandemic. Explains 2020–2022 SE.TER.ENRR gaps for developing countries.

[12] [UNESCO UIS: Background Information on Education Statistics Database (September 2025)](https://download.uis.unesco.org/bdds/202509/background-information-education-statistics-uis-database-en-2025.pdf) — Confirms MYS (OPRI indicator) collected via biennial LITEA survey covering 100+ countries. 2025 release engaged 100+ countries for new data. SDG 4.4.2 (digital skills) has data only to 2017 — shows persistent lag for household-survey-based indicators. SE.TER.ENRR (4.4.3) covers 211 countries 1970–2025.

[13] [van den Akker et al.: Preregistration of Secondary Data Analysis — Template and Tutorial](https://osf.io/preprints/psyarxiv/hvfmr) — The methodological basis for the OSF Secondary Data Preregistration template. Required fields: hypothesis specification, statistical model specification, multiple comparison corrections (Bonferroni explicitly mentioned), power analysis. Available at osf.io/v4z3x.

[14] [preregr R package: Secondary Data Preregistration Template (v1)](https://preregr.opens.science/articles/form_prereg2D_v1.html) — Structured digital template for secondary data preregistration. Explicitly shows Bonferroni correction example: alpha = 0.05/k tests. Mandates specifying all interaction terms a priori. Unspecified analyses become exploratory.

[15] [Giesselmann & Schmidt-Catran (2022): Interactions in Fixed Effects Regression Models](https://www.zora.uzh.ch/entities/publication/d175ed4d-81f7-43b5-b077-f912ad84faa5) — Confirms double-demeaning procedure: demean product of already-demeaned predictors to get true within-unit estimator. Key caveat: efficiency loss when within-unit variation is small or T per unit is low. Monte Carlo simulations show T=10+ recommended.

[16] [OSF: Secondary Data Preregistration Registry Page](https://osf.io/x4gzt/) — OSF registry for secondary data preregistrations. The recommended platform for observational panel data studies in political science.

[17] [AsPredicted: Preregistration Platform](https://aspredicted.org/) — 9-question preregistration platform. Less suited for complex secondary data observational studies with triple interactions and panel data than the OSF Secondary Data Preregistration template.

[18] [Challenges in Pre-registration Using Secondary or Longitudinal Data (PMC 2019)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6840585/) — Documents specific challenges for secondary data preregistration. Recommends disclosing prior knowledge of dataset, hold-out subsamples, and explicitly distinguishing exploratory vs confirmatory claims.

[19] [PUMP: Power Under Multiplicity Project — Power Calculations for Multiple Outcomes](https://arxiv.org/pdf/2112.15273) — Power calculation framework for panel data with fixed effects and multiple outcomes. Relevant for computing power for the Bonferroni-corrected sub-index tests (v2jucomp, v2x_jucon) at the corrected threshold of p < 0.025.

[20] [DIW Discussion Paper 1748: Interactions in Fixed Effects Regression Models](https://www.diw.de/documents/publikationen/73/diw_01.c.594675.de/dp1748.pdf) — Earlier version of Giesselmann & Schmidt-Catran methodology. Confirms double demeaning yields consistent within-unit estimator. Monte Carlo with T=3, 10, 30: precision improves sharply from T=3 to T=10.

## Follow-up Questions

- Can the NSTP codebook PDF (NSTP_Codebook_V1.1.pdf) be downloaded and parsed to extract exact Stata variable names for beneficiary counts, and is there a normalized coverage rate variable or must one be constructed by dividing beneficiary counts by population figures from an external source like World Bank WDI SP.POP.TOTL?
- Given that the standard literature uses a binary dummy (has-at-least-one-program) rather than a coverage rate, should the iteration-4 SocProt measure switch from a continuous coverage rate to this binary indicator for the NSTP-era (1990–2014) period, while retaining ILO SDG 1.3.1 continuous rates for 2015–2022 — and what is the methodological implication for the DD triple interaction estimator of mixing a binary and continuous SocProt variable across periods?
- For the SE.TER.ENRR forward-filling issue: what fraction of the 58 sample democratizer countries have non-missing 2020 and 2021 values in the current WDI download, and would a sensitivity analysis dropping all forward-filled observations meaningfully change the period-2 DD interaction results?

---
*Generated by AI Inventor Pipeline*
