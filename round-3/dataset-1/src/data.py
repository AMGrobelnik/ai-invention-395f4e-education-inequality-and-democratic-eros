#!/usr/bin/env python3
"""Production entry point for the 7-period democratizer panel dataset.

Integrates six data sources into a harmonized country-period panel
for 61 post-1990 developing democratizers (V-Dem regime 0-1→≥2 after 1985,
GDP PPP <$15k at transition), covering 7 periods 1990-2022:

  1. ILO SDG 1.3.1 (SDMX REST API)        — SocProt coverage, 300 countries, 2009-2023
  2. V-Dem via Our World in Data CSVs      — regime classification + libdem/polyarchy
  3. SWIID v9.92 (April 2026)              — disposable-income Gini with uncertainty
  4. UNDP HDR25 mean years of schooling    — education primary proxy, 204 countries 1990-2023
  5. World Bank SE.TER.ENRR                — gross tertiary enrollment, education supplement
  6. World Bank NY.GDP.PCAP.PP.KD          — GDP PPP per capita (filter + economic control)

Output: full_data_out.json (425 country-period rows), data_audit.json
Run aii-json format script to generate mini_data_out.json, preview_data_out.json.
"""

from build_panel import main

if __name__ == "__main__":
    main()
