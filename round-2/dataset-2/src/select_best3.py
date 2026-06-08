# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import json
from pathlib import Path

BASE = Path(__file__).parent
src = json.loads((BASE / "full_data_out.json").read_text())

KEEP = {
    "wvs_wave7_developing_democratizers",
    "vdem_democracy_indicators",
    "ilo_socprot_country_year",
}

src["datasets"] = [d for d in src["datasets"] if d["dataset"] in KEEP]
src["metadata"]["description"] = (
    "Three datasets for micro/macro analysis of public-sector workers, institutional "
    "independence, and social protection in developing democratizers. "
    "(1) WVS Wave 7 individual-level survey (576 examples): employment sector × institutional "
    "trust × SocProt/Gini quadrant — the primary dataset for hypothesis testing. "
    "(2) V-Dem democracy indicators (1102 examples): country-year democracy scores and judicial "
    "corruption for developing democratizer classification and macro context. "
    "(3) ILO SocProt country-year (354 examples): social protection coverage for quadrant "
    "classification validation."
)
src["metadata"]["selected_datasets"] = sorted(KEEP)

out = BASE / "data_out.json"
out.write_text(json.dumps(src, indent=2))
for ds in src["datasets"]:
    print(f"  {ds['dataset']}: {len(ds['examples'])} examples")
print(f"Written: {out} ({out.stat().st_size // 1024} KB)")
