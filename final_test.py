#!/usr/bin/env python3

# Final test to confirm classification results
import os
import sys

# Force reload of util_functions to get latest changes
if 'util_functions' in sys.modules:
    import importlib
    importlib.reload(sys.modules['util_functions'])

from util_functions import *

# Get the PAL files
path_to_pal_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'
all_pal_files = [os.path.join(path_to_pal_data, f) for f in os.listdir(path_to_pal_data) if f.endswith('.nc')]

print(f"Total PAL files found: {len(all_pal_files)}")

# Classify the files
pals_classed_by_region = classify_and_group_files_bounding_box(all_pal_files, region_bounds)

# Print classification summary
print("\n" + "="*50)
print("FINAL PAL CLASSIFICATION SUMMARY")
print("="*50)
total_classified = 0
for region, files in pals_classed_by_region.items():
    if region != "Unclassified":
        print(f"{region}: {len(files)} PALs")
        total_classified += len(files)

if "Unclassified" in pals_classed_by_region and len(pals_classed_by_region["Unclassified"]) > 0:
    print(f"Unclassified: {len(pals_classed_by_region['Unclassified'])} PALs")
    total_classified += len(pals_classed_by_region["Unclassified"])

print("="*50)

# Expected counts from your figure:
expected_counts = {
    "ETNP": 4,   # Extratropical North Pacific
    "TNEP": 20,  # Tropical Northeastern Pacific  
    "TSEP": 6,   # Tropical Southeastern Pacific
    "STNA": 18,  # Subtropical North Atlantic
    "TNIO": 3,   # Tropical North Indian Ocean
    "TNWP": 7    # Tropical Northwestern Pacific
}

print("\nCOMPARISON WITH EXPECTED COUNTS:")
print("-" * 30)
total_expected = sum(expected_counts.values())

for region in expected_counts:
    actual = len(pals_classed_by_region.get(region, []))
    expected = expected_counts[region]
    status = "✓" if actual == expected else "✗"
    print(f"{region}: Expected {expected}, Got {actual} {status}")

print(f"\nTotal Expected: {total_expected}")
print(f"Total Actual: {total_classified}")
print(f"Difference: {total_classified - total_expected}")

unclassified_count = len(pals_classed_by_region.get("Unclassified", []))
print(f"Unclassified: {unclassified_count}")

if unclassified_count == 0 and total_classified == total_expected:
    print("\n🎉 SUCCESS: All PALs are properly classified and counts match expected values!")
else:
    print(f"\n❌ Issue: {unclassified_count} unclassified files or count mismatch")

print("\nFINAL REGION BOUNDARIES:")
print("-" * 30)
for region, bounds in region_bounds.items():
    print(f"{region}: Lat {bounds['lat_min']}-{bounds['lat_max']}, Lon {bounds['lon_min']}-{bounds['lon_max']}")
