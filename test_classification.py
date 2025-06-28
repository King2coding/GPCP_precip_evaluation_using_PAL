#!/usr/bin/env python3

# Simple test script to check PAL classification
import os
import sys

print("Starting test...")

try:
    print("Importing pandas...")
    import pandas as pd
    print("Pandas imported successfully")
    
    print("Importing util_functions...")
    import util_functions as uf
    print("Util_functions imported successfully")
    
    print("Current region bounds:")
    print(uf.region_bounds)
    
    # Get the PAL files
    pal_dir = '/ra1/pubdat/AVHRR_CloudSat_proj/PAL/data_unzipped/PAL_SPURS1_SPURS2_TPOS_Others_202210'
    if os.path.exists(pal_dir):
        print(f"PAL directory exists: {pal_dir}")
        pal_files = [f for f in os.listdir(pal_dir) if f.endswith('.nc')]
        print(f"Found {len(pal_files)} PAL files")
        
        # Prepare full paths
        pal_file_paths = [os.path.join(pal_dir, f) for f in pal_files]
        
        print("Starting classification...")
        classified_files = uf.classify_and_group_files_bounding_box(pal_file_paths, uf.region_bounds)
        print("Classification completed successfully!")
        
        # Print summary
        total_classified = sum(len(files) for files in classified_files.values())
        print(f'\nTotal PAL files: {len(pal_files)}')
        print(f'Total classified files: {total_classified}')
        print(f'Total unclassified files: {len(classified_files.get("Unclassified", []))}')
        print()
        print('PAL CLASSIFICATION SUMMARY')
        print('=' * 50)
        for region, files in classified_files.items():
            if files:  # Only show regions with files
                print(f'{region}: {len(files)} PALs')
    else:
        print(f"PAL directory does not exist: {pal_dir}")
        
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")
