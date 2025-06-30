#!/usr/bin/env python3

# Test the updated plot without the debug parts that might be causing hangs
import matplotlib.pyplot as plt
import numpy as np

# Just test the legend formatting
fig, ax = plt.subplots(figsize=(12, 8))

# Sample data
regions = ['ETNP', 'TNEP', 'TSEP', 'STNA', 'TNIO', 'TNWP']
counts = [4, 20, 6, 18, 3, 7]
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

# Create sample plot
for i, (region, count, color) in enumerate(zip(regions, counts, colors)):
    ax.plot([i, i+1], [0, 1], color=color, linewidth=2, label=f"{region} ({count})")

# Test the new legend format
handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
labels = [f"{region} ({count})" for region, count in zip(regions, counts)]
plt.legend(handles, labels, title="Regions", loc="lower center", bbox_to_anchor=(0.5, -0.15), 
          fontsize=14, title_fontsize=16, ncol=6)

ax.set_title("Test Legend Format", fontsize=16)
plt.tight_layout()
plt.savefig('/home/kkumah/Projects/GPCP_ocean_evaluation_study/codes/test_legend.png', dpi=150, bbox_inches='tight')
print("Test legend saved as test_legend.png")

# Also print some example PAL filenames that might cross boundaries
print("\nExample PAL that might cross boundaries:")
print("PAL_precip_wind_Argo_float_6862_v1.nc")
print("(This is just an example filename from the directory)")
