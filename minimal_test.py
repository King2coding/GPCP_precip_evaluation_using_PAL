#!/usr/bin/env python3
"""
Minimal imports for testing basic functionality - should be very fast
"""

# Just the bare minimum to test if environment works
import numpy as np
import pandas as pd

print("✓ Minimal test successful!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Quick functionality test
arr = np.array([1, 2, 3])
df = pd.DataFrame({'test': [1, 2, 3]})
print("✓ Basic operations working")
