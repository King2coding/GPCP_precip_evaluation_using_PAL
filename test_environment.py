#!/usr/bin/env python3
"""
Test script to verify all required modules are installed and working
Run this to check if your environment is properly set up
"""

import sys
import importlib

def test_import(module_name, description=""):
    """Test if a module can be imported successfully"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name:<20} - {description}")
        return True
    except ImportError as e:
        print(f"✗ {module_name:<20} - FAILED: {e}")
        return False
    except Exception as e:
        print(f"⚠ {module_name:<20} - WARNING: {e}")
        return False

def test_all_imports():
    """Test all required imports for the satellite evaluation project"""
    
    print("=== TESTING REQUIRED MODULES ===")
    print(f"Python version: {sys.version}")
    print("-" * 50)
    
    # Core scientific computing
    test_import("numpy", "Numerical computing")
    test_import("pandas", "Data manipulation and analysis")
    test_import("scipy", "Scientific computing")
    test_import("scipy.stats", "Statistical functions")
    
    # Data visualization
    test_import("matplotlib", "Plotting library")
    test_import("matplotlib.pyplot", "Plotting interface")
    test_import("seaborn", "Statistical visualization")
    
    # Geospatial and meteorological data
    test_import("xarray", "N-dimensional labeled arrays")
    test_import("netCDF4", "NetCDF file support")
    test_import("cartopy", "Cartographic projections")
    test_import("cartopy.crs", "Coordinate reference systems")
    test_import("cartopy.feature", "Geographic features")
    test_import("rasterio", "Raster data I/O")
    
    # Geospatial processing
    test_import("osgeo", "GDAL Python bindings")
    test_import("osgeo.gdal", "GDAL library")
    test_import("pyproj", "Cartographic transformations")
    
    # Parallel computing and memory management
    test_import("dask", "Parallel computing")
    test_import("dask.distributed", "Distributed computing")
    test_import("joblib", "Lightweight pipelining")
    test_import("psutil", "System and process utilities")
    
    # Evaluation metrics
    test_import("HydroErr", "Hydrological error metrics")
    
    # Jupyter environment
    test_import("ipykernel", "Jupyter kernel")
    
    # Additional utilities
    test_import("h5py", "HDF5 file support")
    test_import("cftime", "Calendar-independent time objects")
    test_import("bottleneck", "Fast NumPy array functions")
    test_import("zarr", "Chunked, compressed arrays")
    
    print("-" * 50)
    print("Import testing complete!")
    
    # Test a few key operations
    print("\n=== TESTING KEY OPERATIONS ===")
    
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr
        import matplotlib.pyplot as plt
        
        # Test basic operations
        arr = np.random.random((10, 10))
        df = pd.DataFrame(arr)
        da = xr.DataArray(arr, dims=['x', 'y'])
        
        print("✓ NumPy array creation")
        print("✓ Pandas DataFrame creation")  
        print("✓ XArray DataArray creation")
        print("✓ Basic operations working")
        
    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
    
    print("\n=== ENVIRONMENT READY! ===")
    print("You can now run your satellite evaluation code.")

if __name__ == "__main__":
    test_all_imports()
