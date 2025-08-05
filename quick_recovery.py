#!/usr/bin/env python3
"""
Quick Recovery Script for Satellite Evaluation Analysis
Run this after a kernel crash to quickly get back to where you were
"""

# Quick recovery imports
from memory_management_improvements import (
    setup_memory_management, monitor_memory_usage
)
from kernel_recovery import (
    KernelStateManager, check_what_needs_reloading
)
from util_functions import *
import xarray as xr
import pandas as pd
import numpy as np
import gc

def quick_recovery():
    """Quickly recover from kernel crash"""
    
    print("=== KERNEL CRASH RECOVERY ===")
    print("Checking what needs to be reloaded...")
    
    # Check previous state
    check_what_needs_reloading()
    
    # Set up memory management
    print("\nSetting up memory management...")
    try:
        dask_client = setup_memory_management()
        print("✓ Dask client configured")
    except Exception as e:
        print(f"Warning: Could not set up dask client: {e}")
        dask_client = None
    
    monitor_memory_usage()
    
    print("\n=== RECOVERY OPTIONS ===")
    print("1. If data loading crashed - Run the data loading cells again")
    print("2. If analysis crashed - Skip to analysis section") 
    print("3. If plotting crashed - Skip to plotting section")
    print("4. To avoid future crashes:")
    print("   - Reduce batch_size from 20 to 10")
    print("   - Run cells one at a time instead of all at once")
    print("   - Monitor memory usage with monitor_memory_usage()")
    
    return dask_client

def emergency_cleanup():
    """Emergency cleanup when memory is running low"""
    print("Running emergency cleanup...")
    
    # Close any open datasets
    try:
        # Try to close datasets if they exist in global scope
        import sys
        module = sys.modules[__name__]
        
        for var_name in dir(module):
            var = getattr(module, var_name)
            if hasattr(var, 'close') and hasattr(var, 'data_vars'):
                print(f"Closing dataset: {var_name}")
                var.close()
    except:
        pass
    
    # Force garbage collection
    gc.collect()
    
    # Check memory
    monitor_memory_usage()
    
    print("Emergency cleanup complete")

def restart_kernel_safe():
    """Instructions for safely restarting kernel"""
    print("\n=== SAFE KERNEL RESTART PROCEDURE ===")
    print("1. Save any important variables first")
    print("2. Run: emergency_cleanup()")
    print("3. Restart kernel (Ctrl+Shift+P -> 'Restart Kernel')")
    print("4. Run: quick_recovery()")
    print("5. Load only the sections you need")

if __name__ == "__main__":
    quick_recovery()
