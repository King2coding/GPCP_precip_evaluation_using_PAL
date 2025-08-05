#!/usr/bin/env python3
"""
Fast import test - checks if critical packages work without slow setups
"""

def quick_test():
    """Quick test of essential imports only"""
    print("=== QUICK IMPORT TEST ===")
    
    essential_modules = [
        ('numpy', 'np'),
        ('pandas', 'pd'), 
        ('xarray', 'xr'),
        ('matplotlib.pyplot', 'plt'),
        ('dask', None)
    ]
    
    for module_name, alias in essential_modules:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name} - FAILED: {e}")
            return False
    
    print("✓ All essential imports working!")
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🚀 Ready to run your code!")
    else:
        print("\n❌ Some imports failed. Check your environment.")
