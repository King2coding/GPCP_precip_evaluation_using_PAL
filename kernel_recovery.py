# Kernel Recovery and State Management
# Use this to save/restore your work when kernel crashes occur

import pickle
import os
from datetime import datetime

class KernelStateManager:
    """Manage kernel state to enable recovery from crashes"""
    
    def __init__(self, state_dir='/tmp/jupyter_state'):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.state_file = os.path.join(state_dir, 'kernel_state.pkl')
    
    def save_state(self, variables_dict, description=""):
        """Save important variables to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_data = {
            'timestamp': timestamp,
            'description': description,
            'variables': variables_dict
        }
        
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(state_data, f)
            print(f"State saved at {timestamp}: {description}")
            return True
        except Exception as e:
            print(f"Failed to save state: {e}")
            return False
    
    def load_state(self):
        """Load previously saved state"""
        try:
            with open(self.state_file, 'rb') as f:
                state_data = pickle.load(f)
            print(f"State loaded from {state_data['timestamp']}: {state_data['description']}")
            return state_data['variables']
        except FileNotFoundError:
            print("No saved state found")
            return {}
        except Exception as e:
            print(f"Failed to load state: {e}")
            return {}
    
    def clear_state(self):
        """Clear saved state"""
        try:
            os.remove(self.state_file)
            print("Saved state cleared")
        except FileNotFoundError:
            print("No state file to clear")

# Usage functions
def save_checkpoint(state_manager, variables, description):
    """Save a checkpoint with current important variables"""
    state_manager.save_state(variables, description)

def restore_from_checkpoint(state_manager):
    """Restore variables from last checkpoint"""
    return state_manager.load_state()

# Specific functions for your workflow
def save_data_loading_checkpoint(gpcp_ds_v1pt3_xr=None, gpcp_ds_v3pt2_xr=None, 
                                gpcp_ds_v3pt3_xr=None, imerg_ds_xr=None,
                                pals_classed_by_region=None):
    """Save checkpoint after data loading phase"""
    state_manager = KernelStateManager()
    variables = {}
    
    if gpcp_ds_v1pt3_xr is not None:
        variables['gpcp_v1pt3_loaded'] = True
    if gpcp_ds_v3pt2_xr is not None:
        variables['gpcp_v3pt2_loaded'] = True
    if gpcp_ds_v3pt3_xr is not None:
        variables['gpcp_v3pt3_loaded'] = True
    if imerg_ds_xr is not None:
        variables['imerg_loaded'] = True
    if pals_classed_by_region is not None:
        variables['pals_classed_by_region'] = pals_classed_by_region
    
    save_checkpoint(state_manager, variables, "Data loading complete")

def save_analysis_checkpoint(regional_PAL_GPCP_dfs_daily_mean=None, 
                           regional_PAL_IMERG_dfs_daily_mean=None):
    """Save checkpoint after analysis phase"""
    state_manager = KernelStateManager()
    variables = {}
    
    if regional_PAL_GPCP_dfs_daily_mean is not None:
        variables['regional_PAL_GPCP_dfs_daily_mean'] = regional_PAL_GPCP_dfs_daily_mean
    if regional_PAL_IMERG_dfs_daily_mean is not None:
        variables['regional_PAL_IMERG_dfs_daily_mean'] = regional_PAL_IMERG_dfs_daily_mean
    
    save_checkpoint(state_manager, variables, "Analysis complete")

# Recovery helpers
def check_what_needs_reloading():
    """Check what data needs to be reloaded after kernel restart"""
    state_manager = KernelStateManager()
    state = restore_from_checkpoint(state_manager)
    
    if not state:
        print("No previous state found. Need to run everything from scratch.")
        return
    
    print("Previous state found:")
    for key, value in state.items():
        if isinstance(value, bool) and value:
            print(f"  ✓ {key}")
        elif not isinstance(value, bool):
            print(f"  ✓ {key}: {type(value)}")
    
    print("\nTo recover, run the appropriate sections based on what's missing.")
