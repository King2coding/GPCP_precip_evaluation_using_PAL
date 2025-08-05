# Memory Management Improvements for Satellite Evaluation Code
# Add these configurations at the top of your main script

import os
import gc
import psutil
import dask

# Only import distributed components when needed (slow import)
def _import_distributed():
    """Import distributed components only when needed"""
    try:
        from dask.distributed import Client, LocalCluster
        return Client, LocalCluster
    except ImportError:
        return None, None

def setup_memory_management(use_distributed=False):
    """Configure dask and memory settings to prevent kernel crashes"""
    
    # 1. Configure Dask for better memory management (fast configuration)
    dask.config.set({
        'array.slicing.split_large_chunks': False,
        'array.chunk-size': '128MB',  # Smaller chunks
        'distributed.worker.memory.target': 0.6,  # Use 60% of memory before spilling
        'distributed.worker.memory.spill': 0.7,   # Spill to disk at 70%
        'distributed.worker.memory.pause': 0.8,   # Pause computation at 80%
        'distributed.worker.memory.terminate': 0.95  # Terminate at 95%
    })
    
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available memory: {available_memory_gb:.1f} GB")
    
    # Only start distributed client if explicitly requested (slow)
    if use_distributed:
        print("Setting up distributed dask client (this may take 1-2 minutes)...")
        try:
            # Import distributed components (slow import)
            Client, LocalCluster = _import_distributed()
            if Client is None or LocalCluster is None:
                raise ImportError("Could not import dask.distributed")
            
            # 2. Limit number of workers based on available memory
            n_workers = max(1, min(4, int(available_memory_gb // 2)))  # 2GB per worker
            
            # 3. Start local dask client with memory limits
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=2,
                memory_limit=f'{int(available_memory_gb/n_workers)}GB',
                silence_logs=True  # Reduce startup noise
            )
            client = Client(cluster)
            
            print(f"✓ Started Dask cluster with {n_workers} workers")
            print(f"✓ Memory per worker: {available_memory_gb/n_workers:.1f} GB")
            
            return client
        except Exception as e:
            print(f"Warning: Could not start distributed client: {e}")
            print("Continuing with threaded scheduler (faster startup)")
            return None
    else:
        print("✓ Using dask threaded scheduler (fast startup)")
        print("✓ Memory management configured")
        return None

def memory_safe_batch_processing(files, batch_size=None, process_func=None):
    """Process files in memory-safe batches"""
    
    if batch_size is None:
        # Dynamically determine batch size based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        batch_size = max(10, min(50, int(available_memory_gb * 5)))
    
    results = []
    total_batches = len(files) // batch_size + (1 if len(files) % batch_size > 0 else 0)
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} files)")
        
        # Process batch
        if process_func:
            result = process_func(batch)
            results.append(result)
        
        # Force garbage collection after each batch
        gc.collect()
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            print(f"Warning: Memory usage at {memory_percent:.1f}%. Consider reducing batch size.")
    
    return results

def safe_dataset_operation(operation_func, *args, **kwargs):
    """Safely execute dataset operations with memory monitoring"""
    
    # Check memory before operation
    initial_memory = psutil.virtual_memory().percent
    print(f"Memory usage before operation: {initial_memory:.1f}%")
    
    try:
        result = operation_func(*args, **kwargs)
        return result
    except MemoryError:
        print("Memory error occurred. Forcing garbage collection and retrying...")
        gc.collect()
        # Try with smaller chunks
        if 'chunks' in kwargs:
            kwargs['chunks'] = {'time': 10, 'lat': 50, 'lon': 50}
        return operation_func(*args, **kwargs)
    finally:
        # Always clean up
        gc.collect()
        final_memory = psutil.virtual_memory().percent
        print(f"Memory usage after operation: {final_memory:.1f}%")

def monitor_memory_usage():
    """Print current memory usage"""
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")

def fast_setup():
    """Fast memory management setup without distributed client"""
    print("Setting up fast memory management...")
    
    # Quick dask configuration only
    dask.config.set({
        'array.slicing.split_large_chunks': False,
        'array.chunk-size': '64MB',  # Smaller chunks for safety
    })
    
    monitor_memory_usage()
    print("✓ Fast setup complete!")
    return None
