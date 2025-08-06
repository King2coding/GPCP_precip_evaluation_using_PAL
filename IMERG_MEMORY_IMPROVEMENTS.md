# IMERG Memory-Efficient Processing Improvements

## Overview
The original IMERG processing functions were causing kernel crashes due to excessive memory usage. This document outlines the improvements made to prevent memory overload while maintaining the same preprocessing logic.

## Key Issues Addressed

1. **Excessive Parallel Processing**: Original code used `n_jobs=20`, loading too much data simultaneously
2. **Memory-Intensive Data Types**: Using float64 instead of float32 doubled memory usage unnecessarily
3. **Poor Garbage Collection**: Insufficient cleanup of intermediate variables
4. **Large Batch Processing**: Processing too many files at once without memory monitoring

## New Functions Added

### 1. `read_nc_imger_file_memory_efficient()`
- Uses float32 instead of float64 (halves memory usage)
- Better error handling with try/catch blocks
- Proper context management with `with` statements
- Explicit cleanup of variables

### 2. `process_imerg_memory_efficient()`
- Reduced parallel workers from 20 to 4 (configurable via `max_workers`)
- Uses threading backend instead of default multiprocessing
- Better error handling for individual files
- Explicit garbage collection between operations
- Filters out failed results before processing

### 3. `process_imerg_sequential()`
- Non-parallel version for maximum memory efficiency
- Processes files one by one when memory is very limited
- Periodic garbage collection every 100 files
- Progress monitoring with regular status updates

### 4. `create_xarray_memory_efficient()`
- Converts data to float32 to reduce memory usage
- Implements chunking for better memory management during operations
- Explicit garbage collection after processing

### 5. `simple_process_imerg_batch_memory_efficient()`
- Auto-selects processing mode based on batch size
- Uses sequential processing for large batches (>200 files)
- Uses reduced parallel processing for smaller batches
- Comprehensive error handling with fallback options

## Main Script Improvements

### Updated IMERG Processing Section
```python
# Key improvements in the main processing loop:
1. Reduced IMERG batch size to max 100 files
2. More frequent progress updates (every 10 batches)
3. Auto-selection of processing mode
4. Aggressive garbage collection after each batch
5. Optional memory monitoring with psutil
6. Continues processing even if individual batches fail
```

## Usage Guidelines

### For Normal Memory Situations
Use the default `simple_process_imerg_batch_memory_efficient()` with `processing_mode="auto"`:
```python
processed_batch = simple_process_imerg_batch_memory_efficient(
    batch, 
    product="imerg_fn", 
    processing_mode="auto"
)
```

### For Low Memory Situations
Force sequential processing:
```python
processed_batch = simple_process_imerg_batch_memory_efficient(
    batch, 
    product="imerg_fn", 
    processing_mode="sequential"
)
```

### For Very Low Memory Situations
Use smaller batch sizes and sequential processing:
```python
# Reduce batch size
imerg_batch_size = 50  # or even smaller

# Use sequential processing
processed_batch = process_imerg_sequential(batch, product="imerg_fn")
```

## Memory Monitoring

The updated script includes optional memory monitoring:
```python
try:
    import psutil
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        print(f"Warning: Memory usage at {memory_percent:.1f}%")
except ImportError:
    pass
```

## Performance vs Memory Trade-offs

| Function | Memory Usage | Processing Speed | Use Case |
|----------|--------------|------------------|-----------|
| `process_imerg` (original) | Highest | Fastest | High-memory systems only |
| `process_imerg_memory_efficient` | Medium | Medium | Balanced approach |
| `process_imerg_sequential` | Lowest | Slowest | Low-memory systems |

## Recommended Settings

### High-Memory Systems (>32GB RAM)
- Batch size: 200-500 files
- Processing mode: "parallel"
- Max workers: 4-6

### Medium-Memory Systems (16-32GB RAM)
- Batch size: 100-200 files
- Processing mode: "auto"
- Max workers: 3-4

### Low-Memory Systems (<16GB RAM)
- Batch size: 50-100 files
- Processing mode: "sequential"
- Max workers: N/A (sequential)

## Expected Improvements

1. **Reduced Memory Usage**: 50-70% reduction in peak memory usage
2. **Fewer Kernel Crashes**: Better error handling and memory management
3. **More Stable Processing**: Continues even if individual files fail
4. **Better Monitoring**: Progress updates and memory usage warnings
5. **Maintained Accuracy**: Same preprocessing logic, just more efficient

## Migration

To use the new functions, simply:
1. Ensure `util_functions.py` has been updated with the new functions
2. The main script automatically uses the new `simple_process_imerg_batch_memory_efficient()` function
3. No changes needed to existing workflows - the interface remains the same

The improvements maintain complete compatibility with existing code while providing much better memory management.
