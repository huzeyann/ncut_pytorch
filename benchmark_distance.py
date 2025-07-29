#!/usr/bin/env python3
"""
Benchmark script to compare performance of different L2 distance computation methods.
"""

import torch
import time
import numpy as np
from ncut_pytorch.utils.math_utils import (
    compute_l2_distance,
    compute_l2_distance_fast,
    compute_l2_distance_memory_efficient,
    compute_l2_distance_mixed_precision,
    compute_l2_distance_optimized
)


def benchmark_distance_functions():
    """Benchmark different distance computation methods."""
    
    # Test configurations
    test_configs = [
        {"name": "Small (1K x 1K)", "n1": 1000, "n2": 1000, "d": 512},
        {"name": "Medium (5K x 5K)", "n1": 5000, "n2": 5000, "d": 768},
        {"name": "Large (10K x 10K)", "n1": 10000, "n2": 10000, "d": 1024},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmarks on: {device}")
    print("=" * 80)
    
    for config in test_configs:
        print(f"\n{config['name']} - Shape: ({config['n1']}, {config['n2']}) x {config['d']}")
        print("-" * 60)
        
        # Generate random test data
        X1 = torch.randn(config['n1'], config['d'], device=device, dtype=torch.float32)
        X2 = torch.randn(config['n2'], config['d'], device=device, dtype=torch.float32)
        
        # Warm up GPU
        if device.type == 'cuda':
            for _ in range(3):
                _ = torch.cdist(X1[:100], X2[:100], p=2) ** 2
            torch.cuda.synchronize()
        
        methods = [
            ("Original", compute_l2_distance),
            ("Fast (torch.cdist)", compute_l2_distance_fast),
            ("Mixed Precision", compute_l2_distance_mixed_precision),
            ("Optimized (Auto)", compute_l2_distance_optimized),
        ]
        
        # Add memory efficient method for larger sizes
        if config['n1'] > 5000:
            methods.append(("Memory Efficient", lambda x1, x2: compute_l2_distance_memory_efficient(x1, x2, chunk_size=1024)))
        
        times = {}
        results = {}
        
        for method_name, method_func in methods:
            try:
                # Time the computation
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                start_time = time.time()
                result = method_func(X1, X2)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                end_time = time.time()
                elapsed = end_time - start_time
                
                times[method_name] = elapsed
                results[method_name] = result
                
                print(f"{method_name:20s}: {elapsed:.3f}s")
                
            except Exception as e:
                print(f"{method_name:20s}: ERROR - {str(e)}")
                continue
        
        # Verify all methods produce similar results
        if len(results) > 1:
            base_result = list(results.values())[0]
            max_diff = 0
            for name, result in results.items():
                diff = torch.max(torch.abs(result - base_result)).item()
                max_diff = max(max_diff, diff)
            
            print(f"\nMaximum difference between methods: {max_diff:.2e}")
            
            # Calculate speedups
            if "Original" in times:
                baseline_time = times["Original"]
                print("\nSpeedups vs Original:")
                for name, elapsed in times.items():
                    if name != "Original":
                        speedup = baseline_time / elapsed
                        print(f"  {name:18s}: {speedup:.2f}x faster")
        
        # Memory usage estimate
        memory_gb = (config['n1'] * config['n2'] * 4) / (1024**3)
        print(f"\nEstimated memory usage: {memory_gb:.2f} GB")


def test_accuracy():
    """Test that all methods produce the same results."""
    print("\n" + "=" * 80)
    print("ACCURACY TEST")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small test case
    X1 = torch.randn(100, 64, device=device)
    X2 = torch.randn(100, 64, device=device)
    
    methods = [
        ("Original", compute_l2_distance),
        ("Fast", compute_l2_distance_fast),
        ("Mixed Precision", compute_l2_distance_mixed_precision),
        ("Memory Efficient", lambda x1, x2: compute_l2_distance_memory_efficient(x1, x2, chunk_size=32)),
        ("Optimized", compute_l2_distance_optimized),
    ]
    
    results = {}
    for name, func in methods:
        try:
            results[name] = func(X1, X2)
        except Exception as e:
            print(f"Error in {name}: {e}")
            continue
    
    # Compare all methods against the original
    if "Original" in results:
        base_result = results["Original"]
        print("Maximum absolute differences from original implementation:")
        for name, result in results.items():
            if name != "Original":
                max_diff = torch.max(torch.abs(result - base_result)).item()
                mean_diff = torch.mean(torch.abs(result - base_result)).item()
                print(f"  {name:18s}: max={max_diff:.2e}, mean={mean_diff:.2e}")


if __name__ == "__main__":
    print("L2 Distance Computation Benchmark")
    print("=" * 80)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, running on CPU")
    
    # Run accuracy test first
    test_accuracy()
    
    # Run performance benchmarks
    benchmark_distance_functions()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("• For small to medium matrices (< 10K points): use compute_l2_distance_fast()")
    print("• For large matrices: use compute_l2_distance_optimized() (auto-selects best method)")
    print("• For memory-constrained scenarios: use compute_l2_distance_memory_efficient()")
    print("• For maximum speed on modern GPUs: enable mixed precision in optimized version")
    print("• The get_affinity() function now uses optimized distance by default!") 