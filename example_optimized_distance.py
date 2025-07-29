#!/usr/bin/env python3
"""
Example showing how to use the optimized L2 distance functions.
"""

import torch
import time
from ncut_pytorch.utils.math_utils import (
    compute_l2_distance,
    compute_l2_distance_fast,
    compute_l2_distance_optimized,
    get_affinity
)


def main():
    print("Optimized L2 Distance Computation Example")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data (simulating feature vectors from your NCUT workflow)
    n_samples = 2000
    n_features = 768  # Similar to DINO features
    
    X1 = torch.randn(n_samples, n_features, device=device)
    X2 = torch.randn(n_samples, n_features, device=device)
    
    print(f"\nData shape: {X1.shape}")
    
    # 1. Basic usage - just replace the function call
    print("\n1. Basic optimization - replace function call:")
    
    # Old way (slower)
    start = time.time()
    distances_old = compute_l2_distance(X1, X2)
    time_old = time.time() - start
    
    # New way (faster)
    start = time.time()
    distances_fast = compute_l2_distance_fast(X1, X2)
    time_fast = time.time() - start
    
    # Automatic optimization (best choice)
    start = time.time()
    distances_optimized = compute_l2_distance_optimized(X1, X2)
    time_optimized = time.time() - start
    
    print(f"  Original method:  {time_old:.3f}s")
    print(f"  Fast method:      {time_fast:.3f}s ({time_old/time_fast:.1f}x faster)")
    print(f"  Optimized method: {time_optimized:.3f}s ({time_old/time_optimized:.1f}x faster)")
    
    # Verify results are the same
    max_diff = torch.max(torch.abs(distances_old - distances_fast)).item()
    print(f"  Max difference: {max_diff:.2e} (should be very small)")
    
    # 2. Using with affinity computation (this is where the speedup really matters)
    print("\n2. Using with affinity matrix computation:")
    
    # Old way
    start = time.time()
    affinity_old = get_affinity(X1, X2, gamma=1.0, use_optimized_distance=False)
    time_affinity_old = time.time() - start
    
    # New way (default behavior)
    start = time.time()
    affinity_new = get_affinity(X1, X2, gamma=1.0)  # uses optimized by default
    time_affinity_new = time.time() - start
    
    print(f"  Affinity (old):     {time_affinity_old:.3f}s")
    print(f"  Affinity (new):     {time_affinity_new:.3f}s ({time_affinity_old/time_affinity_new:.1f}x faster)")
    
    # 3. Memory-efficient computation for very large matrices
    print("\n3. Memory-efficient computation:")
    
    # For very large matrices, use chunking
    if device.type == 'cuda':
        try:
            # Create larger test data
            X_large = torch.randn(5000, 512, device=device)
            
            start = time.time()
            distances_chunked = compute_l2_distance_optimized(
                X_large, X_large, 
                chunk_size=1024,  # Process in smaller chunks
                use_mixed_precision=True  # Use half precision for speed
            )
            time_chunked = time.time() - start
            
            print(f"  Large matrix ({X_large.shape[0]}x{X_large.shape[0]}): {time_chunked:.3f}s")
            print(f"  Memory usage reduced by chunking")
            
        except Exception as e:
            print(f"  Large matrix test skipped: {e}")
    
    # 4. Integration with your existing NCUT workflow
    print("\n4. Integration with NCUT workflow:")
    print("   The optimizations are now integrated into get_affinity().")
    print("   Your existing code should work faster without any changes!")
    print("   Just make sure use_optimized_distance=True (default)")
    
    # Example from your hierarchy test
    print("\n5. Example with real NCUT workflow:")
    print("   # Your existing code:")
    print("   # eigvecs = Ncut(n_eig=n_eig, d_gamma=degree, n_neighbors=n_neighbors,")
    print("   #                matmul_chunk_size=chunk_size,")
    print("   #                ).fit_transform(feats)")
    print("   # This will now automatically use the optimized distance computation!")


if __name__ == "__main__":
    main() 