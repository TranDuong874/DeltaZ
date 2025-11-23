#!/usr/bin/env python3
"""
Load and inspect a sample batch from the DeltaZ dataloader.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

import torch
import numpy as np
from dataset.dtu_dataset import create_dataloaders


def main():
    print("=" * 70)
    print("Loading DeltaZ Sample Batch")
    print("=" * 70)
    
    device = torch.device('cpu')  # Use CPU since we don't have CUDA
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            data_root='./model/dataset/dtu_train_ready',
            batch_size=2,  # Small batch for inspection
            num_views=2,
            val_split=0.1,
            device=device,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create dataloaders: {e}")
        return False
    
    print("✓ Dataloaders created")
    
    # Get first batch from training loader
    print("\nLoading first training batch...")
    try:
        for batch_idx, batch in enumerate(train_loader):
            print(f"\n✓ Got batch {batch_idx}:")
            print(f"  Scene: {batch['scene']}")
            print(f"  Frame IDs: {batch['frame_ids']}")
            print(f"  Depth shape: {batch['depth'].shape}")
            print(f"  Intrinsics shape: {batch['intrinsics'].shape}")
            print(f"  Extrinsics shape: {batch['extrinsics'].shape}")
            
            # Details
            depth = batch['depth']
            print(f"\nDepth Statistics:")
            print(f"  Min: {depth.min():.4f}")
            print(f"  Max: {depth.max():.4f}")
            print(f"  Mean: {depth.mean():.4f}")
            print(f"  Median: {torch.median(depth):.4f}")
            print(f"  Dtype: {depth.dtype}")
            print(f"  Device: {depth.device}")
            
            # Intrinsics
            K = batch['intrinsics'][0, 0]
            print(f"\nIntrinsics (first view, first sample):")
            print(K)
            
            # Extrinsics
            extr = batch['extrinsics'][0, 0]
            print(f"\nExtrinsics (first view, first sample):")
            print(extr)
            
            # Check all samples in batch are from same scene
            scenes = batch['scene']
            all_same = all(s == scenes[0] for s in scenes)
            print(f"\n✓ All samples from same scene: {all_same}")
            
            if batch_idx >= 0:  # Just show first batch
                break
            
    except Exception as e:
        print(f"[ERROR] Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ Sample loading successful!")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
