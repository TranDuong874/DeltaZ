#!/usr/bin/env python3
"""
Quick test to verify dataloader and model work correctly.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from dataset.dtu_dataset import create_dataloaders
from architecture.DeltaZUnet import DeltaZUnet
from utils.helpers import get_ray_dirs_mask


def test_dataloader():
    print("=" * 60)
    print("Testing DTU DataLoader")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_root='./model/dataset/dtu_train_ready',
        batch_size=2,
        num_views=2,
        val_split=0.1,
        device=device,
    )
    
    print(f"Number of training batches per epoch: {len(train_loader)}")
    print(f"Number of validation batches per epoch: {len(val_loader)}")
    
    # Get one batch
    print("\nLoading first training batch...")
    for batch in train_loader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"  depth shape: {batch['depth'].shape}")  # (B, V, H, W)
        print(f"  intrinsics shape: {batch['intrinsics'].shape}")  # (B, V, 3, 3)
        print(f"  extrinsics shape: {batch['extrinsics'].shape}")  # (B, V, 3, 4)
        print(f"  frame_ids shape: {batch['frame_ids'].shape}")
        print(f"  scene: {batch['scene']}")
        
        B, V, H, W = batch['depth'].shape
        print(f"\nBatch properties:")
        print(f"  Batch size: {B}")
        print(f"  Views per sample: {V}")
        print(f"  Image height: {H}")
        print(f"  Image width: {W}")
        
        # Check that all samples in batch are from same scene
        print(f"\nAll samples from same scene: {all(s == batch['scene'][0] for s in batch['scene'])}")
        break
    
    return True


def test_model():
    print("\n" + "=" * 60)
    print("Testing DeltaZ Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = DeltaZUnet(in_channels=1, base_channel=32, depth=4, out_confidence=False)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    B, H, W = 2, 256, 256
    dummy_input = torch.randn(B, 1, H, W, device=device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check output
    expected_shape = (B, 1, H, W)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape is correct: {output.shape}")
    
    return True


def test_ray_directions():
    print("\n" + "=" * 60)
    print("Testing Ray Direction Generation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    H, W = 480, 640
    # Create sample intrinsics
    K = torch.tensor([[
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0],
    ]], device=device)
    
    print(f"\nGenerating ray directions for {H}x{W} image...")
    ray_dirs, mask = get_ray_dirs_mask(H, W, K, device=device)
    
    print(f"Ray directions shape: {ray_dirs.shape}")  # (H, W, 3)
    print(f"Mask shape: {mask.shape}")  # (H, W)
    
    # Check values
    print(f"Ray direction range: [{ray_dirs.min():.3f}, {ray_dirs.max():.3f}]")
    print(f"Mask dtype: {mask.dtype}, values: {mask.unique()}")
    
    return True


def test_helpers():
    print("\n" + "=" * 60)
    print("Testing Helper Functions")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from utils.helpers import backproject_depth
    
    # Test backproject_depth
    print("\nTesting backproject_depth...")
    B, H, W = 2, 128, 128
    depth = torch.ones(B, 1, H, W, device=device) * 5.0
    ray_dirs = torch.randn(B, H, W, 3, device=device)
    ray_dirs[..., 2] = 1.0  # Set z=1
    
    K = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    
    points_3d = backproject_depth(depth, ray_dirs, K)
    print(f"Input depth shape: {depth.shape}")
    print(f"Output points_3d shape: {points_3d.shape}")  # (B, 3, H, W)
    
    return True


if __name__ == '__main__':
    try:
        test_dataloader()
        test_model()
        test_ray_directions()
        test_helpers()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
