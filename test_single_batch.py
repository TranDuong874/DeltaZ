#!/usr/bin/env python3
"""
Test a single training batch end-to-end.
Verifies: dataloader → model → losses → backward pass
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add model to path
sys.path.insert(0, str(Path(__file__).parent))

from model.architecture.DeltaZUnet import DeltaZUnet
from model.dataset.dtu_dataset import create_dataloaders
from model.utils.losses import compute_loss
from model.utils.helpers import get_ray_dirs_mask


def test_single_batch():
    print("=" * 70)
    print("Testing Single Batch Training")
    print("=" * 70)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataloaders
    print("\n[1/5] Loading dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            batch_size=4,
            num_views=2,
            train_split=0.9,
            shuffle=True
        )
        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False
    
    # Get first batch
    print("\n[2/5] Loading first batch...")
    try:
        batch = next(iter(train_loader))
        depth = batch['depth'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        scene = batch['scene']
        
        B, V, H, W = depth.shape
        print(f"✓ Batch loaded:")
        print(f"  Depth: {depth.shape} | dtype={depth.dtype} | device={depth.device}")
        print(f"  Intrinsics: {intrinsics.shape}")
        print(f"  Extrinsics: {extrinsics.shape}")
        print(f"  Scene: {scene}")
        print(f"  Batch size: {B}, Views: {V}, Spatial: {H}×{W}")
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize model
    print("\n[3/5] Initializing model...")
    try:
        model = DeltaZUnet(in_channels=1, out_channels=1, base_channels=32)
        model = model.to(device)
        print(f"✓ Model loaded: DeltaZUnet")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Forward pass
    print("\n[4/5] Forward pass...")
    try:
        # Use first view as input, second view for reference
        depth_0 = depth[:, 0:1, :, :]  # (B, 1, H, W)
        depth_1 = depth[:, 1:2, :, :]  # (B, 1, H, W)
        K_0 = intrinsics[:, 0, :, :]   # (B, 3, 3)
        K_1 = intrinsics[:, 1, :, :]   # (B, 3, 3)
        E_0 = extrinsics[:, 0, :, :]   # (B, 3, 4)
        E_1 = extrinsics[:, 1, :, :]   # (B, 3, 4)
        
        print(f"  Input depth_0: {depth_0.shape}")
        print(f"  Reference depth_1: {depth_1.shape}")
        
        # Forward pass
        delta_z_pred = model(depth_0)
        print(f"✓ Model output (delta_z): {delta_z_pred.shape}")
        print(f"  Min: {delta_z_pred.min():.4f}, Max: {delta_z_pred.max():.4f}")
        print(f"  Mean: {delta_z_pred.mean():.4f}, Std: {delta_z_pred.std():.4f}")
        
        # Refined depth
        depth_0_refined = depth_0 + delta_z_pred
        print(f"✓ Refined depth: {depth_0_refined.shape}")
        print(f"  Min: {depth_0_refined.min():.4f}, Max: {depth_0_refined.max():.4f}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compute loss
    print("\n[5/5] Computing loss...")
    try:
        loss = compute_loss(
            delta_z=delta_z_pred,
            depth_pred=depth_0,
            depth_gt=depth_1,
            depth_refined=depth_0_refined,
            K=K_0,
            K_ref=K_1,
            E=E_0,
            E_ref=E_1,
            loss_weights={
                'depth': 1.0,
                'delta': 0.5,
                'magnitude': 0.01,
                'smooth': 0.1,
                'multiview': 0.5,
            }
        )
        
        print(f"✓ Loss computed: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        print(f"✓ Backward pass successful")
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            print(f"✓ Gradients:")
            print(f"  Min grad norm: {min(grad_norms):.6f}")
            print(f"  Max grad norm: {max(grad_norms):.6f}")
            print(f"  Mean grad norm: {sum(grad_norms)/len(grad_norms):.6f}")
        else:
            print(f"✗ No gradients computed")
            return False
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ Single batch test PASSED!")
    print("=" * 70)
    print("\nSystem is ready for training. Run:")
    print("  python3 train.py --device cuda --num-epochs 50")
    
    return True


if __name__ == '__main__':
    try:
        success = test_single_batch()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
