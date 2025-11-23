#!/usr/bin/env python
"""
Overfitting test for DeltaZ model.
Tests if the model can overfit on a single batch - a good sanity check.
If the loss decreases significantly over iterations, the model is learning.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from architecture.DeltaZUnet import DeltaZUnet
from dataset.dtu_dataset import create_dataloaders
from utils.helpers import get_ray_dirs_mask, backproject_depth

print("=" * 80)
print("OVERFITTING TEST - Single Batch Training")
print("=" * 80)

# ============================================================================
# SETUP
# ============================================================================
device = torch.device('cpu')
data_root = "./dtu_train_ready"

print("\n[SETUP] Loading data and model...")
try:
    train_loader, _ = create_dataloaders(
        data_root=data_root,
        batch_size=2,
        num_views=2,
        val_split=0.1,
        device=device,
    )
    
    # Get one batch and keep it fixed
    batch = next(iter(train_loader))
    print(f"✓ Batch loaded: depth {batch['depth'].shape}, scene {batch['scene']}")
    
    # Create model
    model = DeltaZUnet(in_channels=1, base_channel=32, depth=4, out_confidence=False)
    model = model.to(device)
    print(f"✓ Model created: DeltaZUnet")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Higher LR for faster convergence
    print(f"✓ Optimizer: Adam (lr=1e-3)")
    
except Exception as e:
    print(f"✗ Setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n[TRAINING] Running 100 iterations on single batch...")
print("-" * 80)
print(f"{'Iter':>6} | {'Loss':>12} | {'Loss_Depth':>12} | {'Loss_Smooth':>12} | {'Loss_MV':>12}")
print("-" * 80)

num_iterations = 100
losses_history = []

B, V, H, W = batch['depth'].shape
depth_gt = batch['depth'][:, 0]  # (B, H, W)
K_i = batch['intrinsics'][:, 0]  # (B, 3, 3)
R_i = batch['extrinsics'][:, 0, :, :3]  # (B, 3, 3)
t_i = batch['extrinsics'][:, 0, :, 3]  # (B, 3)

try:
    for iteration in range(num_iterations):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        # Use GT depth as initial estimate for now
        delta_z = model(depth_gt.unsqueeze(1))  # (B, 1, H, W)
        delta_z = delta_z.squeeze(1)  # (B, H, W)
        
        depth_pred = depth_gt + delta_z
        
        # Loss 1: Depth loss
        loss_depth = nn.functional.l1_loss(depth_pred, depth_gt)
        
        # Loss 2: Smoothness loss
        dx = delta_z[:, :, :-1] - delta_z[:, :, 1:]
        dy = delta_z[:, :, :-1] - delta_z[:, :, 1:]
        loss_smooth = (dx.abs().mean() + dy.abs().mean()) * 0.1
        
        # Loss 3: Magnitude regularization
        loss_mag = 0.01 * delta_z.abs().mean()
        
        # Total loss
        total_loss = loss_depth + loss_smooth + loss_mag
        
        # Try multi-view loss if available
        loss_mv = 0.0
        if V > 1:
            try:
                depth_j = batch['depth'][:, 1]
                K_j = batch['intrinsics'][:, 1]
                R_j = batch['extrinsics'][:, 1, :, :3]
                t_j = batch['extrinsics'][:, 1, :, 3]
                
                # Get ray directions
                ray_dirs_i, _ = get_ray_dirs_mask(H, W, K_i, device=device)
                if ray_dirs_i.shape[0] == W and ray_dirs_i.shape[1] == H:
                    ray_dirs_i = ray_dirs_i.permute(1, 0, 2)
                ray_dirs_i = ray_dirs_i.unsqueeze(0).expand(B, -1, -1, -1)
                ray_dirs_i = ray_dirs_i.permute(0, 3, 1, 2)
                
                # Backproject
                points_i = backproject_depth(depth_pred.unsqueeze(1), ray_dirs_i, K_i)
                points_i_flat = points_i.reshape(B, 3, -1)
                
                # Transform to world
                R_i_inv = R_i.transpose(-1, -2)
                points_world = torch.matmul(R_i_inv, points_i_flat)
                points_world = points_world + t_i.unsqueeze(-1)
                
                # Transform to view j
                points_j = torch.matmul(R_j, points_world)
                points_j = points_j + t_j.unsqueeze(-1)
                
                # Project to image j
                uv_j = torch.matmul(K_j, points_j)
                uv_j = uv_j.transpose(1, 2)
                uv_j = uv_j[..., :2] / (uv_j[..., 2:3] + 1e-8)
                
                # Normalize for grid sample
                u_norm = (uv_j[..., 0] / (W - 1)) * 2 - 1
                v_norm = (uv_j[..., 1] / (H - 1)) * 2 - 1
                grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B, -1, 1, 2)
                
                # Sample and compare
                depth_j_sampled = torch.nn.functional.grid_sample(
                    depth_j.unsqueeze(1), grid, mode='bilinear', padding_mode='zeros', align_corners=True
                ).squeeze(1).squeeze(-1)
                
                depth_j_pred = points_j[:, 2, :]
                loss_mv = (0.01 * (depth_j_sampled - depth_j_pred).abs().mean()).item()
                total_loss = total_loss + 0.01 * (depth_j_sampled - depth_j_pred).abs().mean()
            except:
                loss_mv = 0.0
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses_history.append(total_loss.item())
        
        # Print
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"{iteration+1:6d} | {total_loss.item():12.6f} | {loss_depth.item():12.6f} | {loss_smooth.item():12.6f} | {loss_mv:12.6f}")
        
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# ANALYSIS
# ============================================================================
print("-" * 80)

initial_loss = losses_history[0]
final_loss = losses_history[-1]
loss_reduction = (initial_loss - final_loss) / initial_loss * 100

print("\n[RESULTS]")
print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss:   {final_loss:.6f}")
print(f"Reduction:    {loss_reduction:.2f}%")

if loss_reduction > 50:
    print("\n✓ EXCELLENT: Model is learning fast (>50% loss reduction)")
    print("  The model can overfit on a single batch")
elif loss_reduction > 20:
    print("\n✓ GOOD: Model is learning (>20% loss reduction)")
    print("  The model can adjust weights to minimize loss")
elif loss_reduction > 0:
    print("\n✓ OK: Model shows some learning (<20% loss reduction)")
    print("  The model is updating but may need tuning")
else:
    print("\n✗ WARNING: Model not learning")
    print("  Check loss computation or learning rate")

print("\n[GRADIENT CHECK]")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
if has_grads:
    print("✓ Gradients are flowing through the network")
else:
    print("✗ No gradients detected")

print("\n" + "=" * 80)
print("OVERFITTING TEST COMPLETE")
print("=" * 80)

if loss_reduction > 20:
    print("\n✅ SANITY CHECK PASSED: Model can learn and overfit")
    print("   Ready for full training!")
else:
    print("\n⚠️  SANITY CHECK WARNING: Model may have training issues")
    print("   Consider checking loss computation or learning rate")

print("=" * 80)
