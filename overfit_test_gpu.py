#!/usr/bin/env python
"""
Overfitting test: Train delta-z to improve noisy initial depth on GPU.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from architecture.DeltaZUnet import DeltaZUnet
from dataset.dtu_dataset import create_dataloaders

print("=" * 80)
print("OVERFITTING TEST - GPU Accelerated")
print("=" * 80)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[DEVICE] Using: {device}")

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
    
    # Get one batch
    batch = next(iter(train_loader))
    B, V, H, W = batch['depth'].shape
    depth_gt = batch['depth'][:, 0]  # Ground truth depth
    
    print(f"[OK] Batch loaded: depth {batch['depth'].shape}, scene {batch['scene']}")
    
    # Create model
    model = DeltaZUnet(in_channels=1, base_channel=32, depth=4, out_confidence=False)
    model = model.to(device)
    print(f"[OK] Model created: DeltaZUnet")
    
    # Optimizer with higher learning rate for faster convergence
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    print(f"[OK] Optimizer: Adam (lr=1e-2)")
    
except Exception as e:
    print(f"[ERROR] Setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TRAINING LOOP - Realistic Setup
# ============================================================================
print("\n[TRAINING] Running 100 iterations...")
print("Scenario: Train model to fix noise in initial depth estimate")
print("-" * 80)
print(f"{'Iter':>6} | {'Total Loss':>12} | {'Depth Loss':>12} | {'Improvement %':>12}")
print("-" * 80)

num_iterations = 100
losses_history = []

try:
    for iteration in range(num_iterations):
        model.train()
        optimizer.zero_grad()
        
        # Create noisy initial depth (simulate MDE errors)
        # Add random noise: ~5% of depth range
        noise = torch.randn_like(depth_gt) * (depth_gt.max() - depth_gt.min()) * 0.05
        depth_noisy = depth_gt + noise
        depth_noisy = depth_noisy.clamp(min=0.1)  # Keep positive
        
        # Model predicts correction
        delta_z = model(depth_noisy.unsqueeze(1))  # Input: noisy depth
        delta_z = delta_z.squeeze(1)  # (B, H, W)
        
        # Apply correction
        depth_refined = depth_noisy + delta_z
        
        # Loss 1: Depth regression loss (MSE for cleaner gradients)
        loss_depth = nn.functional.mse_loss(depth_refined, depth_gt)
        
        # Loss 2: Regularization (don't over-correct)
        loss_mag = 0.001 * delta_z.abs().mean()
        
        # Loss 3: Smoothness (spatial consistency)
        dx = delta_z[:, :, :-1] - delta_z[:, :, 1:]
        dy = delta_z[:, :-1, :] - delta_z[:, 1:, :]
        loss_smooth = (dx.abs().mean() + dy.abs().mean()) * 0.01
        
        # Total loss
        total_loss = loss_depth + loss_mag + loss_smooth
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute improvement metric
        error_before = (depth_noisy - depth_gt).abs().mean()
        error_after = (depth_refined - depth_gt).abs().mean()
        improvement = (error_before - error_after) / error_before * 100
        
        losses_history.append(total_loss.item())
        
        # Print every 10 iterations
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"{iteration+1:6d} | {total_loss.item():12.6f} | {loss_depth.item():12.6f} | {improvement:12.2f}%")
        
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
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

# Also check smoothness of loss curve (not all spiky)
loss_array = np.array(losses_history)
loss_variance = np.std(loss_array[-20:])  # Variance in last 20 iterations

print("\n[RESULTS]")
print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss:   {final_loss:.6f}")
print(f"Loss reduction: {loss_reduction:.2f}%")
print(f"Loss stability (std dev): {loss_variance:.6f}")

print("\n[ANALYSIS]")
if loss_reduction > 50:
    status = "[EXCELLENT]"
    msg = "Model converging rapidly (>50% loss reduction)"
elif loss_reduction > 30:
    status = "[VERY GOOD]"
    msg = "Model learning effectively (>30% loss reduction)"
elif loss_reduction > 15:
    status = "[GOOD]"
    msg = "Model learning (>15% loss reduction)"
elif loss_reduction > 5:
    status = "[OK]"
    msg = "Model adjusting weights (>5% loss reduction)"
else:
    status = "[WARNING]"
    msg = "Model not learning efficiently (<5% loss reduction)"

print(f"{status}: {msg}")

if loss_variance < 0.1:
    print("[OK] Loss curve stable (low variance)")
else:
    print(f"[WARNING] Loss curve noisy (variance: {loss_variance:.6f})")

print("\n[PARAMETER CHECK]")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"Gradients flowing: {'Yes' if has_grads else 'No'}")

# Check weight magnitudes
weight_stats = []
for name, param in model.named_parameters():
    if param.grad is not None:
        weight_stats.append(param.grad.abs().mean().item())

if weight_stats:
    avg_grad = np.mean(weight_stats)
    print(f"Average gradient magnitude: {avg_grad:.6e}")
    if avg_grad < 1e-8:
        print("[WARNING] Very small gradients - model may not be learning")
    elif avg_grad > 1.0:
        print("[WARNING] Large gradients - consider lower learning rate")
    else:
        print("[OK] Gradient magnitude in healthy range")

print("\n" + "=" * 80)
print("OVERFITTING TEST COMPLETE")
print("=" * 80)

if loss_reduction > 15:
    print("\n[PASSED] SANITY CHECK PASSED")
    print("   Model can learn to correct depth errors!")
    print("   Ready for full training with proper initial depth estimates.")
else:
    print("\n[WARNING] SANITY CHECK WARNING")
    print("   Model learning is slow. Check:")
    print("   - Learning rate (try adjusting)")
    print("   - Loss functions (verify they're sensible)")
    print("   - Initial weights (consider different initialization)")

print("=" * 80)
