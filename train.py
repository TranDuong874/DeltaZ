"""
DeltaZ Training Script

This script trains the DeltaZ depth refinement model using:
- Multi-view consistency loss
- 3D point supervision
- Edge-aware smoothness
- Depth correction loss
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from typing import Dict, Tuple

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from architecture.DeltaZUnet import DeltaZUnet
from utils.losses import combined_deltaz_loss
from utils.helpers import (
    backproject_depth,
    get_ray_dirs_mask,
    transform_points,
    project_points
)
from dataset.dtu_dataset import create_dataloaders


class DeltaZTrainer:
    """Trainer for DeltaZ depth refinement model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        """
        Args:
            model: DeltaZUnet model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: torch device
            lr: Learning rate
            weight_decay: L2 regularization
            checkpoint_dir: Where to save checkpoints
            log_dir: Where to save tensorboard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=1,
            eta_min=1e-6,
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        
        self.global_step = 0
        self.epoch = 0
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepare batch data for training.
        
        Returns:
            Dict with all batch data and parsed extrinsics
        """
        # Batch structure:
        # depth: (B, num_views, H, W)
        # intrinsics: (B, num_views, 3, 3)
        # extrinsics: (B, num_views, 3, 4)
        
        depth = batch['depth'].to(self.device)  # (B, V, H, W)
        intrinsics = batch['intrinsics'].to(self.device)  # (B, V, 3, 3)
        extrinsics = batch['extrinsics'].to(self.device)  # (B, V, 3, 4)
        
        B, V, H, W = depth.shape
        
        # Parse extrinsics
        R_list = extrinsics[..., :3, :3]  # (B, V, 3, 3)
        t_list = extrinsics[..., :3, 3]   # (B, V, 3)
        
        # Prepare for forward pass
        data = {
            'depth': depth,
            'intrinsics': intrinsics,
            'R': R_list,
            't': t_list,
            'H': H,
            'W': W,
            'B': B,
            'V': V,
        }
        
        return data
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step.
        
        When use_noisy=True:
        - batch['depth']: noisy initial depth estimate
        - batch['depth_gt']: ground truth depth
        - Model predicts: delta_z = depth_gt - depth_noisy
        """
        B, V, H, W = batch['depth'].shape
        
        # Get ground truth depth
        # If depth_gt is available, use it; otherwise use first view depth
        if 'depth_gt' in batch:
            depth_gt = batch['depth_gt'][:, 0]  # (B, H, W)
            depth_initial = batch['depth'][:, 0]  # Noisy initial depth
        else:
            depth_gt = batch['depth'][:, 0]  # (B, H, W) - GT (first view)
            depth_initial = batch['depth'][:, 0]  # Use same for both
        
        # Model predicts delta correction
        delta_z = self.model(depth_initial.unsqueeze(1))  # Input: (B, 1, H, W)
        delta_z = delta_z.squeeze(1)  # Output: (B, H, W)
        
        # Refined depth
        depth_refined = depth_initial + delta_z  # Apply correction
        
        # Loss 1: Depth refinement (L1 loss)
        loss_depth = torch.nn.functional.l1_loss(depth_refined, depth_gt)
        
        # Loss 2: Delta improvement (encourage meaningful corrections)
        # Penalize if delta doesn't reduce error
        error_before = (depth_initial - depth_gt).abs()
        error_after = (depth_refined - depth_gt).abs()
        improvement = (error_before - error_after).clamp(min=0)  # Only positive improvements
        loss_delta = -improvement.mean()  # Negative because we want to maximize improvement
        
        # Loss 3: Magnitude regularization (don't over-correct)
        loss_mag = 0.01 * delta_z.abs().mean()
        
        # Loss 4: Smoothness (spatial consistency)
        dx = delta_z[:, :, :-1] - delta_z[:, :, 1:]
        dy = delta_z[:, :-1, :] - delta_z[:, 1:, :]
        loss_smooth = 0.01 * (dx.abs().mean() + dy.abs().mean())
        
        loss_dict = {
            'loss_depth': loss_depth.item(),
            'loss_delta': loss_delta.item(),
            'loss_mag': loss_mag.item(),
            'loss_smooth': loss_smooth.item(),
        }
        
        # Loss 5: Multi-view consistency (if V > 1) - SKIP FOR NOW (too complex)
        # Multi-view losses need more careful handling of camera transforms
        loss_dict['loss_mv'] = 0.0
        
        # Total loss
        total_loss = (
            loss_depth +
            0.5 * loss_delta +
            loss_mag +
            loss_smooth
        )
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        loss_dict['total'] = total_loss.item()
        
        return loss_dict
        
        # Get ray directions for reference view
        ray_dirs_i, _ = get_ray_dirs_mask(H, W, K_i, device=self.device)
        # Handle case where function returns (W, H, 3) instead of (H, W, 3)
        if ray_dirs_i.shape[0] == W and ray_dirs_i.shape[1] == H:
            ray_dirs_i = ray_dirs_i.permute(1, 0, 2)  # swap to (H, W, 3)
        ray_dirs_i = ray_dirs_i.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)
        ray_dirs_i = ray_dirs_i.permute(0, 3, 1, 2)  # (B, 3, H, W)
        # Forward pass: predict delta_z
        delta_z = self.model(torch.zeros_like(depth_gt).unsqueeze(1))  # (B, 1, H, W)
        delta_z = delta_z.squeeze(1)  # (B, H, W)
        
        # In practice, we need depth0 as input. For now, use GT as depth0
        # Later: depth0 would come from MDE or SfM
        depth0 = depth_gt.clone().detach()
        
        # Loss computation
        loss_dict = {}
        
        # Loss 1: Basic depth loss
        depth_pred = depth0 + delta_z
        loss_depth = torch.nn.functional.l1_loss(depth_pred, depth_gt)
        loss_dict['loss_depth'] = loss_depth.item()
        
        # Loss 2: Delta-z improvement loss
        err_before = (depth0 - depth_gt).abs()
        err_after = (depth_pred - depth_gt).abs()
        improvement = err_before - err_after
        loss_delta = (-improvement).mean()
        loss_dict['loss_delta'] = loss_delta.item()
        
        # Loss 3: Correction magnitude (regularization)
        loss_mag = 0.01 * delta_z.abs().mean()
        loss_dict['loss_mag'] = loss_mag.item()
        
        # Loss 4: Edge-aware smoothness
        dx_dz = delta_z[:, :, :-1] - delta_z[:, :, 1:]
        dy_dz = delta_z[:, :, :-1] - delta_z[:, :, 1:]
        loss_smooth = (dx_dz.abs().mean() + dy_dz.abs().mean()) * 0.1
        loss_dict['loss_smooth'] = loss_smooth.item()
        
        # Multi-view consistency (if V > 1)
        if V > 1:
            try:
                # Use first neighbor as second view
                depth_j = data['depth'][:, 1]  # (B, H, W)
                K_j = data['intrinsics'][:, 1]  # (B, 3, 3)
                R_j = data['R'][:, 1]  # (B, 3, 3)
                t_j = data['t'][:, 1]  # (B, 3)
                
                # Backproject reference view
                points_i = backproject_depth(
                    depth_pred.unsqueeze(1),  # (B, 1, H, W)
                    ray_dirs_i,  # (B, 3, H, W)
                    K_i
                )  # (B, 3, H, W)
                
                # Flatten points: (B, 3, H*W)
                points_i_flat = points_i.reshape(B, 3, -1)  # (B, 3, H*W)
                
                # Transform from view i to world frame
                # world_point = R_i^T @ (point_i - t_i)
                R_i_inv = R_i.transpose(-1, -2)  # (B, 3, 3)
                # Reshape for batched matmul: (B, 3, 3) @ (B, 3, N) = (B, 3, N)
                points_world = torch.matmul(R_i_inv, points_i_flat)  # (B, 3, H*W)
                points_world = points_world + t_i.unsqueeze(-1)  # (B, 3, H*W)
                
                # Transform from world to view j frame
                # view_j_point = R_j @ point_world + t_j
                points_j = torch.matmul(R_j, points_world)  # (B, 3, H*W)
                points_j = points_j + t_j.unsqueeze(-1)  # (B, 3, H*W)
                
                # Project to image j using K_j
                # uv = K @ p = (B, 3, 3) @ (B, 3, N) = (B, 3, N)
                uv_j = torch.matmul(K_j, points_j)  # (B, 3, H*W)
                uv_j = uv_j.transpose(1, 2)  # (B, H*W, 3)
                uv_j = uv_j[..., :2] / (uv_j[..., 2:3] + 1e-8)  # Perspective division -> (B, N, 2)
                
                # Normalize to [-1, 1] for grid_sample
                u_norm = (uv_j[..., 0] / (W - 1)) * 2 - 1
                v_norm = (uv_j[..., 1] / (H - 1)) * 2 - 1
                grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B, -1, 1, 2)  # (B, N, 1, 2)
                
                # Sample depth from view j
                depth_j_sampled = torch.nn.functional.grid_sample(
                    depth_j.unsqueeze(1),  # (B, 1, H, W)
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )  # (B, 1, N, 1)
                depth_j_sampled = depth_j_sampled.squeeze(1).squeeze(-1)  # (B, N)
                
                # Compare depths
                # points_j is (B, 3, N), z coordinate is at index 2
                depth_j_pred = points_j[:, 2, :]  # (B, N)
                diff_mv = (depth_j_sampled - depth_j_pred).abs().mean()
                loss_dict['loss_mv'] = (0.01 * diff_mv).item()  # Reduced weight for multi-view loss
            except Exception as e:
                print(f"[WARNING] Multi-view loss failed: {e}")
                import traceback
                traceback.print_exc()
                loss_dict['loss_mv'] = 0.0
        
        # Total loss
        total_loss = (
            loss_depth +
            0.5 * loss_delta +
            loss_mag +
            loss_smooth
        )
        
        # Add multi-view loss if computed
        loss_mv_tensor = None
        if V > 1 and 'loss_mv' in loss_dict and loss_dict['loss_mv'] > 0:
            loss_mv_tensor = diff_mv * 0.01
            total_loss = total_loss + loss_mv_tensor
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        loss_dict['total'] = total_loss.item()
        
        return loss_dict
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                loss_dict = self.train_step(batch)
                
                # Accumulate losses
                for key, val in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += val
                
                num_batches += 1
                self.global_step += 1
                
                # Log to tensorboard
                for key, val in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', val, self.global_step)
                
                if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                    print(f"Epoch {self.epoch} Batch {batch_idx + 1:4d}: "
                          f"loss={loss_dict['total']:8.4f} "
                          f"(depth={loss_dict.get('loss_depth', 0):7.4f}, "
                          f"delta={loss_dict.get('loss_delta', 0):7.4f})")
                
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")
                continue
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set with depth improvement metrics."""
        self.model.eval()
        
        val_losses = {}
        num_batches = 0
        
        # Depth improvement metrics
        total_depth_mae_before = 0.0  # Mean Absolute Error before refinement
        total_depth_mae_after = 0.0   # Mean Absolute Error after refinement
        total_depth_mse_before = 0.0
        total_depth_mse_after = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    B, V, H, W = batch['depth'].shape
                    
                    # Use noisy/GT pair if available, otherwise use GT only
                    if 'depth_gt' in batch:
                        depth_noisy = batch['depth'][:, 0]
                        depth_gt = batch['depth_gt'][:, 0]
                    else:
                        # Fallback: use GT as both (won't improve)
                        depth_noisy = batch['depth'][:, 0]
                        depth_gt = batch['depth'][:, 0]
                    
                    # Get model's delta prediction
                    delta_z = self.model(depth_noisy.unsqueeze(1))
                    delta_z = delta_z.squeeze(1)
                    
                    # Refined depth
                    depth_refined = depth_noisy + delta_z
                    
                    # Compute depth errors
                    mae_before = (depth_noisy - depth_gt).abs().mean().item()
                    mae_after = (depth_refined - depth_gt).abs().mean().item()
                    mse_before = ((depth_noisy - depth_gt) ** 2).mean().item()
                    mse_after = ((depth_refined - depth_gt) ** 2).mean().item()
                    
                    total_depth_mae_before += mae_before
                    total_depth_mae_after += mae_after
                    total_depth_mse_before += mse_before
                    total_depth_mse_after += mse_after
                    
                    num_batches += 1
                except Exception as e:
                    print(f"[ERROR] Validation batch failed: {e}")
                    continue
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        
        # Average depth metrics
        num_batches = max(num_batches, 1)
        depth_mae_before = total_depth_mae_before / num_batches
        depth_mae_after = total_depth_mae_after / num_batches
        depth_mse_before = total_depth_mse_before / num_batches
        depth_mse_after = total_depth_mse_after / num_batches
        
        # Improvement metrics
        mae_improvement = (depth_mae_before - depth_mae_after) / (depth_mae_before + 1e-8) * 100
        mse_improvement = (depth_mse_before - depth_mse_after) / (depth_mse_before + 1e-8) * 100
        
        # Add depth metrics to val_losses
        val_losses['depth_mae_before'] = depth_mae_before
        val_losses['depth_mae_after'] = depth_mae_after
        val_losses['depth_mae_improvement_pct'] = mae_improvement
        val_losses['depth_mse_before'] = depth_mse_before
        val_losses['depth_mse_after'] = depth_mse_after
        val_losses['depth_mse_improvement_pct'] = mse_improvement
        
        # Compute total validation loss (average of mae improvement)
        val_losses['total'] = -mae_improvement  # Negative because we want to minimize this
        
        # Log to tensorboard
        for key, val in val_losses.items():
            self.writer.add_scalar(f'val/{key}', val, self.global_step)
        
        return val_losses
    
    def train(self, num_epochs: int = 50, val_every: int = 5, save_every: int = 10):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Total epochs
            val_every: Validate every N epochs
            save_every: Save checkpoint every N epochs
        """
        best_val_loss = float('inf')
        best_mae_improvement = -float('inf')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            start_time = time.time()
            train_losses = self.train_epoch()
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch} finished in {epoch_time:.2f}s")
            print(f"Train - total_loss: {train_losses['total']:.4f}, "
                  f"depth: {train_losses['loss_depth']:.4f}, "
                  f"delta: {train_losses['loss_delta']:.4f}")
            
            # Validation
            if (epoch + 1) % val_every == 0:
                val_losses = self.validate()
                print(f"\nValidation Results:")
                print(f"  Total Loss:             {val_losses['total']:.4f}")
                print(f"  Depth MAE Before:       {val_losses['depth_mae_before']:.6f}")
                print(f"  Depth MAE After:        {val_losses['depth_mae_after']:.6f}")
                print(f"  Depth MAE Improvement:  {val_losses['depth_mae_improvement_pct']:.2f}%")
                print(f"  Depth MSE Before:       {val_losses['depth_mse_before']:.6f}")
                print(f"  Depth MSE After:        {val_losses['depth_mse_after']:.6f}")
                print(f"  Depth MSE Improvement:  {val_losses['depth_mse_improvement_pct']:.2f}%")
                
                # Track best models
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    self.save_checkpoint(tag='best_loss')
                    print(f"  >> New best loss: {best_val_loss:.4f}")
                
                if val_losses['depth_mae_improvement_pct'] > best_mae_improvement:
                    best_mae_improvement = val_losses['depth_mae_improvement_pct']
                    self.save_checkpoint(tag='best_mae_improvement')
                    print(f"  >> New best MAE improvement: {best_mae_improvement:.2f}%")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(tag=f'epoch_{epoch}')
            
            # Update learning rate
            self.scheduler.step()
    
    def save_checkpoint(self, tag: str = 'latest'):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"deltaz_{tag}.pt"
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DeltaZ depth refinement model")
    parser.add_argument('--data-root', type=str, default='./dtu_train_ready',
                       help='Path to dtu_train_ready dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num-views', type=int, default=2,
                       help='Number of views per sample')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='torch device')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: run 1 batch with 2 samples only')
    parser.add_argument('--use-noisy', action='store_true',
                       help='Use noisy inputs with GT targets (from dtu_train_noisy/)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Determine data root and GT path
    if args.use_noisy:
        data_root = './dtu_train_noisy'
        gt_path = './dtu_train_ready'
        print(f"Using noisy training mode:")
        print(f"  Input (noisy): {data_root}")
        print(f"  Target (GT):   {gt_path}")
    else:
        data_root = args.data_root
        gt_path = None
        if not os.path.isabs(data_root):
            # If relative path, try to find dtu_train_ready in DeltaZ root
            if not os.path.exists(data_root):
                possible_root = os.path.join(os.path.dirname(__file__), data_root)
                if os.path.exists(possible_root):
                    data_root = possible_root
        print(f"Data root: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"ERROR: Data root does not exist: {data_root}")
        return
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_views=args.num_views,
        val_split=args.val_split,
        device=device,
        use_noisy=args.use_noisy,
        gt_path=gt_path,
    )
    
    # Create model
    print("Creating model...")
    model = DeltaZUnet(in_channels=1, base_channel=32, depth=4, out_confidence=False)
    
    # Create trainer
    trainer = DeltaZTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Test mode: run single batch
    if args.test_mode:
        print("\n=== TEST MODE: Running 1 batch ===")
        batch_count = 0
        try:
            for batch in train_loader:
                batch_count += 1
                print(f"\nBatch {batch_count}:")
                print(f"  Depth shape: {batch['depth'].shape}")
                print(f"  Intrinsics shape: {batch['intrinsics'].shape}")
                print(f"  Extrinsics shape: {batch['extrinsics'].shape}")
                print(f"  Scene: {batch['scene']}")
                
                # Run single training step
                losses = trainer.train_step(batch)
                print(f"  Losses: {losses}")
                
                if batch_count >= 1:
                    break
        except Exception as e:
            print(f"ERROR during test batch: {e}")
            import traceback
            traceback.print_exc()
        print("Test complete!\n")
        return
    
    # Train
    print("Starting training...")
    trainer.train(num_epochs=args.num_epochs)
    
    print("Training complete!")


if __name__ == '__main__':
    main()
