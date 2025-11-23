import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random


class DTUSceneDataset(Dataset):
    """
    Load a single DTU scene's depth, intrinsics, and extrinsics.
    Returns (depth, intrinsic, extrinsic) tuples for frame pairs.
    """
    
    def __init__(
        self,
        scene_path: str,
        num_views: int = 2,
        max_frames: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            scene_path: Path to a single scan folder (e.g., dtu_train_ready/scan1)
            num_views: Number of views to sample per batch item
            max_frames: If set, only use first N frames from this scene
            device: torch device for loading data
        """
        self.scene_path = scene_path
        self.num_views = num_views
        self.device = device
        
        # Load available frame indices
        depth_dir = os.path.join(scene_path, "depths")
        intrinsics_dir = os.path.join(scene_path, "intrinsics")
        extrinsics_dir = os.path.join(scene_path, "extrinsics")
        
        # Get frame IDs from depth files
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])
        self.frame_ids = [int(f.split(".")[0]) for f in depth_files]
        
        if max_frames is not None:
            self.frame_ids = self.frame_ids[:max_frames]
        
        if len(self.frame_ids) < num_views:
            raise ValueError(
                f"Scene has {len(self.frame_ids)} frames, but num_views={num_views} requested"
            )
        
        self.depth_dir = depth_dir
        self.intrinsics_dir = intrinsics_dir
        self.extrinsics_dir = extrinsics_dir
        
        # Cache for loaded data
        self._depth_cache = {}
        self._intrinsics_cache = {}
        self._extrinsics_cache = {}
    
    def __len__(self) -> int:
        """Number of possible view pairs in this scene."""
        # For each reference frame, we can pick multiple neighbor pairs
        return len(self.frame_ids)
    
    def _load_depth(self, frame_id: int) -> np.ndarray:
        """Load depth map for a frame."""
        if frame_id not in self._depth_cache:
            path = os.path.join(self.depth_dir, f"{frame_id:04d}.npy")
            self._depth_cache[frame_id] = np.load(path)
        return self._depth_cache[frame_id]
    
    def _load_intrinsics(self, frame_id: int) -> np.ndarray:
        """Load intrinsic matrix for a frame."""
        if frame_id not in self._intrinsics_cache:
            path = os.path.join(self.intrinsics_dir, f"{frame_id:04d}.npy")
            self._intrinsics_cache[frame_id] = np.load(path)
        return self._intrinsics_cache[frame_id]
    
    def _load_extrinsics(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load extrinsic (R, t) for a frame."""
        if frame_id not in self._extrinsics_cache:
            path = os.path.join(self.extrinsics_dir, f"{frame_id:04d}.npy")
            extr = np.load(path)
            R = extr[:3, :3]
            t = extr[:3, 3]
            self._extrinsics_cache[frame_id] = (R, t)
        return self._extrinsics_cache[frame_id]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a tuple of frames from the same scene.
        
        Returns:
            Dictionary with keys:
                - 'frame_ids': (num_views,) frame indices
                - 'depth': (num_views, H, W) depth maps
                - 'intrinsics': (num_views, 3, 3) intrinsic matrices
                - 'extrinsics': (num_views, 3, 4) extrinsic matrices [R|t]
        """
        # Sample num_views frames
        ref_frame = self.frame_ids[idx]
        
        # Pick reference + neighbors
        available = [f for f in self.frame_ids if f != ref_frame]
        if len(available) < self.num_views - 1:
            # Not enough neighbors, just duplicate
            neighbors = available * ((self.num_views - 1) // len(available) + 1)
            neighbors = neighbors[:self.num_views - 1]
        else:
            neighbors = random.sample(available, self.num_views - 1)
        
        frame_ids = [ref_frame] + neighbors
        
        # Load data for all frames
        depths = []
        intrinsics = []
        extrinsics = []
        
        for frame_id in frame_ids:
            depth = self._load_depth(frame_id)
            K = self._load_intrinsics(frame_id)
            R, t = self._load_extrinsics(frame_id)
            
            depths.append(torch.from_numpy(depth).float().unsqueeze(0))  # (1, H, W)
            intrinsics.append(torch.from_numpy(K).float())  # (3, 3)
            extrinsics.append(torch.from_numpy(np.hstack([R, t.reshape(3, 1)])).float())  # (3, 4)
        
        # Stack into batch
        depths = torch.stack(depths, dim=0)  # (num_views, 1, H, W)
        depths = depths.squeeze(1)  # (num_views, H, W)
        intrinsics = torch.stack(intrinsics, dim=0)  # (num_views, 3, 3)
        extrinsics = torch.stack(extrinsics, dim=0)  # (num_views, 3, 4)
        
        return {
            'frame_ids': torch.tensor(frame_ids),
            'depth': depths.to(self.device),
            'intrinsics': intrinsics.to(self.device),
            'extrinsics': extrinsics.to(self.device),
            'scene': os.path.basename(self.scene_path),
        }


class SameScenesDataLoader:
    """
    DataLoader that ensures all samples in a batch come from the same scene.
    
    Usage:
        loader = SameScenesDataLoader(
            root_dir='dtu_train_ready',
            batch_size=4,
            num_views=2,
            num_workers=0
        )
        for batch in loader:
            # batch['depth'].shape = (batch_size, num_views, H, W)
            # All frames in batch come from same scene
            pass
    """
    
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 4,
        num_views: int = 2,
        max_frames_per_scene: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0,
        device: torch.device = torch.device("cpu"),
        scans: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir: Path to dtu_train_ready directory
            batch_size: How many samples per batch
            num_views: Number of views per sample
            max_frames_per_scene: Limit frames per scene (None = use all)
            shuffle: Whether to shuffle scene order
            num_workers: DataLoader workers
            device: torch device
            scans: List of specific scan names to use (e.g., ['scan1', 'scan2'])
                  If None, auto-discover all scans
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_views = num_views
        self.num_workers = num_workers
        self.device = device
        
        # Discover scenes
        if scans is None:
            scene_dirs = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        else:
            scene_dirs = scans
        
        self.scene_datasets = {}
        for scene_name in scene_dirs:
            scene_path = os.path.join(root_dir, scene_name)
            try:
                dataset = DTUSceneDataset(
                    scene_path,
                    num_views=num_views,
                    max_frames=max_frames_per_scene,
                    device=device,
                )
                self.scene_datasets[scene_name] = dataset
            except Exception as e:
                print(f"[WARNING] Failed to load scene {scene_name}: {e}")
        
        self.scene_names = list(self.scene_datasets.keys())
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.scene_names)
        
        print(f"Loaded {len(self.scene_datasets)} scenes")
    
    def _create_batch_iterator(self):
        """Create iterator that yields batches from same scene."""
        scene_names = self.scene_names.copy()
        if self.shuffle:
            random.shuffle(scene_names)
        
        for scene_name in scene_names:
            dataset = self.scene_datasets[scene_name]
            
            # Create DataLoader for this scene
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
            
            for batch in loader:
                yield batch
    
    def __iter__(self):
        """Iterate over batches, each batch from a single scene."""
        return self._create_batch_iterator()
    
    def __len__(self) -> int:
        """Total number of batches across all scenes."""
        total_batches = 0
        for dataset in self.scene_datasets.values():
            total_batches += (len(dataset) + self.batch_size - 1) // self.batch_size
        return total_batches
    
    def get_single_scene_loader(self, scene_name: str, batch_size: Optional[int] = None) -> DataLoader:
        """
        Get a DataLoader for a single scene.
        
        Args:
            scene_name: Name of scene (e.g., 'scan1')
            batch_size: Override batch size (default: use self.batch_size)
        
        Returns:
            DataLoader for that scene
        """
        if scene_name not in self.scene_datasets:
            raise ValueError(f"Scene {scene_name} not found. Available: {self.scene_names}")
        
        batch_size = batch_size or self.batch_size
        return DataLoader(
            self.scene_datasets[scene_name],
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )


# Utility function for quick testing
def create_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_views: int = 2,
    val_split: float = 0.1,
    max_frames_per_scene: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[SameScenesDataLoader, SameScenesDataLoader]:
    """
    Create train/val dataloaders from dtu_train_ready directory.
    
    Args:
        data_root: Path to dtu_train_ready
        batch_size: Batch size
        num_views: Views per sample
        val_split: Fraction of scenes for validation (e.g., 0.1 = 10%)
        max_frames_per_scene: Limit frames per scene
        device: torch device
    
    Returns:
        (train_loader, val_loader)
    """
    # Get all scans
    all_scans = sorted([d for d in os.listdir(data_root) 
                       if os.path.isdir(os.path.join(data_root, d))])
    
    # Split into train/val
    num_val = max(1, int(len(all_scans) * val_split))
    val_scans = all_scans[:num_val]
    train_scans = all_scans[num_val:]
    
    train_loader = SameScenesDataLoader(
        data_root,
        batch_size=batch_size,
        num_views=num_views,
        max_frames_per_scene=max_frames_per_scene,
        shuffle=True,
        device=device,
        scans=train_scans,
    )
    
    val_loader = SameScenesDataLoader(
        data_root,
        batch_size=batch_size,
        num_views=num_views,
        max_frames_per_scene=max_frames_per_scene,
        shuffle=False,
        device=device,
        scans=val_scans,
    )
    
    return train_loader, val_loader
