#!/usr/bin/env python3
"""
Visualize depth maps from DTU dataset and export to image files.
Supports multiple visualization modes: viridis, turbo, hot, etc.
"""

import os
import numpy as np
import random
from pathlib import Path


def depth_to_rgb_viridis(depth_map):
    """Convert depth map to RGB using viridis-like colormap."""
    depth_clean = np.copy(depth_map)
    
    # Handle NaN/Inf values
    valid_mask = np.isfinite(depth_clean)
    if not np.any(valid_mask):
        return np.zeros((depth_clean.shape[0], depth_clean.shape[1], 3), dtype=np.uint8)
    
    depth_min = np.min(depth_clean[valid_mask])
    depth_max = np.max(depth_clean[valid_mask])
    
    if depth_max == depth_min:
        normalized = np.ones_like(depth_clean) * 0.5
    else:
        normalized = (depth_clean - depth_min) / (depth_max - depth_min)
    
    # Viridis colormap (purple -> blue -> green -> yellow)
    H, W = depth_clean.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for i in range(H):
        for j in range(W):
            val = normalized[i, j]
            
            # Viridis approximation
            if val < 0.25:
                # Purple to blue
                t = val / 0.25
                rgb[i, j, 0] = int((0.267 - 0.267*t) * 255)
                rgb[i, j, 1] = int(t * 255)
                rgb[i, j, 2] = int((0.329 + 0.47*t) * 255)
            elif val < 0.5:
                # Blue to green
                t = (val - 0.25) / 0.25
                rgb[i, j, 0] = int(0 * 255)
                rgb[i, j, 1] = int((t + 0.25) * 255)
                rgb[i, j, 2] = int((0.799 - 0.299*t) * 255)
            elif val < 0.75:
                # Green to yellow
                t = (val - 0.5) / 0.25
                rgb[i, j, 0] = int(t * 255)
                rgb[i, j, 1] = int(255)
                rgb[i, j, 2] = int((0.5 - 0.5*t) * 255)
            else:
                # Yellow to bright yellow
                t = (val - 0.75) / 0.25
                rgb[i, j, 0] = int((0.993 + 0.007*t) * 255)
                rgb[i, j, 1] = int((0.906 + 0.094*t) * 255)
                rgb[i, j, 2] = int((0.144 - 0.144*t) * 255)
    
    return rgb


def depth_to_rgb_turbo(depth_map):
    """Convert depth map to RGB using turbo-like colormap (red->blue->yellow)."""
    depth_clean = np.copy(depth_map)
    
    # Handle NaN/Inf values
    valid_mask = np.isfinite(depth_clean)
    if not np.any(valid_mask):
        return np.zeros((depth_clean.shape[0], depth_clean.shape[1], 3), dtype=np.uint8)
    
    depth_min = np.min(depth_clean[valid_mask])
    depth_max = np.max(depth_clean[valid_mask])
    
    if depth_max == depth_min:
        normalized = np.ones_like(depth_clean) * 0.5
    else:
        normalized = (depth_clean - depth_min) / (depth_max - depth_min)
    
    H, W = depth_clean.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for i in range(H):
        for j in range(W):
            val = normalized[i, j]
            
            # Turbo approximation (red -> orange -> yellow -> green -> blue -> magenta)
            if val < 0.25:
                t = val / 0.25
                rgb[i, j, 0] = int((1.0 - t * 0.5) * 255)
                rgb[i, j, 1] = int(t * 0.5 * 255)
                rgb[i, j, 2] = int(0)
            elif val < 0.5:
                t = (val - 0.25) / 0.25
                rgb[i, j, 0] = int(0.5 * 255)
                rgb[i, j, 1] = int((0.5 + t * 0.5) * 255)
                rgb[i, j, 2] = int(0)
            elif val < 0.75:
                t = (val - 0.5) / 0.25
                rgb[i, j, 0] = int((0.5 - t * 0.5) * 255)
                rgb[i, j, 1] = int(255)
                rgb[i, j, 2] = int(t * 255)
            else:
                t = (val - 0.75) / 0.25
                rgb[i, j, 0] = int(0)
                rgb[i, j, 1] = int((1.0 - t) * 255)
                rgb[i, j, 2] = int(255)
    
    return rgb


def depth_to_rgb_hot(depth_map):
    """Convert depth map to RGB using hot colormap (black->red->yellow->white)."""
    depth_clean = np.copy(depth_map)
    
    # Handle NaN/Inf values
    valid_mask = np.isfinite(depth_clean)
    if not np.any(valid_mask):
        return np.zeros((depth_clean.shape[0], depth_clean.shape[1], 3), dtype=np.uint8)
    
    depth_min = np.min(depth_clean[valid_mask])
    depth_max = np.max(depth_clean[valid_mask])
    
    if depth_max == depth_min:
        normalized = np.ones_like(depth_clean) * 0.5
    else:
        normalized = (depth_clean - depth_min) / (depth_max - depth_min)
    
    H, W = depth_clean.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for i in range(H):
        for j in range(W):
            val = normalized[i, j]
            
            if val < 1/3:
                t = val / (1/3)
                rgb[i, j, 0] = int(t * 255)
                rgb[i, j, 1] = int(0)
                rgb[i, j, 2] = int(0)
            elif val < 2/3:
                t = (val - 1/3) / (1/3)
                rgb[i, j, 0] = int(255)
                rgb[i, j, 1] = int(t * 255)
                rgb[i, j, 2] = int(0)
            else:
                t = (val - 2/3) / (1/3)
                rgb[i, j, 0] = int(255)
                rgb[i, j, 1] = int(255)
                rgb[i, j, 2] = int(t * 255)
    
    return rgb


def visualize_depth(depth_map, output_path, colormap='viridis'):
    """Convert depth map to colored image and save using PIL."""
    try:
        from PIL import Image
    except ImportError:
        print(f"[ERROR] PIL not installed. Cannot save {output_path}")
        print("Install with: pip install pillow")
        return False
    
    # Select colormap
    if colormap == 'viridis':
        rgb = depth_to_rgb_viridis(depth_map)
    elif colormap == 'turbo':
        rgb = depth_to_rgb_turbo(depth_map)
    elif colormap == 'hot':
        rgb = depth_to_rgb_hot(depth_map)
    else:
        print(f"[WARNING] Unknown colormap '{colormap}', using viridis")
        rgb = depth_to_rgb_viridis(depth_map)
    
    # Save image
    img = Image.fromarray(rgb)
    img.save(output_path)
    
    depth_valid = depth_map[np.isfinite(depth_map)]
    depth_min = np.min(depth_valid) if len(depth_valid) > 0 else 0
    depth_max = np.max(depth_valid) if len(depth_valid) > 0 else 0
    
    print(f"✓ Saved: {output_path}")
    print(f"  Depth range: [{depth_min:.2f}, {depth_max:.2f}]")
    print(f"  Image size: {rgb.shape[1]} × {rgb.shape[0]} (W × H)")
    
    return True


def load_random_sample():
    """Load a random depth map from the dataset."""
    dataset_root = './model/dataset/dtu_train_ready'
    
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Dataset not found: {dataset_root}")
        return None, None, None
    
    all_scenes = sorted([d for d in os.listdir(dataset_root) 
                    if os.path.isdir(os.path.join(dataset_root, d))])
    
    scenes_with_data = []
    for scene in all_scenes:
        depth_dir = os.path.join(dataset_root, scene, 'depths')
        if os.path.exists(depth_dir):
            files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
            if files:
                scenes_with_data.append(scene)
    
    if not scenes_with_data:
        print("[ERROR] No scenes with data found")
        return None, None, None
    
    print(f"Found {len(scenes_with_data)} scenes with data")
    
    scene_name = random.choice(scenes_with_data)
    scene_path = os.path.join(dataset_root, scene_name)
    
    depth_dir = os.path.join(scene_path, 'depths')
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    
    depth_file = random.choice(depth_files)
    depth_path = os.path.join(depth_dir, depth_file)
    frame_idx = int(depth_file.split('.')[0])
    
    depth = np.load(depth_path)
    
    metadata = {
        'scene': scene_name,
        'frame': frame_idx,
        'path': depth_path,
        'shape': depth.shape,
    }
    
    return depth, metadata


def visualize_sample(scene_name=None, frame_idx=None, colormap='viridis'):
    """
    Visualize a specific sample or random sample from DTU dataset.
    
    Args:
        scene_name: Scene to visualize (e.g., 'scan1'), None for random
        frame_idx: Frame index to visualize, None for random
        colormap: Colormap to use ('viridis', 'turbo', 'hot')
    
    Returns:
        Tuple of (png_path, npy_path) on success, (None, None) on failure
    """
    dataset_root = './model/dataset/dtu_train_ready'
    
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Dataset not found: {dataset_root}")
        return None, None
    
    # Load specific or random sample
    if scene_name is None:
        depth, metadata = load_random_sample()
        if depth is None:
            return None, None
    else:
        scene_path = os.path.join(dataset_root, scene_name)
        depth_dir = os.path.join(scene_path, 'depths')
        
        if not os.path.exists(depth_dir):
            print(f"[ERROR] Scene '{scene_name}' not found")
            return None, None
        
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
        if not depth_files:
            print(f"[ERROR] No depth files in {scene_name}")
            return None, None
        
        if frame_idx is None:
            depth_file = random.choice(depth_files)
            frame_idx = int(depth_file.split('.')[0])
        else:
            depth_file = f"{frame_idx}.npy"
            if depth_file not in depth_files:
                print(f"[ERROR] Frame {frame_idx} not found in {scene_name}")
                return None, None
        
        depth_path = os.path.join(depth_dir, depth_file)
        depth = np.load(depth_path)
        
        metadata = {
            'scene': scene_name,
            'frame': frame_idx,
            'path': depth_path,
            'shape': depth.shape,
        }
    
    # Generate output paths
    png_path = f"./depth_viz_{metadata['scene']}_frame{metadata['frame']:04d}_{colormap}.png"
    npy_path = f"./depth_raw_{metadata['scene']}_frame{metadata['frame']:04d}.npy"
    
    # Print info
    print(f"\n{'='*70}")
    print(f"Depth Visualization: {metadata['scene']} Frame {metadata['frame']:04d}")
    print(f"{'='*70}")
    print(f"Shape: {metadata['shape']}")
    print(f"Type: {depth.dtype}")
    print(f"Min: {np.min(depth):.4f}, Max: {np.max(depth):.4f}")
    print(f"Mean: {np.mean(depth):.4f}, Median: {np.median(depth):.4f}")
    print(f"Std: {np.std(depth):.4f}, NaN count: {np.isnan(depth).sum()}")
    
    # Visualize
    print(f"\nGenerating {colormap} visualization...")
    if not visualize_depth(depth, png_path, colormap=colormap):
        return None, None
    
    # Save raw depth
    np.save(npy_path, depth)
    print(f"  Raw data: {npy_path}")
    
    return png_path, npy_path


def main():
    """Main entry point - visualize random sample with all colormaps."""
    print("=" * 70)
    print("DTU Depth Map Visualizer")
    print("=" * 70)
    
    colormaps = ['viridis', 'turbo', 'hot']
    
    for colormap in colormaps:
        png_path, npy_path = visualize_sample(colormap=colormap)
        if png_path is None:
            print(f"[ERROR] Failed to visualize with {colormap} colormap")
            return False
    
    print(f"\n{'='*70}")
    print("✅ Visualization complete! Generated:")
    print(f"   - depth_viz_*.png (colored visualizations)")
    print(f"   - depth_raw_*.npy (raw depth data)")
    print(f"{'='*70}")
    
    return True


if __name__ == '__main__':
    import sys
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
