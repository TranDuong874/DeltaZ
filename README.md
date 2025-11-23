# DeltaZ: Depth Refinement Model

A PyTorch-based depth map refinement system using Δz correction learning with multi-view consistency.

## ✅ Latest Status (Nov 23, 2025)

- ✅ DTU dataset fully processed (108+ scans, 49 views each, with RGB images)
- ✅ Data pipeline verified (loads batches correctly with 2+ views)
- ✅ All loss functions implemented and working (5 losses: depth, delta, mag, smooth, multi-view)
- ✅ Multi-view consistency loss fixed and tested
- ✅ Single batch test successful: 2 samples, 2 views, full loss computation
- ✅ Training pipeline ready for full training runs

## Project Structure

```
DeltaZ/
├── model/
│   ├── __init__.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   └── DeltaZUnet.py          # U-Net model for Δz prediction
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── parse_dtu.py           # DTU dataset converter
│   │   ├── dtu_dataset.py         # PyTorch DataLoaders
│   │   └── dtu_train_ready/       # Processed dataset (119 scans)
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py             # Camera ops, backprojection, etc.
│       └── losses.py              # All training loss functions
├── train.py                        # Main training script
├── test_setup.py                   # Quick verification tests
└── README.md
```

## Installation

```bash
# Create virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support (or CPU)
pip install torch torchvision torchaudio

# Install dependencies
pip install tensorboard numpy pillow imageio
```

## Quick Start: Testing

```bash
# Test with 1 batch of 2 samples (2 views each)
python train.py --test-mode --batch-size 2

# Output:
# Batch 1:
#   Depth shape: torch.Size([2, 2, 128, 160])
#   Losses: {loss_depth, loss_delta, loss_mag, loss_smooth, loss_mv, total}
#   All losses computed successfully!
```

## Quick Start: Full Training

```bash
python train.py \
    --batch-size 4 \
    --num-views 2 \
    --num-epochs 50 \
    --lr 1e-4
```

## Dataset

### DTU MVS (108+ scans, 49 views each)

**Location**: `dtu_train_ready/` (generated from `mvs_training/dtu/`)

**Structure** per scan:
```
scan1/
├── rgb/         (49 RGB images as .png)
├── depths/      (49 depth maps as .npy)
├── intrinsics/  (49 intrinsic matrices as .npy)
└── extrinsics/  (49 extrinsic matrices [R|t] as .npy)
```

### Parsing the Dataset

The dataset is auto-parsed on first load. To re-parse from raw MVSNet data:

```bash
cd model/dataset
python parse_dtu.py --mvsnet_root ./mvs_training/dtu --out_root ../../dtu_train_ready
```

**What parse_dtu.py does**:
- Loads RGB images from `Rectified/` directory
- Loads depth maps from `Depths/` directory  
- Loads camera calibration from `Cameras/` directory
- Handles 0-based camera indexing vs 1-based RGB indexing
- Outputs structured training data with rgb/, depths/, intrinsics/, extrinsics/ folders

## Training Features

✅ **Same-Scene Batching**: All samples in batch from same scene  
✅ **Multi-View Consistency**: Geometric reprojection constraints (working)
✅ **Edge-Aware Smoothness**: Preserves discontinuities  
✅ **Tensorboard Logging**: Real-time monitoring  
✅ **Checkpoint Management**: Save/resume training  
✅ **Test Mode**: Validate setup with `--test-mode` flag  
✅ **GPU/CPU Support**: Automatic device detection

## Key Hyperparameters

```bash
--batch-size 4           # Samples per batch (same scene)
--num-views 2            # Views per sample
--num-epochs 50          # Training epochs
--lr 1e-4                # Adam learning rate
--val-split 0.1          # 10% validation scenes
--test-mode              # Run 1 batch test
--checkpoint-dir ./checkpoints
--log-dir ./logs
--device cuda            # or 'cpu'
```

## DataLoader Details

### SameScenesDataLoader

Ensures all samples in a batch come from the same scene:

```python
from model.dataset.dtu_dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    data_root='./model/dataset/dtu_train_ready',
    batch_size=4,
    num_views=2,
    val_split=0.1,
    device=torch.device('cuda')
)

for batch in train_loader:
    # All samples from same scene
    depth = batch['depth']           # (B, V, H, W)
    intrinsics = batch['intrinsics'] # (B, V, 3, 3)
    extrinsics = batch['extrinsics'] # (B, V, 3, 4)
    scene = batch['scene']           # All same value
```

## Model Architecture

**DeltaZUnet**: 4-layer U-Net encoder-decoder

```python
from model.architecture.DeltaZUnet import DeltaZUnet

model = DeltaZUnet(
    in_channels=1,      # Initial depth
    base_channel=32,
    depth=4,            # Encoder/decoder depth
    out_confidence=False
)
```

**Input**: Initial depth (1 channel)  
**Output**: Correction map Δz (1 channel)  
**Refined**: depth_final = depth_initial + delta_z

## Training Pipeline

### Loss Components (5 Total)

1. **Depth Loss** (L1): Direct depth supervision vs ground truth
2. **Delta-Z Loss**: Encourages depth correction to improve predictions
3. **Magnitude Loss**: Regularizes correction size (0.01 weight)
4. **Smoothness Loss**: Edge-aware spatial smoothness (0.1 weight)
5. **Multi-View Loss**: Geometric reprojection consistency (0.01 weight)

**Tested Configuration**:
```python
Losses: {
    'loss_depth': 0.0405,
    'loss_delta': 0.0405,
    'loss_mag': 0.0004,
    'loss_smooth': 0.0,
    'loss_mv': 7.49,      # Multi-view reprojection
    'total': 7.55
}
```

### Optimizer

- **Adam**: β₁=0.9, β₂=0.999, weight_decay=1e-6
- **Scheduler**: Cosine Annealing with Warm Restarts (T₀=10)
- **Grad Clipping**: max_norm=1.0

## Monitoring Training

```bash
tensorboard --logdir ./logs --port 6006
```

Open: http://localhost:6006

## Helper Functions

### Camera Operations
```python
from model.utils.helpers import (
    get_ray_dirs_mask,      # Ray direction generation
    backproject_depth,      # Depth → 3D points
    transform_points,       # Coordinate transformation
    project_points,         # 3D points → 2D
)
```

## Critical Fixes Applied ✅

### (1) DTU Dataset Parsing
- ✓ Correctly loads RGB from `Rectified/` directory
- ✓ Handles 1-based RGB indexing vs 0-based camera indexing
- ✓ Creates `rgb/` output folder (not just `images/`)

### (2) Ray Direction Shapes
- ✓ Handles `get_ray_dirs_mask()` returning (W, H, 3)
- ✓ Corrects to (H, W, 3) for proper broadcasting
- ✓ Normalizes before backprojection

### (3) Multi-View Loss Computation
- ✓ Fixed tensor shape handling for batched operations
- ✓ Proper coordinate transforms: camera → world → target view
- ✓ Batched matrix multiplication for efficiency
- ✓ Grid sampling with correct normalization

### (4) Loss Integration
- ✓ All 5 losses computed without errors
- ✓ Proper weight scaling for multi-view loss
- ✓ Backward pass and gradient clipping working

## Common Issues

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `--batch-size` to 2 |
| Slow Training | Increase batch size or check GPU util |
| NaN Loss | Check learning rate, data validity |
| Low Accuracy | More epochs, lower learning rate |

## Extensions

### Use Initial Depth (MDE/SfM)

```python
# In train_step:
initial_depth = mde_model(image)
model_input = initial_depth.unsqueeze(1)
delta_z = model(model_input)
```

### Multi-Scale Training

```python
import torch.nn.functional as F

pyramid = [
    F.interpolate(depth, scale_factor=1/2**i)
    for i in range(3)
]
```

### Confidence Estimation

```python
model = DeltaZUnet(..., out_confidence=True)
delta_z, confidence = model(input_depth)
```

---

**Last Updated**: November 23, 2025  
**Status**: ✅ Full Training Ready  
**Test Result**: All 5 losses computed successfully with batch of 2 samples
