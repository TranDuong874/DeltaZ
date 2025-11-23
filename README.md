# DeltaZ: Depth Refinement Model

A PyTorch-based depth map refinement system using Δz correction learning with multi-view consistency.

## ✅ Setup Complete

- ✅ DTU dataset processed (119 scans, 49 views each)
- ✅ Loss functions corrected (4 critical fixes applied)
- ✅ Same-scene dataloader implemented
- ✅ Training pipeline ready

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
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install tensorboard numpy pillow imageio
```

## Quick Start Training

```bash
python3 train.py \
    --data-root ./model/dataset/dtu_train_ready \
    --batch-size 4 \
    --num-views 2 \
    --num-epochs 50 \
    --lr 1e-4 \
    --device cuda
```

## Dataset

### DTU MVS (119 scans)

Located in: `model/dataset/dtu_train_ready/`

Structure per scan:
```
scan1/
├── depths/      (49 depth maps as .npy)
├── intrinsics/  (49 intrinsic matrices as .npy)
└── extrinsics/  (49 extrinsic matrices as .npy)
```

### Re-parse Dataset

If needed:
```bash
cd model/dataset
python3 parse_dtu.py
```

## Training Features

✅ **Same-Scene Batching**: All samples in a batch from same scene  
✅ **Multi-View Consistency**: Geometric constraints  
✅ **Edge-Aware Smoothness**: Preserves discontinuities  
✅ **Tensorboard Logging**: Real-time monitoring  
✅ **Checkpoint Management**: Save/resume training  

## Key Hyperparameters

```bash
--batch-size 4           # Per-scene batch size
--num-views 2            # Views per sample
--num-epochs 50          # Training epochs
--lr 1e-4                # Learning rate
--val-split 0.1          # 10% validation scenes
--checkpoint-dir ./checkpoints
--log-dir ./logs
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

### Loss Components

1. **Depth Loss** (L1): Direct depth supervision
2. **Delta-Z Loss**: Encourages correction improvement
3. **Magnitude Loss**: Regularizes correction size
4. **Smoothness Loss**: Edge-aware spatial smoothness
5. **Multi-View Loss**: Geometric reprojection consistency

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

### (A) Ray Direction Scaling
- ✓ Uses (x, y, 1) convention
- ✓ Correctly normalized

### (B) World Transform
- ✓ Fixed: `X_world = R_i^T @ X_i - R_i^T @ t_i`
- ✓ Correct camera-to-world conversion

### (C) Grid Sampling
- ✓ Improved shape for gradient stability
- ✓ Uses reshape() for better numerics

### (D) Smoothness Loss Dimension
- ✓ Aligned gradient shapes
- ✓ Proper broadcasting

### (E) Delta-Z Gradient Flow
- ✓ Detach depth0 for safe training
- ✓ Works with frozen or learnable depth0

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
**Status**: ✅ Ready for Training
