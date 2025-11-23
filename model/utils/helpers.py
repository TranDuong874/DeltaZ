import torch
import torch.nn.functional as F

import torch
import math
from typing import Optional, Tuple, Union

def get_ray_dirs_mask(
    H: int,
    W: int,
    K: Union[torch.Tensor, None] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    normalize_dirs: bool = False,
    homogeneous: bool = False,
    depth_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-pixel ray directions in camera space (and validity mask).

    Returns:
        dirs: (H, W, 3) or (H, W, 4) if homogeneous=True
              Each vector is the *direction* from camera origin through the pixel,
              expressed in camera coordinates. If normalize_dirs==False then z=1
              convention is used: dir = ((u-cx)/fx, (v-cy)/fy, 1).
              If normalize_dirs==True vectors are unit-length.
        mask: (H, W) uint8 tensor where 1 indicates valid pixel (and if `depth_mask` is provided it is combined)

    Args:
        H, W: output image height and width (pixels)
        K: if provided, either:
           - (3,3) intrinsics tensor for a single camera, or
           - (B,3,3) batched intrinsics (NOT used here except to extract fx,fy,cx,cy for the first batch)
           If K is None, default principal point = center, focal = 2.2*max(H,W) (empirical fallback).
        device, dtype: torch device/type for outputs
        normalize_dirs: whether to L2-normalize direction vectors (True -> unit rays)
        homogeneous: whether to append a 1 as 4th coordinate (useful for some ops)
        depth_mask: optional (H,W) or (1,H,W) or (B,H,W) boolean/uint8 tensor to combine with the returned mask

    Notes:
        - dirs are in camera coordinates (camera space). To get world rays, transform with cam->world.
        - This function uses (u,v) indexing where u is x (cols) and v is y (rows).
    """
    if device is None:
        device = torch.device("cpu")

    # default intrinsics fallback
    if K is None:
        focal = 2.2 * float(max(H, W))
        fx = fy = float(focal)
        cx = float(W) / 2.0
        cy = float(H) / 2.0
    else:
        if K.dim() == 3:
            K0 = K[0]  # batched: take first
        else:
            K0 = K
        fx = float(K0[0, 0].item())
        fy = float(K0[1, 1].item())
        cx = float(K0[0, 2].item())
        cy = float(K0[1, 2].item())

    # pixel coordinate grid (u = x, v = y)
    # note: we produce coordinates in image convention (cols, rows)
    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    u, v = torch.meshgrid(xs, ys, indexing='xy')  # u: W direction, v: H direction
    u = u.t()  # make shape (H, W): meshgrid indexing='xy' yields (W,H), so transpose
    v = v.t()

    # compute directions (z = 1)
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = torch.ones_like(x_cam)

    dirs = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)

    if normalize_dirs:
        norms = torch.norm(dirs, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        dirs = dirs / norms

    if homogeneous:
        ones = torch.ones((H, W, 1), device=device, dtype=dtype)
        dirs = torch.cat([dirs, ones], dim=-1)  # (H, W, 4)

    # base mask: all valid pixels initially
    mask = torch.ones((H, W), device=device, dtype=torch.uint8)

    # combine with optional depth_mask if provided
    if depth_mask is not None:
        dm = depth_mask
        # normalize shape to (H,W)
        if dm.dim() == 3 and dm.shape[0] == 1:  # (1,H,W)
            dm = dm[0]
        if dm.dim() == 3 and dm.shape[0] > 1 and dm.shape[0] != H:
            # Maybe (B,H,W) -> take first if B>1
            dm = dm[0]
        if dm.dim() == 2 and dm.shape != mask.shape:
            dm = torch.nn.functional.interpolate(dm.unsqueeze(0).unsqueeze(0).float(),
                                                 size=(H, W),
                                                 mode='nearest').squeeze()
            dm = (dm > 0.5).to(torch.uint8)
        dm = dm.to(device=device)
        # Ensure boolean-like
        dm_bool = (dm != 0)
        mask = mask & dm_bool.to(torch.uint8)

    return dirs, mask


def backproject_depth(depth, ray_dirs, K=None, u=None, v=None):
    """
    Backproject depth map to 3D points in camera coordinates.
    
    Args:
        depth: (B, 1, H, W) or (B, H, W)
        ray_dirs: (B, 3, H, W) - normalized ray directions OR None (use K)
        K: (B, 3, 3) - intrinsic matrix (if ray_dirs is None)
        u, v: (B, H, W) - pixel coordinates (if K is provided)
    
    Returns:
        points_3d: (B, 3, H, W) or (B, N, 3)
    """
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)  # (B, 1, H, W)
    
    B, _, H, W = depth.shape
    
    if ray_dirs is not None:
        # Use precomputed ray directions
        if ray_dirs.dim() == 3:
            ray_dirs = ray_dirs.unsqueeze(0)  # Add batch dim
        points_3d = depth * ray_dirs  # (B, 3, H, W)
    else:
        # Compute from intrinsics
        assert K is not None, "Either ray_dirs or K must be provided"
        if u is None or v is None:
            # Generate pixel grid
            v, u = torch.meshgrid(
                torch.arange(H, device=depth.device, dtype=depth.dtype),
                torch.arange(W, device=depth.device, dtype=depth.dtype),
                indexing='ij'
            )
            u = u.unsqueeze(0).expand(B, -1, -1)
            v = v.unsqueeze(0).expand(B, -1, -1)
        
        # Backproject using K^-1
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)
        
        x = (u - cx) * depth.squeeze(1) / fx
        y = (v - cy) * depth.squeeze(1) / fy
        z = depth.squeeze(1)
        
        points_3d = torch.stack([x, y, z], dim=1)  # (B, 3, H, W)
    
    return points_3d


def transform_points(points, R, t):
    """
    Transform 3D points using rotation and translation.
    
    Args:
        points: (B, 3, H, W) or (B, N, 3) or (N, 3)
        R: (3, 3) or (B, 3, 3) - rotation matrix
        t: (3,) or (B, 3) - translation vector
    
    Returns:
        transformed_points: same shape as input
    """
    original_shape = points.shape
    
    # Handle different input shapes
    if points.dim() == 2:  # (N, 3)
        points = points.unsqueeze(0)  # (1, N, 3)
        single_batch = True
    elif points.dim() == 4:  # (B, 3, H, W)
        B, _, H, W = points.shape
        points = points.reshape(B, 3, -1).transpose(1, 2)  # (B, N, 3)
        spatial = True
    else:
        single_batch = False
        spatial = False
    
    # Ensure R and t have batch dimension
    if R.dim() == 2:
        R = R.unsqueeze(0)  # (1, 3, 3)
    if t.dim() == 1:
        t = t.unsqueeze(0).unsqueeze(1)  # (1, 1, 3)
    elif t.dim() == 2:
        t = t.unsqueeze(1)  # (B, 1, 3)
    
    # Transform: R @ points^T + t
    points_transformed = torch.matmul(points, R.transpose(-2, -1)) + t
    
    # Restore original shape
    if single_batch:
        points_transformed = points_transformed.squeeze(0)
    if spatial:
        points_transformed = points_transformed.transpose(1, 2).reshape(original_shape)
    
    return points_transformed


def project_points(points, K):
    """
    Project 3D points to 2D pixel coordinates.
    
    Args:
        points: (B, 3, H, W) or (B, N, 3) or (N, 3)
        K: (3, 3) or (B, 3, 3) - intrinsic matrix
    
    Returns:
        uv: (B, N, 2) or (N, 2) - pixel coordinates
        valid_mask: (B, N) or (N,) - mask for points with positive depth
    """
    original_shape = points.shape
    
    # Reshape to (B, N, 3)
    if points.dim() == 2:  # (N, 3)
        points = points.unsqueeze(0)
        single_batch = True
    elif points.dim() == 4:  # (B, 3, H, W)
        B, _, H, W = points.shape
        points = points.reshape(B, 3, -1).transpose(1, 2)
        single_batch = False
    else:
        single_batch = False
    
    if K.dim() == 2:
        K = K.unsqueeze(0)
    
    # Valid points (positive depth)
    z = points[..., 2]
    valid_mask = z > 0
    
    # Project: K @ points^T
    uv_homogeneous = torch.matmul(K, points.transpose(-2, -1))  # (B, 3, N)
    uv_homogeneous = uv_homogeneous.transpose(-2, -1)  # (B, N, 3)
    
    # Normalize by depth
    uv = uv_homogeneous[..., :2] / (uv_homogeneous[..., 2:3] + 1e-8)
    
    if single_batch:
        uv = uv.squeeze(0)
        valid_mask = valid_mask.squeeze(0)
    
    return uv, valid_mask