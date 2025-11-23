
# ============================================================================
# Loss Functions
# ============================================================================

def depth_loss(depth_pred, depth_gt, mask=None):
    """
    L1 depth loss with optional mask.
    
    Args:
        depth_pred: (B, 1, H, W) or (B, H, W)
        depth_gt: (B, 1, H, W) or (B, H, W)
        mask: (B, 1, H, W) or (B, H, W) or None
    """
    diff = (depth_pred - depth_gt).abs()
    
    if mask is None:
        return diff.mean()
    
    mask = mask.float()
    return (diff * mask).sum() / (mask.sum() + 1e-8)


def delta_z_loss(depth0, delta_z, depth_gt, mask=None):
    """
    Encourages depth0 + delta_z to be closer to depth_gt.
    Maximizes improvement from initial depth to corrected depth.
    
    Args:
        depth0: (B, 1, H, W) - initial depth
        delta_z: (B, 1, H, W) - depth correction
        depth_gt: (B, 1, H, W) - ground truth depth
        mask: (B, 1, H, W) or None
    """
    depth_corr = depth0 + delta_z
    
    # Detach depth0 to prevent gradients flowing into backbone if depth0 is frozen
    err_before = (depth0.detach() - depth_gt).abs()
    err_after = (depth_corr - depth_gt).abs()
    
    improvement = err_before - err_after  # should be > 0
    
    if mask is None:
        return (-improvement).mean()
    
    mask = mask.float()
    return ((-improvement) * mask).sum() / (mask.sum() + 1e-8)


def point_3d_loss(points_pred, points_gt, mask=None):
    """
    L2 loss between 3D points in camera space.
    x, y, z loss
    Args:
        points_pred: (B, 3, H, W) or (B, N, 3)
        points_gt: (B, 3, H, W) or (B, N, 3)
        mask: (B, 1, H, W) or (B, N) or None
    """
    diff = (points_pred - points_gt).norm(dim=1, keepdim=True)  # Euclidean distance
    
    if mask is None:
        return diff.mean()
    
    mask = mask.float()
    if mask.shape != diff.shape:
        mask = mask.expand_as(diff)
    
    return (diff * mask).sum() / (mask.sum() + 1e-8)


def multiview_consistency_loss(
    depth_i,
    depth_j,
    K_i, R_i, t_i,
    K_j, R_j, t_j,
    ray_dirs_i=None,
    mask=None
):
    """
    Multi-view consistency loss: backproject view i, transform to view j,
    project to j's image plane, and compare depths.
    
    Args:
        depth_i: (B, 1, H, W) - depth map of reference view i
        depth_j: (B, 1, H, W) - depth map of neighbor view j
        K_i, K_j: (B, 3, 3) - intrinsic matrices
        R_i, t_i: (B, 3, 3), (B, 3) - extrinsics for view i (world to camera)
        R_j, t_j: (B, 3, 3), (B, 3) - extrinsics for view j
        ray_dirs_i: (B, 3, H, W) - ray directions for view i (optional)
        mask: (B, 1, H, W) - valid pixel mask (optional)
    """
    B, _, H, W = depth_i.shape
    
    # 1. Backproject view i to 3D
    points_i = backproject_depth(depth_i, ray_dirs_i, K_i)  # (B, 3, H, W)
    
    # 2. Transform to world coordinates (if needed) then to camera j
    # Assuming R, t are world-to-camera: X_cam = R @ X_world + t
    # So X_world = R^T @ (X_cam - t)
    # Then X_j = R_j @ X_world + t_j
    
    # For simplicity, assume direct transformation from i to j:
    # If R_i, t_i are world→camera, then X_world = R_i^T @ X_i - R_i^T @ t_i
    # transform_points applies: p' = p @ R^T + t
    # So we need R = R_i^T and t = -R_i^T @ t_i
    R_i_inv = R_i.transpose(-2, -1)
    t_i_world = -torch.matmul(R_i_inv, t_i.unsqueeze(-1)).squeeze(-1)
    points_world = transform_points(
        points_i.reshape(B, 3, -1).transpose(1, 2),  # (B, N, 3)
        R_i_inv,
        t_i_world
    )
    
    # Then to camera j: X_j = R_j @ X_world + t_j
    points_j = transform_points(points_world, R_j, t_j)  # (B, N, 3)
    
    # 3. Project to view j pixel coordinates
    uv_j, valid_depth = project_points(points_j, K_j)  # (B, N, 2), (B, N)
    
    # 4. Sample depth from depth_j at projected locations
    # Normalize to [-1, 1] for grid_sample
    u_norm = (uv_j[..., 0] / (W - 1)) * 2 - 1
    v_norm = (uv_j[..., 1] / (H - 1)) * 2 - 1
    # Reshape grid to (B, H_out, W_out, 2) for better gradient stability
    N = u_norm.shape[1]
    grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B, N, 1, 2)
    
    # Check if coordinates are within bounds
    valid_proj = (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
    valid_mask_total = valid_depth & valid_proj
    
    if valid_mask_total.sum() == 0:
        return torch.tensor(0.0, device=depth_i.device, requires_grad=True)
    
    # Sample depth_j
    depth_j_sampled = F.grid_sample(
        depth_j,  # (B, 1, H, W)
        grid,  # (B, N, 1, 2)
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(-1).squeeze(1)  # (B, N)
    
    # 5. Compare depths
    depth_j_pred = points_j[..., 2]  # Z coordinate
    diff = (depth_j_sampled - depth_j_pred).abs()
    
    # Apply masks
    if mask is not None:
        mask_flat = mask.reshape(B, -1)
        valid_mask_total = valid_mask_total & (mask_flat > 0)
    
    valid_mask_total = valid_mask_total.float()
    loss = (diff * valid_mask_total).sum() / (valid_mask_total.sum() + 1e-8)
    
    return loss


def correction_magnitude_loss(delta_z, alpha=1.0):
    """
    L1 penalty on delta_z to encourage minimal corrections.
    
    Args:
        delta_z: (B, 1, H, W)
        alpha: weight for the penalty
    """
    return alpha * delta_z.abs().mean()


def smoothness_loss(delta_z, image, lambda_edge=10.0):
    """
    Edge-aware smoothness loss on Δz.
    Penalizes spatial gradients of Δz, but less so near image edges.
    
    Args:
        delta_z: (B, 1, H, W)
        image: (B, 3, H, W) - RGB image
        lambda_edge: strength of edge weighting
    """
    # Spatial gradients of delta_z
    dx_dz = delta_z[:, :, :, :-1] - delta_z[:, :, :, 1:]  # (B, 1, H, W-1)
    dy_dz = delta_z[:, :, :-1, :] - delta_z[:, :, 1:, :]  # (B, 1, H-1, W)
    
    # Spatial gradients of image (edge detection)
    dx_img = image[:, :, :, :-1] - image[:, :, :, 1:]  # (B, C, H, W-1)
    dy_img = image[:, :, :-1, :] - image[:, :, 1:, :]  # (B, C, H-1, W)
    
    # Edge weighting: lower weight on strong edges
    # Keep dimensions aligned: (B, 1, H, W-1) and (B, 1, H-1, W)
    weight_x = torch.exp(-lambda_edge * dx_img.abs().mean(dim=1, keepdim=True))  # (B, 1, H, W-1)
    weight_y = torch.exp(-lambda_edge * dy_img.abs().mean(dim=1, keepdim=True))  # (B, 1, H-1, W)
    
    loss_x = (dx_dz.abs() * weight_x).mean()
    loss_y = (dy_dz.abs() * weight_y).mean()
    
    return loss_x + loss_y


def edge_weighted_depth_loss(depth_pred, depth_ref, image, lambda_edge=10.0, mask=None):
    """
    Depth loss weighted by image edges.
    Higher weight in smooth regions, lower weight near edges.
    
    Args:
        depth_pred: (B, 1, H, W)
        depth_ref: (B, 1, H, W) - reference depth (GT or SfM)
        image: (B, 3, H, W)
        lambda_edge: edge weighting strength
        mask: (B, 1, H, W) or None
    """
    B, _, H, W = depth_pred.shape
    
    # Compute image gradients
    img_grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    img_grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    
    # Combined gradient magnitude (crop to same size)
    img_grad = torch.sqrt(
        img_grad_x[:, :, :-1, :].pow(2).mean(dim=1, keepdim=True) + 
        img_grad_y[:, :, :, :-1].pow(2).mean(dim=1, keepdim=True) + 
        1e-8
    )
    
    # Edge weight
    weight = torch.exp(-lambda_edge * img_grad)
    
    # Crop depth maps to match gradient size
    depth_pred_crop = depth_pred[:, :, :-1, :-1]
    depth_ref_crop = depth_ref[:, :, :-1, :-1]
    
    # Compute weighted loss
    diff = (depth_pred_crop - depth_ref_crop).abs()
    
    if mask is not None:
        mask_crop = mask[:, :, :-1, :-1].float()
        return (diff * weight * mask_crop).sum() / (mask_crop.sum() + 1e-8)
    
    return (diff * weight).mean()


# ============================================================================
# Combined Loss Function
# ============================================================================

def combined_deltaz_loss(
    depth0,
    delta_z,
    depth_gt,
    image,
    # Multi-view consistency
    depth_j=None,
    K_i=None, R_i=None, t_i=None,
    K_j=None, R_j=None, t_j=None,
    ray_dirs_i=None,
    # Masks
    mask=None,
    # Loss weights
    w_depth=1.0,
    w_delta=0.5,
    w_3d=1.0,
    w_mv=0.5,
    w_mag=0.01,
    w_smooth=0.1,
    w_edge=0.0
):
    """
    Combined loss for delta-z depth correction.
    
    Returns:
        total_loss: scalar
        loss_dict: dictionary of individual loss components
    """
    loss_dict = {}
    total_loss = 0.0
    
    depth_pred = depth0 + delta_z
    
    # 1. Depth supervision
    if depth_gt is not None and w_depth > 0:
        l_depth = depth_loss(depth_pred, depth_gt, mask)
        loss_dict['depth'] = l_depth.item()
        total_loss += w_depth * l_depth
    
    # 2. Delta-z improvement loss
    if depth_gt is not None and w_delta > 0:
        l_delta = delta_z_loss(depth0, delta_z, depth_gt, mask)
        loss_dict['delta'] = l_delta.item()
        total_loss += w_delta * l_delta
    
    # 3. 3D point loss
    if depth_gt is not None and w_3d > 0 and ray_dirs_i is not None:
        points_pred = backproject_depth(depth_pred, ray_dirs_i)
        points_gt = backproject_depth(depth_gt, ray_dirs_i)
        l_3d = point_3d_loss(points_pred, points_gt, mask)
        loss_dict['3d'] = l_3d.item()
        total_loss += w_3d * l_3d
    
    # 4. Multi-view consistency
    if depth_j is not None and w_mv > 0:
        l_mv = multiview_consistency_loss(
            depth_pred, depth_j,
            K_i, R_i, t_i,
            K_j, R_j, t_j,
            ray_dirs_i, mask
        )
        loss_dict['multiview'] = l_mv.item()
        total_loss += w_mv * l_mv
    
    # 5. Correction magnitude regularization
    if w_mag > 0:
        l_mag = correction_magnitude_loss(delta_z, alpha=1.0)
        loss_dict['magnitude'] = l_mag.item()
        total_loss += w_mag * l_mag
    
    # 6. Smoothness
    if w_smooth > 0:
        l_smooth = smoothness_loss(delta_z, image, lambda_edge=10.0)
        loss_dict['smoothness'] = l_smooth.item()
        total_loss += w_smooth * l_smooth
    
    # 7. Edge-weighted depth loss
    if depth_gt is not None and w_edge > 0:
        l_edge = edge_weighted_depth_loss(depth_pred, depth_gt, image, lambda_edge=10.0, mask=mask)
        loss_dict['edge_weighted'] = l_edge.item()
        total_loss += w_edge * l_edge
    
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict