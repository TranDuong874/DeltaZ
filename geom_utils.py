import torch
import torch.nn.functional as F

# Pseudo code
# backproject: depth (B,1,H,W) + intrinsics -> points in camera coords (B,3,H,W)
def backproject_depth_to_camcoords(depth, K_inv):
    # depth: B,1,H,W
    B,_,H,W = depth.shape
    device = depth.device
    # create pixel grid
    u = torch.linspace(0, W-1, W, device=device)
    v = torch.linspace(0, H-1, H, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
    ones = torch.ones_like(grid_u)
    pix = torch.stack([grid_u, grid_v, ones], dim=0).view(3, -1)  # 3, H*W
    pix = pix.unsqueeze(0).repeat(B,1,1)  # B,3,N
    Kinv = K_inv.view(B,3,3)
    dirs = torch.bmm(Kinv, pix)           # B,3,N
    dirs = dirs.view(B,3,H,W)
    cam_pts = dirs * depth
    return cam_pts  # B,3,H,W

# project camera coords into pixel coordinates of another camera
def project_camcoords_to_pixels(cam_pts, K, R, t):
    # cam_pts: in world; if cam_pts is camera coords you must convert accordingly
    # For our pipeline: supply world pts
    B,_,H,W = cam_pts.shape
    cam_pts_flat = cam_pts.view(B,3,-1)  # B,3,N
    # world->cam_j: x_cam = R_j @ X_world + t_j
    x_cam = torch.bmm(R, cam_pts_flat) + t.unsqueeze(-1)  # R: B,3,3, t: B,3
    uvw = torch.bmm(K, x_cam)  # B,3,N
    u = uvw[:,0,:] / (uvw[:,2,:] + 1e-8)
    v = uvw[:,1,:] / (uvw[:,2,:] + 1e-8)
    u = u.view(B,1,H,W)
    v = v.view(B,1,H,W)
    z = uvw[:,2,:].view(B,1,H,W)
    return u, v, z  # in pixel coords
