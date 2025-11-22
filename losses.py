# Supervised depth 
L_depth = smooth_l1(d_pred, d_gt)
L_dz = L1(d_pred - d0, dz_gt)  # equivalent

# 3D point loss
X_pred = backproject(d_pred, K_inv, R_i, c_i)  # world coords
X_gt = backproject(d_gt, K_inv, R_i, c_i)
L_3d = L1(X_pred, X_gt).mean()

# MV consistency loss
L_mag = lambda_mag * L1(d_pred - d0) # Correction magnitude prior
# Edge aware loss
w = exp(-alpha * |grad_I|)
L_smooth = sum(w * |grad(delta_z)|)

# Total loss
L = w_sup*(L_depth + L_3d) + w_mv*L_mv + w_mag*L_mag + w_smooth*L_smooth

# Starting loss
w_sup = 1.0
w_mv = 0.5
w_mag = 0.01
w_smooth = 0.1

# trainer.py pseudo
model = DeltaZUNet(in_ch=11, base_ch=32)
refiner = MultiViewRefiner(feat_ch=64, num_neighbors=4)  # optional
opt = torch.optim.Adam(list(model.parameters()) + list(refiner.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30,60], gamma=0.2)

for epoch in range(epochs):
    for batch in loader:
        rgb, d0, d_gt, K, Kinv, poses = batch  # shapes B,H,W etc
        F = build_input_features(rgb, d0, Kinv, ... )  # returns [B,11,H,W]
        delta = model(F)  # B,1,H,W
        d_pred = d0 + delta
        # optional multi-view: warp neighbor features and call refiner:
        # ref_delta = refiner(ref_feat, warped_feats)
        # d_pred = d_pred + ref_delta

        # compute losses:
        L_sup = L1(d_pred, d_gt)
        L_3d = compute_3d_point_loss(...)
        L_mv = compute_multiview_loss(...)
        L_mag = torch.abs(delta).mean()
        L_smooth = compute_edge_aware_smoothness(delta, rgb)
        loss = w_sup*(L_sup + L_3d) + w_mv*L_mv + w_mag*L_mag + w_smooth*L_smooth

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
