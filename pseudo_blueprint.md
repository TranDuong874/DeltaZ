Input data format:
training_set
    |- train_data
    |   |- scene_1
    |   |   |- images
    |   |   |- depths
    |   |   |- intrinsics
    |   |   |- extrinsics
    |   |   |- est_intrinsics
    |   |   |- est_extrinsics
    |   |- scene_2
    |   |   |- images
    |   |   |- depths
    |   |   |- intrinsics
    |   |   |- extrinsics
    |   |   |- sfm_intrinsics
    |   |   |- sfm_extrinsics
    |   ...
    |- test_data
    |   |- test_scene_1
    |   |   |- images
    |   |   |- depths
    |   |   |- intrinsics
    |   |   |- extrinsics

Pseudo pipeline

images:        [B, 3, H, W]
depth_gt:      [B, 1, H, W]
K_gt:          [B, 3, 3]
E_gt:          [B, 4, 4]   # world→cam (cam→world derived as inverse)

Note:
World → Camera
x_cam = R x_world + t
x_world = R^T (x_cam - t)

for scene in dataset:

    images = scene.load_images()

    # 1. Run SfM once per scene (VGG-SfM, DF-SfM, or COLMAP)
    VGGSfM(images, out_dir=cam_model)

    # 2. Load SfM output
    camera_params, vgg_images, points3D = read_cam_model(cam_model)

    # 3. Extract SfM intrinsics/extrinsics
    est_K, est_E = extract_params(camera_params, vgg_images, points3D)

    # 4. Store them for training (heavy to compute per-iteration)
    scene.save({
        "images": images,
        "depth_gt": scene.depths,
        "K_gt": scene.intrinsics,
        "E_gt": scene.extrinsics,
        "K_sfm": est_K,
        "E_sfm": est_E,
    })

NETWORK INPUT:
class DataLoader:
    def get_sample(self, scene_id, batch_size):

        # Sample batch of views from one scene
        images = scene_id.load_images(batch_size)
        depth_gt = scene_id.load_depths(batch_size)
        K_gt = scene_id.load_intrinsics(batch_size)
        E_gt = scene_id.load_extrinsics(batch_size)
        K_sfm = scene_id.load_est_intrinsics(batch_size)
        E_sfm = scene_id.load_est_extrinsics(batch_size)

        # Predict initial depth on the fly (monocular)
        depth_init = DepthAnythingV2(images)

        # Ground-truth Δz for supervision
        delta_z_gt = depth_gt - depth_init

        # 1. Ray directions per pixel
        ray_dirs = compute_ray_directions(K_gt, H, W)

        # 2. Sparse SfM → densify (KNN)
        sfm_sparse = project_points3D_to_depth(points3D, K_sfm, E_sfm)
        sfm_depth_dense, sfm_dist_map = densify_with_knn(sfm_sparse)

        # Final input tensor to Δz network
        input_tensor = concat([
            images,             # [B,3,H,W]
            depth_init,         # [B,1,H,W]
            ray_dirs,           # [B,3,H,W]
            sfm_depth_dense,    # [B,1,H,W]
            sfm_dist_map        # [B,1,H,W]
        ])

        // Values for loss calculation
        // 1. Direct delta z loss
        L_delta_z = L1(delta_z_pred - delta_z_gt)

        // 2. Depth loss
        depth_pred = depth_init + delta_z_pred
        L_depth = L1(depth_pred - depth_gt)


        // 3. Edge aware loss
        def gradient_x(t):
            return t[..., :, 1:] - t[..., :, :-1]

        def gradient_y(t):
            return t[..., 1:, :] - t[..., :-1, :]

        # depth_pred: [B,1,H,W]
        # image: [B,3,H,W]
        dx = gradient_x(depth_pred)
        dy = gradient_y(depth_pred)

        Ix = gradient_x(image)
        Iy = gradient_y(image)

        edge_x = torch.exp(-torch.mean(torch.abs(Ix), dim=1, keepdim=True))
        edge_y = torch.exp(-torch.mean(torch.abs(Iy), dim=1, keepdim=True))

        L_edge = (dx.abs() * edge_x).mean() + (dy.abs() * edge_y).mean()

        // 4. Surface smoothness
        # First derivatives
        dx = gradient_x(depth_pred)
        dy = gradient_y(depth_pred)

        # Second derivatives
        dxx = gradient_x(dx)
        dyy = gradient_y(dy)

        L_surface = dxx.abs().mean() + dyy.abs().mean()

        // 5. Correction magnitude normalization term
        eps = 1e-3
        L_dz_mag = (torch.abs(delta_z_pred) / (depth_pred + eps)).mean()

        // 6. Multiview consistency
        depth_pred_i = depth_init_i + delta_z_pred_i
        points_i = sample_pixels(depth_pred_i)

        X_i_world = unproject_to_world(points_i, depth_pred_i, K_gt_i, E_gt_i)

        for j in views_except(i):

            depth_pred_j = depth_init_j + delta_z_pred_j

            x_j = project_to_image(X_i_world, K_gt_j, E_gt_j)

            mask_valid = inside_image(x_j)        # mask 1
            x_j = x_j[mask_valid]
            X_i_world_valid = X_i_world[mask_valid]

            d_j = bilinear(depth_pred_j, x_j)

            mask_depth = d_j > 0                  # mask 2
            x_j = x_j[mask_depth]
            X_i_world_valid = X_i_world_valid[mask_depth]
            d_j = d_j[mask_depth]

            X_j_world = unproject_to_world(x_j, d_j, K_gt_j, E_gt_j)

            L_mv += torch.norm(X_i_world_valid - X_j_world, dim=-1).mean()



TRAINING STEP:
L = λz * L_delta_z
  + λd * L_depth
  + λe * L_edge
  + λs * L_surface
  + λm * L_dz_mag
  + λmv * L_mv
