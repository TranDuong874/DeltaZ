import os
import json
import numpy as np
import imageio
from PIL import Image

###############################################
# PFM LOADER
###############################################
def read_pfm(filename):
    with open(filename, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header not in ("PF", "Pf"):
            raise Exception("Not a PFM file.")

        dims = f.readline().decode("utf-8")
        while dims.startswith("#"):
            dims = f.readline().decode("utf-8")

        w, h = map(int, dims.split())
        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(f, endian + "f")
        if header == "PF":
            shape = (h, w, 3)
        else:
            shape = (h, w)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data.astype(np.float32)

###############################################
# CAMERA PARSER
###############################################
def load_cam_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    # extrinsic: lines 1-4
    extr = np.array([[float(x) for x in lines[i].split()] for i in range(1, 5)])[:3]
    R = extr[:, :3]
    t = extr[:, 3]

    # intrinsic: lines 7-9
    intr = np.array([[float(x) for x in lines[i].split()] for i in range(7, 10)])

    return R.astype(np.float32), t.astype(np.float32), intr.astype(np.float32)

###############################################
# PAIR.TXT PARSER
###############################################
def load_pairs(pair_path):
    with open(pair_path, "r") as f:
        lines = f.readlines()

    total = int(lines[0])
    neighbors = {}

    index = 1
    for _ in range(total):
        ref_id = int(lines[index].strip())
        parts = lines[index + 1].split()
        N = int(parts[0])

        pairs = []
        for i in range(N):
            view = int(parts[1 + 2*i])
            score = float(parts[2 + 2*i])
            pairs.append([view, score])

        neighbors[ref_id] = pairs
        index += 2
    
    return neighbors


###############################################
# PROCESS ALL SCENES
###############################################
def convert_all(mvsnet_root, out_root):
    """
    mvsnet_root:    DTU preprocessed root with structure:
                    ├── Cameras/
                    ├── Depths/
                    └── Rectified/
    out_root:       Output GT dataset
    """
    
    cameras_dir = os.path.join(mvsnet_root, "Cameras")
    depths_root = os.path.join(mvsnet_root, "Depths")
    rectified_root = os.path.join(mvsnet_root, "Rectified")

    # Verify all required directories exist
    if not os.path.exists(cameras_dir):
        print(f"[ERROR] Cameras folder not found: {cameras_dir}")
        return
    if not os.path.exists(depths_root):
        print(f"[ERROR] Depths folder not found: {depths_root}")
        return
    if not os.path.exists(rectified_root):
        print(f"[ERROR] Rectified folder not found: {rectified_root}")
        return

    # Find all scan_train folders in depths_root
    scan_dirs = sorted([d for d in os.listdir(depths_root) if d.endswith("_train")])
    
    print(f"Found {len(scan_dirs)} scan folders")

    for scan_dir in scan_dirs:
        # Extract scan number (e.g., "scan1_train" → "scan1")
        scan_name = scan_dir.replace("_train", "")
        
        depth_path = os.path.join(depths_root, scan_dir)
        rectified_path = os.path.join(rectified_root, scan_dir)

        if not os.path.exists(depth_path):
            print(f"[WARNING] No depth folder: {depth_path}")
            continue
        
        if not os.path.exists(rectified_path):
            print(f"[WARNING] No rectified folder: {rectified_path}")
            continue

        convert_scene(cameras_dir, depth_path, rectified_path, scan_name, out_root)


###############################################
# PROCESS ONE SCENE
###############################################
def convert_scene(cameras_dir, depth_scene_path, rectified_scene_path, scan_name, out_root):
    """
    cameras_dir:        Path to Cameras directory with *_cam.txt files
    depth_scene_path:   Path to scan1_train/, scan2_train/, etc. in Depths folder
    rectified_scene_path: Path to scan1_train/, scan2_train/, etc. in Rectified folder
    scan_name:          e.g., 'scan1', 'scan2'
    out_root:           Output directory
    """
    print(f"\n=== Processing {scan_name} ===")

    # Check if required directories exist
    if not os.path.exists(cameras_dir):
        print(f"[WARNING] No camera folder: {cameras_dir}")
        return
    
    if not os.path.exists(depth_scene_path):
        print(f"[WARNING] No depth folder: {depth_scene_path}")
        return
    
    if not os.path.exists(rectified_scene_path):
        print(f"[WARNING] No rectified folder: {rectified_scene_path}")
        return

    # Output dirs
    out_scene = os.path.join(out_root, scan_name)
    os.makedirs(os.path.join(out_scene, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "depths"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "intrinsics"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "extrinsics"), exist_ok=True)

    # Load depth files to determine which cameras to process
    depth_files = sorted([f for f in os.listdir(depth_scene_path) if f.startswith("depth_map_") and f.endswith(".pfm")])
    
    if not depth_files:
        print(f"[WARNING] No depth files found in {depth_scene_path}")
        return
    
    print(f"Found {len(depth_files)} depth files")

    # Process each depth file
    success_count = 0
    for depth_file in depth_files:
        # Extract frame index from depth filename (e.g., "depth_map_0000.pfm" → 0000)
        frame_str = depth_file.replace("depth_map_", "").replace(".pfm", "")
        frame_idx = int(frame_str)
        
        cam_path = os.path.join(cameras_dir, f"{frame_idx:08d}_cam.txt")
        depth_path = os.path.join(depth_scene_path, depth_file)
        
        # Try to find the corresponding RGB image in Rectified folder
        # Pattern: rect_XXX_Y_r5000.png where XXX is the camera number
        # Camera indices are 0-based (0-48) but rect files are 1-based (001-049)
        # So we need to add 1 to the frame_idx to match
        rgb_camera_id = frame_idx + 1
        rgb_found = False
        rgb_path = None
        
        rectified_files = [f for f in os.listdir(rectified_scene_path) if f.endswith(".png")]
        
        for rect_file in rectified_files:
            # Parse rect_001_0_r5000.png → camera_id=001, subset_id=0
            parts = rect_file.replace(".png", "").replace("_r5000", "").split("_")
            if len(parts) >= 2:
                camera_id = int(parts[1])
                if camera_id == rgb_camera_id:
                    rgb_path = os.path.join(rectified_scene_path, rect_file)
                    rgb_found = True
                    break

        if not os.path.exists(cam_path):
            print(f"  [SKIP] {frame_idx:04d} - missing camera file")
            continue
        
        if not rgb_found:
            print(f"  [SKIP] {frame_idx:04d} - missing RGB image")
            continue

        # All files exist, process
        try:
            # Load and save RGB
            rgb = np.array(Image.open(rgb_path))
            imageio.imwrite(os.path.join(out_scene, "rgb", f"{frame_idx:04d}.png"), rgb)

            # Load and save Depth
            depth = read_pfm(depth_path)
            np.save(os.path.join(out_scene, "depths", f"{frame_idx:04d}.npy"), depth)

            # Load and save Camera (intrinsics and extrinsics)
            R, t, K = load_cam_file(cam_path)
            np.save(os.path.join(out_scene, "intrinsics", f"{frame_idx:04d}.npy"), K)
            np.save(os.path.join(out_scene, "extrinsics", f"{frame_idx:04d}.npy"), np.hstack([R, t.reshape(3, 1)]))

            print(f"  OK {frame_idx:04d}")
            success_count += 1
        except Exception as e:
            print(f"  FAIL {frame_idx:04d} - Error: {str(e)}")
            continue

    print(f"Done: {scan_name} ({success_count}/{len(depth_files)} frames processed)")


###############################################
# MAIN
###############################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert DTU MVSNet data to DeltaZ format")
    parser.add_argument("--mvsnet_root", type=str, required=False,
                        default=os.path.join(os.path.dirname(__file__), "mvs_training", "dtu"),
                        help="Path to DTU MVSNet root directory")
    parser.add_argument("--out_root", type=str, required=False,
                        default=os.path.join(os.path.dirname(__file__), "..", "..", "dtu_train_ready"),
                        help="Output directory for converted dataset")
    args = parser.parse_args()
    mvsnet_root = os.path.abspath(args.mvsnet_root)
    out_root = os.path.abspath(args.out_root)
    
    print(f"Processing DTU data from: {mvsnet_root}")
    print(f"Output directory: {out_root}\n")
    
    convert_all(mvsnet_root, out_root)
