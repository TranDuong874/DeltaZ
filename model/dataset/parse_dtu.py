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
# PROCESS ONE SCENE
###############################################
def convert_scene(mvsnet_scene_path, rectified_scene_path, out_root):
    scene_name = os.path.basename(mvsnet_scene_path.rstrip("/"))
    print(f"\n=== Processing {scene_name} ===")

    img_dir = os.path.join(mvsnet_scene_path, "images")
    cam_dir = os.path.join(mvsnet_scene_path, "cams")
    pair_path = os.path.join(mvsnet_scene_path, "pair.txt")

    depth_dir = os.path.join(rectified_scene_path, "depths")

    # check depth folder
    if not os.path.exists(depth_dir):
        print(f"[WARNING] No depth folder: {depth_dir}")
        return

    # output dirs
    out_scene = os.path.join(out_root, scene_name)
    os.makedirs(os.path.join(out_scene, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "depths"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "intrinsics"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "extrinsics"), exist_ok=True)

    # neighbors
    neighbors = load_pairs(pair_path)
    with open(os.path.join(out_scene, "neighbors.json"), "w") as f:
        json.dump(neighbors, f, indent=2)

    # image IDs (00000000.jpg → 0)
    img_ids = sorted([int(f.split(".")[0]) for f in os.listdir(img_dir)])

    for idx in img_ids:
        rgb_path = os.path.join(img_dir, f"{idx:08d}.jpg")
        cam_path = os.path.join(cam_dir, f"{idx:08d}_cam.txt")
        depth_path = os.path.join(depth_dir, f"depth_{idx:04d}.pfm")

        if not os.path.exists(depth_path):
            print(f"[MISSING DEPTH] {depth_path}")
            continue

        # RGB
        rgb = np.array(Image.open(rgb_path))
        imageio.imwrite(os.path.join(out_scene, "images", f"{idx:04d}.png"), rgb)

        # Depth
        depth = read_pfm(depth_path)
        np.save(os.path.join(out_scene, "depths", f"{idx:04d}.npy"), depth)

        # Camera
        R, t, K = load_cam_file(cam_path)
        np.save(os.path.join(out_scene, "intrinsics",  f"{idx:04d}.npy"), K)
        np.save(os.path.join(out_scene, "extrinsics", f"{idx:04d}.npy"), np.hstack([R, t.reshape(3,1)]))

        print(f"  OK frame {idx}")

    print(f"✔ Done: {scene_name}")


###############################################
# PROCESS ALL SCENES
###############################################
def convert_all(mvsnet_root, depths_root, out_root):
    """
    mvsnet_root:    DTU preprocessed root with structure:
                    ├── Cameras/
                    ├── Depths/
                    └── Rectified/
    depths_root:    Depths root containing scan1_train/, scan2_train/, etc.
    out_root:       Output GT dataset
    """

    # Find all scan_train folders in depths_root
    scan_dirs = sorted([d for d in os.listdir(depths_root) if d.endswith("_train")])
    
    print(f"Found {len(scan_dirs)} scan folders in {depths_root}")

    for scan_dir in scan_dirs:
        # Extract scan number (e.g., "scan1_train" → "scan1")
        scan_name = scan_dir.replace("_train", "")
        
        depth_path = os.path.join(depths_root, scan_dir)

        if not os.path.exists(depth_path):
            print(f"[WARNING] No depth folder: {depth_path}")
            continue

        convert_scene_new(mvsnet_root, depth_path, scan_name, out_root)


###############################################
# PROCESS ONE SCENE (UPDATED FOR NEW STRUCTURE)
###############################################
def convert_scene_new(mvsnet_root, depth_scene_path, scan_name, out_root):
    """
    mvsnet_root:        DTU root with Cameras/, Depths/, etc.
    depth_scene_path:   Path to scan1_train/, scan2_train/, etc. in Depths folder
    scan_name:          e.g., 'scan1', 'scan2'
    out_root:           Output directory
    """
    print(f"\n=== Processing {scan_name} ===")

    cam_dir = os.path.join(mvsnet_root, "Cameras")
    depth_dir = depth_scene_path

    # Check if required directories exist
    if not os.path.exists(cam_dir):
        print(f"[WARNING] No camera folder: {cam_dir}")
        return
    
    if not os.path.exists(depth_dir):
        print(f"[WARNING] No depth folder: {depth_dir}")
        return

    # Output dirs
    out_scene = os.path.join(out_root, scan_name)
    os.makedirs(os.path.join(out_scene, "depths"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "intrinsics"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "extrinsics"), exist_ok=True)

    # Load depth files to determine which cameras to process
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".pfm")])
    
    if not depth_files:
        print(f"[WARNING] No depth files found in {depth_dir}")
        return
    
    print(f"Found {len(depth_files)} depth files")

    # Process each depth file
    for depth_file in depth_files:
        # Extract frame index from depth filename (e.g., "depth_map_0000.pfm" or "depth_0000.pfm" → 0000)
        parts = depth_file.split("_")
        if "depth" in parts[0]:
            # Format: depth_map_0000.pfm or depth_0000.pfm
            frame_str = parts[-1].split(".")[0]
        else:
            frame_str = parts[1].split(".")[0]
        
        frame_idx = int(frame_str)
        
        cam_path = os.path.join(cam_dir, f"{frame_idx:08d}_cam.txt")
        depth_path = os.path.join(depth_dir, depth_file)

        if not os.path.exists(cam_path):
            print(f"[MISSING CAMERA] {cam_path}")
            continue

        # Camera file exists, process depth
        try:
            # Load Depth
            depth = read_pfm(depth_path)
            np.save(os.path.join(out_scene, "depths", f"{frame_idx:04d}.npy"), depth)

            # Camera
            R, t, K = load_cam_file(cam_path)
            np.save(os.path.join(out_scene, "intrinsics", f"{frame_idx:04d}.npy"), K)
            np.save(os.path.join(out_scene, "extrinsics", f"{frame_idx:04d}.npy"), np.hstack([R, t.reshape(3, 1)]))

            print(f"  ✓ frame {frame_idx}")
        except Exception as e:
            print(f"  ✗ Error processing frame {frame_idx}: {e}")
            continue

    print(f"✔ Done: {scan_name}")


###############################################
# MAIN
###############################################
if __name__ == "__main__":

    mvsnet_root = "/home/tranduong/DeltaZ/model/dataset/mvs_training/dtu"
    depths_root = os.path.join(mvsnet_root, "Depths")
    out_root = "./dtu_train_ready"
    
    convert_all(mvsnet_root, depths_root, out_root)
