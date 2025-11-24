import os
import glob
import numpy as np
from torch.utils.data import Dataset, Sampler

class MyDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.items = []

        scenes = sorted(os.listdir(folder_path))

        for scene_id in scenes:
            scene_path = os.path.join(folder_path, scene_id)
            if not os.path.isdir(scene_path):
                continue

            # Required paths
            img_dir      = os.path.join(scene_path, "images")
            depth_dir    = os.path.join(scene_path, "depths")
            intr_dir     = os.path.join(scene_path, "intrinsics")
            extr_dir     = os.path.join(scene_path, "extrinsics")

            # Optional paths (may not exist)
            est_intr_dir = os.path.join(scene_path, "est_intrinsics")
            est_extr_dir = os.path.join(scene_path, "est_extrinsics")

            # Normalize: if folder does not exist, set to None
            if not os.path.isdir(est_intr_dir):
                est_intr_dir = None

            if not os.path.isdir(est_extr_dir):
                est_extr_dir = None

            # Loop through views in this scene
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

            for img_path in img_paths:
                view_id = os.path.splitext(os.path.basename(img_path))[0]

                # Required base items
                item = {
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "image": img_path,
                    "depth": os.path.join(depth_dir, f"{view_id}.npy"),
                    "K_gt":  os.path.join(intr_dir, f"{view_id}.npy"),
                    "E_gt":  os.path.join(extr_dir, f"{view_id}.npy"),
                }

                # Optional: add est intrinsics/extrinsics only if they exist
                if est_intr_dir is not None:
                    path_K_sfm = os.path.join(est_intr_dir, f"{view_id}.npy")
                    if os.path.exists(path_K_sfm):
                        item["K_sfm"] = path_K_sfm

                if est_extr_dir is not None:
                    path_E_sfm = os.path.join(est_extr_dir, f"{view_id}.npy")
                    if os.path.exists(path_E_sfm):
                        item["E_sfm"] = path_E_sfm

                self.items.append(item)
            # During dataset initialization

        # Ensure all views sampled are from same group
        self.groups = {}       
        for item in self.items:
            sid = item["scene_id"]
            self.groups.setdefault(sid, []).append(item)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # You can load here or let collate_fn handle loading
        return item

class SceneBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.scenes = list(dataset.groups.keys())

    def __iter__(self):
        import random
        random.shuffle(self.scenes)

        for sid in self.scenes:
            views = self.dataset.groups[sid]
            # sample batch_size views inside scene
            idxs = random.sample(range(len(views)), self.batch_size)
            yield [views[i] for i in idxs]

    def __len__(self):
        return len(self.scenes)
