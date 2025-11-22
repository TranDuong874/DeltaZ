import habitat_sim
import habitat_sim.utils.common as common
import numpy as np
import cv2
import os
import math
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
SCENE = "replica_v1/room0/habitat/mesh_semantic.ply" 
DATASET = "replica_v1/replica.scene_dataset_config.json"

OUTPUT = "output_room0"
NUM_VIEWS = 300       # number of frames to render
HEIGHT = 480
WIDTH = 640
FOV = 90.0

# Utility: save camera pose
def save_pose(filepath, T):
    with open(filepath, "w") as f:
        for row in range(4):
            f.write(" ".join(map(str, T[row])) + "\n")

# Build Habitat-Sim configuration
def make_cfg():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = DATASET
    backend_cfg.scene_id = SCENE

    sensor_spec = []
    
    # RGB sensor
    color_sensor = habitat_sim.CameraSensorSpec()
    color_sensor.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor.resolution = [HEIGHT, WIDTH]
    color_sensor.position = [0, 0, 0]
    color_sensor.hfov = FOV
    sensor_spec.append(color_sensor)

    # Depth sensor
    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [HEIGHT, WIDTH]
    depth_sensor.position = [0, 0, 0]
    depth_sensor.hfov = FOV
    sensor_spec.append(depth_sensor)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_spec

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])

# Main rendering loop
def main():

    cfg = make_cfg()
    sim = habitat_sim.Simulator(cfg)

    # Create output folders
    os.makedirs(f"{OUTPUT}/rgb", exist_ok=True)
    os.makedirs(f"{OUTPUT}/depth", exist_ok=True)
    os.makedirs(f"{OUTPUT}/pose", exist_ok=True)

    # Save intrinsics (simple pinhole model)
    fx = fy = (WIDTH/2) / math.tan((FOV/2) * math.pi/180)
    cx, cy = WIDTH/2, HEIGHT/2
    with open(f"{OUTPUT}/intrinsics.txt", "w") as f:
        f.write(f"{fx} {0} {cx}\n")
        f.write(f"{0} {fy} {cy}\n")
        f.write("0 0 1\n")

    # Generate a circular trajectory around the room
    radius = 2.0
    center = np.array([0.0, 1.2, 0.0])   # approximate center of scene
    height = 1.2

    for t in range(NUM_VIEWS):
        angle = 2 * math.pi * (t / NUM_VIEWS)
        cam_pos = center + np.array([radius * math.cos(angle), height, radius * math.sin(angle)])

        # Look-at target
        target = np.array([0.0, 1.0, 0.0])
        up = np.array([0, 1, 0])

        # Build camera-to-world transform
        forward = (target - cam_pos)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        # Construct 4x4 matrix
        T = np.eye(4)
        T[0, :3] = right
        T[1, :3] = cam_up
        T[2, :3] = -forward
        T[:3, 3] = cam_pos

        # Apply pose
        agent_state = habitat_sim.AgentState()
        agent_state.position = cam_pos
        agent_state.rotation = common.quat_from_matrix(T[:3, :3])
        sim.agents[0].set_state(agent_state)

        # Render
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        depth = obs["depth_sensor"]    # in meters

        # Save outputs
        cv2.imwrite(f"{OUTPUT}/rgb/{t:05d}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(f"{OUTPUT}/depth/{t:05d}.npy", depth)
        save_pose(f"{OUTPUT}/pose/{t:05d}.txt", T)

        print(f"[{t+1}/{NUM_VIEWS}] saved")

    print("Rendering Done.")
    sim.close()

if __name__ == "__main__":
    main()
