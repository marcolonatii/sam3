#!/usr/bin/env python3
"""Debug script: inspect the prev-chain for LiDAR sweeps in nuScenes mini.

Usage:
    python debug_sweeps.py --data-root /path/to/v1.0-mini --scene-name scene-1094
"""

import argparse
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug LiDAR sweep prev-chain")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--scene-name", type=str, default="scene-1094")
    parser.add_argument("--max-keyframes", type=int, default=3)
    parser.add_argument("--nsweeps", type=int, default=10)
    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=args.data_root, verbose=False)

    scene = next(s for s in nusc.scene if s["name"] == args.scene_name)
    sample_token = scene["first_sample_token"]
    kf_idx = 0

    while sample_token and kf_idx < args.max_keyframes:
        sample = nusc.get("sample", sample_token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

        # Walk prev chain manually
        count = 0
        cur = lidar_sd
        while cur["prev"]:
            cur = nusc.get("sample_data", cur["prev"])
            count += 1
        print(f"\nKeyframe {kf_idx}: token={lidar_sd['token'][:16]}...")
        print(f"  is_key_frame: {lidar_sd['is_key_frame']}")
        print(f"  prev chain length: {count}")

        # Use the official devkit multi-sweep
        pc, times = LidarPointCloud.from_file_multisweep(
            nusc, sample, chan="LIDAR_TOP", ref_chan="LIDAR_TOP", nsweeps=args.nsweeps,
        )
        print(f"  from_file_multisweep(nsweeps={args.nsweeps}): {pc.nbr_points()} pts")
        if times.size > 0:
            unique_times = set(times.flatten().round(4))
            print(f"  unique time lags: {len(unique_times)} ({sorted(unique_times)[:5]}...)")

        # Compare to single sweep
        pc1, _ = LidarPointCloud.from_file_multisweep(
            nusc, sample, chan="LIDAR_TOP", ref_chan="LIDAR_TOP", nsweeps=1,
        )
        print(f"  from_file_multisweep(nsweeps=1):  {pc1.nbr_points()} pts")
        print(f"  ratio: {pc.nbr_points() / max(pc1.nbr_points(), 1):.1f}x")

        sample_token = sample.get("next", "")
        kf_idx += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
