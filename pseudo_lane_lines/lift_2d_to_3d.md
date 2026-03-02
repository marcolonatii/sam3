# Plan: Create `pseudo_lane_lines/lift_2d_to_3d.py` + Update Timelines

## Context

SAM3 2D detection is done (produces per-frame lane masks). The next pipeline step is **2D → 3D lifting**: project LiDAR 3D points into the camera image, keep only those falling inside SAM3 lane masks, producing 3D lane points. This is the P0 task at S2.4 in the development plan.

## Task 1: Create `pseudo_lane_lines/lift_2d_to_3d.py`

**Location:** `sam3/laneline/pseudo_lane_lines/lift_2d_to_3d.py`

Also create `sam3/laneline/pseudo_lane_lines/__init__.py` (empty, makes it a package).

### Key Design Decisions

- **Use `nuscenes-devkit`** (already installed, used in `sam3_lane_eval.py`) — provides calibration, ego_pose, LiDAR loading
- **Follow the exact nuScenes 4-step transform chain** from `map_pointcloud_to_image()`:
  1. LiDAR sensor → ego frame (lidar timestamp)
  2. Ego → global (lidar timestamp)
  3. Global → ego (camera timestamp)
  4. Ego → camera sensor frame
  5. Camera → pixels via intrinsics + `view_points()`
- **Output 3D points in ego frame** (at camera timestamp) — ready for downstream `accumulate_observations.py` which will transform ego→world
- **Integrate with SAM3 inference** — import `Sam3LaneInference` and run it inline, or accept pre-computed masks
- **CLI entry point** for standalone testing on nuScenes mini

### Data Structures

```python
@dataclass
class LanePoints3D:
    lane_type: str              # prompt string, e.g. "solid white lane line"
    points_ego: np.ndarray      # (N, 3) xyz in ego frame
    intensities: np.ndarray     # (N,) LiDAR intensity values
    pixel_coords: np.ndarray    # (N, 2) corresponding image pixel (u, v)
    num_points: int

@dataclass
class FrameLiftResult:
    sample_token: str
    camera: str
    lane_points: list[LanePoints3D]
    total_lidar_points: int
    total_lane_points: int
```

### Core Functions

1. **`project_lidar_to_image()`** — Load LiDAR .bin, apply 4-step transform, return pixel coords + depths + ego-frame points + validity mask
2. **`lift_masks_to_3d()`** — Given projected LiDAR points + SAM3 masks, filter to lane-only points, group by lane type
3. **`lift_frame()`** — End-to-end for one frame: load LiDAR → project → query masks → return `FrameLiftResult`
4. **`visualize_lift_result()`** — Camera image overlay (LiDAR points colored by lane type) + BEV scatter plot of 3D points
5. **`main()`** — CLI: parse args, load nuScenes, run SAM3 inference + lifting per frame, save visualizations

### Reuse from Existing Code

- Transform pattern from `sam3_lane_eval.py:81-95` (`_transform_global_to_sensor`) — same Quaternion approach, extended to full 4-step chain
- `view_points()` from `nuscenes.utils.geometry_utils` — already used in eval script
- `Sam3LaneInference` from `sam3_lane_inference.py` — for running SAM3 on each frame
- `TARGET_SCENES`, `LANE_PROMPTS` from `sam3_lane_inference.py` — scene configs
- nuScenes data at `/Users/lilyzhang/Desktop/Archive/Qwen2.5-VL/data_curate/data/v1.0-mini/`

### Edge Cases

- **Sparsity**: LiDAR has ~34K points per scan, but lane lines are thin — expect 10-50 LiDAR hits per lane per frame. Log counts for diagnostics.
- **Depth filtering**: min_depth=1.0m, max_depth=80.0m (beyond 80m LiDAR is too sparse)
- **Multi-mask merging**: A single LiDAR point may hit multiple overlapping SAM3 masks — assign to highest-confidence mask
- **Image bounds**: Filter projected points to valid pixel range [0, W) x [0, H)

## Task 2: Update Section 4.3 Timelines in `3D_Lanelines_Task.md`

**File:** `sam3/laneline/3D_Lanelines_Task.md`

Add a detailed status tracking table below the existing timelines table. For each component, document: **Status**, **Issues**, **Findings**, **Solutions**.

### Files Modified

1. `sam3/laneline/pseudo_lane_lines/__init__.py` — **NEW** (empty)
2. `sam3/laneline/pseudo_lane_lines/lift_2d_to_3d.py` — **NEW** (~250 lines)
3. `sam3/laneline/3D_Lanelines_Task.md` — **EDIT** section 4.3

### Verification

1. Confirm imports resolve: `from nuscenes.nuscenes import NuScenes`, `from nuscenes.utils.geometry_utils import view_points`
2. Run on nuScenes mini with `--data-root /Users/lilyzhang/Desktop/Archive/Qwen2.5-VL/data_curate/data/v1.0-mini/ --max-samples 1` to verify LiDAR loading + projection
3. Check visualization output shows LiDAR points overlaid on camera image with lane points highlighted
