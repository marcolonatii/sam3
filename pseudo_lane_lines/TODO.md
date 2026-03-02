# 3D Lane Line Autolabeler — Task Tracker

Update status here as tasks are completed. Source of truth for current progress.

## What's Been Built (as of 2026-02-27)

### `lift_2d_to_3d.py` — 2D-to-3D Lane Lifting Pipeline

The core pipeline module. Takes SAM3 2D lane masks + LiDAR point clouds, projects LiDAR into the camera, filters points inside masks, and outputs 3D lane points in ego frame.

**Architecture:**
- Clean separation between **projection** (nuScenes-specific) and **filtering** (source-agnostic)
- `ProjectionResult` is the interface boundary — swap the projection backend for production without touching filtering code
- `FrameLiftResult` / `LanePoints3D` carry the output downstream

**Features implemented:**
1. **Multi-sweep LiDAR accumulation** (`--nsweeps`, default=10): Uses the nuScenes devkit's `LidarPointCloud.from_file_multisweep()` to accumulate preceding LiDAR sweeps for ~10x point density on thin lane markings. Note: first keyframe of a scene has 0 prev sweeps — use `--start-frame` to skip it.
2. **Priority-based mask overlay**: Fixed color blending bug where `np.maximum` across BGR channels blended overlapping mask colors to white. Now uses a `mask_type_map` so each pixel gets exactly one color.
3. **Front-frustum BEV visualization**: Shows only the forward half with ego at bottom-center. Auto-fits range to actual lane point extents. Correct nuScenes ego frame mapping (x=forward, y=left).
4. **2-row visualization layout**: Row 1 = [camera overlay | SAM3 masks], Row 2 = [BEV centered].
5. **OpenLane JSON export** (`--export-json`): Saves 3D lane points per frame in OpenLane-compatible format (`xyz` as (3, N), `category`, `visibility`, `track_id`).
6. **CLI flags**: `--scene-name`, `--start-frame`, `--nsweeps`, `--skip-sam3`, `--export-json`, `--visualize`, `--confidence-threshold`, `--min-depth`, `--max-depth`.
7. **Timestamped filenames**: Output images/JSONs include datetime to distinguish runs.

### `sam3_lane_inference.py` — SAM3 2D Segmentation

Wraps SAM3 (Segment Anything 3) for lane detection via text prompts. Produces per-prompt binary masks + confidence scores.

### `sam3_lane_eval.py` — Evaluation vs nuScenes HD Map

Projects nuScenes map `lane_divider` / `road_divider` polylines into camera images, compares with SAM3 masks via IoU / Precision / Recall / F1.

### `debug_sweeps.py` — LiDAR Sweep Diagnostic

Inspects the prev-chain for LiDAR sweeps in nuScenes mini. Confirmed: frame 0 has 0 prev sweeps, frame 1 has ~10, frame 2 has ~20.

### `tests/` — 38 Unit Tests (all passing)

- `TestLiftMasksTo3D` (10 tests): Core point-filtering logic
- `TestParseSam3Results` (6 tests): SAM3 dict parsing
- `TestLiftFromProjection` (4 tests): Integration of parse + lift
- `TestProjectionResult` (3 tests): Data structure sanity
- `TestMultiSweepProjection` (5 tests): Multi-sweep accumulation with mocked devkit
- `TestExportOpenLaneJSON` (10 tests): OpenLane JSON export

All heavy deps (nuScenes, SAM3, OpenCV, pyquaternion) are mocked in `conftest.py` — tests run in CI without GPU or data.

## Commands

```bash
cd sam3/pseudo_lane_lines

# Run tests
python -m pytest tests/ -v

# Full pipeline: SAM3 + multi-sweep + visualization + JSON export
python lift_2d_to_3d.py \
  --data-root /Users/lilyzhang/Desktop/Archive/Qwen2.5-VL/data_curate/data/v1.0-mini \
  --scene-name scene-1094 \
  --start-frame 5 \
  --max-samples 1 \
  --nsweeps 10 \
  --visualize \
  --export-json

# Projection-only (no SAM3, faster iteration)
python lift_2d_to_3d.py \
  --data-root /Users/lilyzhang/Desktop/Archive/Qwen2.5-VL/data_curate/data/v1.0-mini \
  --scene-name scene-0916 \
  --start-frame 5 \
  --max-samples 1 \
  --nsweeps 10 \
  --skip-sam3 \
  --visualize
```

## Pipeline Tasks

| # | Priority | Component | Status |
|---|----------|-----------|--------|
| 1 | P0 | SAM3 2D detection — run SAM3 inference on PV images to produce per-frame 2D lane masks | Done |
| 2 | P0 | SAM3 2D tracking — track lane mask instances across frames for consistent track IDs | In Progress |
| 3 | P0 | 2D-to-3D lifting (`lift_2d_to_3d.py`) — project 2D lane masks to 3D via LiDAR depth | Done |
| 4 | P0 | Multi-sweep LiDAR accumulation — 10-sweep accumulation via devkit | Done |
| 5 | P0 | OpenLane JSON export — save 3D lane points per frame | Done |
| 6 | P0 | Pose-aligned accumulation (`accumulate_observations.py`) — transform per-frame 3D points to world frame | Not Started |
| 7 | P0 | Multi-frame denoising (`denoise.py`) — outlier removal, voxel downsampling | Not Started |
| 8 | P0 | Geometry-based association (`associate_fragments.py`) — re-link track fragments via 3D proximity | Not Started |
| 9 | P0 | Spline fitting (`spline_fit.py`) — fit Catmull-Rom splines, resample to polyline at 0.5 m | Not Started |
| 10 | P0 | End-to-end eval script — 3D lanes vs GT: recall, precision, F-score, X/Z error | Not Started |
| 11 | P0 | CLI integration — extend `generate_pseudo_cuboids_cli.py` + AutoLabelerResult | Not Started |
| 12 | P1 | ALFA eval dataset curation — basic + challenging scene sets | Not Started |
| 13 | P1 | Deploy to dagster image — follow cuboid prelabeler pattern | Not Started |

## Known Issue: LiDAR Sparsity After Mask Filtering

**This is the biggest limitation of the current approach.**

The SAM3 segmentation is excellent — masks cleanly cover lane markings in 2D. But after filtering LiDAR points through these masks, the 3D output is very sparse:
- ~1,900 points across all lane types in a single frame (with 10 sweeps, ~34K total projected)
- ~5.7% hit rate — lane markings are thin (~15 cm) and flat, so few LiDAR beams land on them
- The BEV view shows scattered, fragmented points rather than continuous lines
- At long range (>25 m), LiDAR density drops further — almost no lane hits

Even with 10-sweep accumulation (which helped ~10x over single-sweep), the fundamental problem remains: **LiDAR angular resolution is too coarse for thin lane markings.**

### Potential Solutions (ranked by feasibility)

#### 1. Ground Plane Densification (Recommended Next Step)
**Idea:** Lane markings lie on the road surface. Fit a ground plane (or piecewise surface) from all LiDAR ground points, then for every mask pixel, ray-cast from the camera through that pixel onto the ground surface to get a 3D point.

**Why this is promising:**
- Uses the dense 2D mask information (thousands of pixels) instead of discarding it
- LiDAR provides the ground surface model (robust, no monocular depth ambiguity)
- Simple to implement: RANSAC plane fit on z < threshold, then camera ray intersection
- Can model road curvature with piecewise planes or polynomial surface
- Gives ~100x more 3D lane points than pure LiDAR filtering

**Implementation sketch:**
1. Segment ground points from LiDAR (z < ego_height + threshold, or use ground segmentation)
2. Fit ground surface: RANSAC plane for flat roads, or grid-based piecewise planes for hills
3. For each mask pixel → compute camera ray (inverse intrinsics + extrinsics) → intersect with ground surface → 3D lane point
4. Keep the sparse LiDAR-filtered points as "anchors" for validation

#### 2. Monocular Depth + LiDAR Scale Correction
**Idea:** Run a monocular depth network (Depth Anything V2, ZoeDepth) to get dense per-pixel depth, then use LiDAR points to correct scale/shift.

**Pros:** Dense depth for every pixel, handles non-flat terrain.
**Cons:** Adds another model dependency, monocular depth has systematic errors near ground plane, scale correction can be noisy.

#### 3. Multi-Keyframe Accumulation
**Idea:** Accumulate lane points across many keyframes (not just sweeps within one keyframe) as the vehicle drives, using ego-motion compensation.

**Pros:** As the vehicle moves, different parts of lane markings get hit by LiDAR. Over 10 keyframes (~5 seconds of driving), coverage improves significantly.
**Cons:** Requires ego-pose accuracy, depends on task #6 (pose-aligned accumulation) being built. This is already planned in the pipeline (tasks 6-7) but doesn't solve the per-frame sparsity.

#### 4. Inverse Perspective Mapping (IPM)
**Idea:** Project the 2D mask to BEV using a flat-ground assumption + camera calibration. No LiDAR needed for the mask pixels.

**Pros:** Very simple, no additional models.
**Cons:** Breaks on hills, doesn't give true 3D (assumes z=ground everywhere), distorts at far range.

### Recommendation

**Ground Plane Densification (#1)** is the best next step because:
- It leverages both the dense SAM3 masks AND the LiDAR ground surface
- It's geometrically principled (not learned, no extra model)
- It naturally handles the near-to-far density dropoff
- The sparse LiDAR-filtered points we already have serve as ground-truth anchors
- It can be added as a post-processing step after `lift_masks_to_3d()` without changing the existing pipeline

Then multi-keyframe accumulation (#3, already planned as tasks 6-7) will further densify when aggregating across the drive.

## Open Issues

- SAM3 classification accuracy: fine-grained prompts (e.g. "solid yellow" vs "double solid yellow") produce misclassifications. Consider SAM3 for segmentation only + separate classifier for lane type.
- HF Transformers API instability across versions; MPS/CPU fallback needed for local dev.
- Tracking resets at occlusions fragment lane IDs; downstream association (task #8) will re-link.
- First keyframe of each scene has 0 prev LiDAR sweeps — use `--start-frame >= 1` to skip.
- OpenLane JSON `track_id` is currently a placeholder index — needs real instance association.
- "dashed lane marking" / "solid lane marking" prompts map to generic category 21 — needs mapping refinement or a separate type classifier.
