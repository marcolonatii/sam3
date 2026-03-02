#!/usr/bin/env python3
"""
2D → 3D Lifting: Project LiDAR points into camera image, keep those inside
SAM3 lane masks to produce 3D lane points in ego frame.

Pipeline step S2.4 — sits between SAM3 2D detection and accumulate_observations.

Architecture
------------
The module cleanly separates two concerns:

  1. **Projection** (source-specific): "How do I get LiDAR 3D points into pixel
     space?"  Currently implemented via nuScenes devkit for local prototyping.
     Returns a ``ProjectionResult`` — the only type the filtering step sees.

  2. **Filtering** (source-agnostic): "Given projected points + SAM3 masks,
     which 3D points are lane lines?"  ``lift_masks_to_3d()`` and
     ``lift_from_projection()`` operate only on numpy arrays and
     ``ProjectionResult`` — no dataset dependency.

When integrating into the production AV repo, replace only the projection
function (swap ``project_lidar_to_image_nuscenes`` with the existing production
LiDAR overlay) and feed its output into ``lift_from_projection()``.

Usage (local, nuScenes mini):
    python lift_2d_to_3d.py \\
        --data-root /path/to/nuscenes/v1.0-mini \\
        --max-samples 2 \\
        --visualize
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

# Sibling imports — works when run from this directory or as a package
from sam3_lane_inference import LANE_PROMPTS, TARGET_SCENES, Sam3LaneInference

# ── Constants ────────────────────────────────────────────────────────────────
MIN_DEPTH_M = 1.0
MAX_DEPTH_M = 80.0

# Colors for visualization (BGR for OpenCV)
LANE_COLORS_BGR = [
    (255, 255, 0),   # cyan
    (0, 255, 0),     # green
    (0, 165, 255),   # orange
    (0, 0, 255),     # red
    (255, 0, 255),   # magenta
]
NON_LANE_COLOR_BGR = (128, 128, 128)  # gray for non-lane LiDAR points


# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class ProjectionResult:
    """Source-agnostic output of the LiDAR → image projection step.

    This is the **interface boundary** between projection and filtering.
    Any projection backend (nuScenes devkit, production LiDAR overlay, etc.)
    must produce this struct.  The filtering step (``lift_masks_to_3d``,
    ``lift_from_projection``) consumes only this — no dataset handles.
    """

    pixels: np.ndarray       # (M, 2) int — pixel coordinates (u, v)
    depths: np.ndarray       # (M,) float — depth in camera z
    points_ego: np.ndarray   # (M, 3) float — 3D in ego frame at camera timestamp
    intensities: np.ndarray  # (M,) float — LiDAR intensity per point
    img_w: int
    img_h: int


@dataclass
class LanePoints3D:
    """3D lane points from a single lane type in a single frame."""

    lane_type: str  # prompt string, e.g. "white lane line"
    points_ego: np.ndarray  # (N, 3) xyz in ego frame at camera timestamp
    intensities: np.ndarray  # (N,) LiDAR intensity values
    pixel_coords: np.ndarray  # (N, 2) corresponding image pixel (u, v)
    num_points: int


@dataclass
class FrameLiftResult:
    """Result of 2D→3D lifting for one camera frame."""

    sample_token: str
    camera: str
    lane_points: list[LanePoints3D]
    total_lidar_points: int
    total_lane_points: int


# ── Step 1: Projection (source-specific) ─────────────────────────────────────
#
# This section is the ONLY part that touches nuScenes.  When integrating into
# the production AV repo, replace ``project_lidar_to_image_nuscenes`` with a
# wrapper around the existing production LiDAR overlay that returns a
# ``ProjectionResult``.
#


def project_lidar_to_image_nuscenes(
    nusc: NuScenes,
    sample_token: str,
    camera: str = "CAM_FRONT",
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
    nsweeps: int = 1,
) -> ProjectionResult:
    """nuScenes-specific projection using the devkit's multi-sweep accumulation.

    Uses ``LidarPointCloud.from_file_multisweep()`` to accumulate *nsweeps*
    LiDAR sweeps into the LIDAR_TOP reference frame, then transforms into the
    camera frame for projection.

    When *nsweeps* > 1, preceding sweeps are included to increase point density
    on thin lane markings (~10x with nsweeps=10).

    This is the **local prototyping** implementation.  For production, swap
    this function with one that calls the existing LiDAR overlay code and
    returns the same ``ProjectionResult``.
    """
    if nsweeps < 1:
        raise ValueError(f"nsweeps must be >= 1, got {nsweeps}")

    sample = nusc.get("sample", sample_token)

    # Camera data + calibration
    cam_sd_token = sample["data"][camera]
    cam_sd = nusc.get("sample_data", cam_sd_token)
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_ego = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    intrinsic = np.array(cam_cs["camera_intrinsic"])
    img_w, img_h = cam_sd["width"], cam_sd["height"]

    # ── Accumulate sweeps using devkit ───────────────────────────────────
    # from_file_multisweep returns points in the LIDAR_TOP sensor frame
    # (of the keyframe), with all sweeps ego-motion-compensated.
    pc, times = LidarPointCloud.from_file_multisweep(
        nusc, sample, chan="LIDAR_TOP", ref_chan="LIDAR_TOP", nsweeps=nsweeps,
    )
    # pc.points is (4, N) — x, y, z, intensity in LIDAR_TOP frame
    points_lidar = pc.points[:3].T   # (N, 3)
    intensities_all = pc.points[3]   # (N,)

    if nsweeps > 1:
        n_sweeps_actual = len(set(times.flatten().round(4)))
        print(
            f"    [multi-sweep] requested={nsweeps}, "
            f"loaded={n_sweeps_actual} sweeps, "
            f"{len(intensities_all)} raw pts"
        )

    # ── Transform: LIDAR_TOP → ego (lidar time) → global → ego (cam time) → camera ──
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    # Step 1: LIDAR_TOP sensor → ego (lidar timestamp)
    rot_lidar = Quaternion(lidar_cs["rotation"]).rotation_matrix
    trans_lidar = np.array(lidar_cs["translation"])
    points_ego_lidar = (rot_lidar @ points_lidar.T).T + trans_lidar

    # Step 2: Ego (lidar time) → global
    rot_ego_lidar = Quaternion(lidar_ego["rotation"]).rotation_matrix
    trans_ego_lidar = np.array(lidar_ego["translation"])
    points_global = (rot_ego_lidar @ points_ego_lidar.T).T + trans_ego_lidar

    # Step 3: Global → ego (camera timestamp)
    rot_ego_cam_inv = Quaternion(cam_ego["rotation"]).inverse.rotation_matrix
    trans_ego_cam = np.array(cam_ego["translation"])
    points_ego_cam = (rot_ego_cam_inv @ (points_global - trans_ego_cam).T).T

    # Step 4: Ego → camera sensor
    rot_cam_inv = Quaternion(cam_cs["rotation"]).inverse.rotation_matrix
    trans_cam = np.array(cam_cs["translation"])
    points_cam = (rot_cam_inv @ (points_ego_cam - trans_cam).T).T

    # Project to image using intrinsics
    depths = points_cam[:, 2]
    points_img = view_points(points_cam.T, intrinsic, normalize=True)  # (3, N)
    pixels_all = points_img[:2].T  # (N, 2)

    # Validity mask: in front of camera, within depth range, within image bounds
    valid = (
        (depths > min_depth)
        & (depths < max_depth)
        & (pixels_all[:, 0] >= 0)
        & (pixels_all[:, 0] < img_w)
        & (pixels_all[:, 1] >= 0)
        & (pixels_all[:, 1] < img_h)
    )

    return ProjectionResult(
        pixels=pixels_all[valid].astype(np.int32),
        depths=depths[valid],
        points_ego=points_ego_cam[valid],
        intensities=intensities_all[valid],
        img_w=img_w,
        img_h=img_h,
    )


# ── Step 2: Filtering (source-agnostic) ──────────────────────────────────────
#
# Everything below is independent of how the LiDAR points were projected.
# It operates only on ``ProjectionResult`` and numpy arrays.
# This code stays the same when moving to the production AV repo.
#
def lift_masks_to_3d(
    projection: ProjectionResult,
    masks_by_type: dict[str, list[tuple[np.ndarray, float]]],
) -> list[LanePoints3D]:
    """Filter projected LiDAR points through SAM3 masks, group by lane type.

    Parameters
    ----------
    projection : ProjectionResult from any projection backend.
    masks_by_type : dict mapping lane_type_str → list of (mask, score).
        Each mask is (H, W) binary/uint8.

    Returns
    -------
    List of LanePoints3D, one per lane type that has >0 points.
    """
    pixels = projection.pixels
    points_ego = projection.points_ego
    intensities = projection.intensities

    # Track which lane type each point is assigned to (highest confidence wins)
    best_type = np.full(len(pixels), -1, dtype=np.int32)
    best_score = np.full(len(pixels), -1.0, dtype=np.float64)

    type_names = list(masks_by_type.keys())
    for type_idx, lane_type in enumerate(type_names):
        for mask, score in masks_by_type[lane_type]:
            if mask.ndim == 3:
                mask = mask.squeeze()
            # Look up mask value at each projected pixel
            hit = mask[pixels[:, 1], pixels[:, 0]] > 0.5
            better = hit & (score > best_score)
            best_type[better] = type_idx
            best_score[better] = score

    results: list[LanePoints3D] = []
    for type_idx, lane_type in enumerate(type_names):
        sel = best_type == type_idx
        if not sel.any():
            continue
        results.append(
            LanePoints3D(
                lane_type=lane_type,
                points_ego=points_ego[sel],
                intensities=intensities[sel],
                pixel_coords=pixels[sel],
                num_points=int(sel.sum()),
            )
        )

    return results


def _parse_sam3_results(
    sam3_results: dict,
) -> dict[str, list[tuple[np.ndarray, float]]]:
    """Convert Sam3LaneInference output into masks_by_type for lift_masks_to_3d.

    Handles the case where ``scores`` is missing or shorter than ``masks``
    by defaulting unmatched masks to score 1.0.
    """
    masks_by_type: dict[str, list[tuple[np.ndarray, float]]] = {}
    for prompt, data in sam3_results.items():
        masks = data.get("masks", [])
        scores = data.get("scores", [])
        pairs = []
        for i, mask in enumerate(masks):
            score = scores[i] if i < len(scores) else 1.0
            pairs.append((mask, score))
        if pairs:
            masks_by_type[prompt] = pairs
    return masks_by_type


def lift_from_projection(
    projection: ProjectionResult,
    sam3_results: dict,
    sample_token: str = "",
    camera: str = "",
) -> FrameLiftResult:
    """Source-agnostic lifting: projection already done, just filter by masks.

    This is the function to call from the production AV repo, where projection
    comes from the existing LiDAR overlay code rather than nuScenes devkit.

    Parameters
    ----------
    projection : ProjectionResult from any projection backend.
    sam3_results : output from Sam3LaneInference.run_inference().
    sample_token, camera : metadata passed through to FrameLiftResult.
    """
    masks_by_type = _parse_sam3_results(sam3_results)
    lane_points = lift_masks_to_3d(projection, masks_by_type)
    total_lane = sum(lp.num_points for lp in lane_points)

    return FrameLiftResult(
        sample_token=sample_token,
        camera=camera,
        lane_points=lane_points,
        total_lidar_points=len(projection.pixels),
        total_lane_points=total_lane,
    )


def lift_frame(
    nusc: NuScenes,
    sample_token: str,
    sam3_results: dict,
    camera: str = "CAM_FRONT",
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
    nsweeps: int = 1,
) -> FrameLiftResult:
    """Convenience wrapper: nuScenes projection + filtering in one call.

    For production integration, use ``project_lidar_to_image_nuscenes`` →
    replace with production projection → ``lift_from_projection()``.
    """
    projection = project_lidar_to_image_nuscenes(
        nusc, sample_token, camera, min_depth, max_depth, nsweeps=nsweeps
    )
    return lift_from_projection(projection, sam3_results, sample_token, camera)


# ── Visualization ────────────────────────────────────────────────────────────
def visualize_lift_result(
    nusc: NuScenes,
    result: FrameLiftResult,
    pixels_all: np.ndarray,
    masks_by_type: Optional[dict[str, list[tuple[np.ndarray, float]]]] = None,
    output_path: Optional[Path] = None,
    point_radius: int = 2,
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """Create a 3-panel visualization: camera+masks+points | masks-only | BEV.

    Parameters
    ----------
    nusc : NuScenes handle
    result : FrameLiftResult from lift_frame()
    pixels_all : (M, 2) all projected LiDAR pixel coords (for background)
    masks_by_type : dict mapping lane_type → list of (mask, score).
        If provided, masks are overlaid on panel 1 and shown solo in panel 2.
    output_path : if set, save the visualization image
    point_radius : circle radius for drawing points
    mask_alpha : opacity for mask overlay on camera image (0=transparent, 1=opaque)

    Returns
    -------
    vis : (H, total_W, 3) uint8 BGR image
    """
    sample = nusc.get("sample", result.sample_token)
    cam_sd = nusc.get("sample_data", sample["data"][result.camera])
    img_path = Path(nusc.dataroot) / cam_sd["filename"]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  WARNING: Could not read image {img_path}, skipping visualization")
        return np.zeros((100, 200, 3), dtype=np.uint8)
    h, w = img.shape[:2]

    # Build a combined color mask image from SAM3 masks using priority painting.
    # Each pixel is assigned to exactly one lane type (last type with a
    # high-score mask wins), avoiding channel-wise max that blends to white.
    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    has_masks = masks_by_type is not None and len(masks_by_type) > 0
    if has_masks:
        type_names = list(masks_by_type.keys())
        mask_type_map = np.full((h, w), -1, dtype=np.int32)
        for type_idx, lane_type in enumerate(type_names):
            for mask, score in masks_by_type[lane_type]:
                m = mask.squeeze() if mask.ndim == 3 else mask
                mask_type_map[m > 0.5] = type_idx

        for type_idx, lane_type in enumerate(type_names):
            color = LANE_COLORS_BGR[type_idx % len(LANE_COLORS_BGR)]
            mask_overlay[mask_type_map == type_idx] = color

    # Panel 1: camera image with semi-transparent mask overlay + LiDAR points
    overlay = img.copy()
    if has_masks:
        mask_region = mask_overlay.any(axis=2)
        overlay[mask_region] = cv2.addWeighted(
            img, 1 - mask_alpha, mask_overlay, mask_alpha, 0
        )[mask_region]

    # Draw all projected LiDAR points in gray
    for u, v in pixels_all:
        cv2.circle(overlay, (int(u), int(v)), point_radius, NON_LANE_COLOR_BGR, -1)

    # Draw lane points on top, colored by type
    for idx, lp in enumerate(result.lane_points):
        color = LANE_COLORS_BGR[idx % len(LANE_COLORS_BGR)]
        for u, v in lp.pixel_coords:
            cv2.circle(overlay, (int(u), int(v)), point_radius + 1, color, -1)

    # Add legend
    y_offset = 30
    for idx, lp in enumerate(result.lane_points):
        color = LANE_COLORS_BGR[idx % len(LANE_COLORS_BGR)]
        label = f"{lp.lane_type}: {lp.num_points} pts"
        cv2.putText(overlay, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25

    total_label = (
        f"LiDAR: {result.total_lidar_points} projected, "
        f"{result.total_lane_points} lane ({100 * result.total_lane_points / max(result.total_lidar_points, 1):.1f}%)"
    )
    cv2.putText(overlay, total_label, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 2: masks-only on camera image (no LiDAR points, higher opacity)
    if has_masks:
        masks_panel = img.copy()
        mask_region = mask_overlay.any(axis=2)
        masks_panel[mask_region] = cv2.addWeighted(
            img, 0.3, mask_overlay, 0.7, 0
        )[mask_region]
        # Legend for mask panel
        cv2.putText(masks_panel, "SAM3 Masks", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_off = 55
        if masks_by_type:
            for type_idx, lane_type in enumerate(masks_by_type.keys()):
                color = LANE_COLORS_BGR[type_idx % len(LANE_COLORS_BGR)]
                n_masks = len(masks_by_type[lane_type])
                top_score = max(s for _, s in masks_by_type[lane_type])
                label = f"{lane_type}: {n_masks} mask(s), best={top_score:.2f}"
                cv2.putText(masks_panel, label, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_off += 22
    else:
        masks_panel = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(masks_panel, "No masks (--skip-sam3)", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 3: BEV scatter — front-only view, ego at bottom-center
    # nuScenes ego frame: x=forward, y=left, z=up
    # Shows only forward half (x>0) so lane points fill the panel.
    bev_h_px = h
    bev_w_px = 2 * h  # wide to cover lateral range
    bev = np.zeros((bev_h_px, bev_w_px, 3), dtype=np.uint8)

    # Auto-fit: compute range from actual lane points with padding
    fwd_range = 50.0   # default forward range in meters
    lat_range = 30.0    # default lateral half-range in meters
    if result.lane_points:
        all_ego = np.vstack([lp.points_ego for lp in result.lane_points])
        fwd_range = max(all_ego[:, 0].max() * 1.3, 10.0)
        lat_range = max(
            abs(all_ego[:, 1].min()), abs(all_ego[:, 1].max()), 10.0
        ) * 1.3

    scale_fwd = bev_h_px / fwd_range          # pixels per meter forward
    scale_lat = bev_w_px / (2 * lat_range)    # pixels per meter lateral

    def ego_to_bev(pts: np.ndarray) -> np.ndarray:
        """Convert ego-frame (x=forward, y=left) to BEV pixel coords.

        Ego at bottom-center, forward = up, left = left.
        """
        cx = bev_w_px / 2
        bev_x = (cx - pts[:, 1] * scale_lat).astype(int)
        bev_y = (bev_h_px - pts[:, 0] * scale_fwd).astype(int)
        return np.column_stack([bev_x, bev_y])

    # Draw lane points in BEV
    for idx, lp in enumerate(result.lane_points):
        color = LANE_COLORS_BGR[idx % len(LANE_COLORS_BGR)]
        bev_pts = ego_to_bev(lp.points_ego)
        for bx, by in bev_pts:
            if 0 <= bx < bev_w_px and 0 <= by < bev_h_px:
                cv2.circle(bev, (bx, by), point_radius + 1, color, -1)

    # Draw ego vehicle marker at bottom-center
    ego_bev_x = bev_w_px // 2
    ego_bev_y = bev_h_px - 10
    cv2.drawMarker(bev, (ego_bev_x, ego_bev_y), (255, 255, 255), cv2.MARKER_DIAMOND, 15, 2)
    cv2.putText(bev, "EGO", (ego_bev_x + 10, ego_bev_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw forward-facing range semicircles
    for r_m in [10, 20, 30, 40]:
        r_px_fwd = int(r_m * scale_fwd)
        r_px_lat = int(r_m * scale_lat)
        if r_px_fwd < bev_h_px:
            cv2.ellipse(bev, (ego_bev_x, ego_bev_y), (r_px_lat, r_px_fwd),
                        0, 180, 360, (40, 40, 40), 1)
            label_y = ego_bev_y - r_px_fwd
            if 0 <= label_y < bev_h_px:
                cv2.putText(bev, f"{r_m}m", (ego_bev_x + 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    # Labels
    cv2.putText(bev, f"BEV (front, {fwd_range:.0f}m)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bev, "fwd", (ego_bev_x - 10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    cv2.putText(bev, "left", (10, bev_h_px // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    cv2.putText(bev, "right", (bev_w_px - 45, bev_h_px // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    # Layout: row 1 = [camera overlay | masks], row 2 = [BEV centered]
    masks_resized = cv2.resize(masks_panel, (w, h))
    row1 = np.hstack([overlay, masks_resized])
    row1_w = row1.shape[1]

    # Resize BEV to match row1 width, keeping square aspect then centering
    bev_target_h = h
    bev_resized = cv2.resize(bev, (bev_target_h, bev_target_h))
    if bev_resized.shape[1] < row1_w:
        pad_total = row1_w - bev_resized.shape[1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        bev_row = cv2.copyMakeBorder(
            bev_resized, 0, 0, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
    else:
        bev_row = cv2.resize(bev, (row1_w, bev_target_h))

    vis = np.vstack([row1, bev_row])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis)

    return vis


# ── OpenLane JSON Export ─────────────────────────────────────────────────────

# OpenLane lane category IDs (from OpenLane dataset specification)
LANE_CATEGORY_MAP = {
    "white lane line": 1,
    "yellow lane line": 2,
    "white dashed lane line": 3,
    "yellow dashed lane line": 4,
    "dashed lane line": 5,
    "double white lane line": 6,
    "double yellow lane line": 7,
    "road curb": 20,
    "lane line": 21,       # generic
}
DEFAULT_LANE_CATEGORY = 21  # "lane marking, general case"


def export_openlane_json(
    result: FrameLiftResult,
    image_path: str = "",
    output_path: Optional[Path] = None,
) -> dict:
    """Export a FrameLiftResult to OpenLane-compatible JSON.

    Each ``LanePoints3D`` in the result becomes one entry in ``lane_lines``.
    Note: at this stage each entry is a *point cloud*, not an ordered polyline.
    Downstream association + spline fitting will convert these into ordered
    lane instances.

    Parameters
    ----------
    result : FrameLiftResult from lift_frame().
    image_path : file_path field in the JSON (relative or absolute).
    output_path : if set, write the JSON file here.

    Returns
    -------
    The OpenLane-compatible dict (also written to disk if output_path is set).
    """
    lane_lines = []
    for idx, lp in enumerate(result.lane_points):
        pts = lp.points_ego  # (N, 3)
        n = pts.shape[0]

        # Map lane_type string to OpenLane category ID
        category = LANE_CATEGORY_MAP.get(
            lp.lane_type.lower(), DEFAULT_LANE_CATEGORY
        )

        lane_entry = {
            "xyz": [
                pts[:, 0].tolist(),  # all x values
                pts[:, 1].tolist(),  # all y values
                pts[:, 2].tolist(),  # all z values
            ],
            "category": category,
            "visibility": [1.0] * n,  # all visible (from LiDAR)
            "track_id": idx,          # placeholder, needs association
            "lane_type": lp.lane_type,
            "num_points": n,
        }
        lane_lines.append(lane_entry)

    openlane_dict = {
        "file_path": image_path,
        "sample_token": result.sample_token,
        "camera": result.camera,
        "lane_lines": lane_lines,
        "total_lidar_points": result.total_lidar_points,
        "total_lane_points": result.total_lane_points,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(openlane_dict, f, indent=2)

    return openlane_dict


# ── CLI Entry Point ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D→3D lifting: project LiDAR through SAM3 lane masks."
    )
    script_dir = Path(__file__).parent

    parser.add_argument(
        "--data-root",
        type=str,
        default=str(script_dir / "data" / "v1.0-mini"),
        help="Path to nuScenes mini data root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "data" / "lift_results"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for SAM3 (cuda, mps, cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="CAM_FRONT",
        help="Camera channel",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per scene (for quick tests)",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Process a single scene (e.g., scene-0061). Defaults to all.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="SAM3 confidence threshold",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=MIN_DEPTH_M,
        help="Minimum LiDAR depth in meters",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=MAX_DEPTH_M,
        help="Maximum LiDAR depth in meters",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images",
    )
    parser.add_argument(
        "--skip-sam3",
        action="store_true",
        help="Skip SAM3 inference; only test LiDAR projection (no lane filtering)",
    )
    parser.add_argument(
        "--nsweeps",
        type=int,
        default=10,
        help="Number of LiDAR sweeps to accumulate (default: 10, use 1 for single sweep)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to start processing from (skip earlier frames)",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export 3D lane points in OpenLane JSON format",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes mini...")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(data_root), verbose=False)

    # Optionally initialize SAM3
    inferencer: Optional[Sam3LaneInference] = None
    if not args.skip_sam3:
        print("Initializing SAM3 inference...")
        inferencer = Sam3LaneInference(
            device=args.device,
            confidence_threshold=args.confidence_threshold,
        )

    for scene_config in TARGET_SCENES:
        if args.scene_name and scene_config.name != args.scene_name:
            continue

        if scene_config.token not in {s["token"] for s in nusc.scene}:
            print(f"Scene {scene_config.name} not in dataset, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {scene_config.name}: {scene_config.description}")
        print(f"{'=' * 60}")

        scene_output = output_dir / scene_config.name
        scene_output.mkdir(parents=True, exist_ok=True)

        scene = nusc.get("scene", scene_config.token)
        sample_token = scene["first_sample_token"]
        frame_idx = 0

        # Skip to --start-frame
        while sample_token and frame_idx < args.start_frame:
            sample = nusc.get("sample", sample_token)
            sample_token = sample.get("next", "")
            frame_idx += 1

        end_frame = args.start_frame + (args.max_samples or float("inf"))

        while sample_token:
            if frame_idx >= end_frame:
                break

            sample = nusc.get("sample", sample_token)

            # Run SAM3 inference (or skip)
            if inferencer is not None:
                cam_sd = nusc.get("sample_data", sample["data"][args.camera])
                img_path = Path(nusc.dataroot) / cam_sd["filename"]
                if not img_path.exists():
                    print(f"  Image not found: {img_path}, skipping")
                    sample_token = sample.get("next", "")
                    frame_idx += 1
                    continue
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
                sam3_results = inferencer.run_inference(image, LANE_PROMPTS)
                # Diagnostic: log what SAM3 found
                for prompt, data in sam3_results.items():
                    n_masks = data.get("num_detections", len(data.get("masks", [])))
                    if n_masks > 0:
                        mask_shapes = [m.shape for m in data.get("masks", [])]
                        scores = data.get("scores", [])
                        print(f"    SAM3 '{prompt}': {n_masks} masks, shapes={mask_shapes}, scores={scores}")
                    else:
                        print(f"    SAM3 '{prompt}': 0 detections")
            else:
                # No SAM3 — create empty results to test projection only
                sam3_results = {}

            # Lift 2D masks to 3D
            result = lift_frame(
                nusc,
                sample_token,
                sam3_results,
                camera=args.camera,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                nsweeps=args.nsweeps,
            )

            # Log diagnostics
            print(
                f"  Frame {frame_idx}: "
                f"{result.total_lidar_points} projected LiDAR pts, "
                f"{result.total_lane_points} lane pts "
                f"({100 * result.total_lane_points / max(result.total_lidar_points, 1):.1f}%)"
            )
            for lp in result.lane_points:
                print(f"    {lp.lane_type}: {lp.num_points} pts")

            # Visualization
            if args.visualize:
                proj = project_lidar_to_image_nuscenes(
                    nusc, sample_token, args.camera, args.min_depth, args.max_depth,
                    nsweeps=args.nsweeps,
                )
                masks_by_type = _parse_sam3_results(sam3_results)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                vis_path = scene_output / f"frame_{frame_idx:04d}_lift_{timestamp}.jpg"
                visualize_lift_result(
                    nusc, result, proj.pixels,
                    masks_by_type=masks_by_type,
                    output_path=vis_path,
                )
                print(f"    Saved: {vis_path}")

            # Export OpenLane JSON
            if args.export_json and result.total_lane_points > 0:
                cam_sd = nusc.get("sample_data", sample["data"][args.camera])
                img_rel_path = cam_sd["filename"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_path = scene_output / f"frame_{frame_idx:04d}_lanes_{timestamp}.json"
                export_openlane_json(result, image_path=img_rel_path, output_path=json_path)
                print(f"    Exported JSON: {json_path}")

            sample_token = sample.get("next", "")
            frame_idx += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
