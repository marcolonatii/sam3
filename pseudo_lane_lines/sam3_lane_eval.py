#!/usr/bin/env python3
"""
Evaluate SAM3 lane segmentation against nuScenes HD map lane dividers.

This script projects nuScenes map lane_divider and road_divider polylines into
CAM_FRONT images and compares them to SAM3-predicted lane masks using IoU and
Precision/Recall/F1.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import view_points

from sam3_lane_inference import LANE_PROMPTS, TARGET_SCENES, Sam3LaneInference


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""

    iou: float
    precision: float
    recall: float
    f1: float


def iter_scene_samples(
    nusc: NuScenes,
    scene_token: str,
    camera_name: str = "CAM_FRONT",
) -> Iterable[Tuple[str, str]]:
    """Yield (sample_data_token, image_path) for a scene."""
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]

    while sample_token:
        sample = nusc.get("sample", sample_token)
        sample_data_token = sample["data"][camera_name]
        sample_data = nusc.get("sample_data", sample_data_token)
        image_path = Path(nusc.dataroot) / sample_data["filename"]
        yield sample_data_token, str(image_path)

        sample_token = sample.get("next", "")
        if not sample_token:
            break


def _split_contiguous_segments(
    points: np.ndarray,
    valid_mask: np.ndarray,
) -> List[np.ndarray]:
    """Split points into contiguous segments where valid_mask is True."""
    segments: List[np.ndarray] = []
    current: List[np.ndarray] = []

    for point, is_valid in zip(points, valid_mask):
        if is_valid:
            current.append(point)
        else:
            if len(current) >= 2:
                segments.append(np.vstack(current))
            current = []

    if len(current) >= 2:
        segments.append(np.vstack(current))

    return segments


def _transform_global_to_sensor(
    points: np.ndarray,
    ego_pose: Dict,
    calibrated_sensor: Dict,
) -> np.ndarray:
    """Transform points from global to camera sensor frame."""
    translation = np.array(ego_pose["translation"])
    rotation = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix
    points = (rotation @ (points - translation).T).T

    translation = np.array(calibrated_sensor["translation"])
    rotation = Quaternion(calibrated_sensor["rotation"]).inverse.rotation_matrix
    points = (rotation @ (points - translation).T).T

    return points


def _extract_polyline_points(
    nusc_map: NuScenesMap,
    layer_name: str,
    record_token: str,
) -> List[Tuple[float, float]]:
    """Extract polyline points from a map record."""
    record = nusc_map.get(layer_name, record_token)
    node_tokens = None

    if "line_token" in record and record["line_token"]:
        line = nusc_map.get("line", record["line_token"])
        node_tokens = line.get("node_tokens")
    elif "node_tokens" in record:
        node_tokens = record.get("node_tokens")
    elif "nodes" in record:
        node_tokens = record.get("nodes")

    if not node_tokens:
        return []

    points: List[Tuple[float, float]] = []
    for node_token in node_tokens:
        if isinstance(node_token, dict):
            if "x" in node_token and "y" in node_token:
                points.append((node_token["x"], node_token["y"]))
            continue
        node = nusc_map.get("node", node_token)
        points.append((node["x"], node["y"]))

    return points


def build_lane_gt_mask(
    nusc: NuScenes,
    nusc_map: NuScenesMap,
    sample_data_token: str,
    image_size: Tuple[int, int],
    radius: float = 60.0,
    thickness: int = 2,
    layer_names: Sequence[str] = ("lane_divider", "road_divider"),
) -> np.ndarray:
    """Build a binary GT mask for lane dividers projected into the camera."""
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    sample_data = nusc.get("sample_data", sample_data_token)
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    camera_intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
    ego_x, ego_y, _ = ego_pose["translation"]
    records = nusc_map.get_records_in_radius(
        ego_x, ego_y, radius, layer_names=list(layer_names)
    )

    for layer_name in layer_names:
        for token in records.get(layer_name, []):
            line_points = _extract_polyline_points(nusc_map, layer_name, token)
            if len(line_points) < 2:
                continue

            points = np.array(
                [(x, y, 0.0) for x, y in line_points],
                dtype=np.float32,
            )
            points_cam = _transform_global_to_sensor(points, ego_pose, calibrated_sensor)
            valid_mask = points_cam[:, 2] > 0.1
            segments = _split_contiguous_segments(points_cam, valid_mask)

            for segment in segments:
                points_img = view_points(segment.T, camera_intrinsic, normalize=True)[:2].T
                if points_img.shape[0] < 2:
                    continue
                points_img = np.round(points_img).astype(np.int32)
                points_img[:, 0] = np.clip(points_img[:, 0], 0, width - 1)
                points_img[:, 1] = np.clip(points_img[:, 1], 0, height - 1)
                cv2.polylines(mask, [points_img], isClosed=False, color=1, thickness=thickness)

    return mask


def build_pred_mask(results: Dict, image_size: Tuple[int, int]) -> np.ndarray:
    """Union SAM3 masks into a single binary prediction mask."""
    width, height = image_size
    pred_mask = np.zeros((height, width), dtype=bool)

    for data in results.values():
        for mask in data.get("masks", []):
            if mask.ndim == 3:
                mask = mask.squeeze()
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            pred_mask |= mask > 0.5

    return pred_mask.astype(np.uint8)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> FrameMetrics:
    """Compute IoU and Precision/Recall/F1 for a single frame."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return FrameMetrics(
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM3 lanes on nuScenes mini.")
    script_dir = Path(__file__).parent

    parser.add_argument(
        "--data-root",
        type=str,
        default=str(script_dir / "data" / "v1.0-mini"),
        help="Path to nuScenes mini data root (must include maps/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "data" / "sam3_results"),
        help="Output directory for evaluation summaries",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per scene (for quick tests)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for SAM3 detections",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="CAM_FRONT",
        help="Camera channel to evaluate",
    )
    parser.add_argument(
        "--map-radius",
        type=float,
        default=60.0,
        help="Radius (meters) around ego pose to query map dividers",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Evaluate a single scene by name (e.g., scene-0061). Defaults to all.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Thickness (pixels) for rasterized lane divider lines",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (data_root / "maps").exists():
        raise FileNotFoundError(
            f"Expected maps/ under {data_root}. Please point --data-root to the "
            "nuScenes mini root that contains maps/ and v1.0-mini/."
        )

    print("Loading nuScenes mini...")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(data_root), verbose=False)

    print("Initializing SAM3 inference...")
    inferencer = Sam3LaneInference(
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    scene_name_filter = args.scene_name
    for scene_config in TARGET_SCENES:
        if scene_name_filter and scene_config.name != scene_name_filter:
            continue
        print(f"\n{'=' * 60}")
        print(f"Evaluating {scene_config.name}")
        print(f"{'=' * 60}")

        if scene_config.token not in {scene["token"] for scene in nusc.scene}:
            print(f"Warning: Scene {scene_config.name} not found in dataset, skipping...")
            continue

        scene_output = output_dir / scene_config.name
        scene_output.mkdir(parents=True, exist_ok=True)

        scene_record = nusc.get("scene", scene_config.token)
        log_record = nusc.get("log", scene_record["log_token"])
        location = log_record["location"]
        nusc_map = NuScenesMap(dataroot=str(data_root), map_name=location)

        frame_metrics: List[Dict] = []
        samples_processed = 0

        for sample_data_token, image_path in iter_scene_samples(
            nusc, scene_config.token, camera_name=args.camera
        ):
            if args.max_samples and samples_processed >= args.max_samples:
                break

            if not Path(image_path).exists():
                continue

            with Image.open(image_path) as img:
                image = img.convert("RGB")
            image_size = image.size  # (width, height)

            results = inferencer.run_inference(image, LANE_PROMPTS)
            pred_mask = build_pred_mask(results, image_size)
            gt_mask = build_lane_gt_mask(
                nusc,
                nusc_map,
                sample_data_token,
                image_size,
                radius=args.map_radius,
                thickness=args.line_thickness,
            )

            metrics = compute_metrics(pred_mask, gt_mask)
            frame_metrics.append(
                {
                    "sample_data_token": sample_data_token,
                    "image_path": image_path,
                    "iou": metrics.iou,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )

            samples_processed += 1

        if not frame_metrics:
            print(f"No frames processed for {scene_config.name}")
            continue

        iou_values = [m["iou"] for m in frame_metrics]
        precision_values = [m["precision"] for m in frame_metrics]
        recall_values = [m["recall"] for m in frame_metrics]
        f1_values = [m["f1"] for m in frame_metrics]

        summary = {
            "scene_name": scene_config.name,
            "scene_token": scene_config.token,
            "description": scene_config.description,
            "num_samples_processed": len(frame_metrics),
            "camera": args.camera,
            "map_radius_m": args.map_radius,
            "line_thickness_px": args.line_thickness,
            "prompts_used": LANE_PROMPTS,
            "metrics": {
                "mean_iou": float(np.mean(iou_values)),
                "mean_precision": float(np.mean(precision_values)),
                "mean_recall": float(np.mean(recall_values)),
                "mean_f1": float(np.mean(f1_values)),
            },
            "frames": frame_metrics,
        }

        output_path = scene_output / "eval_metrics.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved evaluation summary to: {output_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
