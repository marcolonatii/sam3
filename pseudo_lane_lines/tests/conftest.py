"""Shared fixtures for lift_2d_to_3d tests.

All fixtures produce synthetic data so tests run without nuScenes, SAM3,
or any GPU.  Heavy deps (nuscenes, sam3_lane_inference) are mocked at
import time via ``sys.modules`` patching — the test module never needs
them installed.
"""

import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Mock heavy dependencies before importing lift_2d_to_3d ──────────────────
# The module-level imports in lift_2d_to_3d.py pull in nuscenes and
# sam3_lane_inference.  We inject lightweight fakes so that importing the
# module succeeds in CI / local dev without those packages.


def _install_mock_modules() -> None:
    """Insert mock modules for nuscenes and sam3_lane_inference."""
    # nuscenes hierarchy
    for mod_name in [
        "nuscenes",
        "nuscenes.nuscenes",
        "nuscenes.utils",
        "nuscenes.utils.data_classes",
        "nuscenes.utils.geometry_utils",
    ]:
        if mod_name not in sys.modules:
            fake = ModuleType(mod_name)
            if mod_name == "nuscenes.nuscenes":
                fake.NuScenes = MagicMock
            if mod_name == "nuscenes.utils.data_classes":
                fake.LidarPointCloud = MagicMock()
            if mod_name == "nuscenes.utils.geometry_utils":
                fake.view_points = MagicMock()
            sys.modules[mod_name] = fake

    # sam3_lane_inference
    if "sam3_lane_inference" not in sys.modules:
        fake_sam3 = ModuleType("sam3_lane_inference")
        fake_sam3.LANE_PROMPTS = ["white lane line", "yellow lane line"]
        fake_sam3.TARGET_SCENES = []
        fake_sam3.Sam3LaneInference = MagicMock
        sys.modules["sam3_lane_inference"] = fake_sam3

    # pyquaternion — only needed for the nuScenes projection path
    if "pyquaternion" not in sys.modules:
        fake_pyq = ModuleType("pyquaternion")
        fake_pyq.Quaternion = MagicMock
        sys.modules["pyquaternion"] = fake_pyq

    # cv2 — only needed for visualization
    if "cv2" not in sys.modules:
        fake_cv2 = ModuleType("cv2")
        fake_cv2.imread = MagicMock(return_value=None)
        fake_cv2.circle = MagicMock()
        fake_cv2.putText = MagicMock()
        fake_cv2.imwrite = MagicMock()
        fake_cv2.resize = MagicMock()
        fake_cv2.drawMarker = MagicMock()
        fake_cv2.FONT_HERSHEY_SIMPLEX = 0
        fake_cv2.MARKER_DIAMOND = 0
        sys.modules["cv2"] = fake_cv2


_install_mock_modules()

# NOW we can safely import from the module under test
from lift_2d_to_3d import ProjectionResult  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────────


def make_mask(
    h: int, w: int, regions: list[tuple[int, int, int, int]] | None = None
) -> np.ndarray:
    """Create an (H, W) binary uint8 mask.

    Parameters
    ----------
    h, w : image dimensions.
    regions : list of (y_start, y_end, x_start, x_end) slices to set to 1.
              If None, the entire mask is 1.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if regions is None:
        mask[:] = 1
    else:
        for y0, y1, x0, x1 in regions:
            mask[y0:y1, x0:x1] = 1
    return mask


# ── Fixtures ────────────────────────────────────────────────────────────────

IMG_W, IMG_H = 1600, 900


@pytest.fixture
def basic_projection() -> ProjectionResult:
    """10 LiDAR points at known pixel locations in a 1600x900 image."""
    rng = np.random.RandomState(42)
    n = 10
    pixels = np.column_stack(
        [
            np.linspace(100, 1500, n, dtype=np.int32),
            np.linspace(100, 800, n, dtype=np.int32),
        ]
    )
    depths = rng.uniform(5.0, 60.0, size=n)
    points_ego = rng.randn(n, 3).astype(np.float64)
    intensities = rng.uniform(0.0, 1.0, size=n)
    return ProjectionResult(
        pixels=pixels,
        depths=depths,
        points_ego=points_ego,
        intensities=intensities,
        img_w=IMG_W,
        img_h=IMG_H,
    )


@pytest.fixture
def empty_projection() -> ProjectionResult:
    """0 LiDAR points — edge case for empty scenes."""
    return ProjectionResult(
        pixels=np.empty((0, 2), dtype=np.int32),
        depths=np.empty(0, dtype=np.float64),
        points_ego=np.empty((0, 3), dtype=np.float64),
        intensities=np.empty(0, dtype=np.float64),
        img_w=IMG_W,
        img_h=IMG_H,
    )


@pytest.fixture
def large_projection() -> ProjectionResult:
    """10K random LiDAR points for performance sanity."""
    rng = np.random.RandomState(99)
    n = 10_000
    pixels = np.column_stack(
        [
            rng.randint(0, IMG_W, size=n),
            rng.randint(0, IMG_H, size=n),
        ]
    ).astype(np.int32)
    depths = rng.uniform(1.0, 80.0, size=n)
    points_ego = rng.randn(n, 3).astype(np.float64)
    intensities = rng.uniform(0.0, 1.0, size=n)
    return ProjectionResult(
        pixels=pixels,
        depths=depths,
        points_ego=points_ego,
        intensities=intensities,
        img_w=IMG_W,
        img_h=IMG_H,
    )


# ── Multi-Sweep Mock NuScenes ──────────────────────────────────────────────


def _make_identity_quat_list() -> list[float]:
    """Return [w, x, y, z] for identity quaternion."""
    return [1.0, 0.0, 0.0, 0.0]


def _make_lidar_points(n_pts: int, x_offset: float = 0.0) -> np.ndarray:
    """Create (4, N) array in LidarPointCloud format: x, y, z, intensity."""
    pts = np.zeros((4, n_pts), dtype=np.float32)
    pts[0] = np.linspace(-1.0, 1.0, n_pts) + x_offset  # x: lateral
    pts[1] = np.linspace(-1.0, 1.0, n_pts)  # y
    pts[2] = np.full(n_pts, 10.0)  # z: forward (depth)
    pts[3] = np.full(n_pts, 0.5)  # intensity
    return pts


@pytest.fixture
def mock_nusc_multi_sweep(tmp_path):
    """Mock NuScenes + LidarPointCloud.from_file_multisweep for testing.

    Mocks the devkit's multi-sweep accumulation to return controlled point
    clouds. Simulates 1-sweep and 3-sweep scenarios by returning different
    point counts based on the nsweeps argument.

    Returns (nusc_mock, sample_token, n_pts_per_sweep).
    """
    class FakeQuaternion:
        """Minimal quaternion that supports .rotation_matrix and .inverse."""

        def __init__(self, q):
            self._q = q

        @property
        def rotation_matrix(self):
            return np.eye(3)

        @property
        def inverse(self):
            return FakeQuaternion(self._q)

    import lift_2d_to_3d as lmod
    original_quat = lmod.Quaternion
    original_lidar_pc = lmod.LidarPointCloud
    lmod.Quaternion = FakeQuaternion

    n_pts = 10

    # Mock LidarPointCloud.from_file_multisweep to return controlled data
    class FakeLidarPC:
        @staticmethod
        def from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=5,
                                 min_distance=1.0):
            """Return nsweeps worth of points (capped at 3 available)."""
            actual_sweeps = min(nsweeps, 3)  # simulate 3 available sweeps
            all_pts = []
            all_times = []
            for i in range(actual_sweeps):
                pts = _make_lidar_points(n_pts, x_offset=float(i * 2))
                all_pts.append(pts)
                all_times.append(np.full((1, n_pts), i * 0.05))

            merged_pts = np.hstack(all_pts)
            merged_times = np.hstack(all_times)

            pc = MagicMock()
            pc.points = merged_pts
            pc.nbr_points.return_value = merged_pts.shape[1]
            return pc, merged_times

    lmod.LidarPointCloud = FakeLidarPC

    # Camera + LiDAR metadata (all identity transforms)
    cam_sd = {
        "token": "cam_sd_0",
        "calibrated_sensor_token": "cam_cs",
        "ego_pose_token": "cam_ego",
        "width": IMG_W,
        "height": IMG_H,
    }
    cam_ego = {
        "token": "cam_ego",
        "rotation": _make_identity_quat_list(),
        "translation": [0.0, 0.0, 0.0],
    }
    cam_cs = {
        "token": "cam_cs",
        "rotation": _make_identity_quat_list(),
        "translation": [0.0, 0.0, 0.0],
        "camera_intrinsic": [
            [800.0, 0.0, 800.0],
            [0.0, 800.0, 450.0],
            [0.0, 0.0, 1.0],
        ],
    }
    lidar_sd = {
        "token": "lidar_sd_0",
        "calibrated_sensor_token": "lidar_cs",
        "ego_pose_token": "lidar_ego",
    }
    lidar_cs = {
        "token": "lidar_cs",
        "rotation": _make_identity_quat_list(),
        "translation": [0.0, 0.0, 0.0],
    }
    lidar_ego = {
        "token": "lidar_ego",
        "rotation": _make_identity_quat_list(),
        "translation": [0.0, 0.0, 0.0],
    }

    sample = {
        "token": "sample_0",
        "data": {
            "CAM_FRONT": "cam_sd_0",
            "LIDAR_TOP": "lidar_sd_0",
        },
    }

    db = {
        ("sample", "sample_0"): sample,
        ("sample_data", "cam_sd_0"): cam_sd,
        ("sample_data", "lidar_sd_0"): lidar_sd,
        ("calibrated_sensor", "cam_cs"): cam_cs,
        ("calibrated_sensor", "lidar_cs"): lidar_cs,
        ("ego_pose", "cam_ego"): cam_ego,
        ("ego_pose", "lidar_ego"): lidar_ego,
    }

    nusc = MagicMock()
    nusc.dataroot = str(tmp_path)
    nusc.get = lambda table, token: db[(table, token)]

    # Mock view_points to do a simple pinhole projection
    def fake_view_points(pts_3xN, intrinsic, normalize=True):
        projected = intrinsic @ pts_3xN
        if normalize:
            projected[:2] /= projected[2:3]
        return projected

    lmod.view_points = fake_view_points

    yield nusc, "sample_0", n_pts

    # Restore originals
    lmod.Quaternion = original_quat
    lmod.LidarPointCloud = original_lidar_pc
