"""Unit tests for lift_2d_to_3d.py — the source-agnostic filtering layer.

Tests cover:
  - lift_masks_to_3d  (core point-filtering logic)
  - _parse_sam3_results (SAM3 dict → masks_by_type)
  - lift_from_projection (integration of parse + lift)
  - ProjectionResult (data structure sanity)

Heavy deps (nuScenes, SAM3, OpenCV) are mocked in conftest.py so every test
runs with only numpy + pytest.
"""

import numpy as np
import pytest

# conftest.py installs mocks before this import resolves
from lift_2d_to_3d import (
    DEFAULT_LANE_CATEGORY,
    FrameLiftResult,
    LANE_CATEGORY_MAP,
    LanePoints3D,
    ProjectionResult,
    _parse_sam3_results,
    export_openlane_json,
    lift_from_projection,
    lift_masks_to_3d,
    project_lidar_to_image_nuscenes,
)

# ── Constants & helpers (duplicated from conftest to avoid import issues) ────
IMG_W, IMG_H = 1600, 900


def make_mask(
    h: int, w: int, regions: list[tuple[int, int, int, int]] | None = None
) -> np.ndarray:
    """Create an (H, W) binary uint8 mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if regions is None:
        mask[:] = 1
    else:
        for y0, y1, x0, x1 in regions:
            mask[y0:y1, x0:x1] = 1
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# TestLiftMasksTo3D — core filtering logic
# ═══════════════════════════════════════════════════════════════════════════


class TestLiftMasksTo3D:
    """Test lift_masks_to_3d: given a projection + masks, return 3D lane points."""

    def test_single_mask_happy_path(self, basic_projection):
        """One mask covering first 2 of 10 pixels → returns exactly those 2."""
        px = basic_projection.pixels
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for u, v in px[:2]:
            mask[v, u] = 1

        masks_by_type = {"white lane line": [(mask, 0.9)]}
        results = lift_masks_to_3d(basic_projection, masks_by_type)

        assert len(results) == 1
        lp = results[0]
        assert lp.lane_type == "white lane line"
        assert lp.num_points == 2
        np.testing.assert_array_equal(lp.pixel_coords, px[:2])
        np.testing.assert_array_equal(
            lp.points_ego, basic_projection.points_ego[:2]
        )

    def test_multiple_types_separate_regions(self, basic_projection):
        """Two lane types with non-overlapping masks → each captures its own points."""
        px = basic_projection.pixels
        mask_white = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        mask_yellow = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for u, v in px[:3]:
            mask_white[v, u] = 1
        for u, v in px[7:]:
            mask_yellow[v, u] = 1

        masks_by_type = {
            "white lane line": [(mask_white, 0.8)],
            "yellow lane line": [(mask_yellow, 0.7)],
        }
        results = lift_masks_to_3d(basic_projection, masks_by_type)

        by_type = {r.lane_type: r for r in results}
        assert by_type["white lane line"].num_points == 3
        assert by_type["yellow lane line"].num_points == 3

    def test_no_mask_hits(self, basic_projection):
        """Mask exists but covers no LiDAR pixel locations → empty result."""
        mask = make_mask(IMG_H, IMG_W, [(0, 5, 0, 5)])
        masks_by_type = {"white lane line": [(mask, 0.9)]}
        results = lift_masks_to_3d(basic_projection, masks_by_type)
        assert results == []

    def test_empty_masks_by_type(self, basic_projection):
        """Empty masks_by_type dict → empty result."""
        results = lift_masks_to_3d(basic_projection, {})
        assert results == []

    def test_empty_projection_with_masks(self, empty_projection):
        """0 LiDAR points + valid masks → empty result."""
        mask = make_mask(IMG_H, IMG_W)
        masks_by_type = {"white lane line": [(mask, 0.9)]}
        results = lift_masks_to_3d(empty_projection, masks_by_type)
        assert results == []

    def test_overlapping_masks_highest_score_wins(self, basic_projection):
        """Two types overlap on same pixels — higher score takes the point."""
        px = basic_projection.pixels
        mask_all = make_mask(IMG_H, IMG_W)
        masks_by_type = {
            "white lane line": [(mask_all, 0.6)],
            "yellow lane line": [(mask_all, 0.9)],
        }
        results = lift_masks_to_3d(basic_projection, masks_by_type)

        # Yellow has higher score → should win all 10 points
        assert len(results) == 1
        assert results[0].lane_type == "yellow lane line"
        assert results[0].num_points == 10

    def test_mask_3d_squeezed(self, basic_projection):
        """(1, H, W) mask is squeezed correctly to (H, W)."""
        px = basic_projection.pixels
        mask_2d = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for u, v in px[:4]:
            mask_2d[v, u] = 1
        mask_3d = mask_2d[np.newaxis, :, :]  # (1, H, W)

        masks_by_type = {"white lane line": [(mask_3d, 0.8)]}
        results = lift_masks_to_3d(basic_projection, masks_by_type)
        assert len(results) == 1
        assert results[0].num_points == 4

    def test_all_points_inside_mask(self, basic_projection):
        """Full-image mask captures all projected points."""
        mask = make_mask(IMG_H, IMG_W)
        masks_by_type = {"road boundary": [(mask, 0.95)]}
        results = lift_masks_to_3d(basic_projection, masks_by_type)
        assert len(results) == 1
        assert results[0].num_points == len(basic_projection.pixels)

    def test_point_assignment_exclusive(self, basic_projection):
        """No point appears in two LanePoints3D results."""
        px = basic_projection.pixels
        mask_white = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        mask_yellow = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for u, v in px[:5]:
            mask_white[v, u] = 1
        for u, v in px[4:]:
            mask_yellow[v, u] = 1

        masks_by_type = {
            "white lane line": [(mask_white, 0.7)],
            "yellow lane line": [(mask_yellow, 0.8)],
        }
        results = lift_masks_to_3d(basic_projection, masks_by_type)

        all_assigned = set()
        for r in results:
            indices = set(map(tuple, r.pixel_coords.tolist()))
            assert all_assigned.isdisjoint(indices), "Point assigned to multiple types"
            all_assigned.update(indices)

    def test_large_projection_perf(self, large_projection):
        """10K points through a full-image mask completes without error."""
        mask = make_mask(IMG_H, IMG_W)
        masks_by_type = {"white lane line": [(mask, 0.8)]}
        results = lift_masks_to_3d(large_projection, masks_by_type)
        assert results[0].num_points == len(large_projection.pixels)


# ═══════════════════════════════════════════════════════════════════════════
# TestParseSam3Results — SAM3 output dict → masks_by_type
# ═══════════════════════════════════════════════════════════════════════════


class TestParseSam3Results:
    """Test _parse_sam3_results: convert SAM3 inference dict → masks_by_type."""

    def test_single_prompt_single_mask(self):
        mask = make_mask(IMG_H, IMG_W, [(100, 200, 300, 500)])
        sam3_results = {
            "white lane line": {
                "masks": [mask],
                "scores": [0.85],
            }
        }
        out = _parse_sam3_results(sam3_results)
        assert "white lane line" in out
        assert len(out["white lane line"]) == 1
        np.testing.assert_array_equal(out["white lane line"][0][0], mask)
        assert out["white lane line"][0][1] == 0.85

    def test_multiple_prompts_multiple_masks(self):
        m1 = make_mask(IMG_H, IMG_W, [(0, 100, 0, 100)])
        m2 = make_mask(IMG_H, IMG_W, [(200, 300, 200, 300)])
        m3 = make_mask(IMG_H, IMG_W, [(400, 500, 400, 600)])
        sam3_results = {
            "white lane line": {"masks": [m1, m2], "scores": [0.9, 0.7]},
            "yellow lane line": {"masks": [m3], "scores": [0.6]},
        }
        out = _parse_sam3_results(sam3_results)
        assert len(out["white lane line"]) == 2
        assert len(out["yellow lane line"]) == 1

    def test_empty_results(self):
        assert _parse_sam3_results({}) == {}

    def test_prompt_with_zero_detections(self):
        """Empty masks list for a prompt → filtered out entirely."""
        sam3_results = {
            "white lane line": {"masks": [], "scores": []},
        }
        out = _parse_sam3_results(sam3_results)
        assert out == {}

    def test_missing_scores_defaults(self):
        """Masks present but no scores key → should default to 1.0 (bug fix)."""
        mask = make_mask(IMG_H, IMG_W, [(50, 150, 50, 150)])
        sam3_results = {
            "white lane line": {"masks": [mask]},
        }
        out = _parse_sam3_results(sam3_results)
        assert "white lane line" in out
        assert len(out["white lane line"]) == 1
        assert out["white lane line"][0][1] == 1.0

    def test_fewer_scores_than_masks(self):
        """Scores list shorter than masks → extra masks default to 1.0."""
        m1 = make_mask(IMG_H, IMG_W, [(0, 50, 0, 50)])
        m2 = make_mask(IMG_H, IMG_W, [(100, 150, 100, 150)])
        sam3_results = {
            "white lane line": {"masks": [m1, m2], "scores": [0.8]},
        }
        out = _parse_sam3_results(sam3_results)
        assert len(out["white lane line"]) == 2
        assert out["white lane line"][0][1] == 0.8
        assert out["white lane line"][1][1] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TestLiftFromProjection — integration of parse + lift
# ═══════════════════════════════════════════════════════════════════════════


class TestLiftFromProjection:
    """Test lift_from_projection: SAM3 dict + projection → FrameLiftResult."""

    def test_happy_path_end_to_end(self, basic_projection):
        px = basic_projection.pixels
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for u, v in px[:5]:
            mask[v, u] = 1

        sam3_results = {
            "white lane line": {"masks": [mask], "scores": [0.9]},
        }
        result = lift_from_projection(
            basic_projection, sam3_results, "tok_123", "CAM_FRONT"
        )

        assert isinstance(result, FrameLiftResult)
        assert result.total_lane_points == 5
        assert len(result.lane_points) == 1
        assert result.lane_points[0].lane_type == "white lane line"

    def test_metadata_passthrough(self, basic_projection):
        sam3_results = {}
        result = lift_from_projection(
            basic_projection, sam3_results, "sample_abc", "CAM_REAR"
        )
        assert result.sample_token == "sample_abc"
        assert result.camera == "CAM_REAR"

    def test_empty_sam3_results(self, basic_projection):
        """No SAM3 detections → 0 lane points, but total_lidar_points is correct."""
        result = lift_from_projection(basic_projection, {}, "tok", "CAM_FRONT")
        assert result.total_lane_points == 0
        assert result.lane_points == []
        assert result.total_lidar_points == len(basic_projection.pixels)

    def test_totals_correct(self, basic_projection):
        """total_lane_points == sum of individual LanePoints3D.num_points."""
        px = basic_projection.pixels
        m1 = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        m2 = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for u, v in px[:3]:
            m1[v, u] = 1
        for u, v in px[7:]:
            m2[v, u] = 1

        sam3_results = {
            "white lane line": {"masks": [m1], "scores": [0.8]},
            "yellow lane line": {"masks": [m2], "scores": [0.7]},
        }
        result = lift_from_projection(basic_projection, sam3_results, "t", "c")
        expected_total = sum(lp.num_points for lp in result.lane_points)
        assert result.total_lane_points == expected_total


# ═══════════════════════════════════════════════════════════════════════════
# TestProjectionResult — data structure sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestProjectionResult:
    """Sanity checks on ProjectionResult data shapes and types."""

    def test_shapes_consistent(self, basic_projection):
        n = len(basic_projection.pixels)
        assert basic_projection.pixels.shape == (n, 2)
        assert basic_projection.depths.shape == (n,)
        assert basic_projection.points_ego.shape == (n, 3)
        assert basic_projection.intensities.shape == (n,)

    def test_dtypes_correct(self, basic_projection):
        assert basic_projection.pixels.dtype == np.int32
        assert np.issubdtype(basic_projection.depths.dtype, np.floating)
        assert np.issubdtype(basic_projection.points_ego.dtype, np.floating)
        assert np.issubdtype(basic_projection.intensities.dtype, np.floating)

    def test_empty_valid(self, empty_projection):
        assert empty_projection.pixels.shape == (0, 2)
        assert empty_projection.depths.shape == (0,)
        assert empty_projection.points_ego.shape == (0, 3)
        assert empty_projection.intensities.shape == (0,)
        assert empty_projection.img_w == IMG_W
        assert empty_projection.img_h == IMG_H


# ═══════════════════════════════════════════════════════════════════════════
# TestMultiSweepProjection — multi-sweep LiDAR accumulation
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiSweepProjection:
    """Test multi-sweep accumulation in project_lidar_to_image_nuscenes."""

    def test_nsweeps_1_backward_compat(self, mock_nusc_multi_sweep):
        """nsweeps=1 loads exactly 1 .bin → same point count as single sweep."""
        nusc, sample_token, n_pts = mock_nusc_multi_sweep
        result = project_lidar_to_image_nuscenes(
            nusc, sample_token, nsweeps=1,
        )
        # With identity transforms and simple intrinsics, all 10 points
        # should project into the image (z=5 > min_depth=1)
        assert len(result.depths) <= n_pts
        assert len(result.depths) > 0

    def test_nsweeps_3_loads_three_sweeps(self, mock_nusc_multi_sweep):
        """nsweeps=3 merges 3 sweeps → up to 3x single-sweep point count."""
        nusc, sample_token, n_pts = mock_nusc_multi_sweep
        result_1 = project_lidar_to_image_nuscenes(
            nusc, sample_token, nsweeps=1,
        )
        result_3 = project_lidar_to_image_nuscenes(
            nusc, sample_token, nsweeps=3,
        )
        # 3 sweeps should have more points than 1 sweep
        assert len(result_3.depths) > len(result_1.depths)
        # At most 3x (some may fall outside image or depth bounds)
        assert len(result_3.depths) <= 3 * n_pts

    def test_nsweeps_exceeds_available(self, mock_nusc_multi_sweep):
        """Requesting more sweeps than available → no error, uses what exists."""
        nusc, sample_token, n_pts = mock_nusc_multi_sweep
        # Only 3 sweeps exist; requesting 10 should silently use 3
        result = project_lidar_to_image_nuscenes(
            nusc, sample_token, nsweeps=10,
        )
        # Should have same count as nsweeps=3 since only 3 exist
        result_3 = project_lidar_to_image_nuscenes(
            nusc, sample_token, nsweeps=3,
        )
        assert len(result.depths) == len(result_3.depths)

    def test_per_sweep_ego_pose_used(self, mock_nusc_multi_sweep):
        """2 sweeps with different ego translations produce different global points."""
        nusc, sample_token, n_pts = mock_nusc_multi_sweep
        result_1 = project_lidar_to_image_nuscenes(
            nusc, sample_token, nsweeps=1,
        )
        result_2 = project_lidar_to_image_nuscenes(
            nusc, sample_token, nsweeps=2,
        )
        # The second sweep has ego_translation=[10,0,0], so its points in
        # global frame differ from sweep 0's ego_translation=[0,0,0].
        # After transforming back to camera ego, the merged set should
        # contain points NOT present in the single-sweep result.
        assert len(result_2.points_ego) > len(result_1.points_ego)

    def test_nsweeps_invalid_raises(self, mock_nusc_multi_sweep):
        """nsweeps=0 or negative → ValueError."""
        nusc, sample_token, _ = mock_nusc_multi_sweep
        with pytest.raises(ValueError, match="nsweeps must be >= 1"):
            project_lidar_to_image_nuscenes(nusc, sample_token, nsweeps=0)
        with pytest.raises(ValueError, match="nsweeps must be >= 1"):
            project_lidar_to_image_nuscenes(nusc, sample_token, nsweeps=-1)


# ═══════════════════════════════════════════════════════════════════════════
# TestExportOpenLaneJSON — OpenLane JSON export
# ═══════════════════════════════════════════════════════════════════════════


class TestExportOpenLaneJSON:
    """Test export_openlane_json: FrameLiftResult → OpenLane JSON format."""

    @staticmethod
    def _make_result(
        lane_types: list[str] | None = None,
        n_points: int = 5,
    ) -> FrameLiftResult:
        """Create a synthetic FrameLiftResult for testing."""
        if lane_types is None:
            lane_types = ["white dashed lane line"]
        rng = np.random.RandomState(0)
        lane_points = []
        total = 0
        for lt in lane_types:
            pts = rng.randn(n_points, 3).astype(np.float64)
            lane_points.append(
                LanePoints3D(
                    lane_type=lt,
                    points_ego=pts,
                    intensities=rng.uniform(0, 1, n_points),
                    pixel_coords=rng.randint(0, 1000, (n_points, 2)),
                    num_points=n_points,
                )
            )
            total += n_points
        return FrameLiftResult(
            sample_token="tok_42",
            camera="CAM_FRONT",
            lane_points=lane_points,
            total_lidar_points=1000,
            total_lane_points=total,
        )

    def test_basic_structure(self):
        """Output has required OpenLane keys: file_path, lane_lines."""
        result = self._make_result()
        out = export_openlane_json(result, image_path="samples/CAM_FRONT/test.jpg")
        assert "file_path" in out
        assert "lane_lines" in out
        assert out["file_path"] == "samples/CAM_FRONT/test.jpg"
        assert len(out["lane_lines"]) == 1

    def test_xyz_shape_is_3_by_n(self):
        """xyz field should be a list of 3 lists (x, y, z), each of length N."""
        result = self._make_result(n_points=7)
        out = export_openlane_json(result)
        lane = out["lane_lines"][0]
        xyz = lane["xyz"]
        assert len(xyz) == 3
        assert len(xyz[0]) == 7
        assert len(xyz[1]) == 7
        assert len(xyz[2]) == 7

    def test_xyz_values_match_points_ego(self):
        """Exported xyz values should exactly match the input points_ego."""
        result = self._make_result(n_points=3)
        out = export_openlane_json(result)
        lane = out["lane_lines"][0]
        pts = result.lane_points[0].points_ego
        np.testing.assert_allclose(lane["xyz"][0], pts[:, 0].tolist())
        np.testing.assert_allclose(lane["xyz"][1], pts[:, 1].tolist())
        np.testing.assert_allclose(lane["xyz"][2], pts[:, 2].tolist())

    def test_category_mapping_known(self):
        """Known lane types map to correct OpenLane category IDs."""
        result = self._make_result(lane_types=["white dashed lane line"])
        out = export_openlane_json(result)
        assert out["lane_lines"][0]["category"] == 3  # white dashed

    def test_category_mapping_unknown(self):
        """Unknown lane type falls back to DEFAULT_LANE_CATEGORY."""
        result = self._make_result(lane_types=["sparkly rainbow line"])
        out = export_openlane_json(result)
        assert out["lane_lines"][0]["category"] == DEFAULT_LANE_CATEGORY

    def test_multiple_lane_types(self):
        """Multiple lane types produce separate lane_lines entries."""
        result = self._make_result(
            lane_types=["white lane line", "yellow lane line"], n_points=4
        )
        out = export_openlane_json(result)
        assert len(out["lane_lines"]) == 2
        assert out["lane_lines"][0]["lane_type"] == "white lane line"
        assert out["lane_lines"][1]["lane_type"] == "yellow lane line"
        assert out["lane_lines"][0]["category"] == 1
        assert out["lane_lines"][1]["category"] == 2

    def test_visibility_all_ones(self):
        """All points come from LiDAR, so visibility should be all 1.0."""
        result = self._make_result(n_points=10)
        out = export_openlane_json(result)
        assert out["lane_lines"][0]["visibility"] == [1.0] * 10

    def test_metadata_passthrough(self):
        """sample_token and camera are included in the output."""
        result = self._make_result()
        out = export_openlane_json(result)
        assert out["sample_token"] == "tok_42"
        assert out["camera"] == "CAM_FRONT"

    def test_write_to_file(self, tmp_path):
        """Output writes valid JSON to disk when output_path is set."""
        import json

        result = self._make_result(n_points=3)
        json_path = tmp_path / "test_export.json"
        export_openlane_json(result, output_path=json_path)
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert len(loaded["lane_lines"]) == 1
        assert len(loaded["lane_lines"][0]["xyz"][0]) == 3

    def test_empty_result_no_lanes(self):
        """FrameLiftResult with no lane points → empty lane_lines list."""
        result = FrameLiftResult(
            sample_token="tok_empty",
            camera="CAM_FRONT",
            lane_points=[],
            total_lidar_points=500,
            total_lane_points=0,
        )
        out = export_openlane_json(result)
        assert out["lane_lines"] == []
        assert out["total_lane_points"] == 0
