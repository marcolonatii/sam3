from dataclasses import dataclass
from typing import Final

from kits.ml.argo.labels.object_label_definitions import UnifiedSegmentationLabels
from kits.ml.labels.lane_line_definitions import LaneLineType
from kits.ml.sam.data_types import SAM3MaskType, SAMType

# Map each LaneLineType to a SAM3 text prompt.
# For better results, use "dashed" instead of OpenLabel's "Broken", "double solid" instead of "DoubleSolid".
LANE_LINE_TYPE_TO_PROMPT: Final[dict[LaneLineType, str]] = {
    LaneLineType.SOLID_YELLOW: "solid yellow lane line",
    LaneLineType.SOLID_WHITE: "solid white lane line",
    LaneLineType.DASHED_YELLOW: "dashed yellow lane line",
    LaneLineType.DASHED_WHITE: "dashed white lane line",
    LaneLineType.DOUBLE_SOLID_YELLOW: "double solid yellow lane line",
    LaneLineType.DOUBLE_SOLID_WHITE: "double solid white lane line",
    LaneLineType.DOUBLE_DASHED_YELLOW: "double dashed yellow lane line",
    LaneLineType.DOUBLE_DASHED_WHITE: "double dashed white lane line",
    LaneLineType.SOLID_DASHED_YELLOW: "solid dashed yellow lane line",
    LaneLineType.SOLID_DASHED_WHITE: "solid dashed white lane line",
    LaneLineType.DASHED_SOLID_YELLOW: "dashed solid yellow lane line",
    LaneLineType.DASHED_SOLID_WHITE: "dashed solid white lane line",
    LaneLineType.DOTTED_YELLOW: "dotted yellow lane line",
    LaneLineType.DOTTED_WHITE: "dotted white lane line",
}


@dataclass(frozen=True)
class LaneLabelingSAM3Config:
    """SAM3 configuration specific to lane line autolabeling."""

    #: Which lane line types to detect.  Defaults to all OpenLabel types.
    lane_line_types: tuple[LaneLineType, ...] = tuple(LaneLineType)

    #: Confidence threshold for filtering SAM3 detections.
    confidence_threshold: float = 0.5

    #: SAM model type for lane-line autolabeling.
    LANELINE_SAM_TYPE: Final[SAMType] = SAMType.SAM3

    #: Recommended SAM3 output mask type for lane-line visualization.
    #: INSTANCE mode returns a separate mask per prompt/detection, so each lane-line
    #: type gets its own mask and can be rendered in a distinct color.
    LANELINE_SAM3_MASK_TYPE: Final[SAM3MaskType] = SAM3MaskType.INSTANCE

    def __post_init__(self) -> None:
        """Validate that every requested type has a SAM3 prompt defined."""
        unsupported = set(self.lane_line_types) - LANE_LINE_TYPE_TO_PROMPT.keys()
        if unsupported:
            raise ValueError(
                f"No SAM3 prompt defined for lane line type(s): "
                f"{sorted(t.name for t in unsupported)}. "
                f"Add entries to LANE_LINE_TYPE_TO_PROMPT or remove them from lane_line_types."
            )

    @property
    def prompts(self) -> list[str]:
        """Get SAM3 text prompts for the configured lane line types."""
        return [LANE_LINE_TYPE_TO_PROMPT[lt] for lt in self.lane_line_types]

    @property
    def prompt_to_lane_line_type(self) -> dict[str, LaneLineType]:
        """Reverse mapping: SAM3 prompt → LaneLineType for post-processing."""
        return {LANE_LINE_TYPE_TO_PROMPT[lt]: lt for lt in self.lane_line_types}


def build_laneline_segmentation_class_prompts(
    lane_line_types: tuple[LaneLineType, ...] | None = None,
) -> dict[UnifiedSegmentationLabels, set[str]]:
    if lane_line_types is None:
        lane_line_types = tuple(LANE_LINE_TYPE_TO_PROMPT.keys())

    return {
        UnifiedSegmentationLabels.ROAD_SURFACE_MARKING: {LANE_LINE_TYPE_TO_PROMPT[lt] for lt in lane_line_types},
    }