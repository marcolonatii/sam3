import torch
from torchvision.ops import masks_to_boxes
import numpy as np
from ..visualization_utils import normalize_bbox
from ..agent.agent_tools import add_prompt_for_session, propagate, get_frames, get_session
from typing import List, Tuple, Dict
from ..model_builder import build_sam3_video_predictor
from langchain.tools import tool
class DetectedObject:
    id: int
    label: str
    bounding_boxes: Dict[int, List[float]] # frame_idx -> [x1, y1, x2, y2]
    center_coordinates: Dict[int, List[float]] # frame_idx -> [x, y]
    img_W: int
    img_H: int
    def __init__(self, label: str, id: int, boxes: Dict[int, List[float]] = {}) -> None:
      self.id = id
      self.label = label
      self.bounding_boxes = boxes
      self.center_coordinates =  [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes.values()]
    def from_outputs_per_frame(self, outputs_per_frame):
      for frame_idx, output in outputs_per_frame.items():
        for obj_id, binary_mask in output.items():
          if obj_id != self.id:
            continue
          if not isinstance(binary_mask, torch.Tensor):
            binary_mask = torch.tensor(binary_mask)
          if not binary_mask.any():
            continue
          box_xyxy = masks_to_boxes(binary_mask.unsqueeze(0)).squeeze()
          box_xyxy = normalize_bbox(box_xyxy, self.img_W, self.img_H)
          self.add_box(frame_idx, box_xyxy)

    def add_box(self, frame_idx, box):
      self.bounding_boxes[frame_idx] = box
      self.center_coordinates[frame_idx] = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    def get_box(self, frame_idx):
      return self.bounding_boxes[frame_idx]
    def get_label(self):
      return self.label
    def get_center(self, frame_idx):
      box = self.bounding_boxes[frame_idx]
      return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    def near(self, frame_idx, object, radius):
      center1 = self.get_center(frame_idx)
      center2 = object.get_center(frame_idx)
      return np.linalg.norm(np.array(center1) - np.array(center2)) < radius



#We have to learn which objects are interacting with each other
class ObjectList:
    objects: List[DetectedObject] = []
    def __init__(self):
        self.objects = []
    def from_outputs_per_frame(self, outputs_per_frame):
      for obj in self.objects:
        obj.from_outputs_per_frame(outputs_per_frame)
    def add_object(self, obj: DetectedObject):
        self.objects.append(obj)
    def contains_object(self, obj: DetectedObject):
        for o in self.objects:
          if o.id == obj.id:
            return True
        return False
    def contains_object_str(self, obj: str):
        for o in self.objects:
          if o.label == obj:
            return True
        return False
    def get_objects(self):
        return self.objects
    def get_object(self, label:str):
      for o in self.objects:
        if o.label == label:
          return o
    def __str__(self) -> str:
      if not self.objects:
          return "ObjectList(empty)"

      lines = ["ObjectList:"]
      for o in self.objects:
          lines.append(
              f"  - id={o.id}, label={o.label}"
          )
      return "\n".join(lines)

    # Define the tool


class Sam3TrackingTool:
    def __init__(self, video_path: str) -> None:
        self.predictor = build_sam3_video_predictor()
        self.video_path = video_path
        self.video_frames_for_vis = get_frames(self.video_path)
        self.session_id = get_session(self.predictor, self.video_path)
        self.object_list = ObjectList()

        #debug purpose
        self.outputs_per_frame = None

    def _add_prompt(self, prompt_text_str: str, bounding_boxes: List[List[float]] = None, bounding_box_labels: List[str] = None) -> None:
        add_prompt_for_session(self.predictor, prompt_text_str, bounding_boxes, bounding_box_labels, self.session_id, self.video_frames_for_vis)
    def _reset_session(self) -> None:
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=self.session_id,
            )
        )
    def _propagate(self) -> None:
        outputs_per_frame = propagate(self.predictor, self.session_id, self.video_frames_for_vis)
        self.object_list.from_outputs_per_frame(outputs_per_frame)
        self.outputs_per_frame = outputs_per_frame
    def _get_object_list(self) -> ObjectList:
        return self.object_list
    def _get_session_id(self) -> str:
        return self.session_id
    def _get_video_path(self) -> str:
        return self.video_path
    def _get_video_frames_for_vis(self) -> List[np.ndarray]:
        return self.video_frames_for_vis
    # def _llm_tools(self):
    #     @tool
    #     def get_object_list(self) -> ObjectList:
    #         return self._get_object_list()
    #     @tool
    #     def add_prompt(self, prompt_text_str: str, bounding_boxes: List[List[float]] = None, bounding_box_labels: List[str] = None) -> None:
    #         self._add_prompt(prompt_text_str, bounding_boxes, bounding_box_labels)
    #     @tool
    #     def reset_session(self) -> None:
    #         self._reset_session()
    #     @tool
    #     def propagate(self) -> None:
    #         self._propagate()
    #     return [get_object_list, add_prompt, reset_session, propagate]