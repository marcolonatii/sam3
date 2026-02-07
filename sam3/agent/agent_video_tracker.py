from pydoc import describe
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
    def __init__(self, label: str, id: int, img_W: int, img_H: int, boxes: Dict[int, List[float]] = None) -> None:
      self.img_W = img_W
      self.img_H = img_H
      self.id = id
      self.label = label
      self.bounding_boxes = boxes if boxes is not None else {}
      self.center_coordinates = {
        frame_idx: [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        for frame_idx, box in self.bounding_boxes.items()
      }
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
      #todo: possible to get key error
      try:
        center1 = self.get_center(frame_idx)
        center2 = object.get_center(frame_idx)
      except KeyError:
        return False
      return np.linalg.norm(np.array(center1) - np.array(center2)) < radius
    def above(self, frame_idx, object, threshold):
      try:
        center1 = self.get_center(frame_idx)
        center2 = object.get_center(frame_idx)
      except KeyError:
        return False
      return center1[1] > center2[1] + threshold
    def below(self, frame_idx, object, threshold):
      try:
        center1 = self.get_center(frame_idx)
        center2 = object.get_center(frame_idx)
      except KeyError:
        return False
      return center1[1] < center2[1] - threshold
    def colliding(self, frame_idx, object, threshold):
      #todo: codex's code
      try:
        box1 = self.get_box(frame_idx)
        box2 = object.get_box(frame_idx)
      except KeyError:
        return False
      x1_min, y1_min, x1_max, y1_max = box1
      x2_min, y2_min, x2_max, y2_max = box2

      x_overlap = min(x1_max, x2_max) >= max(x1_min, x2_min)
      y_overlap = min(y1_max, y2_max) >= max(y1_min, y2_min)

      if x_overlap and y_overlap:
        return True

      vertical_touch = (
        y_overlap
        and (
          abs(x1_max - x2_min) <= threshold
          or abs(x2_max - x1_min) <= threshold
        )
      )
      horizontal_touch = (
        x_overlap
        and (
          abs(y1_max - y2_min) <= threshold
          or abs(y2_max - y1_min) <= threshold
        )
      )
      return vertical_touch or horizontal_touch




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
    def label_objects(self, labels: Dict[int, str]):
        for obj_id, label in labels.items():
            for o in self.objects:
                if o.id == obj_id:
                    o.label = label
                    break
    def contains_object(self, obj: DetectedObject):
        for o in self.objects:
          if o.id == obj.id:
            return True
        return False
    def contains_object_str(self, label: str):
        for o in self.objects:
          if o.label == label:
            return True
        return False
    def get_objects(self):
        return self.objects
    def get_object_by_label(self, label:str):
      for o in self.objects:
        if o.label == label:
          return o
    def new_id(self):
        return len(self.objects)
    #todo: would this cause memory leak? May need to deep copy
    def merge(self, other: "ObjectList"):
        for o in other.objects:
            if self.contains_object(o):
                o.id = self.new_id()
            self.add_object(o)
    def __str__(self) -> str:
      if not self.objects:
          return "ObjectList(empty)"

      lines = ["ObjectList:"]
      for o in self.objects:
          lines.append(
              f"  - id={o.id}, label={o.label}, tracked: {o.bounding_boxes is not None}"
          )
      return "\n".join(lines)

    # Define the tool


from PIL import Image
from io import BytesIO
import base64

class Frame:
    frame_np: np.ndarray
    saving_path: str
    frame_idx: int
    def _numpy_to_data_url(self, frame_np):
        img = Image.fromarray(self.frame_np)  # assumes RGB
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"
    def __init__(self, frame_np: np.ndarray, saving_path: str):
        self.frame_np = frame_np
    def to_data_url(self):
        return self._numpy_to_data_url()
    def save(self):
        Image.fromarray(self.frame_np).save(self.saving_path+"/frame_"+str(self.frame_idx)+".png")



    




class Sam3TrackingTool:
    def __init__(self, video_path: str, bpe_path: str) -> None:
        self.predictor = build_sam3_video_predictor(bpe_path=bpe_path)
        self.video_path = video_path
        self.video_frames_for_vis = get_frames(self.video_path)
        self.session_id = get_session(self.predictor, self.video_path)
        self.object_list = ObjectList()

        #debug purpose
        self.outputs_per_frame = None

    #todo: recursively refine the object list
    def _add_prompt(self, prompt_text_str: str, bounding_boxes: List[List[float]] = None, bounding_box_labels: List[int] = None) -> None:
        #todo: add objects here
        response = add_prompt_for_session(self.predictor, prompt_text_str, bounding_boxes, bounding_box_labels, self.session_id, self.video_frames_for_vis)
        return response
    def _reset_session(self) -> None:
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=self.session_id,
            )
        )
    def _propagate(self) -> None:
        outputs_per_frame = propagate(self.predictor, self.session_id, self.video_frames_for_vis)
        new_objects = ObjectList()
        new_objects.from_outputs_per_frame(outputs_per_frame)
        self.object_list.merge(new_objects)
        self.outputs_per_frame = outputs_per_frame
    def _get_object_list(self) -> ObjectList:
        return self.object_list
    def _get_session_id(self) -> str:
        return self.session_id
    def _get_video_path(self) -> str:
        return self.video_path
    def _get_video_frames_for_vis(self) -> List[np.ndarray]:
        return self.video_frames_for_vis
    def _detect_interaction(self, object1: str, object2: str, interaction_type: str, threshold: float = 0.05) -> List[int]:
        obj1 = self.object_list.get_object_by_label(object1)
        obj2 = self.object_list.get_object_by_label(object2)
        if obj1 is None or obj2 is None:
            return []
        fn = getattr(obj1, interaction_type)
        return [frame_idx for frame_idx in range(len(self.video_frames_for_vis)) if fn(frame_idx, obj2, threshold)]
            
    def _llm_tools(self):
        add_prompt_description = """
            Add a prompt to the SAM3 video tracker, \
            input boxes are expected to be [xmin, ymin, width, height] format\
            in normalized coordinates of range 0~1, \
            bounding_box_labels should be a list of integers, 1 stands including the object, 0 stands excluding the object,\
            the output will be saved in ./frames_output/frame_0.png \
            """
        add_prompt_description_temp = """
            text prompt to add to SAM3 video tracker, \
            the text prompt should be nouns separated by comma, \
            for example, "person, car, tree" \
            the output will be saved in ./frames_output/frame_0.png \
        """
        @tool(description="Get the list of objects detected in the video")
        def get_object_list() -> str:
            return "".join(self._get_object_list().__str__())
        @tool(description=add_prompt_description_temp)
        # def add_prompt(prompt_text_str: str, bounding_boxes: List[List[float]] = None, bounding_box_labels: List[str] = None) -> str:
        def add_prompt(prompt_text_str: str) -> str:
            response = self._add_prompt(prompt_text_str)
            for obj_id in response['outputs']['out_obj_ids']:
                #todo: add box for first frame
                self.object_list.add_object(DetectedObject(label="", id=obj_id, img_W=self.video_frames_for_vis[0].shape[1], img_H=self.video_frames_for_vis[0].shape[0]))
            #todo: shouldn't return success if no objects are added
            return "now the tracked objects are: \n" + "".join(self._get_object_list().__str__())
        @tool(description="Reset the video tracker session")
        def reset_session() -> str:
            self._reset_session()
            return "Session reset successfully"
        @tool(description="Propagate the video tracker")
        def propagate() -> str:
            self._propagate()
            return "Propagated successfully"
        @tool(description="Detect interaction (near, above, below, colliding) between two objects, return frames if the interaction happens")
        def detect_interaction(object1: str, object2: str, interaction_type: str, threshold: float = 0.05) -> str:
            if self.object_list.contains_object_str(object1) and self.object_list.contains_object_str(object2):
                return ",".join(self._detect_interaction(object1, object2, interaction_type, threshold))
            else:
                return "Objects not found"
        return [get_object_list, add_prompt, reset_session, propagate, detect_interaction]