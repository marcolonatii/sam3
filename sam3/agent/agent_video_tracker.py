from pydoc import describe
import json
import os
import torch
from torchvision.ops import masks_to_boxes
import numpy as np
from ..visualization_utils import normalize_bbox, prepare_masks_for_visualization
from ..agent.agent_tools import (
    add_prompt_for_session,
    propagate,
    get_frames,
    get_session,
    iou_mask,
    normalized_box_to_mask,
    xywh_to_xyxy,
    visualize_formatted_frame_output,
)
from typing import List, Tuple, Dict
from ..model_builder import build_sam3_video_predictor
from langchain.tools import tool
from tqdm import tqdm

from PIL import Image
from io import BytesIO
import base64


class Frame:
    frame_np: np.ndarray  # not normalized
    frame_pil: Image.Image
    saving_path: str
    frame_idx: int
    img_W: int
    img_H: int

    # object_dictionary: Dict[str, List[int]] # label -> coordinates of the object in the frame
    def _numpy_to_data_url(self, frame_np: np.ndarray):
        # img = Image.fromarray(self.frame_np)  # assumes RGB
        img = Image.fromarray(frame_np)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"

    def to_data_url(self):
        return self._numpy_to_data_url(self.frame_np)

    def __init__(self, frame_np: np.ndarray, saving_path: str):
        self.frame_np = frame_np
        self.frame_pil = Image.fromarray(frame_np)
        self.img_W = frame_np.shape[1]
        self.img_H = frame_np.shape[0]
        self.saving_path = saving_path

    def from_pil(self, frame_pil: Image.Image):
        self.frame_pil = frame_pil
        self.frame_np = np.array(frame_pil)
        self.img_W = frame_pil.width
        self.img_H = frame_pil.height

    def save(self, path=None):
        if path is None:
            path = self.saving_path
        os.makedirs(path, exist_ok=True)
        Image.fromarray(self.frame_np).save(
            os.path.join(path, "frame_" + str(self.frame_idx) + ".png")
        )

    def get_saving_path(self):
        return self.saving_path

    def get_frame_np(self):
        return self.frame_np

    def get_frame_pil(self):
        return self.frame_pil

    def get_img_W(self):
        return self.img_W

    def get_img_H(self):
        return self.img_H

    def get_frame_idx(self):
        return self.frame_idx

    def denormalized_box(self, box: List[float]):
        """
        box in normalized coordinates -> pixel coordinates
        """
        x1, y1, x2, y2 = box
        return [x1 * self.img_W, y1 * self.img_H, x2 * self.img_W, y2 * self.img_H]

    def get_normalized_box(self, box: List[float]):
        """
        box in pixel coordinates -> normalized coordinates
        """
        x1, y1, x2, y2 = box
        return [x1 / self.img_W, y1 / self.img_H, x2 / self.img_W, y2 / self.img_H]

    def cropped_frame_data_url(self, box: List[float]):
        """
        box in normalized coordinates -> data url
        """
        x1, y1, x2, y2 = self.denormalized_box(box)
        return self._numpy_to_data_url(self.frame_np[y1:y2, x1:x2])


# an decorator to note the image in the output string
class ImageDecorator:
    def __init__(self, decorator_str: str = "__image__") -> None:
        self.decorator_str = decorator_str

    def get_decorator_str(self) -> str:
        return self.decorator_str

    # todo: get_images
    def get_image_from_string(self, string: str):
        return string.split(self.decorator_str)[1]
    def get_text_from_string(self, string: str) -> str:
        return string.split(self.decorator_str)[0]

    def append_image_to_string(self, string: str, frame: Frame) -> str:
        return string + self.decorator_str + frame.to_data_url()
    def image_count(self, string: str) -> int:
        return string.count(self.decorator_str)


class DetectedObject:
    id: int
    label: str
    bounding_boxes: Dict[int, List[float]]  # frame_idx -> [x1, y1, x2, y2]
    center_coordinates: Dict[int, List[float]]  # frame_idx -> [x, y]
    masks: Dict[int, np.ndarray]  # frame_idx -> mask
    img_W: int
    img_H: int

    def __init__(
        self,
        label: str,
        id: int,
        img_W: int,
        img_H: int,
        boxes: Dict[int, List[float]] = None,
        masks: Dict[int, np.ndarray] = None,
    ) -> None:
        """
        boxes: [x1, y1, x2, y2] in normalized coordinates of range 0~1
        """
        self.img_W = img_W
        self.img_H = img_H
        self.id = id
        self.label = label
        self.bounding_boxes = boxes if boxes is not None else {}
        self.masks = (
            masks
            if masks is not None
            else {
                frame_idx: normalized_box_to_mask(
                    self.bounding_boxes[frame_idx], self.img_W, self.img_H
                )
                for frame_idx in self.bounding_boxes.keys()
            }
        )
        self.center_coordinates = {
            frame_idx: [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            for frame_idx, box in self.bounding_boxes.items()
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        meta_path = os.path.join(path, f"object_{self.id}.json")
        masks_path = os.path.join(path, f"masks_{self.id}.npz")

        masks_payload = {str(frame_idx): mask for frame_idx, mask in self.masks.items()}
        if masks_payload:
            np.savez_compressed(masks_path, **masks_payload)
        else:
            masks_path = None

        data = {
            "id": int(self.id),
            "label": self.label,
            "img_W": int(self.img_W),
            "img_H": int(self.img_H),
            "bounding_boxes": {
                str(k): [float(x) for x in v] for k, v in self.bounding_boxes.items()
            },
            "center_coordinates": {
                str(k): [float(x) for x in v]
                for k, v in self.center_coordinates.items()
            },
            # relative path to the json file
            "masks_path": masks_path.split("/")[-1],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, id: int) -> "DetectedObject":
        meta_path = os.path.join(path, f"object_{id}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        bounding_boxes = {int(k): v for k, v in data.get("bounding_boxes", {}).items()}
        masks: Dict[int, np.ndarray] = {}
        masks_path = data.get("masks_path")
        # todo: fix the path format
        if masks_path:
            # Resolve relative paths relative to the JSON file's directory
            if not os.path.isabs(masks_path):
                masks_path = os.path.join(path, masks_path)
            if os.path.exists(masks_path):
                with np.load(masks_path, allow_pickle=False) as npz:
                    for key in npz.files:
                        masks[int(key)] = npz[key]

        return cls(
            label=data.get("label", ""),
            id=data.get("id", 0),
            img_W=data.get("img_W", 0),
            img_H=data.get("img_H", 0),
            boxes=bounding_boxes,
            masks=masks,
        )

    def from_outputs_per_frame(self, outputs_per_frame, need_box: bool = True):
        for frame_idx, output in tqdm(outputs_per_frame.items()):
            for obj_id, binary_mask in output.items():
                if obj_id != self.id:
                    continue
                if not isinstance(binary_mask, torch.Tensor):
                    binary_mask = torch.tensor(binary_mask)
                if not binary_mask.any():
                    continue
                if need_box:
                    box_xyxy = masks_to_boxes(binary_mask.unsqueeze(0)).squeeze()
                    box_xyxy = normalize_bbox(box_xyxy, self.img_W, self.img_H)
                    self.add_box(frame_idx, box_xyxy)
                self.add_mask(frame_idx, binary_mask)

    def add_box(self, frame_idx, box):
        self.bounding_boxes[frame_idx] = box
        self.center_coordinates[frame_idx] = [
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
        ]

    def add_mask(self, frame_idx, mask):
        self.masks[frame_idx] = mask

    def get_mask(self, frame_idx):
        return self.masks[frame_idx]

    def get_box(self, frame_idx):
        return self.bounding_boxes[frame_idx]

    def get_label(self):
        return self.label

    def get_center(self, frame_idx):
        box = self.bounding_boxes[frame_idx]
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def near(self, frame_idx, object, radius):
        # todo: possible to get key error
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
        # todo: codex's code
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

        vertical_touch = y_overlap and (
            abs(x1_max - x2_min) <= threshold or abs(x2_max - x1_min) <= threshold
        )
        horizontal_touch = x_overlap and (
            abs(y1_max - y2_min) <= threshold or abs(y2_max - y1_min) <= threshold
        )
        return vertical_touch or horizontal_touch

    def overlapping(self, frame_idx, object, threshold):
        try:
            mask1 = self.get_mask(frame_idx)
            mask2 = object.get_mask(frame_idx)
        except KeyError:
            return False
        # todo: shouldn't we use iom_mask?
        return iou_mask(mask1, mask2) > threshold

    def update_bounding_boxes(self, bounding_box: Dict[int, List[float]]):
        for frame_idx, box in bounding_box.items():
            self.add_box(frame_idx, box)

    def update_masks(self, masks: Dict[int, np.ndarray]):
        for frame_idx, mask in masks.items():
            self.add_mask(frame_idx, mask)


# We have to learn which objects are interacting with each other
class ObjectList:
    objects: List[DetectedObject] = []

    def __init__(self):
        self.objects = []

    def from_outputs_per_frame(self, outputs_per_frame, need_box: bool = True):
        for obj in self.objects:
            obj.from_outputs_per_frame(outputs_per_frame, need_box=need_box)

    def add_object(self, obj: DetectedObject):
        if not self.contains_object(obj):
            self.objects.append(obj)

    def save_objects(self, path: str):
        os.makedirs(path, exist_ok=True)
        for obj in self.objects:
            obj.save(path)

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

    def get_object_by_label(self, label: str):
        for o in self.objects:
            if o.label == label:
                return o

    def get_object_by_id(self, object_id: int):
        for o in self.objects:
            if o.id == object_id:
                return o

    def new_id(self):
        return len(self.objects)

    # todo: would this cause memory leak? May need to deep copy
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
                f"  - id={o.id}, label={o.label}, tracked: {len(o.bounding_boxes) > 1}"
            )
        return "\n".join(lines)

    # Define the tool


json_schema = {"total_pullup_count": "<number>"}
agent_system_msg = f"""
You are doing sport analysis on videos. Proceed with the tools
1.List the objects of interest you want to track in order to answer the question
2.Verify the object are tracked successfully by calling get_tracked_objects_info
4.propagate the video with the functions
5.Then after you get the tracks of the objects, use tools to analyze the position of the objects
6.return your answer in <answer> ... <answer>
7.the output format should be in json format with {json_schema}"""


class Sam3TrackingTool:
    def __init__(self, video_path: str, bpe_path: str) -> None:
        self.predictor = build_sam3_video_predictor(bpe_path=bpe_path)
        self.video_path = video_path
        self.video_frames_for_vis = get_frames(self.video_path)
        self.session_id = get_session(self.predictor, self.video_path)
        self.object_list = ObjectList()
        self.object_to_track = ObjectList()
        if isinstance(self.video_frames_for_vis[0], np.ndarray):
            self.frame_dict = {
                frame_idx: Frame(
                    frame_np=self.video_frames_for_vis[frame_idx],
                    saving_path=os.path.join(
                        video_path, "frames", f"frame_{frame_idx}.png"
                    ),
                )
                for frame_idx in range(len(self.video_frames_for_vis))
            }
        else:
            self.frame_dict = {
                frame_idx: Frame(
                    frame_np=np.array(Image.open(self.video_frames_for_vis[frame_idx])),
                    saving_path=os.path.join(
                        video_path, "frames", f"frame_{frame_idx}.png"
                    ),
                )
                for frame_idx in range(len(self.video_frames_for_vis))
            }
        self.prompt = None

        # debug purpose
        self.outputs_per_frame = None
        self.image_decorator = ImageDecorator()

    # todo: recursively refine the object list
    def _add_prompt(
        self,
        prompt_text_str: str,
        bounding_boxes: List[List[float]] = None,
        bounding_box_labels: List[int] = None,
    ) -> None:
        # todo: add objects here
        self.object_to_track = ObjectList()
        self.prompt = prompt_text_str
        response = add_prompt_for_session(
            predictor=self.predictor,
            prompt_text_str=prompt_text_str,
            frame_idx=0,
            bounding_boxes=bounding_boxes,
            bounding_box_labels=bounding_box_labels,
            obj_ids=[],
            session_id=self.session_id,
            video_frames_for_vis=self.video_frames_for_vis,
        )
        for i in range(len(response["outputs"]["out_obj_ids"])):
            self.object_to_track.add_object(
                DetectedObject(
                    label=self.prompt,
                    id=response["outputs"]["out_obj_ids"][i],
                    img_W=self.video_frames_for_vis[0].shape[1],
                    img_H=self.video_frames_for_vis[0].shape[0],
                    boxes={0: xywh_to_xyxy(response["outputs"]["out_boxes_xywh"][i])},
                    masks={0: response["outputs"]["out_binary_masks"][i]},
                )
            )
        return response

    def _reset_session(self) -> None:
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=self.session_id,
            )
        )

    # note: don't delete this
    def _propagate(self, need_box: bool = True) -> None:
        outputs_per_frame = propagate(
            self.predictor, self.session_id, self.video_frames_for_vis
        )

        # Add any new objects from outputs_per_frame that aren't in object_to_track
        all_obj_ids = set()
        for output in outputs_per_frame.values():
            all_obj_ids.update(output.keys())
        tracked_ids = {obj.id for obj in self.object_to_track.objects}
        new_ids = all_obj_ids - tracked_ids

        img_H, img_W = (
            self.video_frames_for_vis[0].shape[0],
            self.video_frames_for_vis[0].shape[1],
        )
        self.object_to_track.from_outputs_per_frame(outputs_per_frame, need_box=need_box)
        self.object_list.merge(self.object_to_track)
        for obj_id in new_ids:
            new_obj = DetectedObject(
                label=self.prompt or "",
                id=obj_id,
                img_W=img_W,
                img_H=img_H,
                boxes={},
                masks={},
            )
            new_obj.from_outputs_per_frame(outputs_per_frame, need_box=need_box)
            self.object_list.add_object(new_obj)

        self.outputs_per_frame = outputs_per_frame

    def _get_all_objects(self) -> ObjectList:
        return self.object_list

    def _get_object_to_track(self) -> ObjectList:
        return self.object_to_track

    def _save_objects(self, path: str) -> None:
        self.object_to_track.save_objects(path)

    def _restart_session(self) -> None:
        self.session_id = get_session(self.predictor, self.video_path)
        self.object_to_track = ObjectList()

    def _get_session_id(self) -> str:
        return self.session_id

    def _get_video_path(self) -> str:
        return self.video_path

    def _get_video_frames_for_vis(self) -> List[np.ndarray]:
        return self.video_frames_for_vis

    def _detect_interaction_by_label(
        self, object1: str, object2: str, interaction_type: str, threshold: float = 0.05
    ) -> List[int]:
        # todo: multiple objects with the same label
        obj1 = self.object_list.get_object_by_label(object1)
        obj2 = self.object_list.get_object_by_label(object2)
        if obj1 is None or obj2 is None:
            return []
        fn = getattr(obj1, interaction_type)
        return [
            frame_idx
            for frame_idx in range(len(self.video_frames_for_vis))
            if fn(frame_idx, obj2, threshold)
        ]

    def _save_visualizations(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for frame_idx in range(len(self.video_frames_for_vis)):
            visualize_formatted_frame_output(
                frame_idx=frame_idx,
                video_frames=self.video_frames_for_vis,
                outputs_list=self.outputs_per_frame,
                titles=["SAM 3 Dense Tracking outputs"],
                figsize=(6, 4),
                show=False,
                save_path=os.path.join(path, f"frame_{frame_idx}.png"),
            )

    def _llm_tools(self):
        add_prompt_description = """
            Add a prompt to the SAM3 video tracker, \
            input boxes are expected to be [xmin, ymin, width, height] format\
            in normalized coordinates of range 0~1, \
            bounding_box_labels should be a list of integers, 1 stands including the object, 0 stands excluding the object,\
            the output will be saved in ./frames_output/frame_0.png \
            """
        add_prompt_description_temp = """
            prompt to identify a single object in the video, \
            the text prompt should be noun of a single object\
        """

        @tool(description="Get the information of the tracked objects")
        def get_tracked_objects_info() -> str:
            return self._get_all_objects().__str__()

        @tool(description="Get the list of objects you are tracking")
        def get_tracking_objects() -> str:
            return self._get_object_to_track().__str__()

        @tool(description=add_prompt_description_temp)
        # def add_prompt(prompt_text_str: str, bounding_boxes: List[List[float]] = None, bounding_box_labels: List[str] = None) -> str:
        def identify_object_by_prompt(prompt_text_str: str) -> str:
            response = self._add_prompt(prompt_text_str)
            msg = "objects identified: \n" + "".join(
                self._get_object_to_track().__str__()
            )
            output_lists = prepare_masks_for_visualization({0: response["outputs"]})
            annotated_img = visualize_formatted_frame_output(
                frame_idx=0,
                video_frames=self.video_frames_for_vis,
                outputs_list=output_lists,
                titles=None,
                figsize=(6, 4),
                show=False,
            )
            return self.image_decorator.append_image_to_string(msg, Frame(annotated_img, ""))
        @tool(description="Reset the video tracker session")
        def reset_tracker() -> str:
            self._reset_session()
            return "Session reset successfully"

        @tool(description="Track the object in the video")
        def track_objects(object_name: str) -> str:
            response = self._add_prompt(object_name)
            self._propagate()
            return "Object " + object_name + " tracked successfully"

        @tool(
            description="Detect interaction (near, above, below, colliding) between two objects, return frames if the interaction happens"
        )
        def detect_interaction(
            object1_id: int,
            object2_id: int,
            interaction_type: str,
            threshold: float = 0.05,
        ) -> str:
            obj1 = self.object_list.get_object_by_id(object1_id)
            obj2 = self.object_list.get_object_by_id(object2_id)
            if obj1 is None or obj2 is None:
                return "Objects not found"
            fn = getattr(obj1, interaction_type)
            return str(
                [
                    frame_idx
                    for frame_idx in range(len(self.video_frames_for_vis))
                    if fn(frame_idx, obj2, threshold)
                ]
            )

        @tool(description="Get the frame by frame index")
        def get_frame(frame_idx: int) -> str:
            if frame_idx not in self.frame_dict:
                return "Frame not found"
            return self.frame_dict[frame_idx].to_data_url()

        @tool(description="Get the bounding box of an object by object id")
        def get_object_boudingbox(object_id: int, frame_idx: int) -> str:
            return self.object_list.get_object_by_id(object_id).get_box(frame_idx)

        return [
            get_tracking_objects,
            identify_object_by_prompt,
            reset_tracker,
            track_objects,
            detect_interaction,
            get_frame,
            get_object_boudingbox,
            get_tracked_objects_info,
        ]
