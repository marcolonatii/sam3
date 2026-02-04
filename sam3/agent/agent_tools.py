# from msvcrt import open_osfhandle
import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
# Use relative imports since we're working with local source code
# from .. import build_sam3_image_model
from ..model.box_ops import box_xywh_to_cxcywh
from ..model.sam3_image_processor import Sam3Processor
from ..model_builder import build_sam3_video_predictor
from ..model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from ..visualization_utils import (
    draw_box_on_image,
    normalize_bbox,
    plot_results,
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
    plot_bbox,
    plot_mask
)
from torchvision.ops import masks_to_boxes
# predictor = build_sam3_video_predictor()

import matplotlib.pyplot as plt
import torch
from torchvision.ops import masks_to_boxes
import numpy as np

import cv2



# font size for axes titles
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12

def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    # print(outputs_per_frame)

    return outputs_per_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


def visualize_formatted_frame_output(
    frame_idx,
    video_frames,
    outputs_list,
    titles=None,
    points_list=None,
    points_labels_list=None,
    figsize=(12, 8),
    title_suffix="",
    prompt_info=None,
    save_path=None,
):
    """
    Visualize segmentation masks on a video frame and optionally save to file.
    """
    # --- Handle outputs_list ---
    if isinstance(outputs_list, dict) and frame_idx in outputs_list:
        outputs_list = [outputs_list]
    elif isinstance(outputs_list, dict) and not any(isinstance(k, int) for k in outputs_list.keys()):
        outputs_list = [{frame_idx: outputs_list}]

    num_outputs = len(outputs_list)
    if titles is None:
        titles = [f"Set {i+1}" for i in range(num_outputs)]
    assert len(titles) == num_outputs, "titles length must match outputs_list"

    fig, axes = plt.subplots(1, num_outputs, figsize=figsize)
    if num_outputs == 1:
        axes = [axes]

    img = video_frames[frame_idx]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img_H, img_W = img.shape[:2]

    # --- Colormap ---
    cmap = plt.cm.get_cmap("tab20")  # 20 distinct colors

    for idx in range(num_outputs):
        ax, outputs_set, ax_title = axes[idx], outputs_list[idx], titles[idx]
        ax.set_title(f"Frame {frame_idx} - {ax_title}{title_suffix}")
        ax.imshow(img)

        if frame_idx in outputs_set:
            _outputs = outputs_set[frame_idx]
        else:
            print(f"Warning: Frame {frame_idx} not in outputs_set")
            continue

        objects_drawn = 0
        for obj_id, binary_mask in _outputs.items():
            # Convert to torch tensor if needed
            if not isinstance(binary_mask, torch.Tensor):
                binary_mask = torch.tensor(binary_mask)

            if not binary_mask.any():
                continue

            # Bounding box
            box_xyxy = masks_to_boxes(binary_mask.unsqueeze(0)).squeeze()
            box_xyxy = normalize_bbox(box_xyxy, img_W, img_H)

            # Use colormap instead of global COLORS
            color = cmap(int(obj_id) % 20)[:3]

            plot_bbox(
                img_H,
                img_W,
                box_xyxy,
                text=f"(id={obj_id})",
                box_format="XYXY",
                color=color,
                ax=ax,
            )

            mask_np = binary_mask.numpy()
            plot_mask(mask_np, color=color, ax=ax)
            objects_drawn += 1

        if objects_drawn == 0:
            ax.text(
                0.5,
                0.5,
                "No objects detected",
                transform=ax.transAxes,
                fontsize=16,
                ha="center",
                va="center",
                color="red",
                weight="bold",
            )

        # Draw additional points if provided
        if points_list is not None and points_list[idx] is not None:
            show_points(points_list[idx], points_labels_list[idx], ax=ax, marker_size=200)

        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved frame {frame_idx} to {save_path}")
    else:
        plt.show()


# load "video_frames_for_vis" for visualization purposes (they are not used by the model)
# video_path="/content/football.mp4"
def get_frames(video_path):
  if isinstance(video_path, str) and video_path.endswith(".mp4"):
      cap = cv2.VideoCapture(video_path)
      video_frames_for_vis = []
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      cap.release()
  else:
      video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
      try:
          # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
          video_frames_for_vis.sort(
              key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
          )
      except ValueError:
          # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
          print(
              f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
              f"falling back to lexicographic sort."
          )
          video_frames_for_vis.sort()
  return video_frames_for_vis



def get_session(predictor, video_path):
  response = predictor.handle_request(
      request=dict(
          type="start_session",
          resource_path=video_path,
          stride=5
      )
  )
  session_id = response["session_id"]
  return session_id
# prompt_text_str = "player in white"
def add_prompt_for_session(predictor, prompt_text_str, bounding_boxes, bounding_box_labels, session_id, video_frames_for_vis):
  frame_idx = 0  # add a text prompt on frame 0
  response = predictor.handle_request(
      request=dict(
          type="add_prompt",
          session_id=session_id,
          frame_index=frame_idx,
          text=prompt_text_str,
          bounding_boxes=bounding_boxes,
          bounding_box_labels=bounding_box_labels
      )
  )
  out = response["outputs"]

  plt.close("all")
  visualize_formatted_frame_output(
      frame_idx,
      video_frames_for_vis,
      outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
      titles=["SAM 3 Dense Tracking outputs"],
      figsize=(6, 4),
  )
  return response




def propagate(predictor, session_id, video_frames_for_vis):
  # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
  outputs_per_frame = propagate_in_video(predictor, session_id)

  # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
  outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

  vis_frame_stride = 60
  plt.close("all")
  for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
      visualize_formatted_frame_output(
          frame_idx,
          video_frames_for_vis,
          outputs_list=[outputs_per_frame],
          titles=["SAM 3 Dense Tracking outputs"],
          figsize=(6, 4),
      )
  return outputs_per_frame



