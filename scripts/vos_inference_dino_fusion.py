"""
VOS inference with SAM3 + DINOv3 cross-attention fusion.

This script is the SAM3/SAM3.1 analogue of SAM2's vos_inference_dino_fusion.py.
It runs semi-supervised VOS inference (mask prompt on the first frame, propagate
to all other frames) with the DINOv3 fusion modules attached to the tracker.

The script directly uses the underlying Sam3TrackerPredictor API (same as SAM2's
SAM2VideoPredictor), bypassing the higher-level SAM3VideoInference detector wrapper,
because:
  (a) VOS evaluation requires mask (not text) prompts.
  (b) The DINO fusion is injected inside Sam3TrackerBase.track_step, so we need to
      call that code path directly.

Two checkpoints are required:
  1. A pretrained SAM3/SAM3.1 checkpoint (auto-downloaded from HuggingFace if omitted).
     Loaded with strict=False so the new DINO modules are initialised fresh.
  2. A DINO fusion checkpoint produced by scripts/train_dino_fusion.py, which stores
     "dino_encoder" and "cross_attention_fuser" state dicts.

The DINO fusion is activated automatically: build_sam3_predictor(..., use_dino_fusion=True,
dino_checkpoint_path=...) calls tracker.set_dino_fusion(dino_encoder, cross_attn_fuser),
and the fusion runs on every frame inside track_step.

Usage:
  # Run from /home/marcol01/sam3/
  python scripts/vos_inference_dino_fusion.py \\
      --version sam3 \\
      --dino_fusion_checkpoint ./checkpoints_dino_fusion/dino_fusion_best.pt \\
      --base_video_dir /path/to/videos \\
      --input_mask_dir /path/to/annotations \\
      --output_mask_dir /path/to/output

  # SAM3.1 (multiplex) variant:
  python scripts/vos_inference_dino_fusion.py \\
      --version sam3.1 \\
      --dino_fusion_checkpoint ./checkpoints_dino_fusion_multiplex/dino_fusion_best.pt \\
      --base_video_dir /path/to/videos \\
      --input_mask_dir /path/to/annotations \\
      --output_mask_dir /path/to/output

Notes on SAM3 vs SAM3.1:
  - SAM3 uses build_sam3_video_model; the tracker lives at model.tracker
    (a Sam3TrackerPredictor that inherits Sam3TrackerBase).
  - SAM3.1 uses build_sam3_multiplex_video_predictor; the tracker lives at
    predictor.model.tracker.model (a VideoTrackingDynamicMultiplex, which inherits
    Sam3TrackerBase via VideoTrackingMultiplex).
  - The DINO fusion is injected into whichever Sam3TrackerBase instance is found
    by _attach_dino_fusion_sam3 / _attach_dino_fusion_sam31 inside model_builder.py.
"""

import argparse
import os
import sys

# Ensure the sam3 repo root is on the path when running as a script
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import getpass
import numpy as np
import torch
from PIL import Image


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = (
    b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80"
    b"\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80"
    b"\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00"
    b"\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0"
    b"\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80"
    b"@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0"
    b"\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0"
    b"@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0"
    b"@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80"
    b"\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`"
    b"\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00"
    b" @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80"
    b"\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0"
    b" \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0"
    b"\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0"
    b"@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0"
    b"\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0"
    b"\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0"
    b"\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0"
    b"\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80"
    b"\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0"
    b" \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00"
    b"\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0"
    b"  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00"
    b"\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`"
    b"\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`"
    b"\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0"
    b"@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`"
    b"\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0"
    b"\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"
)


# ---------------------------------------------------------------------------
# Mask I/O helpers
# ---------------------------------------------------------------------------

def load_ann_png(path):
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    return {object_id: (mask == object_id) for object_id in object_ids}


def put_per_obj_mask(per_obj_mask, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for object_id in sorted(per_obj_mask)[::-1]:
        m = per_obj_mask[object_id].reshape(height, width)
        mask[m] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0
    return per_obj_input_mask, input_palette


def save_masks_to_dir(
    output_mask_dir, video_name, frame_name, per_obj_output_mask,
    height, width, per_obj_png_file, output_palette,
):
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name), exist_ok=True
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


# ---------------------------------------------------------------------------
# Tracker extractor
# ---------------------------------------------------------------------------

def _get_tracker(predictor, version):
    """Return the underlying tracker model (with dino_encoder attribute) for DINO fusion checks."""
    from sam3.model.sam3_tracker_base import Sam3TrackerBase

    if version == "sam3":
        # Sam3VideoPredictorMultiGPU → self.model (Sam3VideoInferenceWithInstanceInteractivity)
        # → self.tracker (Sam3TrackerPredictor(Sam3TrackerBase))
        model = getattr(predictor, "model", None)
        if model is not None:
            tracker = getattr(model, "tracker", None)
            if isinstance(tracker, Sam3TrackerBase):
                return tracker
    elif version == "sam3.1":
        # Sam3MultiplexVideoPredictor → self.model (Sam3MultiplexTrackingWithInteractivity)
        # → self.tracker (Sam3MultiplexPredictorWrapper) → self.model (Sam3VideoTrackingMultiplexDemo)
        # VideoTrackingMultiplex is NOT a Sam3TrackerBase subclass; check for dino_encoder attr.
        model = getattr(predictor, "model", None)
        if model is not None:
            wrapper = getattr(model, "tracker", None)
            if wrapper is not None:
                inner = getattr(wrapper, "model", None)
                if inner is not None and hasattr(inner, "dino_encoder"):
                    return inner
    # Fallback: search all submodules for one that has dino_encoder
    for module in predictor.modules():
        if hasattr(module, "dino_encoder"):
            return module
    return None


# ---------------------------------------------------------------------------
# VOS inference  — SAM3 (version == "sam3")
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference_sam3(
    tracker,          # Sam3TrackerPredictor with DINO fusion attached
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    """Semi-supervised VOS inference using the SAM3 tracker directly.

    Uses tracker.init_state / add_new_mask / propagate_in_video — the same code
    path as Sam2VideoPredictor.  DINO fusion activates automatically because it
    is injected inside Sam3TrackerBase.track_step.
    """
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg")
    ]
    frame_names.sort(key=lambda p: int(p))

    inference_state = tracker.init_state(video_path=video_dir)
    height = inference_state["video_height"]
    width  = inference_state["video_width"]
    input_palette = None

    if not use_all_masks:
        input_frame_inds = [0]
    else:
        input_frame_inds = sorted(set(
            idx
            for idx, name in enumerate(frame_names)
            if not per_obj_png_file and os.path.exists(
                os.path.join(input_mask_dir, video_name, f"{name}.png")
            )
        ))
        if not input_frame_inds:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}."
            )

    object_ids_set = None
    for input_frame_idx in input_frame_inds:
        per_obj_input_mask, input_palette = load_masks_from_dir(
            input_mask_dir=input_mask_dir,
            video_name=video_name,
            frame_name=frame_names[input_frame_idx],
            per_obj_png_file=per_obj_png_file,
        )
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        for object_id, object_mask in per_obj_input_mask.items():
            tracker.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=torch.as_tensor(object_mask, dtype=torch.bool),
            )

    if not object_ids_set:
        raise RuntimeError(f"In {video_name=}, no object ids found.")

    # Preflight: encode memories for conditioning frames
    tracker.propagate_in_video_preflight(inference_state, run_mem_encoder=True)

    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}

    for (
        out_frame_idx, out_obj_ids, _low_res_masks, video_res_masks, _obj_scores
    ) in tracker.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=inference_state["num_frames"],
        reverse=False,
    ):
        per_obj_output_mask = {
            out_obj_id: (video_res_masks[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask

    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )


# ---------------------------------------------------------------------------
# VOS inference  — SAM3.1 (version == "sam3.1")
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference_sam31(
    predictor,        # Sam3MultiplexVideoPredictor with DINO fusion attached
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    """Semi-supervised VOS inference using the SAM3.1 multiplex tracker directly.

    We bypass the high-level SAM3.1 detection + tracking stack and call the
    underlying ``Sam3VideoTrackingMultiplexDemo`` tracker directly — the same
    model whose ``_track_step_aux`` now injects DINOv3 features.

    Because ``Sam3VideoTrackingMultiplexDemo`` overrides ``init_state`` to take
    ``(video_height, video_width, num_frames)`` (without loading frames from
    disk), we explicitly call the parent class ``VideoTrackingMultiplexDemo.init_state``
    which accepts ``video_path`` and populates ``inference_state["images"]``.
    Everything else (``add_new_masks``, ``propagate_in_video_preflight``,
    ``propagate_in_video``) is called on ``tracker_model`` directly.
    """
    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg")
    ]
    frame_names.sort(key=lambda p: int(p))

    # Access the underlying tracker model (Sam3VideoTrackingMultiplexDemo)
    demo_model = predictor.model          # Sam3MultiplexTrackingWithInteractivity
    tracker_wrapper = demo_model.tracker  # Sam3MultiplexPredictorWrapper
    tracker_model = tracker_wrapper.model # Sam3VideoTrackingMultiplexDemo

    # Load frames by calling the parent-class init_state (accepts video_path,
    # populates inference_state["images"]).  Sam3VideoTrackingMultiplexDemo
    # overrides this with a different signature so we call the parent directly.
    inference_state = VideoTrackingMultiplexDemo.init_state(
        tracker_model,
        video_path=video_dir,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    )
    height = inference_state["video_height"]
    width  = inference_state["video_width"]
    input_palette = None

    if not use_all_masks:
        input_frame_inds = [0]
    else:
        input_frame_inds = sorted(set(
            idx
            for idx, name in enumerate(frame_names)
            if not per_obj_png_file and os.path.exists(
                os.path.join(input_mask_dir, video_name, f"{name}.png")
            )
        ))
        if not input_frame_inds:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}."
            )

    # Collect all masks for the first input frame then call add_new_masks (batched).
    # SAM3.1 uses add_new_masks (plural) which takes stacked [N, H, W] tensors and
    # initialises the multiplex_state on the first call.
    object_ids_set = None
    for input_frame_idx in input_frame_inds:
        per_obj_input_mask, input_palette = load_masks_from_dir(
            input_mask_dir=input_mask_dir,
            video_name=video_name,
            frame_name=frame_names[input_frame_idx],
            per_obj_png_file=per_obj_png_file,
        )
        if not per_obj_input_mask:
            raise RuntimeError(
                f"No masks found for {video_name=} at frame {frame_names[input_frame_idx]!r}."
            )
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        obj_ids = sorted(per_obj_input_mask)
        masks_stacked = torch.stack(
            [torch.as_tensor(per_obj_input_mask[oid], dtype=torch.float32) for oid in obj_ids]
        )  # [N, H, W]
        tracker_model.add_new_masks(
            inference_state=inference_state,
            frame_idx=input_frame_idx,
            obj_ids=obj_ids,
            masks=masks_stacked,
        )

    if not object_ids_set:
        raise RuntimeError(f"In {video_name=}, no object ids found.")

    # Encode memories for conditioning frames before propagation.
    tracker_model.propagate_in_video_preflight(inference_state, run_mem_encoder=True)

    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}

    # propagate_in_video for Sam3VideoTrackingMultiplexDemo yields a 4-tuple:
    # (frame_idx, obj_ids, low_res_masks, video_res_masks)
    for (
        out_frame_idx, out_obj_ids, _low_res_masks, video_res_masks
    ) in tracker_model.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=len(frame_names),
        reverse=False,
    ):
        per_obj_output_mask = {
            out_obj_id: (video_res_masks[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask

    torch.cuda.synchronize()

    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )

    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VOS inference with SAM3/SAM3.1 + DINOv3 cross-attention fusion"
    )
    parser.add_argument(
        "--version", type=str, default="sam3", choices=["sam3", "sam3.1"],
        help="Model version to use (default: sam3).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to pretrained SAM3/SAM3.1 checkpoint (auto-downloads from HuggingFace if omitted).",
    )
    parser.add_argument(
        "--dino_fusion_checkpoint", type=str, required=True,
        help=(
            "Path to the DINO fusion checkpoint produced by scripts/train_dino_fusion.py "
            "(contains 'dino_encoder' and 'cross_attention_fuser' state dicts)."
        ),
    )
    parser.add_argument(
        "--base_video_dir", type=str, required=True,
        help="Directory containing videos (as JPEG files) to run VOS prediction on.",
    )
    parser.add_argument(
        "--input_mask_dir", type=str, required=True,
        help="Directory containing input masks (first-frame PNG files) of each video.",
    )
    parser.add_argument(
        "--output_mask_dir", type=str, required=True,
        help="Directory to save the output masks (as PNG files).",
    )
    parser.add_argument(
        "--video_list_file", type=str, default=None,
        help="Text file containing video names to process (one per line).",
    )
    parser.add_argument(
        "--score_thresh", type=float, default=0.0,
        help="Threshold for the output mask logits (default: 0.0).",
    )
    parser.add_argument(
        "--use_all_masks", action="store_true",
        help="Use all available PNG files in input_mask_dir as input "
             "(default: only the first frame's mask).",
    )
    parser.add_argument(
        "--per_obj_png_file", action="store_true",
        help="Use separate per-object PNG files for input and output masks "
             "(default: all objects packed into a single DAVIS-format PNG per frame).",
    )
    # DINOv3 architecture options (must match training)
    parser.add_argument(
        "--dino_model", type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HuggingFace model ID for DINOv3 (must match the one used during training).",
    )
    parser.add_argument(
        "--cross_attn_num_heads", type=int, default=8,
        help="Number of heads in cross-attention fuser (must match training, default: 8).",
    )
    args = parser.parse_args()

    username = getpass.getuser()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_{username}"

    # ── Build predictor with DINO fusion ──────────────────────────────────
    print(f"Building {args.version} predictor with DINOv3 fusion ...")
    from sam3 import build_sam3_predictor

    build_kwargs = dict(
        version=args.version,
        compile=False,
        async_loading_frames=False,
        use_fa3=False,
        use_dino_fusion=True,
        dino_model_name=args.dino_model,
        dino_freeze_backbone=True,
        dino_checkpoint_path=args.dino_fusion_checkpoint,
    )
    if args.checkpoint:
        build_kwargs["checkpoint_path"] = args.checkpoint

    predictor = build_sam3_predictor(**build_kwargs)
    print(f"  Predictor ready.\n")

    # ── Video list ─────────────────────────────────────────────────────────
    if args.video_list_file is not None:
        with open(args.video_list_file) as f:
            video_names = [v.strip() for v in f if v.strip()]
    else:
        video_names = sorted(
            p for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        )

    print(f"Running VOS prediction on {len(video_names)} videos")

    # For SAM3 we call the tracker directly; for SAM3.1 we go through the predictor.
    if args.version == "sam3":
        tracker = _get_tracker(predictor, version="sam3")
        if tracker is None:
            raise RuntimeError(
                "Could not find Sam3TrackerBase in the SAM3 predictor. "
                "Check that the model was built correctly."
            )
        if tracker.dino_encoder is None:
            raise RuntimeError(
                "DINO fusion was not attached to the tracker. "
                "Verify that --dino_fusion_checkpoint points to a valid checkpoint."
            )

        for n_video, video_name in enumerate(video_names):
            print(f"\n{n_video + 1}/{len(video_names)} - {video_name}")
            try:
                vos_inference_sam3(
                    tracker=tracker,
                    base_video_dir=args.base_video_dir,
                    input_mask_dir=args.input_mask_dir,
                    output_mask_dir=args.output_mask_dir,
                    video_name=video_name,
                    score_thresh=args.score_thresh,
                    use_all_masks=args.use_all_masks,
                    per_obj_png_file=args.per_obj_png_file,
                )
            except Exception as e:
                print(f"  Error on {video_name}: {e}")
                torch.cuda.empty_cache()
    else:
        tracker_model = _get_tracker(predictor, version="sam3.1")
        if tracker_model is None:
            raise RuntimeError(
                "Could not find the SAM3.1 tracker model. "
                "Check that the model was built correctly."
            )
        if getattr(tracker_model, "dino_encoder", None) is None:
            raise RuntimeError(
                "DINO fusion was not attached to the SAM3.1 tracker. "
                "Verify that --dino_fusion_checkpoint points to a valid checkpoint."
            )
        for n_video, video_name in enumerate(video_names):
            print(f"\n{n_video + 1}/{len(video_names)} - {video_name}")
            try:
                vos_inference_sam31(
                    predictor=predictor,
                    base_video_dir=args.base_video_dir,
                    input_mask_dir=args.input_mask_dir,
                    output_mask_dir=args.output_mask_dir,
                    video_name=video_name,
                    score_thresh=args.score_thresh,
                    use_all_masks=args.use_all_masks,
                    per_obj_png_file=args.per_obj_png_file,
                )
            except Exception as e:
                print(f"  Error on {video_name}: {e}")
                torch.cuda.empty_cache()

    print(
        f"\nCompleted VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


if __name__ == "__main__":
    main()
