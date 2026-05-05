"""
SAM3 / SAM3.1 VOS inference — mask prompt on the first frame, propagate.

Analogous to scripts/vos_inference_dino_fusion.py but uses the plain
SAM3 / SAM3.1 architecture without DINOv3 cross-attention fusion.

The GT mask for the first frame is loaded from ``input_mask_dir`` and used
as the conditioning prompt.  All subsequent frames are produced by propagation.

Predicted masks are saved in DAVIS palette-PNG format, one PNG per frame, under:
  output_mask_dir/
    video_name/
      00000.png
      00001.png
      ...

Usage:
  python scripts/run_vos_text_prompt.py \\
    --base_video_dir /Experiments/marcol01/frames \\
    --input_mask_dir /path/to/annotations \\
    --output_mask_dir /home/marcol01/sam3/sam3_predictions

  # SAM3.1 variant:
  python scripts/run_vos_text_prompt.py \\
    --version sam3.1 \\
    --base_video_dir /Experiments/marcol01/frames \\
    --input_mask_dir /path/to/annotations \\
    --output_mask_dir /home/marcol01/sam3/sam3_predictions

  # Override checkpoint:
  python scripts/run_vos_text_prompt.py \\
    --base_video_dir /Experiments/marcol01/frames \\
    --input_mask_dir /path/to/annotations \\
    --output_mask_dir /home/marcol01/sam3/sam3_predictions \\
    --checkpoint /path/to/checkpoint.pt
"""

import argparse
import getpass
import os
import sys

# Ensure the sam3 repo root is on the path when running as a script
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# DAVIS palette (identical to the one used in vos_inference.py)
# ---------------------------------------------------------------------------
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


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
# VOS inference  — SAM3 (version == "sam3")
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference_sam3(
    tracker,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
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
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg")
    ]
    frame_names.sort(key=lambda p: int(p))

    # Access the underlying tracker model (Sam3VideoTrackingMultiplexDemo)
    demo_model = predictor.model
    tracker_wrapper = demo_model.tracker
    tracker_model = tracker_wrapper.model

    # tracker_model.backbone is set to None during build_sam3_multiplex_video_predictor
    # because the backbone is shared with the detector.  For standalone VOS inference
    # (without the detection pipeline) we need the tracker to run forward_image, so we
    # temporarily point its backbone at the detector's backbone — same architecture,
    # same trained weights (checkpoint is loaded into demo_model.detector.backbone).
    if tracker_model.backbone is None:
        tracker_model.backbone = demo_model.detector.backbone

    # Call parent-class init_state (accepts video_path, loads frames from disk).
    # Sam3VideoTrackingMultiplexDemo overrides init_state with a different signature
    # so we call the parent directly.
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

    tracker_model.propagate_in_video_preflight(inference_state, run_mem_encoder=True)

    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}

    # Sam3VideoTrackingMultiplexDemo.propagate_in_video yields a 4-tuple:
    # (frame_idx, obj_ids, low_res_masks, video_res_masks)
    for (
        out_frame_idx, out_obj_ids, _low_res_masks, video_res_masks, *_
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
        description="SAM3 / SAM3.1 VOS inference — mask prompt on the first frame"
    )
    parser.add_argument(
        "--version", type=str, default="sam3.1", choices=["sam3", "sam3.1"],
        help="Model version to use (default: sam3.1).",
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
        "--checkpoint", type=str, default=None,
        help="Path to pretrained SAM3/SAM3.1 checkpoint (auto-downloads from HuggingFace if omitted).",
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
    args = parser.parse_args()

    username = getpass.getuser()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_{username}"

    print(f"Building {args.version} predictor ...")
    from sam3 import build_sam3_predictor
    build_kwargs = dict(
        version=args.version,
        compile=False,
        async_loading_frames=False,
        use_fa3=False,
    )
    if args.checkpoint:
        build_kwargs["checkpoint_path"] = args.checkpoint
    predictor = build_sam3_predictor(**build_kwargs)
    print(f"  Predictor ready.\n")

    if args.video_list_file is not None:
        with open(args.video_list_file) as f:
            video_names = [v.strip() for v in f if v.strip()]
    else:
        video_names = sorted(
            p for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        )

    print(f"Running VOS prediction on {len(video_names)} videos")

    if args.version == "sam3":
        from sam3.model.sam3_tracker_base import Sam3TrackerBase
        model = getattr(predictor, "model", None)
        tracker = getattr(model, "tracker", None) if model is not None else None
        if not isinstance(tracker, Sam3TrackerBase):
            raise RuntimeError("Could not find Sam3TrackerBase in the SAM3 predictor.")

        # The SAM3 tracker is built without a backbone (the backbone lives in
        # model.detector).  Assign it so that _get_image_feature can run the
        # backbone on frame 0 when features are not yet cached.
        if tracker.backbone is None and hasattr(model, "detector") and hasattr(model.detector, "backbone"):
            tracker.backbone = model.detector.backbone

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
