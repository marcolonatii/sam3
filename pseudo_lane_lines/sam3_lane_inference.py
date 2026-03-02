#!/usr/bin/env python3
"""
SAM3 Lane Detection Inference on nuScenes Mini Dataset

This script runs SAM3 text-prompted segmentation on 5 selected nuScenes scenes
to detect lane markings and road boundaries, with comprehensive visualization.

Supports two inference modes:
- frame: Per-frame inference (no temporal tracking, faster for testing)
- video: Session-based video inference with temporal tracking/propagation

Prerequisites:
1. Accept the SAM3 license at: https://huggingface.co/facebook/sam3
2. Login to HuggingFace: huggingface-cli login
3. Install dependencies:
   pip install git+https://github.com/huggingface/transformers torchvision
   # For video mode, install sam3 from source:
   # pip install -e /path/to/sam3_repo

Target Scenes:
- scene-0061: Easy daylight sanity check
- scene-0553: Urban clutter/occlusions stress
- scene-0103: Hard visibility/lighting stress  
- scene-0916: Topology complexity stress
- scene-1094: Adverse condition (night + rain)
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Conditional imports for different modes
# These are only needed for frame mode with HuggingFace transformers
Sam3Model = None
Sam3Processor = None
snapshot_download = None

def _lazy_import_hf():
    """Lazily import HuggingFace dependencies only when needed."""
    global Sam3Model, Sam3Processor, snapshot_download
    if Sam3Model is None:
        try:
            from transformers import Sam3Model as _Sam3Model, Sam3Processor as _Sam3Processor
            from huggingface_hub import snapshot_download as _snapshot_download
            Sam3Model = _Sam3Model
            Sam3Processor = _Sam3Processor
            snapshot_download = _snapshot_download
        except ImportError as e:
            raise ImportError(
                f"Could not import HuggingFace transformers. For frame mode, install:\n"
                f"  pip install git+https://github.com/huggingface/transformers\n"
                f"Original error: {e}"
            )


@dataclass
class SceneConfig:
    """Configuration for a nuScenes scene to process."""
    name: str
    token: str
    description: str
    test_purpose: str
    pass_criteria: str


# Target scenes based on the plan
TARGET_SCENES = [
    SceneConfig(
        name="scene-0061",
        token="cc8c0bf57f984915a77078b10eb33198",
        description="Parked truck, construction, intersection, turn left, following a van",
        test_purpose="Easy daylight driving sanity check",
        pass_criteria="Both lane boundaries visible in near/mid field; masks not fragmented",
    ),
    SceneConfig(
        name="scene-0553",
        token="6f83169d067343658251f72e1dd17dbc",
        description="Wait at intersection, bicycle, large truck, peds crossing crosswalk",
        test_purpose="Urban clutter/occlusions stress",
        pass_criteria="Doesn't jump to car edges; doesn't label crosswalk stripes as lanes",
    ),
    SceneConfig(
        name="scene-0103",
        token="fcbccedd61424f1b85dcbf8f897f9754",
        description="Many peds right, wait for turning car, long bike rack left, cyclist",
        test_purpose="Hard visibility/lighting stress",
        pass_criteria="Dominant lane structure retained in near-field despite lighting",
    ),
    SceneConfig(
        name="scene-0916",
        token="325cef682f064c55a255f2625c533b75",
        description="Parking lot, bicycle rack, parked bicycles, bus, many peds",
        test_purpose="Topology complexity stress",
        pass_criteria="Split lines not incorrectly merged; main lane stable",
    ),
    SceneConfig(
        name="scene-1094",
        token="de7d80a1f5fb4c3e82ce8a4f213b450a",
        description="Night, after rain, many peds, PMD, jaywalker, truck, scooter",
        test_purpose="Adverse condition (rain + night)",
        pass_criteria="Performance under wet reflections and night glare",
    ),
]

# Default lane-related text prompts (can be overridden by config.yaml)
DEFAULT_LANE_PROMPTS = [
    "white lane line",
    "yellow lane line", 
    "dashed lane marking",
    "solid lane marking",
]


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml in same directory.
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}, using defaults")
        return {
            "prompts": DEFAULT_LANE_PROMPTS,
            "inference": {"prompt_frame": 0, "alpha": 0.5, "show_frame_numbers": True},
            "output": {"timestamp_prefix": True},
        }
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    return config


# For backward compatibility
LANE_PROMPTS = DEFAULT_LANE_PROMPTS


class NuScenesLoader:
    """Simple loader for nuScenes mini dataset."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.meta_root = self.data_root / "v1.0-mini"
        self.samples_root = self.data_root / "samples"
        
        # Load metadata
        with open(self.meta_root / "scene.json") as f:
            self.scenes = {s["token"]: s for s in json.load(f)}
        
        with open(self.meta_root / "sample.json") as f:
            self.samples = {s["token"]: s for s in json.load(f)}
        
        with open(self.meta_root / "sample_data.json") as f:
            self.sample_data = {s["token"]: s for s in json.load(f)}
    
    def _get_channel_from_filename(self, filename: str) -> str:
        """Extract channel name from filename path.
        
        Examples:
            samples/CAM_FRONT/xxx.jpg -> CAM_FRONT
            sweeps/RADAR_FRONT/xxx.pcd -> RADAR_FRONT
        """
        parts = filename.split("/")
        if len(parts) >= 2:
            return parts[1]  # e.g., CAM_FRONT
        return ""
    
    def get_scene_samples(self, scene_token: str, camera: str = "CAM_FRONT") -> list[dict]:
        """Get all camera samples for a scene."""
        scene = self.scenes[scene_token]
        samples = []
        
        # Walk through the sample chain
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = self.samples[sample_token]
            
            # Find the camera sample_data
            for sd_token in self._get_sample_data_tokens(sample_token):
                sd = self.sample_data[sd_token]
                channel = self._get_channel_from_filename(sd["filename"])
                if channel == camera:
                    image_path = self.data_root / sd["filename"]
                    if image_path.exists():
                        samples.append({
                            "token": sd_token,
                            "timestamp": sd["timestamp"],
                            "image_path": str(image_path),
                            "filename": sd["filename"],
                        })
                    break
            
            sample_token = sample.get("next", "")
            if not sample_token:
                break
        
        return samples
    
    def _get_sample_data_tokens(self, sample_token: str) -> list[str]:
        """Get all sample_data tokens for a sample."""
        return [
            sd["token"] for sd in self.sample_data.values()
            if sd.get("sample_token") == sample_token
        ]


class Sam3LaneInference:
    """SAM3-based lane detection inference (per-frame mode, no temporal tracking).
    
    Uses native SAM3 implementation on CUDA (same as video mode).
    Falls back to HuggingFace Transformers on CPU/MPS if available.
    """
    
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        use_native_sam3: bool = True,  # Default to native for consistency with video mode
    ):
        """Initialize SAM3 lane inference.
        
        Args:
            model_id: HuggingFace model ID for SAM3
            device: Device to use (cuda, mps, cpu). Auto-detected if None.
            confidence_threshold: Confidence threshold for detections
            use_native_sam3: If True and CUDA available, use native sam3 repo implementation.
                           If False or CUDA not available, use HuggingFace transformers.
        """
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Determine which implementation to use
        # On CUDA, prefer native SAM3 (same as video mode) for consistency
        self.use_native = False
        if self.device == "cuda":
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor as NativeSam3Processor
                self.use_native = True
                print("Using native SAM3 implementation (CUDA)")
            except ImportError as e:
                print(f"Native SAM3 not available ({e}), falling back to HuggingFace transformers")
                self.use_native = False
        
        if self.use_native:
            self._init_native_sam3()
        else:
            self._init_huggingface_sam3()
    
    def _init_huggingface_sam3(self):
        """Initialize using HuggingFace Transformers implementation."""
        _lazy_import_hf()  # Ensure HF imports are available
        print("Loading SAM3 model via HuggingFace Transformers...")
        model_path = self._prepare_model_dir()
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def _init_native_sam3(self):
        """Initialize using native SAM3 repo implementation (CUDA only)."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as NativeSam3Processor
        
        print("Loading SAM3 model via native implementation...")
        self.model = build_sam3_image_model(device=self.device)
        self.native_processor = NativeSam3Processor(
            self.model, 
            confidence_threshold=self.confidence_threshold
        )
        print("Model loaded successfully!")

    def _prepare_model_dir(self) -> str:
        """Download the model and ensure processor config naming is compatible."""
        _lazy_import_hf()  # Ensure HF imports are available
        cache_dir = Path(__file__).resolve().parent / "data" / "sam3_model"
        cache_dir.mkdir(parents=True, exist_ok=True)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        local_dir = snapshot_download(
            self.model_id,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )

        processor_cfg = Path(local_dir) / "processor_config.json"
        preprocessor_cfg = Path(local_dir) / "preprocessor_config.json"
        if processor_cfg.exists() and not preprocessor_cfg.exists():
            shutil.copy(processor_cfg, preprocessor_cfg)

        return local_dir
    
    @torch.no_grad()
    def run_inference(
        self,
        image: Image.Image,
        prompts: list[str],
    ) -> dict:
        """Run SAM3 inference with text prompts.
        
        Args:
            image: PIL Image to process
            prompts: List of text prompts
            
        Returns:
            Dictionary with masks, boxes, and scores for each prompt
        """
        if self.use_native:
            return self._run_inference_native(image, prompts)
        else:
            return self._run_inference_huggingface(image, prompts)
    
    def _run_inference_native(self, image: Image.Image, prompts: list[str]) -> dict:
        """Run inference using native SAM3 implementation."""
        results = {}
        
        # Set the image once
        state = self.native_processor.set_image(image)
        
        for prompt in prompts:
            # Reset prompts for new query
            self.native_processor.reset_all_prompts(state)
            
            # Re-set image backbone (reset clears it)
            state = self.native_processor.set_image(image, state)
            
            # Run text prompt
            state = self.native_processor.set_text_prompt(prompt, state)
            
            # Extract masks and scores
            masks = []
            scores = []
            
            if "masks" in state and state["masks"] is not None:
                mask_tensor = state["masks"]
                score_tensor = state.get("scores", None)
                
                for i in range(mask_tensor.shape[0]):
                    mask = mask_tensor[i, 0].cpu().numpy().astype(np.uint8)
                    masks.append(mask)
                    if score_tensor is not None and i < len(score_tensor):
                        scores.append(float(score_tensor[i].cpu()))
                    else:
                        scores.append(1.0)
            
            results[prompt] = {
                "masks": masks,
                "scores": scores,
                "num_detections": len(masks),
            }
        
        return results
    
    def _run_inference_huggingface(self, image: Image.Image, prompts: list[str]) -> dict:
        """Run inference using HuggingFace Transformers implementation."""
        results = {}
        original_size = image.size  # (width, height)
        
        for prompt in prompts:
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Run model
            outputs = self.model(**inputs)
            
            # Post-process using instance segmentation
            # target_sizes should be (height, width)
            target_sizes = [(original_size[1], original_size[0])]
            
            processed = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.confidence_threshold,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )
            
            # Extract masks and scores from processed output
            masks = []
            scores = []
            
            if processed and len(processed) > 0:
                result = processed[0]
                
                # Result contains 'segmentation' (combined mask) and 'segments_info'
                if "segments_info" in result:
                    for segment in result["segments_info"]:
                        segment_id = segment.get("id", 0)
                        score = segment.get("score", 1.0)
                        
                        # Extract mask for this segment from segmentation
                        if "segmentation" in result:
                            seg_map = result["segmentation"]
                            if isinstance(seg_map, torch.Tensor):
                                seg_map = seg_map.cpu().numpy()
                            mask = (seg_map == segment_id).astype(np.uint8)
                            masks.append(mask)
                            scores.append(float(score))
                
                # Alternative: direct mask output
                elif "masks" in result:
                    for i, mask in enumerate(result["masks"]):
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        masks.append(mask)
                        score = result.get("scores", [1.0] * len(result["masks"]))[i]
                        scores.append(float(score))
            
            results[prompt] = {
                "masks": masks,
                "scores": scores,
                "num_detections": len(masks),
            }
        
        return results


class Sam3VideoInference:
    """SAM3-based video inference with temporal tracking and propagation.
    
    Uses the SAM3 video predictor API for session-based inference,
    which maintains temporal consistency across frames.
    
    NOTE: Requires CUDA and triton. Will raise an error if CUDA is not available.
    """
    
    def __init__(self, gpus_to_use: Optional[list[int]] = None):
        """Initialize the video predictor.
        
        Args:
            gpus_to_use: List of GPU indices to use. If None, uses all available GPUs.
        
        Raises:
            RuntimeError: If CUDA is not available (video mode requires CUDA + triton)
        """
        # Check CUDA availability first
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM3 video mode requires CUDA. Please use --mode frame on non-CUDA devices, "
                "or run on a machine with NVIDIA GPU."
            )
        
        # Import sam3 video predictor (requires sam3 installed from source)
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as e:
            raise ImportError(
                "sam3 package not found or missing dependencies. Please install from source:\n"
                "  cd sam3_repo && pip install -e .\n"
                f"Original error: {e}"
            )
        
        if gpus_to_use is None:
            gpus_to_use = list(range(torch.cuda.device_count()))
        
        print(f"Initializing SAM3 video predictor with GPUs: {gpus_to_use}")
        self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
        self.session_id = None
        print("Video predictor initialized!")
    
    def start_session(self, video_path: str) -> str:
        """Start a new video session.
        
        Args:
            video_path: Path to video file (MP4) or directory of JPEG frames
            
        Returns:
            Session ID
        """
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        self.session_id = response["session_id"]
        return self.session_id
    
    def reset_session(self):
        """Reset the current session (clear all prompts)."""
        if self.session_id:
            self.predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=self.session_id,
                )
            )
    
    def add_text_prompt(self, frame_index: int, text: str) -> dict:
        """Add a text prompt on a specific frame.
        
        Args:
            frame_index: Frame index to add the prompt on
            text: Text prompt describing what to segment
            
        Returns:
            Output dict with masks for the prompted frame
        """
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self.session_id,
                frame_index=frame_index,
                text=text,
            )
        )
        return response["outputs"]
    
    def propagate_in_video(self) -> dict[int, dict]:
        """Propagate masks through the entire video.
        
        Returns:
            Dictionary mapping frame_index -> outputs
        """
        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=self.session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
        return outputs_per_frame
    
    def close_session(self):
        """Close the current session and free resources."""
        if self.session_id:
            self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=self.session_id,
                )
            )
            self.session_id = None
    
    def shutdown(self):
        """Shutdown the predictor and free all resources."""
        self.predictor.shutdown()
    
    @staticmethod
    def prepare_frames_directory(image_paths: list[str], temp_dir: str) -> str:
        """Prepare a directory of numbered JPEG frames for video inference.
        
        SAM3 video predictor expects frames named as <frame_index>.jpg
        
        Args:
            image_paths: List of image paths in order
            temp_dir: Temporary directory to store renamed frames
            
        Returns:
            Path to the frames directory
        """
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        for i, src_path in enumerate(image_paths):
            dst_path = frames_dir / f"{i:05d}.jpg"
            # Copy or symlink the frame
            shutil.copy(src_path, dst_path)
        
        return str(frames_dir)
    
    @staticmethod
    def extract_masks_from_output(output: dict, image_size: tuple[int, int]) -> dict:
        """Extract masks from video predictor output format.
        
        Args:
            output: Output from video predictor
            image_size: (height, width) of the image
            
        Returns:
            Dictionary with masks and object IDs
        """
        masks = []
        obj_ids = []
        
        if output is None:
            return {"masks": masks, "obj_ids": obj_ids}
        
        # Video predictor returns masks keyed by object ID
        for obj_id, mask_data in output.items():
            if isinstance(mask_data, dict) and "mask" in mask_data:
                mask = mask_data["mask"]
            elif isinstance(mask_data, (np.ndarray, torch.Tensor)):
                mask = mask_data
            else:
                continue
            
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Ensure mask is correct size
            if mask.shape != image_size:
                mask = cv2.resize(mask.astype(np.float32), (image_size[1], image_size[0]))
            
            masks.append((mask > 0.5).astype(np.uint8))
            obj_ids.append(obj_id)
        
        return {"masks": masks, "obj_ids": obj_ids}


class Visualizer:
    """Visualization utilities for lane detection results."""
    
    # Color palette for different prompts
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    @staticmethod
    def create_overlay(
        image: np.ndarray,
        results: dict,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create mask overlay on image.
        
        Args:
            image: Original image as numpy array (H, W, 3)
            results: Dictionary of results from inference
            alpha: Transparency for overlay
            
        Returns:
            Image with mask overlays
        """
        overlay = image.copy()
        
        for i, (prompt, data) in enumerate(results.items()):
            color = Visualizer.COLORS[i % len(Visualizer.COLORS)]
            
            for mask in data["masks"]:
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                # Create colored mask
                mask_binary = mask > 0.5
                overlay[mask_binary] = (
                    alpha * np.array(color) + 
                    (1 - alpha) * overlay[mask_binary]
                ).astype(np.uint8)
        
        return overlay
    
    @staticmethod
    def create_side_by_side(
        original: np.ndarray,
        overlay: np.ndarray,
    ) -> np.ndarray:
        """Create side-by-side comparison.
        
        Args:
            original: Original image
            overlay: Overlay image
            
        Returns:
            Side-by-side image
        """
        return np.hstack([original, overlay])
    
    @staticmethod
    def create_legend(prompts: list[str], height: int = 100) -> np.ndarray:
        """Create a legend for the visualization."""
        width = 400
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        y_offset = 20
        for i, prompt in enumerate(prompts):
            color = Visualizer.COLORS[i % len(Visualizer.COLORS)]
            # Draw color box
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
            # Draw text
            cv2.putText(
                legend, prompt, (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
            y_offset += 20
        
        return legend


def create_video_from_frames(
    frame_paths: list[str],
    output_path: str,
    fps: float = 12.0,
) -> None:
    """Create MP4 video from frames.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Output video path
        fps: Frames per second
    """
    if not frame_paths:
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for path in frame_paths:
        frame = cv2.imread(path)
        writer.write(frame)
    
    writer.release()
    print(f"Video saved to: {output_path}")


def run_frame_mode(args, loader: NuScenesLoader, target_scenes: list[SceneConfig]):
    """Run per-frame inference mode (no temporal tracking)."""
    print("Running in FRAME mode (per-frame inference, no temporal tracking)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inferencer
    use_native = getattr(args, 'use_native_sam3', False)
    print("Initializing SAM3 frame inference...")
    inferencer = Sam3LaneInference(
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        use_native_sam3=use_native,
    )
    
    visualizer = Visualizer()
    
    # Process each target scene
    for scene_config in target_scenes:
        print(f"\n{'='*60}")
        print(f"Processing {scene_config.name}")
        print(f"Purpose: {scene_config.test_purpose}")
        print(f"{'='*60}")
        
        # Check if scene exists
        if scene_config.token not in loader.scenes:
            print(f"Warning: Scene {scene_config.name} not found in dataset, skipping...")
            continue
        
        # Create output directories
        scene_output = output_dir / scene_config.name
        overlays_dir = scene_output / "overlays"
        comparisons_dir = scene_output / "comparisons"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        # Get samples for scene
        samples = loader.get_scene_samples(scene_config.token)
        
        # Apply frame selection if specified
        if args.frames:
            frame_indices = [int(f) for f in args.frames.split(",")]
            samples = [samples[i] for i in frame_indices if i < len(samples)]
        elif args.max_samples:
            samples = samples[:args.max_samples]
        
        print(f"Processing {len(samples)} samples")
        
        overlay_paths = []
        
        # Process each sample
        for idx, sample in enumerate(tqdm(samples, desc=f"Processing {scene_config.name}")):
            try:
                # Load image
                image = Image.open(sample["image_path"])
                image_np = np.array(image)
                
                # Run inference
                results = inferencer.run_inference(image, LANE_PROMPTS)
                
                # Create visualizations
                overlay = visualizer.create_overlay(image_np, results)
                comparison = visualizer.create_side_by_side(image_np, overlay)
                
                # Save visualizations
                frame_name = Path(sample["filename"]).stem
                
                overlay_path = overlays_dir / f"{frame_name}_overlay.jpg"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                overlay_paths.append(str(overlay_path))
                
                comparison_path = comparisons_dir / f"{frame_name}_comparison.jpg"
                cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing {sample['filename']}: {e}")
                continue
        
        # Create video from overlays
        if overlay_paths and not args.frames:
            video_path = scene_output / f"{scene_config.name}_lanes.mp4"
            create_video_from_frames(overlay_paths, str(video_path))
        
        # Save scene summary
        summary = {
            "scene_name": scene_config.name,
            "description": scene_config.description,
            "test_purpose": scene_config.test_purpose,
            "pass_criteria": scene_config.pass_criteria,
            "num_samples_processed": len(samples),
            "prompts_used": LANE_PROMPTS,
            "mode": "frame",
        }
        
        with open(scene_output / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {scene_output}")


def run_video_mode(args, loader: NuScenesLoader, target_scenes: list[SceneConfig]):
    """Run session-based video inference with temporal tracking."""
    print("Running in VIDEO mode (session-based with temporal tracking)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize video inferencer
    print("Initializing SAM3 video inference...")
    inferencer = Sam3VideoInference()
    
    visualizer = Visualizer()
    
    try:
        # Process each target scene
        for scene_config in target_scenes:
            print(f"\n{'='*60}")
            print(f"Processing {scene_config.name}")
            print(f"Purpose: {scene_config.test_purpose}")
            print(f"{'='*60}")
            
            # Check if scene exists
            if scene_config.token not in loader.scenes:
                print(f"Warning: Scene {scene_config.name} not found in dataset, skipping...")
                continue
            
            # Create output directories
            scene_output = output_dir / scene_config.name / "video_mode"
            overlays_dir = scene_output / "overlays"
            comparisons_dir = scene_output / "comparisons"
            overlays_dir.mkdir(parents=True, exist_ok=True)
            comparisons_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all samples for scene
            all_samples = loader.get_scene_samples(scene_config.token)
            print(f"Found {len(all_samples)} total samples")
            
            # Determine which frames to output
            if args.frames:
                output_frame_indices = [int(f) for f in args.frames.split(",")]
            elif args.max_samples:
                output_frame_indices = list(range(min(args.max_samples, len(all_samples))))
            else:
                output_frame_indices = list(range(len(all_samples)))
            
            # Create temp directory with numbered frames for video predictor
            with tempfile.TemporaryDirectory() as temp_dir:
                print("Preparing frames for video inference...")
                image_paths = [s["image_path"] for s in all_samples]
                frames_dir = Sam3VideoInference.prepare_frames_directory(image_paths, temp_dir)
                
                # Start video session
                print("Starting video session...")
                inferencer.start_session(frames_dir)
                
                # Process each prompt separately (video mode handles one concept at a time)
                all_prompt_results = {}
                
                for prompt in LANE_PROMPTS:
                    print(f"Processing prompt: '{prompt}'")
                    
                    # Reset session for new prompt
                    inferencer.reset_session()
                    
                    # Add prompt on first frame
                    prompt_frame = args.prompt_frame if hasattr(args, 'prompt_frame') else 0
                    print(f"  Adding prompt on frame {prompt_frame}...")
                    inferencer.add_text_prompt(prompt_frame, prompt)
                    
                    # Propagate through video
                    print("  Propagating through video...")
                    outputs = inferencer.propagate_in_video()
                    all_prompt_results[prompt] = outputs
                
                # Generate visualizations for selected frames
                print(f"Generating visualizations for frames: {output_frame_indices}")
                overlay_paths = []
                
                for frame_idx in tqdm(output_frame_indices, desc="Creating visualizations"):
                    if frame_idx >= len(all_samples):
                        continue
                    
                    sample = all_samples[frame_idx]
                    image = Image.open(sample["image_path"])
                    image_np = np.array(image)
                    image_size = (image_np.shape[0], image_np.shape[1])
                    
                    # Combine results from all prompts
                    combined_results = {}
                    for prompt, outputs in all_prompt_results.items():
                        if frame_idx in outputs:
                            mask_data = Sam3VideoInference.extract_masks_from_output(
                                outputs[frame_idx], image_size
                            )
                            combined_results[prompt] = {
                                "masks": mask_data["masks"],
                                "scores": [1.0] * len(mask_data["masks"]),
                                "num_detections": len(mask_data["masks"]),
                            }
                        else:
                            combined_results[prompt] = {
                                "masks": [],
                                "scores": [],
                                "num_detections": 0,
                            }
                    
                    # Create visualizations
                    overlay = visualizer.create_overlay(image_np, combined_results)
                    comparison = visualizer.create_side_by_side(image_np, overlay)
                    
                    # Save visualizations
                    frame_name = Path(sample["filename"]).stem
                    
                    overlay_path = overlays_dir / f"{frame_idx:03d}_{frame_name}_overlay.jpg"
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    overlay_paths.append(str(overlay_path))
                    
                    comparison_path = comparisons_dir / f"{frame_idx:03d}_{frame_name}_comparison.jpg"
                    cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                
                # Close the session
                inferencer.close_session()
            
            # Create video from overlays if we have more than a few frames
            if len(overlay_paths) > 3 and not args.frames:
                video_path = scene_output / f"{scene_config.name}_lanes_video.mp4"
                create_video_from_frames(overlay_paths, str(video_path))
            
            # Save scene summary
            summary = {
                "scene_name": scene_config.name,
                "description": scene_config.description,
                "test_purpose": scene_config.test_purpose,
                "pass_criteria": scene_config.pass_criteria,
                "num_samples_total": len(all_samples),
                "frames_output": output_frame_indices,
                "prompts_used": LANE_PROMPTS,
                "mode": "video",
            }
            
            with open(scene_output / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"Results saved to: {scene_output}")
    
    finally:
        # Cleanup
        inferencer.shutdown()


def render_mask_only_frame(img, outputs, frame_idx=None, alpha=0.5, color_map=None):
    """Render masks only (no bounding boxes or labels) on a frame.
    
    Args:
        img: np.ndarray, shape (H, W, 3), uint8 or float32 in [0,255] or [0,1]
        outputs: dict with keys: out_obj_ids, out_binary_masks
        frame_idx: int or None, for overlaying frame index text
        alpha: float, mask overlay alpha
        color_map: dict mapping obj_id ranges to RGB colors. If None, uses default colors.
                   Format: {(start_id, end_id): [R, G, B], ...}
                   Or: {type_idx: [R, G, B], ...} where type_idx = obj_id // 100
    Returns:
        overlay: np.ndarray, shape (H, W, 3), uint8
    """
    from sam3.visualization_utils import COLORS
    
    if img.dtype == np.float32 or img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img = img[..., :3]  # drop alpha if present
    overlay = img.copy()
    
    if "out_binary_masks" not in outputs or len(outputs["out_binary_masks"]) == 0:
        return overlay
    
    for i in range(len(outputs["out_obj_ids"])):
        obj_id = outputs["out_obj_ids"][i]
        
        # Determine color based on lane type
        # Object ID format: prompt_idx * 1000 + prompt_frame_idx * 100 + detection_idx
        # So prompt_idx (lane type) = obj_id // 1000
        if color_map is not None:
            type_idx = obj_id // 1000  # Extract lane type from object ID
            if type_idx in color_map:
                color255 = np.array(color_map[type_idx], dtype=np.uint8)
            else:
                # Fallback to default colors
                color = COLORS[obj_id % len(COLORS)]
                color255 = (color * 255).astype(np.uint8)
        else:
            color = COLORS[obj_id % len(COLORS)]
            color255 = (color * 255).astype(np.uint8)
        
        mask = outputs["out_binary_masks"][i]
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[..., c][mask_bool] = (
                alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
            ).astype(np.uint8)
    
    # Overlay frame index at the top-left corner (optional)
    if frame_idx is not None:
        cv2.putText(
            overlay,
            f"Frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    
    return overlay


def save_mask_only_video(video_frames, outputs, out_path, alpha=0.5, fps=10, show_frame_numbers=True, color_map=None):
    """Save video with mask overlays only (no bounding boxes).
    
    Args:
        video_frames: list of video frame data
        outputs: dict mapping frame_idx -> outputs dict
        out_path: output video path
        alpha: mask overlay alpha
        fps: frames per second
        show_frame_numbers: whether to show frame numbers on video
        color_map: dict mapping type_idx to RGB colors for lane types
    """
    import subprocess
    from sam3.visualization_utils import load_frame
    
    # Read first frame to get size
    first_img = load_frame(video_frames[0])
    height, width = first_img.shape[:2]
    if first_img.dtype == np.float32 or first_img.max() <= 1.0:
        first_img = (first_img * 255).astype(np.uint8)
    
    # Use 'mp4v' for initial encoding
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("temp.mp4", fourcc, fps, (width, height))
    
    outputs_list = [
        (video_frames[frame_idx], frame_idx, outputs[frame_idx])
        for frame_idx in sorted(outputs.keys())
    ]
    
    for frame, frame_idx, frame_outputs in tqdm(outputs_list, desc="Rendering video"):
        img = load_frame(frame)
        frame_num = frame_idx if show_frame_numbers else None
        overlay = render_mask_only_frame(img, frame_outputs, frame_idx=frame_num, alpha=alpha, color_map=color_map)
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    writer.release()
    
    # Re-encode the video for compatibility using ffmpeg
    subprocess.run(["ffmpeg", "-y", "-i", "temp.mp4", "-loglevel", "error", out_path])
    print(f"Video saved to {out_path}")
    
    import os
    os.remove("temp.mp4")  # Clean up temporary file


def run_video_file_frame_mode(args, config: dict):
    """Run per-frame inference on a video file (no temporal tracking).
    
    Each frame is processed independently. Faster but no temporal consistency.
    """
    from datetime import datetime
    
    video_path = Path(args.video_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running SAM3 FRAME-BY-FRAME inference on: {video_path}")
    print(f"(No temporal tracking - each frame processed independently)")
    print(f"{'='*60}\n")
    
    # Use lane_types from config (new format) or fall back to prompts (old format)
    lane_types = config.get("lane_types", None)
    
    if args.prompts:
        # CLI override - simple prompts list
        prompts = [p.strip() for p in args.prompts.split(",")]
        lane_types = None
    elif lane_types:
        # New config format with lane types and colors
        prompts = [lt["prompt"] for lt in lane_types]
        print("\nUsing lane types:")
        for lt in lane_types:
            color_str = f"RGB{tuple(lt['color'])}"
            print(f"  - {lt['name']}: '{lt['prompt']}' -> {color_str}")
    elif "prompts" in config:
        # Old config format - simple prompts list
        prompts = config["prompts"]
        lane_types = None
    else:
        prompts = DEFAULT_LANE_PROMPTS
        lane_types = None
    
    if not lane_types:
        print(f"Using prompts: {prompts}")
    
    # Build color map from lane types
    color_map = None
    if lane_types:
        color_map = {i: lt["color"] for i, lt in enumerate(lane_types)}
    
    # Get inference settings
    inference_config = config.get("inference", {})
    alpha = inference_config.get("alpha", 0.5)
    show_frame_numbers = inference_config.get("show_frame_numbers", True)
    confidence_threshold = inference_config.get("confidence_threshold", 0.3)
    
    # Initialize frame inferencer
    print("\nInitializing SAM3 frame inference...")
    use_native = getattr(args, 'use_native_sam3', False)
    inferencer = Sam3LaneInference(
        device=args.device,
        confidence_threshold=confidence_threshold,
        use_native_sam3=use_native,
    )
    
    # Load video frames
    print("Loading video frames...")
    video_frames = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    total_frames = len(video_frames)
    print(f"Loaded {total_frames} frames at {fps:.1f} fps")
    
    if total_frames == 0:
        print("Error: No frames found in video")
        return
    
    # Process each frame
    all_outputs = {}
    
    for frame_idx, frame_rgb in enumerate(tqdm(video_frames, desc="Processing frames")):
        image = Image.fromarray(frame_rgb)
        
        # Run inference with all prompts
        results = inferencer.run_inference(image, prompts)
        
        # Convert to output format
        masks = []
        obj_ids = []
        probs = []
        
        obj_id_counter = 0
        for prompt_idx, (prompt, data) in enumerate(results.items()):
            for i, mask in enumerate(data["masks"]):
                masks.append(mask)
                obj_ids.append(prompt_idx * 100 + i)  # Unique ID per prompt
                probs.append(data["scores"][i] if i < len(data["scores"]) else 1.0)
                obj_id_counter += 1
        
        all_outputs[frame_idx] = {
            "out_obj_ids": np.array(obj_ids),
            "out_probs": np.array(probs),
            "out_binary_masks": np.stack(masks, axis=0) if masks else np.array([]),
            "out_boxes_xywh": np.array([]),  # Not used in mask-only mode
        }
    
    # Generate output filename
    output_config = config.get("output", {})
    use_timestamp = output_config.get("timestamp_prefix", True)
    
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = output_dir / f"{timestamp}_{video_path.stem}_frame_mode.mp4"
        summary_path = output_dir / f"{timestamp}_{video_path.stem}_summary.json"
    else:
        timestamp = None
        output_video_path = output_dir / f"{video_path.stem}_frame_mode.mp4"
        summary_path = output_dir / f"{video_path.stem}_summary.json"
    
    # Save output video
    print(f"\nSaving output video to: {output_video_path}")
    save_mask_only_video(
        video_frames,
        all_outputs,
        str(output_video_path),
        alpha=alpha,
        fps=fps,
        show_frame_numbers=show_frame_numbers,
        color_map=color_map,
    )
    
    # Save individual frame images if enabled
    save_frame_images = output_config.get("save_frame_images", False)
    frame_output_dir = None
    if save_frame_images:
        # Create sam3_frame_output/<timestamp>/ folder structure
        frame_output_base = output_dir / "sam3_frame_output"
        frame_output_dir = frame_output_base / (timestamp if timestamp else "default")
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving frame images to: {frame_output_dir}")
        for frame_idx in tqdm(sorted(all_outputs.keys()), desc="Saving frames"):
            frame_rgb = video_frames[frame_idx]
            frame_outputs = all_outputs[frame_idx]
            # Render frame without frame number overlay
            overlay = render_mask_only_frame(
                frame_rgb, frame_outputs, frame_idx=None, alpha=alpha, color_map=color_map
            )
            # Save as JPEG
            frame_path = frame_output_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        print(f"Saved {len(all_outputs)} frame images")
    
    # Save summary
    summary = {
        "input_video": str(video_path),
        "output_video": str(output_video_path),
        "frame_output_dir": str(frame_output_dir) if frame_output_dir else None,
        "timestamp": timestamp,
        "total_frames": total_frames,
        "fps": fps,
        "prompts_used": prompts,
        "lane_types": lane_types if lane_types else None,
        "mode": "frame",
        "config_file": str(args.config) if args.config else "default",
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Frame-by-frame processing complete!")
    print(f"Output video: {output_video_path}")
    if frame_output_dir:
        print(f"Frame images: {frame_output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")


def run_video_file_tracking(args, config: dict):
    """Run video tracking on a single video file (not nuScenes dataset).
    
    This mode directly processes a video file with SAM3 video predictor
    for lane line tracking with temporal consistency.
    """
    import glob
    from datetime import datetime
    
    video_path = Path(args.video_file)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running SAM3 VIDEO TRACKING on: {video_path}")
    print(f"(Session-based with temporal propagation)")
    print(f"{'='*60}\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: Video tracking mode requires CUDA. Please run on a CUDA-enabled machine.")
        print("Falling back to frame-by-frame mode...")
        run_video_file_frame_mode(args, config)
        return
    
    # Import sam3 video predictor
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as e:
        print(f"Error: Could not import SAM3. Make sure sam3 is installed: {e}")
        return
    
    # Load video frames for visualization
    print("Loading video frames...")
    video_frames_for_vis = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    total_frames = len(video_frames_for_vis)
    print(f"Loaded {total_frames} frames at {fps:.1f} fps")
    
    if total_frames == 0:
        print("Error: No frames found in video")
        return
    
    # Get image dimensions
    img_h, img_w = video_frames_for_vis[0].shape[:2]
    print(f"Frame size: {img_w}x{img_h}")
    
    # Build video predictor
    print("\nInitializing SAM3 video predictor...")
    gpus_to_use = list(range(torch.cuda.device_count()))
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    
    try:
        # Start video session
        print("Starting video session...")
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        session_id = response["session_id"]
        print(f"Session started: {session_id}")
        
        # Use lane_types from config (new format) or fall back to prompts (old format)
        lane_types = config.get("lane_types", None)
        
        if args.prompts:
            # CLI override - simple prompts list
            prompts = [p.strip() for p in args.prompts.split(",")]
            lane_types = None  # Will use default colors
        elif lane_types:
            # New config format with lane types and colors
            prompts = [lt["prompt"] for lt in lane_types]
            print("\nUsing lane types:")
            for lt in lane_types:
                color_str = f"RGB{tuple(lt['color'])}"
                print(f"  - {lt['name']}: '{lt['prompt']}' -> {color_str}")
        elif "prompts" in config:
            # Old config format - simple prompts list
            prompts = config["prompts"]
            lane_types = None
        else:
            prompts = DEFAULT_LANE_PROMPTS
            lane_types = None
        
        if not lane_types:
            print(f"\nUsing prompts: {prompts}")
        
        # Build color map from lane types (maps type_idx to RGB color)
        color_map = None
        if lane_types:
            color_map = {i: lt["color"] for i, lt in enumerate(lane_types)}
        
        # Get inference settings from config
        inference_config = config.get("inference", {})
        alpha = inference_config.get("alpha", 0.5)
        show_frame_numbers = inference_config.get("show_frame_numbers", True)
        
        # Determine prompt frames to use
        # Priority: prompt_interval > prompt_frames > prompt_frame > args.prompt_frame
        prompt_interval = inference_config.get("prompt_interval", None)
        prompt_frames_config = inference_config.get("prompt_frames", None)
        single_prompt_frame = inference_config.get("prompt_frame", args.prompt_frame)
        
        if prompt_interval is not None:
            # Generate frames at regular intervals: 0, interval, 2*interval, ...
            prompt_frames = list(range(0, total_frames, prompt_interval))
            print(f"\nUsing prompt_interval={prompt_interval}: detecting on {len(prompt_frames)} frames")
            print(f"Prompt frames: {prompt_frames[:10]}{'...' if len(prompt_frames) > 10 else ''}")
        elif prompt_frames_config is not None:
            prompt_frames = prompt_frames_config
            print(f"\nUsing multiple prompt frames: {prompt_frames}")
        else:
            prompt_frames = [single_prompt_frame]
            print(f"\nUsing single prompt frame: {single_prompt_frame}")
        
        # Process each prompt on each prompt frame
        all_outputs = {}
        
        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n--- Processing prompt {prompt_idx + 1}/{len(prompts)}: '{prompt}' ---")
            
            total_detections = 0
            
            for pf_idx, prompt_frame in enumerate(prompt_frames):
                # Reset session for each prompt frame
                predictor.handle_request(
                    request=dict(
                        type="reset_session",
                        session_id=session_id,
                    )
                )
                
                if len(prompt_frames) > 1:
                    print(f"  Frame {prompt_frame} ({pf_idx + 1}/{len(prompt_frames)})...", end=" ")
                else:
                    print(f"Adding text prompt on frame {prompt_frame}...")
                
                response = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=prompt_frame,
                        text=prompt,
                    )
                )
                
                # Check initial detection
                initial_out = response["outputs"]
                num_objects = len(initial_out.get("out_obj_ids", []))
                
                if len(prompt_frames) > 1:
                    print(f"{num_objects} detection(s)")
                else:
                    print(f"Initial detection: {num_objects} object(s)")
                
                if num_objects == 0:
                    continue
                
                total_detections += num_objects
                
                # Propagate through video from this prompt frame
                outputs_per_frame = {}
                for prop_response in predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                    )
                ):
                    outputs_per_frame[prop_response["frame_index"]] = prop_response["outputs"]
                
                # Merge outputs (combine all prompts and prompt frames)
                # Use unique offset: prompt_idx * 1000 + pf_idx * 100 to avoid ID collisions
                obj_id_offset = prompt_idx * 1000 + pf_idx * 100
                
                for frame_idx, frame_out in outputs_per_frame.items():
                    if frame_idx not in all_outputs:
                        all_outputs[frame_idx] = {
                            "out_obj_ids": [],
                            "out_probs": [],
                            "out_binary_masks": [],
                            "out_boxes_xywh": [],
                        }
                    
                    for i, obj_id in enumerate(frame_out["out_obj_ids"]):
                        new_obj_id = int(obj_id) + obj_id_offset
                        all_outputs[frame_idx]["out_obj_ids"].append(new_obj_id)
                        
                        if "out_probs" in frame_out and len(frame_out["out_probs"]) > i:
                            all_outputs[frame_idx]["out_probs"].append(frame_out["out_probs"][i])
                        else:
                            all_outputs[frame_idx]["out_probs"].append(1.0)
                        
                        if "out_binary_masks" in frame_out and len(frame_out["out_binary_masks"]) > i:
                            all_outputs[frame_idx]["out_binary_masks"].append(frame_out["out_binary_masks"][i])
                        
                        if "out_boxes_xywh" in frame_out and len(frame_out["out_boxes_xywh"]) > i:
                            all_outputs[frame_idx]["out_boxes_xywh"].append(frame_out["out_boxes_xywh"][i])
            
            if len(prompt_frames) > 1:
                print(f"  Total detections for '{prompt}': {total_detections}")
        
        # Convert lists to numpy arrays for visualization
        for frame_idx in all_outputs:
            frame_out = all_outputs[frame_idx]
            frame_out["out_obj_ids"] = np.array(frame_out["out_obj_ids"])
            frame_out["out_probs"] = np.array(frame_out["out_probs"])
            if frame_out["out_binary_masks"]:
                # Stack masks - they should be numpy arrays or tensors
                masks = []
                for m in frame_out["out_binary_masks"]:
                    if isinstance(m, torch.Tensor):
                        masks.append(m.cpu().numpy())
                    else:
                        masks.append(m)
                frame_out["out_binary_masks"] = np.stack(masks, axis=0) if masks else np.array([])
            else:
                frame_out["out_binary_masks"] = np.array([])
            
            if frame_out["out_boxes_xywh"]:
                boxes = []
                for b in frame_out["out_boxes_xywh"]:
                    if isinstance(b, torch.Tensor):
                        boxes.append(b.cpu().numpy())
                    else:
                        boxes.append(b)
                frame_out["out_boxes_xywh"] = np.stack(boxes, axis=0) if boxes else np.array([])
            else:
                frame_out["out_boxes_xywh"] = np.array([])
        
        # Close session
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        
        if not all_outputs:
            print("\nNo detections found for any prompt")
            return
        
        # Generate timestamp and create output folder structure
        # Structure: sam3_video_output/<timestamp>/
        #   - video.mp4
        #   - summary.json
        #   - frame_XXXX.jpg (extracted frames)
        output_config = config.get("output", {})
        use_timestamp = output_config.get("timestamp_prefix", True)
        
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = "default"
        
        # Create output folder: sam3_video_output/<timestamp>/
        video_output_base = output_dir / "sam3_video_output"
        video_output_dir = video_output_base / timestamp
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_video_path = video_output_dir / f"{video_path.stem}_lane_tracking.mp4"
        summary_path = video_output_dir / "summary.json"
        
        # Save output video (mask only, no bounding boxes)
        print(f"\nSaving output video to: {output_video_path}")
        save_mask_only_video(
            video_frames_for_vis,
            all_outputs,
            str(output_video_path),
            alpha=alpha,
            fps=fps,
            show_frame_numbers=show_frame_numbers,
            color_map=color_map,
        )
        
        # Save individual frame images if enabled (all frames)
        save_frame_images = output_config.get("save_frame_images", False)
        if save_frame_images:
            print(f"\nSaving all frame images to: {video_output_dir}")
            for frame_idx in tqdm(sorted(all_outputs.keys()), desc="Saving frames"):
                frame_rgb = video_frames_for_vis[frame_idx]
                frame_outputs = all_outputs[frame_idx]
                # Render frame without frame number overlay
                overlay = render_mask_only_frame(
                    frame_rgb, frame_outputs, frame_idx=None, alpha=alpha, color_map=color_map
                )
                # Save as JPEG
                frame_path = video_output_dir / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            print(f"Saved {len(all_outputs)} frame images")
        
        # Save summary
        summary = {
            "input_video": str(video_path),
            "output_video": str(output_video_path),
            "output_dir": str(video_output_dir),
            "timestamp": timestamp,
            "total_frames": total_frames,
            "fps": fps,
            "prompts_used": prompts,
            "lane_types": lane_types if lane_types else None,
            "prompt_frames": prompt_frames,
            "prompt_interval": prompt_interval,
            "config_file": str(args.config) if args.config else "default",
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Video tracking complete!")
        print(f"Output folder: {video_output_dir}")
        print(f"Video: {output_video_path.name}")
        print(f"Summary: {summary_path.name}")
        if save_frame_images:
            print(f"Frames: frame_0000.jpg - frame_{total_frames-1:04d}.jpg")
        print(f"{'='*60}")
    
    finally:
        # Shutdown predictor
        predictor.shutdown()


def run_single_frame_inference(args, config: dict):
    """Run inference on a single frame from a video file.
    
    This is much faster than processing all frames when you only need one.
    """
    from datetime import datetime
    from PIL import Image
    
    video_path = Path(args.video_file)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = args.single_frame
    
    print(f"\n{'='*60}")
    print(f"Running SAM3 SINGLE FRAME inference on: {video_path}")
    print(f"Frame index: {frame_idx}")
    print(f"{'='*60}\n")
    
    # Extract the single frame from video
    print(f"Extracting frame {frame_idx}...")
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_idx >= total_frames:
        print(f"Error: Frame {frame_idx} out of range (video has {total_frames} frames)")
        cap.release()
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    print(f"Frame size: {image.size[0]}x{image.size[1]}")
    
    # Get lane types from config
    lane_types = config.get("lane_types", None)
    if lane_types:
        prompts = [lt["prompt"] for lt in lane_types]
        print("\nUsing lane types:")
        for lt in lane_types:
            color_str = f"RGB{tuple(lt['color'])}"
            print(f"  - {lt['name']}: '{lt['prompt']}' -> {color_str}")
        color_map = {i: lt["color"] for i, lt in enumerate(lane_types)}
    elif "prompts" in config:
        prompts = config["prompts"]
        color_map = None
        print(f"Using prompts: {prompts}")
    else:
        prompts = ["white paint on road"]
        color_map = None
        print(f"Using default prompts: {prompts}")
    
    # Get inference settings
    inference_config = config.get("inference", {})
    alpha = inference_config.get("alpha", 0.7)
    confidence_threshold = inference_config.get("confidence_threshold", 0.3)
    
    # Initialize inferencer
    print("\nInitializing SAM3 inference...")
    inferencer = Sam3LaneInference(
        device=args.device,
        confidence_threshold=confidence_threshold,
    )
    
    # Run inference
    print(f"Running inference on frame {frame_idx}...")
    results = inferencer.run_inference(image, prompts)
    
    # Convert results to output format
    masks = []
    obj_ids = []
    probs = []
    
    for prompt_idx, (prompt, data) in enumerate(results.items()):
        print(f"  '{prompt}': {data['num_detections']} detection(s)")
        for i, mask in enumerate(data["masks"]):
            masks.append(mask)
            obj_ids.append(prompt_idx * 100 + i)
            probs.append(data["scores"][i] if i < len(data["scores"]) else 1.0)
    
    frame_outputs = {
        "out_obj_ids": np.array(obj_ids),
        "out_probs": np.array(probs),
        "out_binary_masks": np.stack(masks, axis=0) if masks else np.array([]),
    }
    
    # Render the frame with masks
    overlay = render_mask_only_frame(
        frame_rgb, frame_outputs, frame_idx=None, alpha=alpha, color_map=color_map
    )
    
    # Generate output path with timestamp
    output_config = config.get("output", {})
    use_timestamp = output_config.get("timestamp_prefix", True)
    
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_output_base = output_dir / "sam3_frame_output"
        frame_output_dir = frame_output_base / timestamp
    else:
        timestamp = None
        frame_output_base = output_dir / "sam3_frame_output"
        frame_output_dir = frame_output_base / "default"
    
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the frame
    frame_path = frame_output_dir / f"frame_{frame_idx:04d}.jpg"
    cv2.imwrite(str(frame_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"\n{'='*60}")
    print("Single frame inference complete!")
    print(f"Output: {frame_path}")
    print(f"Detections: {len(obj_ids)} object(s)")
    print(f"{'='*60}")


def extract_frames_from_video(args):
    """Extract specific frames from an existing video inference result.
    
    Usage:
        python sam3_lane_inference.py --extract-frames sam3_video_output/20260117_221308 --frames 24,50
    """
    import glob
    
    output_folder = Path(args.extract_frames)
    
    # Check if it's a relative path under output_dir
    if not output_folder.exists():
        # Try relative to output_dir
        output_folder = Path(args.output_dir) / args.extract_frames
    
    if not output_folder.exists():
        print(f"Error: Output folder not found: {args.extract_frames}")
        print(f"Tried: {Path(args.extract_frames)} and {Path(args.output_dir) / args.extract_frames}")
        return
    
    # Find the video file
    video_files = list(output_folder.glob("*.mp4"))
    if not video_files:
        print(f"Error: No video file found in {output_folder}")
        return
    
    video_path = video_files[0]
    print(f"\n{'='*60}")
    print(f"Extracting frames from: {video_path}")
    print(f"{'='*60}\n")
    
    # Parse frame indices
    if not args.frames:
        print("Error: --frames is required (e.g., --frames 24,50)")
        return
    
    frame_indices = [int(f.strip()) for f in args.frames.split(",")]
    print(f"Extracting frames: {frame_indices}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video has {total_frames} frames")
    
    extracted_count = 0
    for frame_idx in frame_indices:
        if frame_idx >= total_frames:
            print(f"Warning: Frame {frame_idx} out of range (max {total_frames-1}), skipping")
            continue
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}, skipping")
            continue
        
        # Save frame to same folder
        frame_path = output_folder / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        print(f"Saved: {frame_path.name}")
        extracted_count += 1
    
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"Extracted {extracted_count} frame(s) to: {output_folder}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    import argparse
    
    # Get the script directory for default paths
    script_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(description="SAM3 Lane Detection on nuScenes or Video Files")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["frame", "video", "video-file"],
        default="frame",
        help="Inference mode: 'frame' for per-frame (no tracking), 'video' for session-based with temporal tracking, 'video-file' for direct video file tracking",
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default=None,
        help="Path to video file for video-file mode (MP4, MOV, etc.)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated text prompts (e.g., 'white lane line,yellow lane line'). Overrides config file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file. Defaults to config.yaml in script directory.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(script_dir / "data" / "v1.0-mini"),
        help="Path to nuScenes mini data root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "data" / "sam3_results"),
        help="Output directory for results",
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
        help="Maximum samples per scene (for testing)",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated frame indices to process (e.g., '0,4,9' for frames 1,5,10)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Process only this scene (e.g., 'scene-1094')",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Frame index to add text prompt on (for video mode)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--use-native-sam3",
        action="store_true",
        help="Use native SAM3 repo implementation instead of HuggingFace (requires CUDA + triton)",
    )
    parser.add_argument(
        "--single-frame",
        type=int,
        default=None,
        help="Process only a single frame (e.g., --single-frame 24). Saves just that frame image.",
    )
    parser.add_argument(
        "--extract-frames",
        type=str,
        default=None,
        help="Extract specific frames from existing video result. Provide path to output folder (e.g., sam3_video_output/20260117_221308) and frame indices with --frames.",
    )
    
    args = parser.parse_args()
    
    # Handle extract-frames mode (extract frames from existing video result)
    if args.extract_frames is not None:
        extract_frames_from_video(args)
        return
    
    # Handle video-file mode
    if args.mode == "video-file":
        if not args.video_file:
            print("Error: --video-file is required for video-file mode")
            return
        
        # Load config to determine video vs frame mode
        config = load_config(args.config)
        
        # Handle single-frame mode
        if args.single_frame is not None:
            run_single_frame_inference(args, config)
            return
        
        video_mode = config.get("mode", "video")  # Default to video tracking
        
        print(f"Config mode: {video_mode}")
        
        if video_mode == "frame":
            run_video_file_frame_mode(args, config)
        else:  # "video" mode (default)
            run_video_file_tracking(args, config)
        return
    
    # Setup paths for nuScenes mode
    data_root = Path(args.data_root)
    
    # Initialize loader
    print("Initializing nuScenes loader...")
    loader = NuScenesLoader(str(data_root))
    
    # Filter scenes if specified
    if args.scene:
        target_scenes = [s for s in TARGET_SCENES if s.name == args.scene]
        if not target_scenes:
            print(f"Error: Scene '{args.scene}' not found in TARGET_SCENES")
            return
    else:
        target_scenes = TARGET_SCENES
    
    # Run appropriate mode
    if args.mode == "frame":
        run_frame_mode(args, loader, target_scenes)
    elif args.mode == "video":
        # Check CUDA availability for video mode
        if not torch.cuda.is_available():
            print("\n" + "="*60)
            print("WARNING: CUDA not available. Video mode requires CUDA + triton.")
            print("Falling back to frame mode (no temporal tracking).")
            print("To use video mode with temporal tracking, run on a CUDA-enabled machine.")
            print("="*60 + "\n")
            run_frame_mode(args, loader, target_scenes)
        else:
            run_video_mode(args, loader, target_scenes)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
