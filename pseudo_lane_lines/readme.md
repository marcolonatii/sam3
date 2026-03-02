## SAM3 Lane Detection

Lane detection inference using SAM3 (Segment Anything Model 3) on nuScenes driving scenes.

### Frame Mode (CPU/MPS/CUDA)
Per-frame inference without temporal tracking. Works on all devices:

```bash
# Process specific frames from a scene
python sam3_lane_inference.py --mode frame --scene scene-1094 --frames "0,4,9"

# Process all frames from a scene
python sam3_lane_inference.py --mode frame --scene scene-1094

# Process all target scenes
python sam3_lane_inference.py --mode frame
```

### Video Mode (CUDA only)
Session-based video inference with temporal tracking. Requires CUDA + triton:

```bash
# Process specific frames with temporal tracking
python sam3_lane_inference.py --mode video --scene scene-1094 --frames "0,4,9"

# Process full scene with temporal tracking
python sam3_lane_inference.py --mode video --scene scene-1094
```

### Additional Options
```bash
# single frame mode
python laneline/sam3_lane_inference.py --mode video-file --video-file /workspace/data/nuscenes/turn.mov --output-dir /workspace/data/nuscenes/sam3_output --single-frame 50

# video mode
# Make sure config.yaml has: mode: "video"
python laneline/sam3_lane_inference.py --mode video-file --video-file /workspace/data/nuscenes/turn.mov --output-dir /workspace/data/nuscenes/sam3_output

# extract frame 
python laneline/sam3_lane_inference.py --extract-frames /workspace/data/nuscenes/sam3_output/sam3_video_output/20260117_230848 --frames 24 --output-dir /workspace/data/nuscenes/sam3_output 2>&1

# Use native SAM3 implementation (CUDA only, potentially faster)

python sam3_lane_inference.py --mode frame --use-native-sam3

# Custom confidence threshold
python sam3_lane_inference.py --mode frame --confidence-threshold 0.5

# Specify device manually
python sam3_lane_inference.py --mode frame --device cuda
```

### Target Scenes
- `scene-0061`: Easy daylight driving (sanity check)
- `scene-0553`: Urban clutter/occlusions stress
- `scene-0103`: Hard visibility/lighting stress
- `scene-0916`: Topology complexity stress
- `scene-1094`: Adverse condition (night + rain)


