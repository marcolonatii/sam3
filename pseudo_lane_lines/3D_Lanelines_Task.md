# PT-2070 3D Lane Line Autolabeler

# 1\. Introduction

Why 3D lane lines are important for AV, below are 4 different consumers requires accurate 3D lane-line representation across the AV stack:

* For onboard perception: We need large volumes of ground-truth 3D lane-line annotations to train the onboard unified perception model to detect lane lines reliably.

* For map generation and maintenance: 3D lane lines serve as geometric anchors to build, validate, and update the vector map.

* For localization: 3D lane lines improve lane-level localization by matching observed lane geometry to the map, and this is particularly important under GNSS degradation or challenging visual conditions.

* For behavior and motion planning: 3D lane lines provide lane-level constraints (lane assignment, boundaries, curvature, grade) that improve trajectory selection in merges, splits, and complex interchanges.

3D lane lines can be produced either onboard (real-time) or offboard (offline labeling/map pipelines). This design doc focuses on an offboard 3D lane-line auto-labeling method. Offboard labels directly support map generation and maintenance, and they also improve the other consumers by providing the large, high-quality 3D lane-line annotations needed to train and iterate the onboard perception model.


Currently, we don’t use any model to auto-label 3D lane lines. The diagram above shows the current labeling workflow for producing 3D lane-line annotations, which takes approximately X time per slice, Y time for triage. We want to reduce iteration time and labeling cost by introducing an offboard auto-labeling system.

The UM BEV model performs lane-line detection, but it is a lightweight online model and is not designed for high-quality offboard labeling. For scalable, high-quality auto-labeling, we plan to explore either adding a lane-line head to HulkUM v2 or using an off-the-shelf foundation model. Building on our success with offboard 3D cuboid labeling, this approach can generate high-quality pseudo-labels that complement human labeling and further reduce cost and turnaround time.

# 2\. Requirements

* The method will need to predict the following attributes:  
  * 3D coordinates of lane polyline in vehicle frame coordinates.      
  * Color of each point of the lane line.   
  * Pattern of each point of lane line.   
  * Lane line type (one of CenterLine/LaneLine/EdgeLine).   
  * Lane line ID that can be used to look up merge, split and connection.  
* This method will need to be robust to the following failure modes:  
  * Weather conditions (snow, rain, etc)  
  * Lane merging and splitting  
  * Large road curves  
  * Construction zones (where traffic cones may indicate a different, temporary lane line)  
  * Instances where human trajectories intersect with or are close to the lane lines/lane

### Out of Scope 

### Reference

SparseLaneSTP, 4.1. Auto-labeling strategy, [Supp talks about SLA vs PNA, TCA](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Pittner_SparseLaneSTP_Leveraging_Spatio-Temporal_ICCV_2025_supplemental.pdf)

# 3\. System Design

## 3.3 SAM3 \+ Lidar

Pipeline Reference: [https://github.com/HKUST-Aerial-Robotics/MonoLaneMapping](https://github.com/HKUST-Aerial-Robotics/MonoLaneMapping)

Inputs:

* Camera frames (all cameras)  
* LiDAR scans  
* Camera-LiDAR calibration (extrinsics, intrinsics)  
* Ego pose per frame (from SLA cam-lidar joint calibrated)

Outputs:

* 3D lane polylines in world coordinates  
* Global lane instance ID per polyline (unique within slice)  
* Per-polyline attributes: lane type (center/lane/edge), color, pattern (solid/dashed)  
* Timestamps for each polyline (frame range)

Regardless of which front-end detector is used, raw per-frame outputs are insufficient for auto-labeling and require a second refinement stage. 

1. Accumulating observations along the trajectory: Multiple frames viewing the same lane are fused in world coordinates, filling spatial gaps and increasing point density.  
2. Denoising via redundancy: Outliers (false positives, misaligned points) are suppressed when they don't recur across frames; consensus geometry emerges from repeated observations.  
3. Re-associating across tracking resets: Geometry-based association (3D proximity, tangent consistency) links fragments that the front-end tracker separated, producing stable global IDs.  
4. Fitting smooth curves: Raw point clouds are converted into parametric curves (splines or polylines) that are smooth, continuous, and easy to edit.

### 3.3.1 Architecture

### 3.3.2 Frontend Detector Selection

| Criterion | Option 1: Add lane-line head to HulkUM v2 | Option 2: SAM3 \+ LiDAR lifting |
| ----- | ----- | ----- |
| Delta | Detection module is trained in-house (3D/BEV output) | Detection module is off-the-shelf (2D masks \+ tracking) |
| Time | Medium (needs training integration \+ iterations) | Fast (mostly inference \+ projection \+ fitting) |
| Shared required work | Same for both: pose alignment → aggregation/denoise → association/re-ID → polyline/spline fitting | Same for both: pose alignment → aggregation/denoise → association/re-ID → polyline/spline fitting |
| Engineering \+ compute cost | High (training infra \+ GPU cycles) | Low (inference only) |
| Impact to existing 3DOD | High (observed negative transfer/regression in the past) | Low (decoupled from 3DOD training) |
| Performance | Higher if we have good quality labels and can iterate safely | Good for MVP; may need finetune/training to match best-in-class |

Trade-off: Option 1 vs 2 is mainly a choice of the front-end detector \+ impact on 3DOD \+ infra cost. The post-detection refinement pipeline (aggregation, association, polyline fitting) is required either way.

Context: In HulkUMv1, adding a lane-line-specific head ([BEV-laneDet](https://github.com/latai-pd/av/blob/main/kits/ml/pytorch/heads/lanes/bev_lane_head.py)) impacted object detection performance. Even with detached gradients, regression was observed. Alternating training regimes (training 3DOD head for N iterations with lane head fixed, then vice versa) could mitigate this.

### 3.2.2 Backend Laneline Refinement

**1: Pose-Aligned Accumulation**  
Transform per-frame lane points from ego/LiDAR frame to a shared world frame, enabling multi-frame fusion.

Process:

1. For each frame t, receive 3D lane points P\_t in ego frame and ego pose T\_world\_ego(t).  
2. Transform: P\_world \= T\_world\_ego(t) · P\_t  
3. Associate points by front-end track\_id (within tracking window).  
4. Accumulate points across frames, grouped by track\_id.  
   

Output: Per-track point clouds in world coordinates.

**2\. Multi-Frame Denoising**  
Remove outliers and reduce noise in the accumulated point clouds.

| Methods | Description | When to Use |
| :---- | :---- | :---- |
| Statistical outlier removal | Remove points with mean neighbor distance \> k·σ | Always; removes isolated noise |
| Voxel downsampling | Grid-based averaging to uniform density | Always; reduces redundant points |
| RANSAC line/curve fitting | Fit model, reject inliers outside threshold | If significant outliers remain |
| Observation count filtering | Remove lanes with \< N observations | Always; suppresses false positives |

Parameters (initial values):

* Voxel size: 0.2 m (lateral) × 0.5 m (longitudinal)  
* Min observations per lane: 4 frames  
* Outlier threshold: 2σ from local mean

Output: Cleaned, downsampled point clouds per track.

**3\. Geometry-Based Association**  
Link track fragments into globally consistent lane instances. The front-end tracker provides locally consistent IDs, but tracking resets (occlusions, scene cuts) fragment physical lanes into multiple tracks. This stage re-associates fragments using geometric cues.

Association Criteria:

| Criterion | Implementation |
| :---- | :---- |
| 3D proximity | Endpoint distance \< threshold (e.g., 2 m) |
| Tangent consistency | Direction at endpoints aligns within ±15° |
| Overlap ratio | If tracks overlap spatially, require \> 70% point overlap to merge |
| Lateral sequence consistency | If lane A is left of lane B in one segment, it should remain left of B in the associated segment |

Association Algorithm:

1. Build candidate pairs: (track\_i, track\_j) where endpoints are within proximity threshold.  
2. Score each pair by tangent alignment \+ overlap.  
3. Greedy or Hungarian matching to assign global lane IDs.  
4. Handle merges/splits: If a track's points diverge (e.g., at a lane split), create new global IDs for the branches.

Output: Global lane ID assigned to each point cloud.

**4\. Curve Fitting**  
Convert the fused, denoised point cloud into a smooth, parametric curve suitable for labeling output.

* Fitting Method: Control-point-based spline (B-spline or Catmull-Rom).

Why Splines?

* Smooth and continuous (C1 or C2)  
* Editable: labelers can adjust control points  
* Compact: fewer parameters than dense polyline  
* Well-suited for curve optimization (can add smoothness priors)

Process:

1. Control point initialization: Place control points at regular intervals (e.g., 3 m chord length) along the point cloud skeleton.  
2. Spline fitting: Optimize control point positions to minimize distance to data points \+ smoothness regularization.  
3. Resampling: Evaluate spline at fixed intervals (e.g., 0.5 m) to produce final polyline output.

Output Format:

* poly3d label type, layer lane\_line3d\_bev\_user  
* List of (x, y, z) coordinates, ordered in direction of travel  
* Attributes: type, color, pattern

# 4\. Development and Deployment Plan

## 4.1 Current Cuboid Labeling Systems

From a highlevel, the multi-stage autolabeling runs in below call chain:

1. [generate\_pseudo\_cuboids\_cli.py](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/generate_pseudo_cuboids_cli.py#L11) (Entry point for autolabeling)

 	→ 2\. [pseudo\_label\_openlabel\_file()](https://github.com/latai-pd/av/blob/142642d0120cc84a84418263d95e0dcd2677b04d/autonomy/perception/labels/pseudo_cuboids/generate_pseudo_cuboids_cli.py#L77)   
→ 3\. [detect\_hulk\_um\_tracks()](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/detect_pseudo_cuboids.py#L177)   
→ 4\. [process\_data\_row\_um()](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/detect_pseudo_cuboids.py#L235)  
→ 5\. [process\_data\_row\_um\_state\_estimation\_dynamic\_tracker()](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/detect_pseudo_cuboids.py#L311)  
→ 6\. Returns [AutoLabelerResult](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/result_types.py#L37)

From 3\. [detect\_hulk\_um\_tracks()](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/detect_pseudo_cuboids.py#L177), it runs for a single log slice

1. Parses OpenLabel data into UnifiedModelData.  
2. Resolves the tracker method based on platform/speed/locale.  
3. Loads the detector model (Hulk UM).  
4. Runs process\_data\_row\_um() to get detections \+ tracks.

From 4\. [process\_data\_row\_um()](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/detect_pseudo_cuboids.py#L235) , it runs the detector and hands off to tracking

1. Calls um\_inferencer.process\_row() — forward pass through the BEV (bird's-eye view) neural network, producing per-frame 3D bounding box detections.  
2. Trims context frames (non-inference frames used only for temporal context).  
3. Delegates to process\_um\_results() that routes to the appropriate tracker implementation. 

   
From 5\. [process\_data\_row\_um\_state\_estimation\_dynamic\_tracker()](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/detect_pseudo_cuboids.py#L311), this is tracking logic

1. Split detections into stationary and dynamic objects.  
2. Refine & track each group separately — stationary objects are refined in place (map frame), dynamic objects are tracked across frames (vehicle frame).  
3. Optionally refine dynamic tracks with a secondary model.  
4. Package both groups (dynamic\_objects\_vehicle\_frame, static\_objects\_map\_frame) into the final result.

## 4.2 New Laneline Labeling Systems

1\. The CLI [generate\_pseudo\_cuboids\_cli.py](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/pseudo_cuboids/generate_pseudo_cuboids_cli.py#L11) already orchestrates the full pipeline. It is most straightforward to add the mode flag to perform autolabeling in different modes. In later phrase, we could consider rename it to be generic generate\_autolabels\_cli.py

- Cuboids and lanelines (P0)  
- Cuboid only (P2)  
- Lane lines only (P2). [Akash Baskaran](mailto:abaskaran@lat.ai) suggested that It might be difficult to have lane lines only, because AutolabelerResult needs UM data as a requirement, and we get that only from cuboids.

Per [PT-2092 \- SALSA: Slice AutoLabeling Service on Arrival](https://docs.google.com/document/d/1HrNz-LiiJCpz1X9CN3A8cWPg9tqKxUIv3CSxI3f7jiw/edit?usp=sharing) and [PT-2061 \- Share Hulk UM Inference Between Prelabeling and Active Learning Follow-up](https://docs.google.com/document/d/1alauRL0oPcUOsD_dZn-yODkS6jb8sdc6aoAd9r-WsLw/edit?tab=t.0#heading=h.7flbfqtlomc8), SML has some ongoing plans to move HulkUM inference to pseudo cuboids datasets. We could further discuss if this is absolutely required right now to add an option for the labeling team to run lane-line autolabeling independently. This would allow them to optimize GPU usage by reusing the pseudo-cuboids dataset for the cuboid-labeling job. If the goal is to flush out the auto-labeling pipeline for 3D lane lines. Having lane line autolabeling being a part of this script reduces a lot of the overhead of setting up the inference pipeline, data loading and many other things. It does add the additional baggage that we need to wait for cuboid detection to complete before we can do lane line labeling.

Yukai Wang suggested independent dataset loading for lane lines (decoupled from cuboid data hydration). For the first version, we should aim for pipeline completeness by reusing the existing data loading path. If independent hydration can bring a meaningful speedup, it can come as a second-stage optimization, we can leverage Yukai's help for that refinement.

2\. Add lane\_lines\_map\_frame to **AutoLabelerResult** 

```py
@dataclass
class AutoLabelerResult:
   #: Dynamic objects in vehicle coordinate.
   dynamic_objects_vehicle_frame: list[PseudoCuboidObjectFrame]
   #: Static objects in map coordinate. 
   static_objects_map_frame: Optional[list[PseudoCuboid]]
   #: Ground truth objects in vehicle coordinate.
   ground_truth_objects_vehicle_frame: Optional[list[CuboidLabelObjectFrame]]
   #: Results from the detector.
   um_results: UMDetections
   #: Vehicle platform of the input.
   platform: str | None
   #: NEW Add 3D lane lines in map coordinate. PseudoLaneLine3D is an existing datatype can be reused
   lane_lines_map_frame: Optional[list["PseudoLaneLine3D"]] = None
```

3\. Create pseudo\_lane\_lines folder

```py
autonomy/perception/labels/
├── pseudo_cuboids/              # Existing
│   ├── detect_pseudo_cuboids.py
│   └── ...
├── pseudo_lane_lines/                  # NEW
│   ├── BUILD.bazel
│   └── detect_pseudo_lanelines.py    # Orchestrator: calls stages in order
│   ├── lift_2d_to_3d.py              # 2D → 3D points (vehicle frame)
│   ├── accumulate_observations.py    # Multi-frame fusion in world coordinates
│   ├── denoise.py                    # Outlier suppression via cross-frame consensus
│   ├── associate_fragments.py        # Geometry-based re-association + global IDs
│   └── spline_fit.py                 # Point cloud → splines/polylines

```

In detect\_pseudo\_lanelines.py, it calls **detect\_lane\_lines\_for\_slice** which orchestrates the multi-stage laneline pipeline: SAM3 → lift → accumulate → denoise → associate → fit

Following the existing cuboid pipeline pattern, **detect\_lane\_lines\_for\_slice()**is consumed in **detect\_pseudo\_cuboids.py** at the same level where cuboid results are assembled — inside **process\_data\_row\_um\_state\_estimation\_dynamic\_tracker()**.

New Call chain

```py
generate_pseudo_cuboids_cli.py
  → pseudo_label_openlabel_file()
    → detect_hulk_um_tracks() # Can be renamed as detect_pseudo_labels
      → process_data_row_um() 
        → process_um_results()
          → process_data_row_um_state_estimation_dynamic_tracker()
            ├── ... existing cuboid detection & tracking ...
            ├── detect_lane_lines_for_slice()          # NEW — called here
            └── AutoLabelerResult(lane_lines_map_frame=...)  # stored here
      → get_tracks_with_only_keyframes()               # Extending for lane lines
    → write to OpenLabel JSON                           # Extending for lane lines

```

Note that: The function name **detect\_hulk\_um\_tracks** can be renamed as **detect\_pseudo\_labels,** because it is the top-level orchestrator , which will cover cuboids \+ lane lines. This is not super urgent and can be left for later refinement.

Deployment reference: 

1. How we deploy the the new hulk um model: [https://latitudeai.atlassian.net/wiki/spaces/SE/pages/441352265/Training+and+Releasing+the+Hulk+UM+prelabeler](https://latitudeai.atlassian.net/wiki/spaces/SE/pages/441352265/Training+and+Releasing+the+Hulk+UM+prelabeler)  
2. How we tested the dagster image: [https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/dagster/dags/prelabeling/README.adoc](https://github.com/latai-pd/av/blob/main/autonomy/perception/labels/dagster/dags/prelabeling/README.adoc)

## 4.3 Timelines

The below targets assume 1 engineer working sequentially. With a second engineer, the evaluation tasks (end-to-end eval script, ALFA dataset curation, online vs. offboard benchmark) can be developed in parallel since they have no code dependency on the pipeline.

| Priority | Component | Target |
| :---- | :---- | :---- |
| P0 | SAM3 2D detection — run SAM3 inference on PV images to produce per-frame 2D lane masks | Done |
| P0 | SAM3 2D tracking — track lane mask instances across frames to produce consistent track IDs | S2.3 |
| P0 | 2D → 3D lifting (lift\_2d\_to\_3d.py) — project 2D lane masks to 3D via LiDAR depth \+ calibration | S2.4 |
| P0 | Pose-aligned accumulation (accumulate\_observations.py) — transform per-frame 3D points to world frame, group by track\_id | S3.1 |
| P0 | Multi-frame denoising (denoise.py) — outlier removal, voxel downsampling, observation count filtering | S3.1 |
| P0 | Geometry-based association (associate\_fragments.py) — re-associate track fragments via 3D proximity, tangent consistency, overlap | S3.2 |
| P0 | Spline fitting (spline\_fit.py) — fit Catmull-Rom splines (chord 3.0 m, tau 0.5), resample to polyline at 0.5 m | S3.2 |
| P0 | Implement end-to-end eval script (4.4.2) — final 3D lanes vs GT: recall, precision, F-score, X/Z error | S3.2 |
| P0 | CLI integration — add mode flag to generate\_pseudo\_cuboids\_cli.py, extend AutoLabelerResult, OpenLabel JSON output | S3.3 |
| P0 | ALFA eval dataset curation (4.4.4) — curate basic \+ challenging scene sets for failure mode analysis | S3.3 |
| P0 | Run end-to-end benchmarking — evaluate on ALFA eval sets, report metrics | S3.3 |
| P1 | Online vs. offboard baseline benchmark (4.4.3) — compare offboard auto-labels against onboard UM on same slices | S3.4 |
| P0 | Deploy to dagster image — follow existing cuboid prelabeler pattern, validate on test slices in staging | S3.5 |

### 4.3.1 Detailed Status Tracking

| Component | Status | Issues | Findings | Solutions |
| :---- | :---- | :---- | :---- | :---- |
| SAM3 2D detection (`sam3_lane_inference.py`) | Done | HF Transformers API instability across versions; MPS/CPU fallback needed for local dev | Native SAM3 repo on CUDA is more stable than HF Transformers wrapper; text prompts outperform point prompts for lane lines | Dual backend: native SAM3 on CUDA, HF Transformers on MPS/CPU; lazy imports to avoid hard dependency |
| SAM3 2D tracking | In Progress | Tracking resets at occlusions fragment lane IDs; video mode requires native SAM3 repo (not HF) | SAM3 video mode propagates masks across frames with consistent object IDs; prompt_interval=10 balances coverage vs speed | Use SAM3 video mode with periodic re-prompting; downstream association stage (S3.2) will re-link fragments |
| 2D → 3D lifting (`lift_2d_to_3d.py`) | Done | Lane lines are thin — expect 10-50 LiDAR hits per lane per frame; overlapping masks from multiple prompts | nuScenes 4-step transform chain validated against devkit `map_pointcloud_to_image()`; ego-frame output aligns with downstream accumulation | Depth filter 1-80m; highest-confidence mask wins for overlapping regions; diagnostics log point counts per lane type |
| Pose-aligned accumulation | Not Started | — | — | — |
| Multi-frame denoising | Not Started | — | — | — |
| Geometry-based association | Not Started | — | — | — |
| Spline fitting | Not Started | — | — | — |
| End-to-end eval script | Not Started | — | — | — |
| CLI integration | Not Started | — | — | — |
| ALFA eval dataset curation | Not Started | — | — | — |
| End-to-end benchmarking | Not Started | — | — | — |
| Online vs. offboard benchmark | Not Started | — | — | — |
| Deploy to dagster image | Not Started | — | — | — |

## **4.4 Evaluation Metrics**

### 4.4.1 End-to-End Metrics

Evaluate final polyline output against GT 3D lane annotations.

| Metric | Description |
| :---- | :---- |
| Lane recall | Fraction of GT lanes matched by a predicted lane |
| Lane precision | Fraction of predicted lanes matched by a GT lane |
| F-score | Harmonic mean of recall and precision |
| Category accuracy | Fraction of matched lanes with correct type/color/pattern |
| Chamfer distance (m) | Mean point-to-point distance for matched lanes |
| X-error close (m) | Lateral error for points in 0–40 m range |
| X-error far (m) | Lateral error for points in 40+ m range |
| Z-error close (m) | Height error for points in 0–40 m range |
| Z-error far (m) | Height error for points in 40+ m range |

### 4.4.2 Online vs. Offboard Baseline Benchmark (P1)

The offboard auto-labels must be strictly more accurate than the onboard UM BEV lane detection. Benchmark both on the same slices by extracting 3D lane outputs from online UM inference and evaluating against GT using the same end-to-end metrics (4.4.2). Requires additional work to extract and format 3D lane outputs from the online UM model for evaluation.

### 4.4.3 Eval Curation and Failure Mode Analysis

Per requirements (Section 2), all metrics above should be reported separately on basic and challenging evaluation sets:

* weather conditions (snow, rain), lane merging and splitting, large road curves.  
* We will use ALFA to curate evaluation datasets for each of these scenarios.

