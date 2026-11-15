# SaoChep head-swap CUSTOM-NODE slice

Only the reachable reference preparation, head/pose masks, anchor selection, alignment, and conditioning contract adapters from `SAOCHEP_HEADSWAP_CODE_ONLY/workflow_base.json` are ported here. Job creation, tier/profile rules, graph construction, media lifecycle, duration/prompt/audio handling, bounded generation, postprocessing and export remain caller-owned. No endpoint or model download is performed by these nodes.

## Namespaced node IDs and sockets

| Registered ID | Required sockets → outputs |
|---|---|
| `SaoChepHeadSwapHandGuides` | `image: IMAGE`, `max_pixels: INT` → `hand_boxes: BOUNDING_BOX`, `hand_landmarks_json: STRING` |
| `SaoChepHeadSwapCrop` | `image: IMAGE`, `driving_images: IMAGE`, `hood_mask: MASK`, `hand_mask: MASK`, `hand_landmarks_json: STRING` → `clip_rgb: IMAGE`, `reference_rgb_hq: IMAGE`, `reference_alpha_hq: MASK`, tight RGB/MASK, report STRING, native RGBA IMAGE |
| `SaoChepHeadSwapBackfaceMask` | `images: IMAGE`, `track_data: SAM3_TRACK_DATA` → `head: MASK`, `pose_mask: IMAGE` (blue-on-white), face core, forbidden/accessory/skin/neck masks and geometry JSON |
| `SaoChepHeadSwapAnchor` | `images: IMAGE`, `geometry: STRING` → selected one-frame IMAGE, report STRING, selected geometry STRING |
| `SaoChepHeadSwapStableAnchor` | `images: IMAGE`, `geometry: STRING` → selected one-frame IMAGE and reports |
| `SaoChepHeadSwapChinEyeAlign` | `reference: IMAGE`, `alpha: MASK`, `driving: IMAGE`, thresholds, optional geometry → aligned IMAGE, aligned MASK, route report STRING |
| `SaoChepHeadSwapReferenceMask` | `mask: MASK` → blue-on-black `IMAGE` for SCAIL reference conditioning |
| `SaoChepHeadSwapFrameFit` | `images: IMAGE`, `target_frames: INT` (raw requested SCAIL frames) → `images: IMAGE`, `aligned_frames: INT` (smallest 4n+1 count ≥ requested; tail padding ≤4) |
| `SaoChepHeadSwapContracts` | driving IMAGE/mask IMAGE, reference IMAGE/alpha MASK/reference-mask IMAGE, crop/policy JSON → validated driving/mask/reference IMAGE and report |

FrameFit receives the raw requested count (`round(seconds * fps)`, e.g. 24 at 1s/24fps); it aligns conditioning IMAGE/mask inputs to 25 but does not change SCAIL `frames=24`. `aligned_frames` is available for `video_frames` fallback only. If source conditioning is more than four frames short of that aligned count, it fails rather than repeating unboundedly. It changes conditioning images only, not source duration/audio. All registered IDs are package-prefixed to avoid collision with the bundle's unqualified IDs. The graph owner must map its logical `HSCommercialResolutionCrop`, `HSBackfaceHeadMaskTrial`, `HSFirstFrameAnchorTrial`, `HSAdaptiveChinEyeR2`, `HSFinalReferenceMask`, `HSFinal20261109Contracts`, and `HSFinal20261109HandGuides` nodes to these IDs. Existing `AWGMemoryGenerate` contracts are unchanged: `pose_video_mask` is required `IMAGE`; `reference_image_mask` remains optional `IMAGE`. The graph wires them directly; no AWG handle-to-IMAGE materializer or full-memory fallback is added. Reference RGB and masks retain the exact driving canvas dimensions, including source-aspect native dimensions (crop target dimensions come from `driving_images`, not fixed portrait constants).

## Frozen source and model assets

`frozen_crop_sha256.json` is the adapted-code integrity boundary: it fingerprints the exact vendored source/config files required by the crop runtime, not the unused original audit/client/job-runner code or its original whole-directory manifest. Runtime verifies this boundary before crop inference. Adapted paths resolve from the package directory relative to `__file__` (the ComfyUI clone folder may be renamed); no `/content` paths or runtime downloads remain. The bundle's external weights remain external and their original sizes/hashes are pinned in `EXTERNAL_ASSETS.json`; the worker provisioning/readiness step checks them. Place them beneath the installed node at `custom_nodes/SaoChepCommercial/saochep_headswap/frozen_crop/models/` (resolved relative to this package's `__file__`, so clone folder names do not affect lookup). The canonical node-relative location is required for every model and its configs:

| Asset | Destination relative to `custom_nodes/SaoChepCommercial/saochep_headswap/` | SHA-256 |
|---|---|---|
| `bisenet_face_parsing.onnx` | `frozen_crop/models/bisenet_face_parsing.onnx` | `95995975429e40f0001b60571c7b3f3ff5d707c07d5a8afc90fdded71c84ff67` |
| `yoloface_8n.onnx` | `frozen_crop/models/yoloface_8n.onnx` | `821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929` |
| YuNet `face_detection_yunet_2023mar.onnx` | `frozen_crop/models/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_landmarker.task` | `frozen_crop/models/face_landmarker.task` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| `hand_landmarker.task` | `frozen_crop/models/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `hair.tflite`, `multi.tflite` | `frozen_crop/models/{hair,multi}.tflite` | See bundle `EXTERNAL_ASSETS.json` |
| BiRefNet Lite `model.safetensors` | `frozen_crop/models/birefnet_lite/model.safetensors` | `ce8bcfc045e336322c0424a5863dcfb7e9ce8fed0a5fd4d1b2b20adf12d97243` |
| ViTMatte `model.safetensors` | `frozen_crop/models/vitmatte_small/model.safetensors` | `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb` |

Corresponding model config files are vendored with the crop source. The new slice does not include CodeFormer or alternate checkpoints. External dependency prerequisites are the already-provisioned ComfyUI PyTorch, NumPy, Pillow, OpenCV, ONNX Runtime, MediaPipe Tasks, Transformers/safetensors, SAM3 tracking node, and model weights. Provenance/licensing review of model/code assets remains a release gate.

## Verification / limits

`SaoChepHeadSwapCrop.validate_dependencies()` checks MediaPipe Tasks, ONNX Runtime, local Transformers/ViTMatte, timm, kornia, and imports the vendored BiRefNet module without loading weights or running inference. Run `python3 -m pytest -q tests/test_headswap_nodes.py` plus the README CPU commands in the repository root. Head-swap regression tests require NumPy, pytest, and CPU OpenCV. The alignment-crop regression executes the real OpenCV resize to verify both scale axes; model/Torch/MediaPipe execution remains mocked. CPU tests also cover schemas, IDs, integrity manifest, AWG socket compatibility and mocked tensor behavior. They do not prove model execution, ComfyUI registration, SAM3 runtime, CUDA/VRAM fit, or rendered head-swap quality. Those require the provisioned GPU ComfyUI environment and real-media acceptance.
