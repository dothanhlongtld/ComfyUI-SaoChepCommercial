# ComfyUI-SaoChepCommercial

Custom node extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) providing optical skin-tone calibration, CIELAB reference chroma restoration, delivery aspect-ratio resizing, and color telemetry logging.

Developed for the **Saochep.net Commercial Wan 2.1 Video Production Pipeline**.

---

## Features

1. **`ReferenceChromaLightTest` / `SaoChepReferenceChroma`**:
   - **CIELAB Skin Tone Harmonization**: Measures cheek core color (P20–P80 percentile) on the reference image and driving frames using OpenCV YuNet ONNX face landmarking.
   - **Strict Luminance Lock ($L_{out} = L_{in}$)**: Keeps 100% lighting, shadow volume, and environmental depth intact; shifts only chromaticity ($\Delta a^*, \Delta b^*$).
   - **Air-gapped & Offline**: Bundles `face_detection_yunet_2023mar.onnx` (227 KB) directly with zero external network downloads.
   - **GPU Accelerated**: Ultra-fast PyTorch CUDA tensor batching (~0.5s for 220+ frames).
   - **Thread-safe**: Protected by threading re-entrant locks for multi-worker / concurrent GPU execution.

2. **`SaoChepDeliveryResize`**:
   - High-quality Lanczos delivery resampling preserving exact aspect ratio for commercial delivery ($720\times 1280$ or $1080\times 1920$).

3. **`SaoChepSaveReport`**:
   - Telemetry exporter saving structured color telemetry JSON with path validation and safe tenant isolation.

---

## Installation

### Automatic / Git Clone
Navigate to your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI-SaoChepCommercial.git
```

### Dependencies
All required libraries are part of standard ComfyUI installations:
```bash
pip install -r requirements.txt
```
(Requires: `torch`, `opencv-python`, `pillow`, `numpy`)

---

## Node Mappings

| Internal Class Type | Display Name | Category | Function |
|---|---|---|---|
| `ReferenceChromaLightTest` | Reference Chroma Light Test (Optimized V2.2) | OpticalStudioEnhance | CIELAB skin tone compensation |
| `SaoChepReferenceChroma` | SaoChep — Reference color (opt-in) | OpticalStudioEnhance | Thread-safe CIELAB skin tone compensation |
| `SaoChepDeliveryResize` | SaoChep — Delivery resolution | SaoChep/Commercial | Aspect-safe delivery upscale |
| `SaoChepSaveReport` | SaoChep — Save color telemetry | SaoChep/Commercial | Color telemetry JSON export |

---

## License & Notes
- YuNet model weights `face_detection_yunet_2023mar.onnx` are provided under OpenCV Zoo Apache 2.0 license.
- Inherited MediaPipe implementation is bypassed in favor of native OpenCV YuNet for Python 3.10–3.13 compatibility.
