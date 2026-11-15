"""External asset destinations are relative to this custom-node checkout."""
from .frozen_crop.paths import asset_path

EXTERNAL_ASSET_DESTINATIONS = {
    "bisenet_face_parsing.onnx": "saochep_headswap/frozen_crop/models/bisenet_face_parsing.onnx",
    "yoloface_8n.onnx": "saochep_headswap/frozen_crop/models/yoloface_8n.onnx",
    "face_detection_yunet_2023mar.onnx": "saochep_headswap/frozen_crop/models/face_detection_yunet_2023mar.onnx",
    "face_landmarker.task": "saochep_headswap/frozen_crop/models/face_landmarker.task",
    "hand_landmarker.task": "saochep_headswap/frozen_crop/models/hand_landmarker.task",
    "hair.tflite": "saochep_headswap/frozen_crop/models/hair.tflite",
    "multi.tflite": "saochep_headswap/frozen_crop/models/multi.tflite",
    "birefnet_lite/model.safetensors": "saochep_headswap/frozen_crop/models/birefnet_lite/model.safetensors",
    "vitmatte_small/model.safetensors": "saochep_headswap/frozen_crop/models/vitmatte_small/model.safetensors",
}
