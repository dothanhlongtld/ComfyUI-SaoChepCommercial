"""CPU business contracts for the locally registered head-swap adapters."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfy_custom_nodes.SaoChepCommercial.saochep_headswap"


def load_package():
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    for name in ("comfy_custom_nodes", "comfy_custom_nodes.SaoChepCommercial"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / "saochep_headswap")]
            sys.modules[name] = module
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, ROOT / "saochep_headswap" / "__init__.py",
        submodule_search_locations=[str(ROOT / "saochep_headswap")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


class Tensor(np.ndarray):
    """Small NumPy tensor facade for node logic without installing PyTorch."""
    device = "cpu"

    def __new__(cls, value):
        return np.asarray(value, dtype=np.float32).view(cls)

    def __array_finalize__(self, obj):
        if obj is not None:
            self.device = getattr(obj, "device", "cpu")

    def to(self, dtype):
        return self.astype(dtype).view(Tensor)

    def cpu(self):
        return self

    def float(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def expand(self, shape):
        return np.broadcast_to(self, shape).copy().view(Tensor)

    def max(self, dim=None, **kwargs):
        return types.SimpleNamespace(values=np.asarray(self).max(axis=dim))

    def mean(self, dim=None, **kwargs):
        return np.asarray(self).mean(axis=dim)


def install_torch_facade(monkeypatch):
    torch = types.SimpleNamespace(
        float32=np.float32,
        zeros=lambda shape, **kwargs: Tensor(np.zeros(shape, np.float32)),
        ones=lambda shape, **kwargs: Tensor(np.ones(shape, np.float32)),
        from_numpy=lambda value: Tensor(value),
        cat=lambda values, dim=0: Tensor(np.concatenate(values, axis=dim)),
        isfinite=np.isfinite,
        equal=lambda left, right: np.array_equal(left, right),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_registry_ids_are_namespaced_and_preserve_graph_socket_types():
    mapping = load_package().NODE_CLASS_MAPPINGS
    assert set(mapping) == {
        "SaoChepHeadSwapCrop", "SaoChepHeadSwapReferenceMask", "SaoChepHeadSwapHandGuides",
        "SaoChepHeadSwapAnchor", "SaoChepHeadSwapStableAnchor", "SaoChepHeadSwapChinEyeAlign",
        "SaoChepHeadSwapBackfaceMask", "SaoChepHeadSwapContracts", "SaoChepHeadSwapFrameFit",
    }
    crop = mapping["SaoChepHeadSwapCrop"]
    assert crop.RETURN_TYPES == ("IMAGE", "IMAGE", "MASK", "IMAGE", "MASK", "STRING", "IMAGE")
    assert crop.INPUT_TYPES()["required"].keys() == {
        "image", "driving_images", "hood_mask", "hand_mask", "hand_landmarks_json"}
    mask = mapping["SaoChepHeadSwapBackfaceMask"]
    assert mask.RETURN_TYPES == ("MASK", "IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK", "STRING")
    assert mask.INPUT_TYPES()["required"]["track_data"] == ("SAM3_TRACK_DATA",)
    fit = mapping["SaoChepHeadSwapFrameFit"]
    assert fit.INPUT_TYPES()["required"].keys() == {"images", "target_frames"}
    assert fit.RETURN_TYPES == ("IMAGE", "INT")


def test_crop_readiness_validates_import_apis_without_loading_models(monkeypatch):
    package = load_package()
    mapping = package.NODE_CLASS_MAPPINGS
    vision = types.SimpleNamespace(**{name: type(name, (), {"create_from_options": staticmethod(lambda **kwargs: None)})
        for name in ("FaceLandmarker", "HandLandmarker", "ImageSegmenter")})
    for name in ("FaceLandmarkerOptions", "HandLandmarkerOptions", "ImageSegmenterOptions"):
        setattr(vision, name, type(name, (), {}))
    mp = types.ModuleType("mediapipe")
    mp.tasks = types.SimpleNamespace(BaseOptions=type("BaseOptions", (), {}), vision=vision)
    ort = types.SimpleNamespace(SessionOptions=type("SessionOptions", (), {}), InferenceSession=type("InferenceSession", (), {}))
    timm = types.ModuleType("timm")
    timm.layers = types.SimpleNamespace(**{name: object() for name in ("DropPath", "to_2tuple", "trunc_normal_")})
    kornia = types.ModuleType("kornia")
    kornia.filters = types.SimpleNamespace(laplacian=object())
    transformers = types.SimpleNamespace(**{name: type(name, (), {"from_pretrained": staticmethod(lambda *a, **kw: pytest.fail("must not load model"))})
        for name in ("AutoModelForImageSegmentation", "VitMatteImageProcessor", "VitMatteForImageMatting")})
    local_module = PACKAGE_NAME + ".frozen_crop.models.birefnet_lite.birefnet"
    engine_module = PACKAGE_NAME + ".frozen_crop.crop.master_head_engine_v100"
    pipeline_module = PACKAGE_NAME + ".frozen_crop.crop_pipeline_v101"
    cv2 = types.ModuleType("cv2")
    cv2.FaceDetectorYN = types.SimpleNamespace(create=lambda *args, **kwargs: None)
    for name in ("warpAffine", "resize", "cvtColor", "distanceTransform", "fillPoly"):
        setattr(cv2, name, object())
    replacements = {
        "mediapipe": mp, "onnxruntime": ort, "timm": timm, "timm.layers": timm.layers,
        "kornia": kornia, "kornia.filters": kornia.filters, "transformers": transformers,
        "cv2": cv2, "torch": types.ModuleType("torch"),
        "PIL": types.ModuleType("PIL"), local_module: types.SimpleNamespace(BiRefNet=type("BiRefNet", (), {})),
        engine_module: types.SimpleNamespace(MasterHeadEngineV100=type("MasterHeadEngineV100", (), {})),
        pipeline_module: types.ModuleType(pipeline_module),
    }
    replacements["PIL"].Image = object()
    for name, module in replacements.items():
        monkeypatch.setitem(sys.modules, name, module)
    report = mapping["SaoChepHeadSwapCrop"].validate_dependencies()
    assert report["local_birefnet"] == "imported; model weights not loaded"
    assert report["local_crop_engine"] == "imported; model weights not loaded"
    assert "ImageSegmenter" in report["mediapipe_tasks"]


def test_awg_generator_reference_inputs_remain_directly_compatible():
    source = (ROOT / "bounded_video_nodes.py").read_text()
    assert '"pose_video_mask": ("IMAGE",)' in source
    assert '"reference_image_mask": ("IMAGE",)' in source
    assert "reference_image_mask:POSITIONAL_OR_KEYWORD:None" in source


def test_clean_source_tree_manifest_excludes_caches_and_preserves_asset_hashes(tmp_path):
    frozen = ROOT / "saochep_headswap" / "frozen_crop"
    manifest = json.loads((frozen / "frozen_crop_sha256.json").read_text())
    assert manifest and "runtime.py" in manifest and "crop/master_head_engine_v100.py" in manifest
    assert all("__pycache__" not in path and not path.endswith((".pyc", ".pyo")) for path in manifest)
    for relative, digest in manifest.items():
        source = frozen / relative
        assert source.is_file()
        import hashlib
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        clean_copy = tmp_path / relative
        clean_copy.parent.mkdir(parents=True, exist_ok=True)
        clean_copy.write_bytes(source.read_bytes())
    clean_package = tmp_path / "clean_custom_node"
    for relative in manifest:
        destination = clean_package / "frozen_crop" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((frozen / relative).read_bytes())
    (clean_package / "frozen_crop" / "frozen_crop_sha256.json").write_bytes(
        (frozen / "frozen_crop_sha256.json").read_bytes())
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_verify_crop_sources"])
    nodes._verify_crop_sources(clean_package)
    # The bundle's required identities remain pinned separately from source files.
    pins = json.loads((ROOT / "saochep_headswap" / "EXTERNAL_ASSETS.json").read_text())
    assert pins["custom_nodes/SaoChepCommercial/saochep_headswap/frozen_crop/models/bisenet_face_parsing.onnx"]["sha256"] == "95995975429e40f0001b60571c7b3f3ff5d707c07d5a8afc90fdded71c84ff67"
    expected_assets = {
        "bisenet_face_parsing.onnx": (53199247, "95995975429e40f0001b60571c7b3f3ff5d707c07d5a8afc90fdded71c84ff67"),
        "yoloface_8n.onnx": (12659761, "821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929"),
        "face_detection_yunet_2023mar.onnx": (232589, "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
        "face_landmarker.task": (3758596, "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
        "hand_landmarker.task": (7819105, "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
        "hair.tflite": (781618, "2628cf3ce5f695f604cbea2841e00befcaa3624bf80caf3664bef2656d59bf84"),
        "multi.tflite": (16371837, "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"),
        "birefnet_lite/model.safetensors": (88965896, "ce8bcfc045e336322c0424a5863dcfb7e9ce8fed0a5fd4d1b2b20adf12d97243"),
        "vitmatte_small/model.safetensors": (103294572, "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"),
    }
    for name, (size, digest) in expected_assets.items():
        record = next(record for path, record in pins.items() if path.endswith("/" + name))
        assert (record["bytes"], record["sha256"]) == (size, digest)
    assert not list(tmp_path.rglob("*.pyc"))


def test_crop_uses_actual_driving_canvas_and_reference_alpha_layout(monkeypatch):
    package = load_package()
    calls = {}
    class FakeAdapter:
        def isolate(self, **kwargs):
            calls.update(kwargs)
            return ("clip", "reference", "alpha", "tight", "tight_alpha", "{}", "rgba")
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".frozen_crop.adapter",
                        types.SimpleNamespace(SaoChepCropHeadHandSafe=FakeAdapter))
    image = types.SimpleNamespace(ndim=4, shape=(1, 800, 600, 3))
    class Driving:
        ndim = 4
        shape = (11, 704, 384, 3)
        def __len__(self): return self.shape[0]
        def __getitem__(self, key): return self
    result = package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapCrop"]().crop(
        image, Driving(), "hood", "hands", "[]")
    assert calls["target_width"] == 384 and calls["target_height"] == 704
    assert calls["driving_images"].shape == (11, 704, 384, 3)
    assert json.loads(result[5])["release"] == "SaoChepCommercial-headswap-adapter-v1"


def test_reference_mask_and_contract_measure_real_blue_on_white_head_area(monkeypatch):
    install_torch_facade(monkeypatch)
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["HSFinalReferenceMask"])
    mapping = load_package().NODE_CLASS_MAPPINGS
    head = Tensor([[[1., 0.], [0., 0.]]])
    pose = mapping["SaoChepHeadSwapBackfaceMask"]
    colored = nodes._colored(head)
    assert np.all(colored[..., 2] == 1)  # blue alone includes white background
    assert np.array_equal((colored[..., 2] > .9) & (np.asarray(colored[..., :2]).max(axis=-1) < .1), head > .5)
    ref_mask = mapping["SaoChepHeadSwapReferenceMask"]().make(head)[0]
    driving = Tensor(np.zeros((5, 2, 2, 3), np.float32))
    pose_frames = Tensor(np.broadcast_to(colored, (5, 2, 2, 3)))
    reference = Tensor(np.zeros((1, 2, 2, 3), np.float32))
    alpha = Tensor([[[1., 0.], [0., 0.]]])
    report = mapping["SaoChepHeadSwapContracts"]().check(
        driving, pose_frames, reference, alpha, Tensor(ref_mask), "{}", "{}")[-1]
    assert json.loads(report)["mask_area_min"] == .25


def test_stable_anchor_registered_node_uses_vendored_selection(monkeypatch):
    package = load_package()
    stable = __import__(PACKAGE_NAME + ".stable_anchor", fromlist=["choose_index"])
    # Middle face is more frontal/sharp and both neighbors are repaired-invalid.
    boxes = [[0, 0, 20, 20], [1, 0, 20, 20], [2, 0, 20, 20]]
    assert stable.choose_index(boxes, [0.4, 0.05, 0.4], [2, 10, 2], [True, False, True]) == 1
    assert package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapStableAnchor"].FUNCTION == "select"
    class Detector:
        def setInputSize(self, size): pass
        def detect(self, image): return 0, np.array([[2, 2, 12, 12, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]], np.float32)
    detector = Detector()
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (detector, None))))
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2GRAY = 0; cv2.CV_64F = 6
    cv2.resize = lambda image, size: image
    cv2.cvtColor = lambda image, code: image[..., 0]
    cv2.Laplacian = lambda image, depth: np.array([1., 3.])
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    images = Tensor(np.ones((2, 20, 20, 3), np.float32))
    geometry = json.dumps([{"face_box": [2, 2, 12, 12]}, {"face_box": [2, 2, 12, 12]}])
    result = package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapStableAnchor"]().select(images, geometry)
    assert result[0].shape[0] == 1
    assert json.loads(result[1])["selected_frame"] == 0


def test_nested_namespace_face_parse_preserves_selected_tracking_candidate(monkeypatch):
    package = load_package()
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_parse_selected_face"])
    selected = np.arange(15, dtype=np.float32)
    calls = {}
    def parser(image, landmarks, width, height, session, mean, std):
        calls.update(landmarks=landmarks.copy(), width=width, height=height)
        return {"labels": np.array([[1]])}
    parsed = nodes._parse_selected_face(np.zeros((4, 4, 3)), selected, object(), parser)
    assert parsed["labels"][0, 0] == 1
    assert calls["width"] == selected[2] and calls["height"] == selected[3]
    assert np.array_equal(calls["landmarks"], selected[4:14].reshape(5, 2))
    assert not any(name == "saochep_headswap" or name.startswith("saochep_headswap.") for name in sys.modules)
    assert package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapBackfaceMask"].FUNCTION == "assemble"


def test_alignment_crop_resize_sets_both_opencv_scales_and_preserves_geometry():
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_resize_alignment_crop"])
    import cv2

    image = np.zeros((200, 100, 3), dtype=np.uint8)
    resized = nodes._resize_alignment_crop(image)
    assert resized.shape == (512, 256, 3)

    small = nodes._resize_alignment_crop(np.zeros((50, 25, 3), dtype=np.uint8))
    assert small.shape == (512, 256, 3)


def test_chin_eye_semantic_input_stays_normalized_float(monkeypatch):
    package = load_package()
    module_name = PACKAGE_NAME + ".adaptive_semantic"
    captured = {}
    def parse_driving(image, landmarks):
        captured["image"] = image
        return np.ones((2, 2), np.float32), np.ones((2, 2), np.float32)
    monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(parse_driving=parse_driving))
    driver = np.full((2, 2, 3), .5, dtype=np.float32)
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_parse_driver_semantics"])
    nodes._parse_driver_semantics(driver, np.zeros((468, 2), np.float32))
    assert captured["image"].dtype == np.float32
    assert captured["image"].min() >= 0 and captured["image"].max() <= 1
    assert np.array_equal(captured["image"], driver)
    assert package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapChinEyeAlign"].FUNCTION == "align"


def test_model_path_policy_is_single_node_relative_root():
    package = load_package()
    paths = __import__(PACKAGE_NAME + ".frozen_crop.paths", fromlist=["model_roots"])
    roots = paths.model_roots()
    assert roots == [ROOT / "saochep_headswap" / "frozen_crop" / "models"]
    pipeline = (ROOT / "saochep_headswap" / "frozen_crop" / "crop_pipeline_v101.py").read_text()
    assert "model_directory()" in pipeline


def test_frame_fit_pads_or_trims_to_generator_count_and_rejects_invalid(monkeypatch):
    install_torch_facade(monkeypatch)
    fit = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapFrameFit"]()
    frames = Tensor(np.arange(24, dtype=np.float32).reshape(24, 1, 1, 1))
    padded, aligned_frames = fit.fit(frames, 24)
    assert padded.shape[0] == aligned_frames == 25
    assert np.array_equal(padded[-2:, 0, 0, 0], [23, 23])
    assert len(frames) == 24  # Requested generation duration is not changed.
    longer = Tensor(np.arange(30, dtype=np.float32).reshape(30, 1, 1, 1))
    trimmed, aligned_frames = fit.fit(longer, 24)
    assert aligned_frames == 25
    assert np.array_equal(trimmed[:, 0, 0, 0], np.arange(25))
    with pytest.raises(ValueError, match="maximum tail padding is four"):
        fit.fit(Tensor(np.arange(20, dtype=np.float32).reshape(20, 1, 1, 1)), 24)
    with pytest.raises(ValueError, match="positive requested"):
        fit.fit(frames, 0)
    with pytest.raises(ValueError, match="non-empty"):
        fit.fit(Tensor(np.zeros((0, 1, 1, 3))), 5)
