"""CPU business contracts for the locally registered head-swap adapters."""
import importlib.util
import json
import math
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
            sys.modules[name] = module
        module = sys.modules[name]
        module.__path__ = [str(ROOT.parent)] if name == "comfy_custom_nodes" else [str(ROOT)]
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
        if dtype == "cpu":
            return self
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
        "SaoChepHeadSwapTarget", "SaoChepHeadSwapActiveSlice", "SaoChepHeadSwapPreflight", "SaoChepHeadSwapExportSource",
    }
    crop = mapping["SaoChepHeadSwapCrop"]
    assert crop.RETURN_TYPES == ("IMAGE", "IMAGE", "MASK", "IMAGE", "MASK", "STRING", "IMAGE")
    assert crop.INPUT_TYPES()["required"].keys() == {
        "image", "driving_images", "hood_mask", "hand_mask", "hand_landmarks_json"}
    mask = mapping["SaoChepHeadSwapBackfaceMask"]
    assert mask.RETURN_TYPES == ("MASK", "IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK", "STRING")
    assert mask.INPUT_TYPES()["required"]["track_data"] == ("SAM3_TRACK_DATA",)
    assert mask.INPUT_TYPES()["optional"] == {"actor_mask": ("MASK",)}
    fit = mapping["SaoChepHeadSwapFrameFit"]
    assert fit.INPUT_TYPES()["required"].keys() == {"images", "target_frames"}
    assert fit.RETURN_TYPES == ("IMAGE", "INT")
    target = mapping["SaoChepHeadSwapTarget"]
    assert target.INPUT_TYPES()["required"].keys() == {"images", "track_data"}
    assert target.RETURN_TYPES == ("MASK", "STRING")
    assert target.RETURN_NAMES == ("actor_mask", "target_report")
    assert target.INPUT_TYPES()["optional"]["selection_policy"][0] == ["legacy_stable_body", "face_visible_actor"]
    active_slice = mapping["SaoChepHeadSwapActiveSlice"]
    assert active_slice.INPUT_TYPES()["required"].keys() == {"images", "actor_mask", "target_report"}
    assert active_slice.RETURN_TYPES == ("IMAGE", "MASK", "STRING")
    preflight = mapping["SaoChepHeadSwapPreflight"]
    assert preflight.OUTPUT_NODE is True and preflight.RETURN_TYPES == ()
    assert preflight.INPUT_TYPES()["required"].keys() == {"target_report", "native_width", "native_height", "delivery_width", "delivery_height"}
    assert preflight.INPUT_TYPES()["optional"]["delivery_scale"][1]["default"] == 1
    source = mapping["SaoChepHeadSwapExportSource"]
    assert source.RETURN_TYPES == ("VHS_FILENAMES",) and source.FUNCTION == "resolve"
    assert source.OUTPUT_NODE is False
    assert source.INPUT_TYPES()["required"].keys() == {"video", "attempt_prefix"}


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


def test_registered_first_frame_anchor_reuses_geometry_excludes_hidden_and_legacy_falls_back(monkeypatch):
    package = load_package()
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["HSFirstFrameAnchorTrial"])
    measured = []
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".selection", types.SimpleNamespace(
        measure_frame=lambda image, face, polygons, repaired=False: measured.append(face.copy()) or {
            "box": face[:4].tolist(), "yaw_proxy": 0., "roll": 0., "pitch_proxy": 0.,
            "hand_overlap": 0., "sharpness": 1., "repaired": repaired},
        choose_frame=lambda metrics: (0, {"candidate_count": len(metrics)}),
        choose_ranked_frame=lambda metrics: (0, {"candidate_count": len(metrics)})))
    class Detector:
        calls = 0
        def setInputSize(self, size): pass
        def detect(self, image):
            self.calls += 1
            return 0, np.array([vector], np.float32)
    vector = [2, 2, 12, 12, 4, 5, 10, 5, 7, 8, 5, 12, 9, 12, .98]
    detector = Detector()
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (detector, None))))
    class Hands:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def detect(self, image): return types.SimpleNamespace(hand_landmarks=[])
    vision = types.SimpleNamespace(
        HandLandmarkerOptions=lambda **kwargs: kwargs,
        HandLandmarker=types.SimpleNamespace(create_from_options=lambda options: Hands()),
        BaseOptions=lambda **kwargs: kwargs, ImageFormat=types.SimpleNamespace(SRGB=0),
    )
    mp = types.ModuleType("mediapipe")
    mp.tasks = types.SimpleNamespace(vision=vision, BaseOptions=vision.BaseOptions)
    mp.Image = lambda **kwargs: types.SimpleNamespace(**kwargs)
    mp.ImageFormat = vision.ImageFormat
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".frozen_crop.paths",
                        types.SimpleNamespace(asset_path=lambda name: name))
    cv2 = types.ModuleType("cv2")
    cv2.resize = lambda image, *args, **kwargs: image
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    images = Tensor(np.ones((3, 20, 20, 3), np.float32))
    records = [
        {"face_box": vector[:4], "yunet_face": vector, "face_visible": True},
        {"face_box": vector[:4], "face_visible": False, "temporal_repaired": True},
        {"face_box": vector[:4], "yunet_face": vector, "face_visible": True},
    ]
    result = package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapAnchor"]().select(images, json.dumps(records))
    assert detector.calls == 0
    assert len(measured) == 2
    assert result[0].shape[0] == 1

    # Old geometry without the measured face vector retains detection fallback.
    legacy = [{"face_box": vector[:4]}]
    package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapAnchor"]().select(images[:1], json.dumps(legacy))
    assert detector.calls == 1


@pytest.mark.parametrize('reference_depth,reference_span,roll,expected', [
    (80., 60., 0., -8.),   # aligned chin would leave eyes 20 pixels low
    (120., 60., 0., 8.),   # the opposite mismatch is handled symmetrically
    (90., 60., 0., -3.),
    (102., 60., 0., 0.),   # proportional deadband
    (80., 30., 0., 0.),    # strong pose/scale mismatch
    (80., 60., 25., 0.),   # incompatible eye-line roll
    (np.nan, 60., 0., 0.),
])
def test_chin_balance_uses_relative_geometry_and_bounds_neck_shift(reference_depth, reference_span, roll, expected):
    nodes = __import__(PACKAGE_NAME + '.nodes', fromlist=['_balance_aligned_chin'])

    def mesh(depth, span, eye_rise):
        points = np.zeros((468, 2), np.float32)
        points[[33, 133]] = [-span / 2, -eye_rise / 2]
        points[[263, 362]] = [span / 2, eye_rise / 2]
        points[152] = [0, depth]
        return points

    offset = np.array([7., 15.], np.float32)
    result, signed_shift = nodes._balance_aligned_chin(
        mesh(reference_depth, reference_span, roll), mesh(100., 60., 0.), offset,
    )
    assert signed_shift == pytest.approx(expected)
    np.testing.assert_allclose(result, offset + [0., expected])
    np.testing.assert_array_equal(offset, [7., 15.])


def test_alignment_landmarks_retries_tracked_crop_and_rejects_unowned_face(monkeypatch):
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_alignment_landmarks"])
    monkeypatch.setattr(nodes, "_resize_alignment_crop", lambda crop: crop)
    mp = types.SimpleNamespace(Image=lambda **kw: kw, ImageFormat=types.SimpleNamespace(SRGB=0))

    def face(x, y):
        points = [types.SimpleNamespace(x=x, y=y) for _ in range(468)]
        for i in (263, 362):
            points[i] = types.SimpleNamespace(x=x + .1, y=y)
        return points

    bystander = face(.5, .5)
    bystander[1] = types.SimpleNamespace(x=.5, y=.15)
    answers = iter([[bystander], [face(.05, .05)], [face(.5, .5)]])
    calls = []

    def detect(image):
        calls.append(image["data"].shape[:2])
        return types.SimpleNamespace(face_landmarks=next(answers))

    tracked = [20, 10, 20, 20, 30, 20, 33, 20, 30, 20, 30, 20, 30, 20, .99]
    result = nodes._alignment_landmarks(np.zeros((48, 64, 3), np.uint8), tracked[:4], tracked,
                                        types.SimpleNamespace(detect=detect), mp)
    points, method = result
    assert method == "tracked_face_roi"
    np.testing.assert_allclose(points[33], [30, 20])
    assert calls == [(40, 40), (48, 64), (30, 30)]


def test_stable_anchor_reuses_exact_selected_yunet_geometry_without_redetection(monkeypatch):
    package = load_package()
    class Detector:
        def setInputSize(self, size): pytest.fail("selected geometry must avoid duplicate YuNet inference")
        def detect(self, image): pytest.fail("selected geometry must avoid duplicate YuNet inference")
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (Detector(), None))))
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2GRAY = 0; cv2.CV_64F = 6
    cv2.resize = lambda image, size: image
    cv2.cvtColor = lambda image, code: image[..., 0]
    cv2.Laplacian = lambda image, depth: np.array([1., 3.])
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    vector = np.array([2, 2, 12, 12, 4, 5, 10, 5, 7, 8, 5, 12, 9, 12, .98], np.float32)
    geometry = json.dumps([{"face_box": vector[:4].tolist(), "yunet_face": vector.tolist()},
                           {"face_box": vector[:4].tolist(), "yunet_face": vector.tolist()}])
    images = Tensor(np.ones((2, 20, 20, 3), np.float32))
    result = package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapStableAnchor"]().select(images, geometry)
    assert result[0].shape[0] == 1
    assert json.loads(result[1])["selected_frame"] in (0, 1)


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


def test_confirmed_face_gaps_remain_absent_and_require_matching_endpoints():
    tracking = __import__(PACKAGE_NAME + ".tracking", fromlist=["preserve_confirmed_gaps"])
    left = np.array([10., 10., 20., 20., *range(10)])
    right = left.copy(); right[0] += 1

    # Leading gaps fail closed with a frame range; no fabricated initial face.
    with pytest.raises(ValueError, match=r"frames 0\.\.0"):
        tracking.preserve_confirmed_gaps([None, left], max_gap=3)
    observed, repaired = tracking.preserve_confirmed_gaps([left, None, right], max_gap=3)
    assert repaired == [1] and observed[1] is None
    # A prolonged gap or implausible actor transition is never bridged.
    with pytest.raises(ValueError, match=r"frames 1\.\.5"):
        tracking.preserve_confirmed_gaps([left, None, None, None, None, None, right], max_gap=3)
    switched = right.copy(); switched[:2] += 500
    with pytest.raises(ValueError, match=r"Unconfirmed identity across tracking gap at frames 1\.\.1"):
        tracking.preserve_confirmed_gaps([left, None, switched], max_gap=3)
    # Trailing absence is handled by the assembler's confirmed SAM path, not
    # repaired into a face; the tracker helper itself rejects unbounded gaps.
    with pytest.raises(ValueError, match=r"frames 1\.\.1"):
        tracking.preserve_confirmed_gaps([left, None], max_gap=3)


def test_backface_assembly_preallocated_masks_match_expected_and_keep_sam_gap_owner(monkeypatch):
    package = load_package()
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["HSBackfaceHeadMaskTrial"])
    layers = __import__(PACKAGE_NAME + ".layers", fromlist=["face_ownership"])
    install_torch_facade(monkeypatch)
    h = w = 8
    face_a = np.array([1, 1, 6, 6, 2, 3, 6, 3, 4, 4, 3, 6, 5, 6, .99], np.float32)
    face_b = face_a.copy(); face_b[:2] = 60
    class Detector:
        def __init__(self): self.frames = iter(([face_a], [face_b], [face_a]))
        def setInputSize(self, size): pass
        def detect(self, image): return 0, np.array(next(self.frames), np.float32)
    detector = Detector()
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (detector, object()))))
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.aligned_parse", types.SimpleNamespace(aligned_parse=lambda *args: None))
    monkeypatch.setattr(nodes, "_choose_face", lambda image: (face_a, None))
    monkeypatch.setattr(nodes, "_parse_selected_face", lambda *args: {
        "labels": np.ones((h, w), np.int32), "hair": np.zeros((h, w)), "hat": np.zeros((h, w))})
    monkeypatch.setattr(layers, "face_ownership", lambda labels, hair, hat, box:
                        (np.ones((h, w), bool), np.zeros((h, w), bool)))
    monkeypatch.setattr(layers, "component_family", lambda seed, parts, scale: np.asarray(seed, bool))
    cv2 = types.ModuleType("cv2")
    cv2.INTER_NEAREST = 0
    cv2.resize = lambda image, size, interpolation=None: image
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    tracker = types.ModuleType("comfy.ldm.sam3.tracker")
    tracker.unpack_masks = lambda packed: packed
    for name in ("comfy", "comfy.ldm", "comfy.ldm.sam3"):
        parent = types.ModuleType(name); parent.__path__ = []
        monkeypatch.setitem(sys.modules, name, parent)
    monkeypatch.setitem(sys.modules, "comfy.ldm.sam3.tracker", tracker)
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".sam_guard",
                        types.SimpleNamespace(validate_transition=lambda before, after: None))
    images = Tensor(np.ones((3, h, w, 3), np.float32))
    packed = Tensor(np.ones((3, 1, h, w), np.float32))
    result = package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapBackfaceMask"]().assemble(images, {"packed_masks": packed})
    expected_pose = np.zeros((3, h, w, 3), np.float32); expected_pose[..., 2] = 1
    expected = [
        np.ones((3, h, w), np.float32),
        expected_pose,
        np.stack([np.ones((h, w)), np.zeros((h, w)), np.ones((h, w))]).astype(np.float32),
        np.zeros((3, h, w), np.float32), np.zeros((3, h, w), np.float32),
        np.stack([np.ones((h, w)), np.zeros((h, w)), np.ones((h, w))]).astype(np.float32),
        np.zeros((3, h, w), np.float32),
    ]
    assert len(result) == 8
    for actual, want in zip(result[:7], expected):
        assert np.array_equal(np.asarray(actual), want)
    geometry = json.loads(result[7])
    assert geometry[1]["face_visible"] is False
    assert geometry[1]["face_box"] == face_a[:4].tolist()
    assert geometry[1]["tracking_source"] == "sam_confirmed_actor"


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


def install_unpack_masks(monkeypatch, calls=None):
    tracker = types.ModuleType("comfy.ldm.sam3.tracker")
    def unpack(packed):
        if calls is not None:
            calls.append(packed.shape[0])
        return packed
    tracker.unpack_masks = unpack
    for name in ("comfy", "comfy.ldm", "comfy.ldm.sam3"):
        if name not in sys.modules:
            parent = types.ModuleType(name); parent.__path__ = []
            monkeypatch.setitem(sys.modules, name, parent)
    monkeypatch.setitem(sys.modules, "comfy.ldm.sam3.tracker", tracker)


def run_target(monkeypatch, raw_masks, calls=None):
    install_torch_facade(monkeypatch)
    install_unpack_masks(monkeypatch, calls)
    node = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapTarget"]()
    frames = Tensor(np.zeros((len(raw_masks), raw_masks.shape[-2], raw_masks.shape[-1], 3)))
    return node.select(frames, {"packed_masks": Tensor(raw_masks)})


def test_active_slice_preserves_locked_actor_identity_and_localizes_report(monkeypatch):
    install_torch_facade(monkeypatch)
    node = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapActiveSlice"]()
    images = Tensor(np.arange(6 * 4 * 5 * 3, dtype=np.float32).reshape(6, 4, 5, 3))
    masks = np.zeros((6, 4, 5), np.float32)
    masks[2:5, 1:3, 2:4] = 1
    report = json.dumps({"version":1,"decision":"render","reason":"TARGET_CONFIRMED","frame_count":6,"target_index":1,"active_start_frame":2,"active_end_frame":5})
    active_images, active_masks, local_text = node.slice(images, Tensor(masks), report)
    assert np.array_equal(np.asarray(active_images), np.asarray(images)[2:5])
    assert np.array_equal(np.asarray(active_masks), masks[2:5])
    assert json.loads(local_text) == {"version":1,"decision":"render","reason":"TARGET_CONFIRMED","frame_count":3,"target_index":1,"active_start_frame":0,"active_end_frame":3}


def test_active_slice_rejects_mask_occupancy_mismatching_reported_bounds(monkeypatch):
    install_torch_facade(monkeypatch)
    node = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapActiveSlice"]()
    images = Tensor(np.zeros((6, 3, 3, 3), np.float32))
    masks = np.zeros((6, 3, 3), np.float32); masks[1:5, 0, 0] = 1
    report = json.dumps({"version":1,"decision":"render","reason":"TARGET_CONFIRMED",
                         "frame_count":6,"target_index":0,"active_start_frame":2,"active_end_frame":5})
    with pytest.raises(ValueError, match="does not match reported"):
        node.slice(images, Tensor(masks), report)


def test_active_slice_rejects_empty_holes_and_malformed_intervals(monkeypatch):
    install_torch_facade(monkeypatch)
    node = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapActiveSlice"]()
    images = Tensor(np.zeros((4, 3, 3, 3), np.float32))
    report = json.dumps({"version":1,"decision":"render","reason":"TARGET_CONFIRMED","frame_count":4,"target_index":0})
    masks = np.zeros((4, 3, 3), np.float32); masks[0, 0, 0] = 1; masks[2, 0, 0] = 1
    with pytest.raises(ValueError, match="contains empty frames"):
        node.slice(images, Tensor(masks), report)
    malformed = json.dumps({"version":1,"decision":"render","reason":"TARGET_CONFIRMED","frame_count":4,"target_index":0,"active_start_frame":1})
    with pytest.raises(ValueError, match="schema|together"):
        node.slice(images, Tensor(masks), malformed)


def test_active_slice_rejects_boolean_bounds_and_native_interval(monkeypatch):
    install_torch_facade(monkeypatch)
    node = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapActiveSlice"]()
    images = Tensor(np.zeros((4, 3, 3, 3), np.float32))
    masks = np.ones((4, 3, 3), np.float32)
    base = {"version":1,"decision":"render","reason":"TARGET_CONFIRMED","frame_count":4,"target_index":0}
    for bounds in ({"active_start_frame":True,"active_end_frame":3},
                   {"active_start_frame":0,"active_end_frame":5}):
        with pytest.raises(ValueError, match="bounds|interval"):
            node.slice(images, Tensor(masks), json.dumps({**base, **bounds}))
    native = {"version":1,"decision":"native","reason":"TARGET_UNCERTAIN","frame_count":4,
              "target_index":None,"active_start_frame":0,"active_end_frame":4}
    with pytest.raises(ValueError, match="render target|confirmed"):
        node.slice(images, Tensor(masks), json.dumps(native))


def test_face_visible_policy_selects_sustained_face_owned_track_and_reports_interval(monkeypatch):
    import cv2
    masks = np.zeros((6, 2, 12, 12), np.float32)
    masks[:, 0, :2, :] = 1; masks[:, 0, -2:, :] = 1
    masks[:, 0, :, :2] = 1; masks[:, 0, :, -2:] = 1  # larger border-only track
    masks[1:5, 1, 2:10, 2:10] = 1
    class Detector:
        def setInputSize(self, size):
            self.size = size
        def detect(self, image):
            row = np.zeros((1, 15), np.float32)
            row[0, :4] = [4, 3, 4, 4]
            row[0, 4:14] = [5, 4, 7, 4, 6, 5, 5, 6, 7, 6]
            row[0, 14] = .92
            return 1, row
    monkeypatch.setattr(cv2, "FaceDetectorYN", types.SimpleNamespace(create=lambda *args, **kwargs: Detector()))
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (Detector(), None))))
    images = Tensor(np.zeros((6, 12, 12, 3), np.float32))
    install_torch_facade(monkeypatch)
    install_unpack_masks(monkeypatch)
    node = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapTarget"]()
    actor, text = node.select(images, {"packed_masks": Tensor(masks)}, "face_visible_actor")
    report = json.loads(text)
    assert report["target_index"] == 1 and report["active_start_frame"] == 1 and report["active_end_frame"] == 5
    assert not np.asarray(actor)[0].any() and not np.asarray(actor)[5].any()
    assert np.asarray(actor)[1:5].any()


def test_face_visible_policy_extends_body_interval_and_preserves_native_prefix(monkeypatch):
    import cv2
    masks = np.zeros((10, 1, 12, 12), np.float32)
    masks[:, 0, 2:10, 2:10] = 1
    class Detector:
        def setInputSize(self, size): pass
        def detect(self, image):
            if int(image[0, 0, 0]) in (0, 3, 7, 8, 9): return 1, None
            row = np.zeros((1, 15), np.float32)
            row[0, :4] = [4, 3, 4, 4]
            row[0, 4:14] = [5, 4, 7, 4, 6, 5, 5, 6, 7, 6]
            row[0, 14] = .93
            return 1, row
    monkeypatch.setattr(cv2, "FaceDetectorYN", types.SimpleNamespace(create=lambda *args, **kwargs: Detector()))
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (Detector(), None))))
    image_values = np.zeros((10, 12, 12, 3), np.float32)
    image_values[:, 0, 0, 2] = np.arange(10) / 255
    images = Tensor(image_values)
    install_torch_facade(monkeypatch); install_unpack_masks(monkeypatch)
    actor, text = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapTarget"]().select(
        images, {"packed_masks": Tensor(masks)}, "face_visible_actor")
    report = json.loads(text)
    assert (report["active_start_frame"], report["active_end_frame"]) == (4, 10)
    assert not np.asarray(actor)[:4].any()
    assert np.all(np.asarray(actor)[4:, 2:10, 2:10] == 1)


def test_face_visible_policy_reports_tied_eligible_tracks_ambiguous(monkeypatch):
    import cv2
    masks = np.zeros((4, 2, 12, 12), np.float32)
    masks[:, 0, 2:10, 2:6] = 1; masks[:, 1, 2:10, 6:10] = 1
    class Detector:
        def setInputSize(self, size): pass
        def detect(self, image):
            rows = np.zeros((2, 15), np.float32)
            for row, x in zip(rows, (2, 6)):
                row[:4] = [x, 3, 4, 4]; row[4:14] = [x+1,4,x+3,4,x+2,5,x+1,6,x+3,6]; row[14] = .94
            return 2, rows
    monkeypatch.setattr(cv2, "FaceDetectorYN", types.SimpleNamespace(create=lambda *args, **kwargs: Detector()))
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (Detector(), None))))
    install_torch_facade(monkeypatch); install_unpack_masks(monkeypatch)
    _, text = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapTarget"]().select(
        Tensor(np.zeros((4, 12, 12, 3))), {"packed_masks": Tensor(masks)}, "face_visible_actor")
    assert json.loads(text)["reason"] == "TARGET_AMBIGUOUS"


def test_identity_observations_clamps_normalized_comfy_image_before_bgr_conversion():
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_identity_observations"])
    captured = []

    class Detector:
        def setInputSize(self, size):
            assert size in ((2, 1), (1, 1))
        def detect(self, image):
            captured.append(image.copy())
            return 0, None

    image = np.array([[[1.05, -.05, .5], [.5, 1., 0.]]], dtype=np.float32)
    result, owners = nodes._identity_observations(
        image, None, Detector(), 2, 1, 0, None, None, np, None)
    assert np.array_equal(result, np.array([[[128, 0, 255], [0, 255, 128]]], dtype=np.uint8))
    assert np.array_equal(captured[0], result)
    assert owners == {}

    # Healthy normalized values retain _rgb's round-to-nearest conversion.
    healthy = np.array([[[.1, .5, .9]]], dtype=np.float32)
    nodes._identity_observations(healthy, None, Detector(), 1, 1, 0, None, None, np, None)
    assert np.array_equal(captured[1], np.rint(healthy[..., ::-1].clip(0, 1) * 255).astype(np.uint8))


@pytest.mark.parametrize("image", [
    np.array([[[np.nan, 0., 0.]]], dtype=np.float32),
    np.zeros((1, 1, 4), dtype=np.float32),
])
def test_identity_observations_rejects_invalid_images_before_detector(image):
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_identity_observations"])

    class Detector:
        def setInputSize(self, size):
            pytest.fail("invalid image reached detector")
        def detect(self, image):
            pytest.fail("invalid image reached detector")

    with pytest.raises(ValueError, match="Invalid target-selection image frame"):
        nodes._identity_observations(image, None, Detector(), 1, 1, 0, None, None, np, None)


def _identity_target_fixture(monkeypatch, tmp_path, frame_count=5, track_count=1, marker=.2,
                             visible_tracks=None, embedding_error=None, frame_markers=None):
    import cv2
    if visible_tracks is None:
        visible_tracks = [set(range(frame_count)) for _ in range(track_count)]
    faces_by_frame = []
    for frame in range(frame_count):
        rows = []
        for track in range(track_count):
            if frame not in visible_tracks[track]:
                continue
            row = np.zeros(15, np.float32)
            x = 1 if track == 0 else 7
            row[:4] = [x, 2, 4, 5]
            row[4:14] = [x + .5, 3, x + 3.5, 3, x + 2, 4, x + .5, 6, x + 3.5, 6]
            row[14] = .99
            rows.append(row)
        faces_by_frame.append(np.asarray(rows, np.float32).reshape((-1, 15)) if rows else None)

    class Detector:
        def setInputSize(self, size): pass
        def detect(self, image):
            # The synthetic image carries its source frame in the green channel,
            # so deterministic rescans can revisit nonsequential frame IDs.
            frame = int(image[0, 0, 1])
            faces = faces_by_frame[frame]
            return len(faces) if faces is not None else 0, faces

    detector = Detector()
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (detector, None))))
    install_torch_facade(monkeypatch)
    install_unpack_masks(monkeypatch)
    output_root = tmp_path / "comfy-output"
    output_root.mkdir(exist_ok=True)
    models_root = tmp_path / "models"
    models_root.mkdir(exist_ok=True)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(output_root)
    folder_paths.models_dir = str(models_root)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    package = load_package()
    node = package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapTarget"]()
    identity = __import__(PACKAGE_NAME + ".identity", fromlist=["embed"])
    monkeypatch.setattr(identity, "create_recognizer", lambda _path: object())
    embed_calls = []
    def fake_embed(_recognizer, bgr, _face):
        if embedding_error is not None:
            raise embedding_error
        embed_calls.append((int(bgr[0, 0, 1]), int(bgr[0, 0, 0])))
        feature = np.zeros(128, np.float32)
        feature[0 if bgr[0, 0, 0] < 100 else 1] = 1
        return feature.tolist()
    monkeypatch.setattr(identity, "embed", fake_embed)
    masks = np.zeros((frame_count, track_count, 12, 12), np.float32)
    for frame in range(frame_count):
        for track in range(track_count):
            if frame in visible_tracks[track]:
                if track == 0:
                    masks[frame, track, 1:11, 0:6] = 1
                else:
                    masks[frame, track, 1:11, 6:12] = 1
    markers = [marker] * frame_count if frame_markers is None else frame_markers
    image_values = np.stack([np.full((12, 12, 3), value, np.float32) for value in markers])
    image_values[:, 0, 0, 1] = np.arange(frame_count, dtype=np.float32) / 255
    images = Tensor(image_values)
    return node, images, Tensor(masks), output_root, embed_calls


def _identity_select(node, images, masks, prefix):
    return node.select(images, {"packed_masks": masks}, "face_visible_actor",
                       identity_lock=True, identity_attempt_prefix=prefix)










def test_identity_lock_seeds_selected_original_track_and_matches_across_calls(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "d" * 32
    node, images, masks, output_root, embed_calls = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=5, track_count=2,
        visible_tracks=[{0, 1, 2}, {0, 1, 2, 3, 4}])
    _, seed_text = _identity_select(node, images, masks, prefix)
    seed_report = json.loads(seed_text)
    assert seed_report["decision"] == "render" and seed_report["target_index"] == 1
    template = output_root / prefix / ".identity" / "source.json"
    assert template.is_file() and len(json.loads(template.read_text())["templates"]) >= 3
    assert "source.json" not in seed_text and "templates" not in seed_text

    # The subsequent scene has a different SAM track index but the same synthetic
    # SFace identity; identity lock compares features, not body-track numbering.
    next_node, next_images, next_masks, _, _ = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=4, marker=.2)
    _, matched_text = _identity_select(next_node, next_images, next_masks, prefix)
    matched = json.loads(matched_text)
    assert matched["decision"] == "render" and matched["target_index"] == 0
    assert len(embed_calls) == 5  # seed extraction only; matching uses another fixture


@pytest.mark.parametrize("case,expected", [
    ("same-actor", "render"), ("different-actor", "native"),
    ("insufficient-overlap", "native"), ("no-strict-anchor", "native"),
    ("initial-selection", "native"), ("second-track", "native"),
])
def test_identity_lock_recovers_sparse_landmarks_only_on_one_continuous_owned_track(
    monkeypatch, tmp_path, case, expected,
):
    prefix = "aiworkergrid/" + "7" * 32
    if case != "initial-selection":
        seed_node, seed_images, seed_masks, _, _ = _identity_target_fixture(monkeypatch, tmp_path)
        assert json.loads(_identity_select(seed_node, seed_images, seed_masks, prefix)[1])["decision"] == "render"
    node, images, masks, _, _ = _identity_target_fixture(
        monkeypatch, tmp_path, track_count=2 if case == "second-track" else 1,
        marker=.8 if case == "different-actor" else .2,
    )
    # Three landmarks (not four) remain inside; 3/4 of the face box overlaps
    # the original track. One untouched frame provides a strict anchor.
    start = 0 if case == "no-strict-anchor" else 1
    masks[start:, :, :, 4:] = 0
    if case == "insufficient-overlap":
        masks[start:, :, :, 2:] = 0
    if case == "second-track":
        masks[:, 1, 1:11, 11] = 1
    output, text = _identity_select(node, images, masks, prefix)
    report = json.loads(text)
    assert report["decision"] == expected
    if expected == "render":
        assert (report["active_start_frame"], report["active_end_frame"]) == (0, 5)
        np.testing.assert_array_equal(output, masks[:, 0])
    else:
        assert not np.any(output)


def test_identity_seed_samples_long_run_uniformly_with_five_feature_calls(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "a" * 32
    node, images, masks, output_root, embed_calls = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=11)
    _, text = _identity_select(node, images, masks, prefix)
    assert json.loads(text)["decision"] == "render"
    assert [frame for frame, _ in embed_calls] == [0, 2, 5, 8, 10]
    assert len(embed_calls) <= 5
    vectors = json.loads((output_root / prefix / ".identity" / "source.json").read_text())["templates"]
    assert len(vectors) == 5


def test_identity_seed_rescan_missing_selected_track_fails_closed(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "b" * 32
    node, images, masks, output_root, _ = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=5, track_count=2,
        visible_tracks=[set(range(5)), set(range(4))])
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_identity_observations"])
    original = nodes._identity_observations
    calls = 0
    def changing(*args, **kwargs):
        nonlocal calls
        calls += 1
        bgr, owned = original(*args, **kwargs)
        # The selected first track disappears, while a different credible
        # observation remains uniquely owned by the other track.
        if calls > 5:
            return bgr, {1: owned[1]} if 1 in owned else {}
        return bgr, owned
    monkeypatch.setattr(nodes, "_identity_observations", changing)
    identity = __import__(PACKAGE_NAME + ".identity", fromlist=["IdentityError"])
    with pytest.raises(identity.IdentityError, match="Source actor observations changed during identity seeding"):
        _identity_select(node, images, masks, prefix)
    assert not (output_root / prefix / ".identity" / "source.json").exists()


def test_identity_seed_qualifying_three_frame_run_uses_all_observations(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "c" * 32
    node, images, masks, output_root, embed_calls = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=3)
    _, text = _identity_select(node, images, masks, prefix)
    assert json.loads(text)["decision"] == "render"
    assert [frame for frame, _ in embed_calls] == [0, 1, 2]
    templates = json.loads((output_root / prefix / ".identity" / "source.json").read_text())["templates"]
    assert len(templates) == 3


def test_identity_seed_with_fewer_than_three_usable_features_stays_native(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "c" * 32
    node, images, masks, output_root, embed_calls = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=5)
    identity = __import__(PACKAGE_NAME + ".identity", fromlist=["embed"])
    real_embed = identity.embed
    calls = 0
    def partly_unusable(*args):
        nonlocal calls
        calls += 1
        return None if calls <= 3 else real_embed(*args)
    monkeypatch.setattr(identity, "embed", partly_unusable)
    _, text = _identity_select(node, images, masks, prefix)
    report = json.loads(text)
    assert report["decision"] == "native" and report["reason"] == "TARGET_UNCERTAIN"
    assert calls == 5
    assert not (output_root / prefix / ".identity" / "source.json").exists()


def test_identity_lock_women_only_scene_is_absent_and_does_not_reseed(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "e" * 32
    node, images, masks, output_root, _ = _identity_target_fixture(monkeypatch, tmp_path)
    _, seed_text = _identity_select(node, images, masks, prefix)
    template = output_root / prefix / ".identity" / "source.json"
    before = template.read_bytes()
    woman_node, woman_images, woman_masks, _, _ = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=4, marker=.8)
    _, text = _identity_select(woman_node, woman_images, woman_masks, prefix)
    report = json.loads(text)
    assert report["decision"] == "native" and report["reason"] == "TARGET_ABSENT"
    assert template.read_bytes() == before


def test_identity_lock_strict_torch_from_numpy_accepts_no_match_and_render(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "9" * 32
    node, images, masks, _, _ = _identity_target_fixture(monkeypatch, tmp_path)
    _identity_select(node, images, masks, prefix)  # Seed before enforcing strict API types.

    def enforce_strict_torch_api():
        torch = sys.modules["torch"]
        original_from_numpy = torch.from_numpy
        calls = []

        def strict_from_numpy(value):
            if isinstance(value, Tensor):
                raise TypeError("expected np.ndarray, got Tensor")
            calls.append(value)
            return original_from_numpy(value)

        monkeypatch.setattr(torch, "from_numpy", strict_from_numpy)
        with pytest.raises(TypeError, match="expected np.ndarray, got Tensor"):
            torch.from_numpy(Tensor(np.zeros((1,), np.float32)))
        calls.clear()
        return calls

    absent_node, absent_images, absent_masks, output_root, _ = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=4, marker=.8)
    absent_calls = enforce_strict_torch_api()
    absent_path = output_root / prefix / ".identity" / "source.json"
    seeded = absent_path.read_bytes()
    mask, report_text = _identity_select(absent_node, absent_images, absent_masks, prefix)
    report = json.loads(report_text)
    assert report["decision"] == "native" and report["reason"] == "TARGET_ABSENT"
    assert np.asarray(mask).shape == (4, 12, 12) and not np.asarray(mask).any()
    assert absent_path.read_bytes() == seeded
    assert absent_calls == []

    match_node, match_images, match_masks, _, _ = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=4)
    matched_calls = enforce_strict_torch_api()
    matched_mask, matched_text = _identity_select(match_node, match_images, match_masks, prefix)
    assert json.loads(matched_text)["decision"] == "render"
    assert np.asarray(matched_mask).shape == (4, 12, 12)
    assert matched_calls and all(isinstance(value, np.ndarray) and not isinstance(value, Tensor)
                                 for value in matched_calls)

    ambiguous_node, ambiguous_images, ambiguous_masks, _, _ = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=4, track_count=2)
    ambiguous_calls = enforce_strict_torch_api()
    ambiguous_mask, ambiguous_text = _identity_select(
        ambiguous_node, ambiguous_images, ambiguous_masks, prefix)
    assert json.loads(ambiguous_text)["reason"] == "TARGET_AMBIGUOUS"
    assert np.asarray(ambiguous_mask).shape == (4, 12, 12)
    assert not np.asarray(ambiguous_mask).any()
    assert ambiguous_calls == []


def test_identity_lock_ambiguous_matching_tracks_route_native(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "f" * 32
    node, images, masks, output_root, _ = _identity_target_fixture(monkeypatch, tmp_path)
    _identity_select(node, images, masks, prefix)
    ambiguous_node, ambiguous_images, ambiguous_masks, _, _ = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=4, track_count=2)
    _, text = _identity_select(ambiguous_node, ambiguous_images, ambiguous_masks, prefix)
    report = json.loads(text)
    assert report["decision"] == "native" and report["reason"] == "TARGET_AMBIGUOUS"


def test_identity_lock_seeds_from_longest_qualifying_face_run_not_early_fragments(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "2" * 32
    visible = [{0, 1, 3, 4, 6, 8, 9, 10}]
    markers = [.2, .2, .2, .2, .2, .2, .2, .2, .8, .8, .8]
    node, images, masks, output_root, embed_calls = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=11, visible_tracks=visible,
        frame_markers=markers)
    _, text = _identity_select(node, images, masks, prefix)
    assert [frame for frame, _ in embed_calls] == [8, 9, 10]
    report = json.loads(text)
    assert report["decision"] == "render" and report["active_start_frame"] == 8
    template = output_root / prefix / ".identity" / "source.json"
    vectors = json.loads(template.read_text())["templates"]
    assert len(vectors) == 3 and all(vector[1] == 1 for vector in vectors)

    later_node, later_images, later_masks, _, _ = _identity_target_fixture(
        monkeypatch, tmp_path, frame_count=4, marker=.8)
    _, later_text = _identity_select(later_node, later_images, later_masks, prefix)
    assert json.loads(later_text)["decision"] == "render"


def test_identity_lock_sface_runtime_error_propagates_from_node(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "3" * 32
    package = load_package()
    production_identity = __import__(PACKAGE_NAME + ".identity", fromlist=["embed"])
    real_embed = production_identity.embed
    node, images, masks, _, _ = _identity_target_fixture(monkeypatch, tmp_path)
    class BrokenSFace:
        def alignCrop(self, *_args): raise RuntimeError("private runtime detail")
    monkeypatch.setattr(production_identity, "create_recognizer", lambda _path: BrokenSFace())
    monkeypatch.setattr(production_identity, "embed", real_embed)
    with pytest.raises(production_identity.IdentityError, match="SFace alignment"):
        _identity_select(node, images, masks, prefix)


def test_identity_lock_model_and_corrupt_template_errors_propagate(monkeypatch, tmp_path):
    prefix = "aiworkergrid/" + "1" * 32
    node, images, masks, output_root, _ = _identity_target_fixture(monkeypatch, tmp_path)
    identity = __import__(PACKAGE_NAME + ".identity", fromlist=["IdentityError"])
    def fail_model(_path): raise identity.IdentityError("bad pinned model")
    monkeypatch.setattr(identity, "create_recognizer", fail_model)
    with pytest.raises(identity.IdentityError, match="bad pinned model"):
        _identity_select(node, images, masks, prefix)

    monkeypatch.undo()
    node, images, masks, output_root, _ = _identity_target_fixture(monkeypatch, tmp_path)
    _identity_select(node, images, masks, prefix)
    template = output_root / prefix / ".identity" / "source.json"
    template.write_text('{"version":1,"version":1}')
    template.chmod(0o600)
    later_node, later_images, later_masks, _, _ = _identity_target_fixture(monkeypatch, tmp_path)
    identity = __import__(PACKAGE_NAME + ".identity", fromlist=["IdentityError"])
    with pytest.raises(identity.IdentityError, match="Malformed"):
        _identity_select(later_node, later_images, later_masks, prefix)


def test_preflight_propagates_only_present_validated_active_bounds(monkeypatch):
    install_torch_facade(monkeypatch)
    node = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapPreflight"]()
    report = {"version":1,"decision":"render","reason":"TARGET_CONFIRMED","frame_count":8,
              "target_index":2,"active_start_frame":1,"active_end_frame":7}
    ui = node.check(json.dumps(report), 8, 6, 8, 6)["ui"]["headswap_preflight"][0]
    assert ui["active_start_frame"] == 1 and ui["active_end_frame"] == 7
    native_with_bounds = {"version":1,"decision":"native","reason":"TARGET_UNCERTAIN","frame_count":8,
                          "target_index":None,"active_start_frame":1,"active_end_frame":7}
    with pytest.raises(ValueError, match="render target"):
        node.check(json.dumps(native_with_bounds), 8, 6, 8, 6)
    legacy = dict(report); legacy.pop("active_start_frame"); legacy.pop("active_end_frame")
    assert "active_start_frame" not in node.check(json.dumps(legacy), 8, 6, 8, 6)["ui"]["headswap_preflight"][0]


def test_target_accepts_pinned_empty_track_sentinel_as_native_absent(monkeypatch):
    install_torch_facade(monkeypatch)
    install_unpack_masks(monkeypatch)
    images = Tensor(np.zeros((4, 8, 12, 3), np.float32))
    sentinel = {"packed_masks": None, "n_frames": 4, "scores": []}
    mask, report_text = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapTarget"]().select(images, sentinel)
    assert json.loads(report_text) == {
        "version": 1, "decision": "native", "reason": "TARGET_ABSENT",
        "frame_count": 4, "target_index": None,
    }
    assert np.asarray(mask).shape == (4, 8, 12)
    assert not np.asarray(mask).any()


def test_target_rejects_malformed_or_mistimed_empty_track_sentinel(monkeypatch):
    install_torch_facade(monkeypatch)
    install_unpack_masks(monkeypatch)
    images = Tensor(np.zeros((4, 8, 12, 3), np.float32))
    valid = {"packed_masks": None, "n_frames": 4, "scores": []}
    invalid = [
        {"n_frames": 4, "scores": []},
        {"packed_masks": None, "scores": []},
        {"packed_masks": None, "n_frames": "4", "scores": []},
        {"packed_masks": None, "n_frames": True, "scores": []},
        {"packed_masks": None, "n_frames": 3, "scores": []},
        {"packed_masks": None, "n_frames": 4, "scores": [0.9]},
        {"packed_masks": None, "n_frames": 4, "scores": ()},
        {"packed_masks": None, "n_frames": 4},
    ]
    node = load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapTarget"]()
    for track_data in invalid:
        with pytest.raises(ValueError):
            node.select(images, track_data)
    # Exact sentinel remains narrowly recognized; real packed masks keep normal processing.
    assert json.loads(node.select(images, valid)[1])["reason"] == "TARGET_ABSENT"
    shorter = {"packed_masks": Tensor(np.zeros((3, 1, 8, 12), np.float32)), "n_frames": 4, "scores": []}
    with pytest.raises(ValueError, match="timing mismatch"):
        node.select(images, shorter)


def test_target_selects_largest_stable_person_track_once(monkeypatch):
    masks = np.zeros((4, 2, 8, 8), np.float32)
    masks[:, 0, 1:3, 1:3] = 1
    masks[:, 1, 3:7, 3:7] = 1
    actor_mask, report_text = run_target(monkeypatch, masks)
    report = json.loads(report_text)
    assert report == {"version": 1, "decision": "render", "reason": "TARGET_CONFIRMED", "frame_count": 4, "target_index": 1}
    assert np.all(np.asarray(actor_mask)[:, 3:7, 3:7] == 1)
    assert np.all(np.asarray(actor_mask)[:, :3, :] == 0)


def test_target_tie_absent_and_insufficient_track_are_native(monkeypatch):
    tied = np.zeros((4, 2, 8, 8), np.float32)
    tied[:, 0, :2, :] = 1; tied[:, 1, 6:, :] = 1
    _, report = run_target(monkeypatch, tied)
    assert json.loads(report)["reason"] == "TARGET_AMBIGUOUS"
    absent = np.zeros((3, 1, 8, 8), np.float32)
    mask, report = run_target(monkeypatch, absent)
    assert json.loads(report)["reason"] == "TARGET_ABSENT"
    assert not np.asarray(mask).any()
    brief = np.zeros((5, 1, 8, 8), np.float32); brief[0, 0, :3, :3] = 1
    _, report = run_target(monkeypatch, brief)
    assert json.loads(report)["reason"] == "TARGET_UNCERTAIN"


def test_transient_large_entrant_cannot_outscore_sustained_principal_and_decode_is_frame_bounded(monkeypatch):
    masks = np.zeros((5, 2, 12, 12), np.float32)
    masks[:, 0, 2:6, 2:6] = 1
    masks[0, 1, :, :] = 1
    calls = []
    _, report_text = run_target(monkeypatch, masks, calls)
    report = json.loads(report_text)
    assert report["decision"] == "render" and report["target_index"] == 0
    assert calls == [1] * 5 + [1] * 5


def test_target_near_complete_coverage_beats_larger_sixty_percent_entrant(monkeypatch):
    masks = np.zeros((10, 2, 12, 12), np.float32)
    masks[:, 0, 1:3, 1:3] = 1
    masks[4:, 1, :, :] = 1
    output, report_text = run_target(monkeypatch, masks)
    report = json.loads(report_text)
    assert report["decision"] == "render" and report["target_index"] == 0
    assert np.asarray(output)[0, 1:3, 1:3].all()


def test_target_ninety_percent_candidate_is_uncertain_when_any_frame_mask_is_missing(monkeypatch):
    near_complete = np.zeros((10, 1, 8, 8), np.float32)
    near_complete[1:, 0, 1:3, 1:3] = 1
    output, report_text = run_target(monkeypatch, near_complete)
    assert json.loads(report_text)["reason"] == "TARGET_UNCERTAIN"
    assert not np.asarray(output).any()
    below_tier = np.zeros((10, 1, 8, 8), np.float32)
    below_tier[:8, 0, 1:3, 1:3] = 1
    _, report_text = run_target(monkeypatch, below_tier)
    assert json.loads(report_text)["reason"] == "TARGET_UNCERTAIN"


def test_target_rejects_nonfinite_or_out_of_range_person_masks(monkeypatch):
    invalid_nan = np.zeros((3, 1, 4, 4), np.float32); invalid_nan[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run_target(monkeypatch, invalid_nan)
    invalid_range = np.zeros((3, 1, 4, 4), np.float32); invalid_range[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        run_target(monkeypatch, invalid_range)


def test_short_shot_policy_is_explicit_native_uncertain(monkeypatch):
    for count in (1, 2):
        masks = np.ones((count, 1, 4, 4), np.float32)
        _, report_text = run_target(monkeypatch, masks)
        assert json.loads(report_text)["reason"] == "TARGET_UNCERTAIN"


def test_actor_mask_filters_initial_candidates_before_ranking():
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_filter_faces_by_mask", "_rank_face_candidates", "_validated_actor_mask"])
    mask = np.zeros((40, 40), np.float32); mask[10:20, 10:20] = 1
    outside = np.array([0, 8, 18, 20] + [0] * 11, dtype=np.float32)
    inside = np.array([11, 11, 8, 8] + [0] * 11, dtype=np.float32)
    candidates = nodes._filter_faces_by_mask([outside, inside], mask)
    assert len(candidates) == 1 and np.array_equal(candidates[0], inside)
    selected = nodes._rank_face_candidates(np.zeros((40, 40, 3)), candidates, hint=mask)
    assert np.array_equal(selected, inside)
    valid = nodes._validated_actor_mask(Tensor(np.zeros((2, 40, 40))), (2, 40, 40, 3))
    assert valid.shape == (2, 40, 40)
    invalid = np.zeros((2, 40, 40), np.float32); invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        nodes._validated_actor_mask(Tensor(invalid), (2, 40, 40, 3))


def actor_backface_fixture(monkeypatch, detections, body_masks=None, sam_masks=None,
                           component_family_fn=None, face_ownership_fn=None, parser_fn=None):
    package = load_package()
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["HSBackfaceHeadMaskTrial"])
    install_torch_facade(monkeypatch)
    h = w = 8
    face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    class Detector:
        def __init__(self): self.index = 0
        def setInputSize(self, size): pass
        def detect(self, image):
            current = detections[self.index]; self.index += 1
            return 0, np.stack(current) if current else None
    detector = Detector()
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.skin_node",
                        types.SimpleNamespace(AGDrivingSkinColor=types.SimpleNamespace(models=lambda: (detector, object()))))
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".skin.aligned_parse", types.SimpleNamespace(aligned_parse=lambda *args: None))
    monkeypatch.setattr(nodes, "_parse_selected_face", parser_fn or (lambda *args: {
        "labels": np.ones((h, w), np.int32), "hair": np.zeros((h, w)), "hat": np.zeros((h, w))}))
    quality = type("QualityFailure", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, PACKAGE_NAME + ".layers", types.SimpleNamespace(
        QualityFailure=quality,
        face_ownership=face_ownership_fn or (lambda labels, hair, hat, box: (np.pad(np.ones((6, 6), bool), ((1, 1), (1, 1))), np.zeros((h, w), bool))),
        component_family=component_family_fn or (lambda seed, parts, scale: np.logical_or.reduce([np.asarray(seed, bool)] + [np.asarray(part, bool) for part in parts]))))
    cv2 = types.ModuleType("cv2"); cv2.INTER_NEAREST = 0; cv2.resize = lambda image, size, interpolation=None: image
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    tracker = types.ModuleType("comfy.ldm.sam3.tracker"); tracker.unpack_masks = lambda packed: packed
    for name in ("comfy", "comfy.ldm", "comfy.ldm.sam3"):
        parent = types.ModuleType(name); parent.__path__ = []
        monkeypatch.setitem(sys.modules, name, parent)
    monkeypatch.setitem(sys.modules, "comfy.ldm.sam3.tracker", tracker)
    frames = len(detections)
    images = Tensor(np.ones((frames, h, w, 3), np.float32))
    bodies = np.zeros((frames, h, w), np.float32) if body_masks is None else np.asarray(body_masks, np.float32)
    if body_masks is None: bodies[:, 1:7, 1:6] = 1
    masks = np.zeros((frames, 1, h, w), np.float32) if sam_masks is None else np.asarray(sam_masks, np.float32)
    if sam_masks is None: masks[:, 0, 1:7, 1:7] = 1
    node = package.NODE_CLASS_MAPPINGS["SaoChepHeadSwapBackfaceMask"]()
    return node.assemble(images, {"packed_masks": Tensor(masks)}, actor_mask=Tensor(bodies)), face, quality


def recovery_fixture(monkeypatch, body_masks=None, component_family_fn=None):
    face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    detections = [[face], [face]]
    def expanded(seed, parts, scale):
        return np.ones((8, 8), bool)
    return actor_backface_fixture(monkeypatch, detections, body_masks=body_masks,
                                  component_family_fn=component_family_fn or expanded)


def test_actor_mask_recovers_nonanchor_from_owned_sam_and_clears_face_outputs(monkeypatch):
    bodies = np.zeros((2, 8, 8), np.float32)
    bodies[0, 1:7, 1:7] = 1
    bodies[1, 1:4, 1:7] = 1
    result, _, _ = recovery_fixture(monkeypatch, bodies)
    assert np.array_equal(np.asarray(result[0][1]), np.pad(np.ones((6, 6)), ((1, 1), (1, 1))))
    for output in result[2:7]:
        assert not np.asarray(output[1]).any()
    geometry = json.loads(result[7])[1]
    assert geometry["face_visible"] is False and geometry["temporal_repaired"] is True
    assert geometry["tracking_source"] == "sam_confirmed_actor"


def test_anchor_retries_largest_failed_ownership_then_accepts_alternate_same_body(monkeypatch):
    largest = np.array([1, 1, 5, 5, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    alternate = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    bodies = np.zeros((2, 8, 8), np.float32); bodies[:, 1:7, 2:6] = 1
    parsed = []
    def parser(image, face, *args):
        parsed.append(tuple(face[:4]))
        return {"labels": np.ones((8, 8), np.int32), "hair": np.zeros((8, 8)), "hat": np.zeros((8, 8))}
    def expansion(seed, parts, scale):
        return np.ones((8, 8), bool) if scale > 4 else np.asarray(seed, bool)
    result, _, _ = actor_backface_fixture(monkeypatch, [[largest], [alternate]], body_masks=bodies,
                                          component_family_fn=expansion, parser_fn=parser)
    assert len(result) == 8
    assert parsed.count(tuple(largest[:4])) == 2  # candidate evaluation + normal non-anchor processing
    assert parsed.count(tuple(alternate[:4])) == 1  # accepted anchor reused during propagation
    assert np.array_equal(np.asarray(result[0][1]), np.pad(np.ones((6, 6)), ((1, 1), (1, 1))))
    assert np.asarray(result[2][1]).any()  # accepted anchor core was committed
    assert not np.asarray(result[2][0]).any()  # recovered rejected candidate has no semantic output


def test_anchor_all_candidates_fail_with_actual_typed_ownership_reason(monkeypatch):
    face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    bodies = np.zeros((2, 8, 8), np.float32); bodies[:, 1:7, 1:7] = 1
    with pytest.raises(RuntimeError, match="^no_sam_head_owned_by_anchor:1$"):
        actor_backface_fixture(monkeypatch, [[face], [face]], body_masks=bodies,
                               sam_masks=np.zeros((2, 1, 8, 8), np.float32),
                               component_family_fn=lambda seed, parts, scale: np.asarray(seed, bool))


def test_anchor_unknown_parser_exception_propagates_without_retry(monkeypatch):
    face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    calls = []
    def explode(*args):
        calls.append(1)
        raise LookupError("parser backend failure")
    with pytest.raises(LookupError, match="parser backend failure"):
        actor_backface_fixture(monkeypatch, [[face], [face]], parser_fn=explode)
    assert len(calls) == 1


def test_anchor_validation_candidate_cap_is_eight(monkeypatch):
    detections = []
    for _ in range(10):
        detections.append([np.array([0, 0, 6, 6, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)])
    bodies = np.ones((10, 8, 8), np.float32)
    parsed = []
    def parser(*args):
        parsed.append(1)
        return {"labels": np.ones((8, 8), np.int32), "hair": np.zeros((8, 8)), "hat": np.zeros((8, 8))}
    with pytest.raises(RuntimeError, match="no_sam_head_owned_by_anchor"):
        actor_backface_fixture(monkeypatch, detections, body_masks=bodies,
                               sam_masks=np.zeros((10, 1, 8, 8), np.float32),
                               component_family_fn=lambda seed, parts, scale: np.asarray(seed, bool),
                               parser_fn=parser)
    assert len(parsed) == 8


def test_actor_mask_actor_detection_ambiguity_fails_closed_before_anchor_retry(monkeypatch):
    face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    competing = face.copy(); competing[0] = 1
    bodies = np.zeros((2, 8, 8), np.float32); bodies[:, 1:7, 1:7] = 1
    with pytest.raises(RuntimeError, match="actor_face_ambiguous:0:ambiguous_driving_actor"):
        actor_backface_fixture(monkeypatch, [[face, competing], [face]], body_masks=bodies)


def test_actor_mask_excludes_low_confidence_dress_face_before_ambiguity_ranking(monkeypatch):
    true_face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .939], np.float32)
    dress_face = np.array([1, 1, 5, 5, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .523], np.float32)
    bodies = np.ones((2, 8, 8), np.float32)
    result, _, _ = actor_backface_fixture(monkeypatch, [[true_face, dress_face], [true_face]], body_masks=bodies)
    geometry = json.loads(result[7])
    assert geometry[0]["face_box"] == true_face[:4].tolist()
    assert geometry[0]["face_visible"] is True


def test_shared_credible_face_filter_rejects_low_confidence_out_of_bounds_and_malformed(monkeypatch):
    nodes = __import__(PACKAGE_NAME + ".nodes", fromlist=["_credible_face_detections"])
    credible_face = np.array([1, 1, 4, 4, 1, 1, 3, 1, 2, 2, 1, 3, 3, 3, .95], np.float32)
    low = credible_face.copy(); low[14] = .52
    out_of_bounds = credible_face.copy(); out_of_bounds[0] = 5
    result = nodes._credible_face_detections(np.stack([credible_face, low, out_of_bounds]), 8, 8)
    assert len(result) == 1
    assert np.array_equal(result[0][1], credible_face)

    for malformed in (np.zeros((2, 14), np.float32), np.full((1, 15), np.nan, np.float32)):
        with pytest.raises(ValueError, match="Malformed YuNet detections"):
            nodes._credible_face_detections(malformed, 8, 8)


def test_actor_mask_recovery_rejects_candidate_with_insufficient_body_overlap(monkeypatch):
    bodies = np.zeros((2, 8, 8), np.float32)
    bodies[0, 1:7, 1:7] = 1
    bodies[1, 6:8, 6:8] = 1
    with pytest.raises(RuntimeError, match="sam_head_not_owned_by_selected_body:1"):
        recovery_fixture(monkeypatch, bodies)


def test_actor_mask_recovery_transition_failure_is_typed_quality_failure(monkeypatch):
    load_package()
    guard = __import__(PACKAGE_NAME + ".sam_guard", fromlist=["validate_transition"])
    monkeypatch.setattr(guard, "validate_transition", lambda before, after, **kwargs: (_ for _ in ()).throw(ValueError("uncertain")))
    bodies = np.zeros((2, 8, 8), np.float32)
    bodies[0, 1:7, 1:7] = 1
    bodies[1, 1:4, 1:7] = 1
    with pytest.raises(RuntimeError, match="sam_head_continuity_unconfirmed:1:uncertain"):
        recovery_fixture(monkeypatch, bodies)




def test_normal_process_real_guard_failure_keeps_guard_decision(monkeypatch):
    load_package()
    guard = __import__(PACKAGE_NAME + ".sam_guard", fromlist=["validate_transition"])
    original_validate = guard.validate_transition
    calls = []
    def observe_real_guard(previous, current, **kwargs):
        calls.append((previous.copy(), current.copy(), kwargs.copy()))
        return original_validate(previous, current, **kwargs)
    monkeypatch.setattr(guard, "validate_transition", observe_real_guard)
    bodies = np.ones((2, 8, 8), np.float32)
    sam_masks = np.zeros((2, 1, 8, 8), np.float32)
    sam_masks[0, 0, :, :4] = 1
    sam_masks[1, 0, :, 4:] = 1
    cores = [np.zeros((8, 8), bool), np.zeros((8, 8), bool)]
    cores[0][:, :4] = True
    cores[1][:, 4:] = True
    core_index = iter(cores)
    def ownership(*args):
        return next(core_index), np.zeros((8, 8), bool)
    actor_backface_fixture(monkeypatch, [[np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)]] * 2,
                           body_masks=bodies, sam_masks=sam_masks,
                           component_family_fn=lambda seed, parts, scale: np.asarray(seed, bool),
                           face_ownership_fn=ownership)
    assert len(calls) == 1
    assert np.count_nonzero(calls[0][0]) == np.count_nonzero(calls[0][1]) == 32
    assert calls[0][2] == {"motion_compensated": True}




def test_actor_mask_anchor_is_not_recovered_and_unknown_component_error_propagates(monkeypatch):
    bodies = np.zeros((2, 8, 8), np.float32)
    bodies[:, 1:7, 1:7] = 1
    bodies[0] = 0
    bodies[0, 1, 1] = 1
    with pytest.raises(RuntimeError, match="sam_head_not_owned_by_selected_body:0"):
        recovery_fixture(monkeypatch, bodies)
    def explode(seed, parts, scale):
        raise LookupError("unexpected component failure")
    bodies[:] = 0
    bodies[:, 1:7, 1:7] = 1
    with pytest.raises(LookupError, match="unexpected component failure"):
        recovery_fixture(monkeypatch, bodies, component_family_fn=explode)


def test_actor_mask_path_uses_later_anchor_and_bidirectional_sam_for_long_gaps(monkeypatch):
    face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    detections = [[], [], [], [face], [], [], [], [face], []]
    intruder = np.array([5, 1, 3, 5, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    detections[4] = [face, intruder]  # Larger crossing face is outside the selected body mask.
    result, _, _ = actor_backface_fixture(monkeypatch, detections)
    geometry = json.loads(result[7])
    assert len(geometry) == 9
    assert geometry[3]["face_visible"] is True and geometry[7]["face_visible"] is True
    assert geometry[4]["face_visible"] is True
    assert geometry[4]["face_box"] == face[:4].tolist()
    for index in (0, 1, 2, 5, 6, 8):
        assert geometry[index]["face_visible"] is False
        assert geometry[index]["face_box"] is None and "yunet_face" not in geometry[index]
        assert geometry[index]["mouth_y"] is None
    expected = np.zeros((9, 8, 8), np.float32); expected[:, 1:7, 1:7] = 1
    assert np.array_equal(np.asarray(result[0]), expected)
    assert len(result) == 8


def test_registered_anchors_skip_real_hidden_records_and_map_selection_to_original_frame(monkeypatch):
    import importlib
    cv2 = importlib.import_module("cv2")
    face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    detections = [[], [], [], [face], [], [], [], [face], []]
    result, _, _ = actor_backface_fixture(monkeypatch, detections)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    selection_name = PACKAGE_NAME + ".selection"
    spec = importlib.util.spec_from_file_location(selection_name, ROOT / "saochep_headswap" / "selection.py")
    selection = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, selection_name, selection)
    spec.loader.exec_module(selection)
    monkeypatch.setattr(selection, "cv2", cv2)
    paths = importlib.import_module(PACKAGE_NAME + ".frozen_crop.paths")
    monkeypatch.setattr(paths, "asset_path", lambda name: f"/mock/{name}")
    class Hands:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def detect(self, image): return types.SimpleNamespace(hand_landmarks=[])
    vision = types.SimpleNamespace(
        HandLandmarkerOptions=lambda **kwargs: kwargs,
        HandLandmarker=types.SimpleNamespace(create_from_options=lambda options: Hands()),
        FaceLandmarkerOptions=lambda **kwargs: kwargs,
        FaceLandmarker=types.SimpleNamespace(create_from_options=lambda options: Hands()),
    )
    mp = types.ModuleType("mediapipe")
    mp.tasks = types.SimpleNamespace(BaseOptions=lambda **kwargs: kwargs, vision=vision)
    mp.Image = lambda **kwargs: types.SimpleNamespace(**kwargs)
    mp.ImageFormat = types.SimpleNamespace(SRGB="SRGB")
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    selector_images = Tensor(np.stack([np.full((8, 8, 3), i / 10, np.float32) for i in range(9)]))
    geometry_text = result[7]
    nodes = importlib.import_module(PACKAGE_NAME + '.nodes')
    attempts = []

    def mesh_probe(image, box, face, mesh, mp):
        attempts.append(round(float(image[0, 0, 0]) * 10 / 255))
        return object() if attempts[-1] == 7 else None

    monkeypatch.setattr(nodes, '_alignment_landmarks', mesh_probe)
    mapping = load_package().NODE_CLASS_MAPPINGS
    first_image, first_report, first_geometry = mapping["SaoChepHeadSwapAnchor"]().select(selector_images, geometry_text)
    assert attempts == [3, 7]
    assert json.loads(first_report)['selection'] == 'tracked_mesh_retry'
    assert json.loads(first_report)['selected_frame'] == 7
    stable_image, stable_report, stable_geometry = mapping["SaoChepHeadSwapStableAnchor"]().select(selector_images, geometry_text)
    for image, report_text, geometry_out in ((first_image, first_report, first_geometry), (stable_image, stable_report, stable_geometry)):
        report = json.loads(report_text); selected_index = report["selected_frame"]
        assert selected_index in (3, 7)
        assert int(round(float(image[0, 0, 0, 0]) * 10)) == selected_index
        selected_record = json.loads(geometry_out)[0]
        assert selected_record["face_visible"] is True
        assert selected_record["face_box"] is not None






def test_actor_mask_path_fails_typed_for_missing_sam_body_or_anchor(monkeypatch):
    face = np.array([2, 2, 4, 4, 2, 3, 5, 3, 3, 4, 2, 5, 5, 5, .99], np.float32)
    detections = [[], [face], [], []]
    masks = np.zeros((4, 1, 8, 8), np.float32); masks[:, 0, 1:7, 1:7] = 1; masks[2] = 0
    with pytest.raises(RuntimeError, match="sam_actor_track_missing:2"):
        actor_backface_fixture(monkeypatch, detections, sam_masks=masks)
    empty_body = np.zeros((4, 8, 8), np.float32); empty_body[:, 1:7, 1:6] = 1; empty_body[2] = 0
    with pytest.raises(RuntimeError, match="selected_body_track_empty_frame"):
        actor_backface_fixture(monkeypatch, detections, body_masks=empty_body)
    with pytest.raises(RuntimeError, match="no_observed_face_anchor"):
        actor_backface_fixture(monkeypatch, [[], [], [], []])


def test_preflight_ui_has_bounded_validated_report_and_rejects_bad_schema(monkeypatch):
    install_torch_facade(monkeypatch)
    mapping = load_package().NODE_CLASS_MAPPINGS
    node = mapping["SaoChepHeadSwapPreflight"]()
    target = json.dumps({"version": 1, "decision": "native", "reason": "TARGET_ABSENT", "frame_count": 12, "target_index": None})
    result = node.check(target, 720, 1280, 720, 1280)
    report = result["ui"]["headswap_preflight"][0]
    assert report == {"version": 1, "decision": "native", "reason": "TARGET_ABSENT", "frame_count": 12, "target_index": None, "native_width": 720, "native_height": 1280, "delivery_width": 720, "delivery_height": 1280, "prepared": False}
    scaled = node.check(target, 720, 1280, 720, 1280, delivery_scale=2)["ui"]["headswap_preflight"][0]
    assert (scaled["delivery_width"], scaled["delivery_height"]) == (1440, 2560)
    with pytest.raises(ValueError, match="delivery_scale"):
        node.check(target, 720, 1280, 720, 1280, delivery_scale=True)
    with pytest.raises(ValueError, match="Scaled delivery"):
        node.check(target, 720, 1280, 8193, 1280, delivery_scale=2)
    assert result["result"] == ()
    render_target = json.dumps({"version": 1, "decision": "render", "reason": "TARGET_CONFIRMED", "frame_count": 12, "target_index": 2})
    reference_alpha = Tensor(np.zeros((1, 6, 8), np.float32)); reference_alpha[:, 2:4, 3:5] = 1
    prepared_mask = mapping["SaoChepHeadSwapReferenceMask"]().make(reference_alpha)[0]
    alignment = json.dumps({"mode": "translation_only", "scale_correction": 1.0, "translation": [0.0, 1.0], "driving_detection": "tracked_face_roi"})
    prepared = node.check(render_target, 8, 6, 8, 6, prepared_report=alignment, reference_mask=Tensor(prepared_mask))
    assert prepared["ui"]["headswap_preflight"][0]["prepared"] is True
    with pytest.raises(ValueError, match="together"):
        node.check(render_target, 8, 6, 8, 6, reference_mask=Tensor(prepared_mask))
    with pytest.raises(ValueError, match="recognized"):
        node.check(render_target, 8, 6, 8, 6, prepared_report='{}', reference_mask=Tensor(prepared_mask))
    with pytest.raises(ValueError, match="exact"):
        node.check(render_target, 8, 6, 8, 6, prepared_report=alignment, reference_mask=Tensor(np.zeros((2, 6, 8, 3))))
    with pytest.raises(ValueError, match="black/blue"):
        node.check(render_target, 8, 6, 8, 6, prepared_report=alignment, reference_mask=Tensor(np.zeros((1, 6, 8, 3))))
    arbitrary = np.asarray(prepared_mask).copy(); arbitrary[0, 0, 0, 0] = .2
    with pytest.raises(ValueError, match="black/blue"):
        node.check(render_target, 8, 6, 8, 6, prepared_report=alignment, reference_mask=Tensor(arbitrary))
    nonbinary_blue = np.asarray(prepared_mask).copy(); nonbinary_blue[0, 0, 0, 2] = .4
    with pytest.raises(ValueError, match="black/blue"):
        node.check(render_target, 8, 6, 8, 6, prepared_report=alignment, reference_mask=Tensor(nonbinary_blue))
    bad_alignment = json.dumps({"mode": "translation_only", "scale_correction": float("nan"), "translation": [0, 0], "driving_detection": "full_frame"})
    with pytest.raises(ValueError, match="scale"):
        node.check(render_target, 8, 6, 8, 6, prepared_report=bad_alignment, reference_mask=Tensor(prepared_mask))
    with pytest.raises(ValueError, match="Native target"):
        node.check(target, 8, 6, 8, 6, prepared_report=alignment, reference_mask=Tensor(prepared_mask))
    with pytest.raises(ValueError):
        node.check(target + "{}", 720, 1280, 720, 1280)
    with pytest.raises(ValueError):
        node.check(target, 0, 1280, 720, 1280)
    with pytest.raises(ValueError):
        node.check(target, True, 1280, 720, 1280)
    with pytest.raises(ValueError, match="track index"):
        node.check(json.dumps({"version": 1, "decision": "render", "reason": "TARGET_CONFIRMED", "frame_count": 1, "target_index": 64}), 8, 6, 8, 6)
    with pytest.raises(ValueError, match="version"):
        node.check(json.dumps({"version": True, "decision": "native", "reason": "TARGET_ABSENT", "frame_count": 1, "target_index": None}), 8, 6, 8, 6)


def install_export_source_node(monkeypatch, output_root):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(output_root)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    return load_package().NODE_CLASS_MAPPINGS["SaoChepHeadSwapExportSource"]()


def test_export_source_returns_exact_vhs_structure_for_owned_mp4(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    attempt = output_root / "runs" / "attempt-7"
    attempt.mkdir(parents=True)
    source = attempt / "lossless.mp4"
    source.write_bytes(b"already encoded")
    node = install_export_source_node(monkeypatch, output_root)
    result = node.resolve("runs/attempt-7/lossless.mp4", "runs/attempt-7")
    assert result == ((True, [str(source.resolve())]),)
    assert math.isnan(node.IS_CHANGED())


def test_export_source_rejects_traversal_symlinks_missing_and_unowned_paths(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    attempt = output_root / "attempt"; attempt.mkdir(parents=True)
    outside = tmp_path / "outside.mp4"; outside.write_bytes(b"do not read")
    node = install_export_source_node(monkeypatch, output_root)
    invalid = [
        (str(outside), "attempt"),
        ("../outside.mp4", "attempt"),
        ("attempt/../outside.mp4", "attempt"),
        ("attempt\\video.mp4", "attempt"),
        ("https://example.invalid/video.mp4", "attempt"),
        ("attempt/video.mov", "attempt"),
        ("attempt/missing.mp4", "attempt"),
        ("other/video.mp4", "attempt"),
        ("attempt/video.mp4", "../attempt"),
        ("attempt/" + "a" * 2050 + ".mp4", "attempt"),
    ]
    for video, prefix in invalid:
        with pytest.raises(ValueError) as error:
            node.resolve(video, prefix)
        assert str(error.value) == "Unable to resolve an MP4 in the requested attempt"
        assert video not in str(error.value)
    linked_attempt = output_root / "linked-attempt"
    linked_video = attempt / "linked.mp4"
    try:
        linked_attempt.symlink_to(tmp_path, target_is_directory=True)
        linked_video.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation unavailable on this platform")
    with pytest.raises(ValueError):
        node.resolve("linked-attempt/outside.mp4", "linked-attempt")
    with pytest.raises(ValueError):
        node.resolve("attempt/linked.mp4", "attempt")
