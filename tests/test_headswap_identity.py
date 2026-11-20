import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name, relative_path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _identity():
    # Avoid importing the legacy top-level package and polluting namespace tests.
    return _module("_saochep_identity_test", "saochep_headswap/identity.py")


def _vector(value=0.0):
    vector = [0.0] * _identity().FEATURES
    vector[0] = value
    vector[1] = (1.0 - value * value) ** .5
    return vector


def _payload(vectors):
    identity = _identity()
    return {"version": 1, "model_sha256": identity.MODEL_SHA256,
            "dimension": identity.FEATURES, "templates": vectors}


def test_attempt_template_is_exclusive_private_and_round_trips(tmp_path):
    identity = _identity()
    path = identity.template_path(tmp_path, "aiworkergrid/" + "a" * 32)
    vectors = [_vector(.3), _vector(.4), _vector(.5)]
    identity.save_templates_once(path, vectors)
    assert identity.load_templates(path) == vectors
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(identity.IdentityError, match="concurrently"):
        identity.save_templates_once(path, vectors)


def test_identity_prefix_rejects_traversal(tmp_path):
    identity = _identity()
    with pytest.raises(identity.IdentityError):
        identity.template_path(tmp_path, "../outside")


def test_malformed_oversized_and_symlink_templates_fail_closed(tmp_path):
    identity = _identity()
    path = identity.template_path(tmp_path, "aiworkergrid/" + "b" * 32)
    path.write_text(json.dumps(_payload([_vector()] * 3)))
    with pytest.raises(identity.IdentityError):
        identity.load_templates(path)
    path.unlink()
    path.write_bytes(b" " * 32769)
    path.chmod(0o600)
    with pytest.raises(identity.IdentityError, match="oversized"):
        identity.load_templates(path)
    path.unlink()
    target = path.parent / "target"
    target.write_text("{}")
    target.chmod(0o600)
    path.symlink_to(target)
    with pytest.raises(identity.IdentityError):
        identity.load_templates(path)


def test_identity_schema_rejects_bad_digest_nonfinite_and_nonunit_features():
    identity = _identity()
    valid = _payload([_vector()] * 3)
    identity._validate_templates(valid)
    invalid_values = (
        {**valid, "model_sha256": "0" * 64},
        _payload([[float("nan")] * identity.FEATURES]),
        _payload([[0.0] * identity.FEATURES]),
        _payload([_vector()] * 9),
    )
    for invalid in invalid_values:
        with pytest.raises(identity.IdentityError):
            identity._validate_templates(invalid)


def test_persisted_template_rejects_duplicate_keys_and_noninteger_dimension(tmp_path):
    identity = _identity()
    path = identity.template_path(tmp_path, "aiworkergrid/" + "c" * 32)
    path.write_text('{"version":1,"version":1}')
    path.chmod(0o600)
    with pytest.raises(identity.IdentityError, match="Malformed"):
        identity.load_templates(path)
    path.write_text(json.dumps({**_payload([_vector()] * 3), "dimension": 128.0}))
    path.chmod(0o600)
    with pytest.raises(identity.IdentityError, match="dimensions"):
        identity.load_templates(path)
    path.write_text(json.dumps({**_payload([_vector()] * 3), "dimension": True}))
    path.chmod(0o600)
    with pytest.raises(identity.IdentityError, match="dimensions"):
        identity.load_templates(path)


def test_deeply_nested_template_is_classified_as_malformed_and_preserved(tmp_path):
    identity = _identity()
    path = identity.template_path(tmp_path, "aiworkergrid/" + "4" * 32)
    payload = ("[" * 10000 + "0" + "]" * 10000).encode("ascii")
    assert len(payload) < 32768
    path.write_bytes(payload)
    path.chmod(0o600)
    with pytest.raises(identity.IdentityError, match="Malformed identity template"):
        identity.load_templates(path)
    assert path.read_bytes() == payload


def test_multiframe_cosine_threshold_and_eligible_track_ambiguity():
    identity = _identity()
    templates = [_vector(0.0)] * 3
    same = [_vector(0.0)] * 3
    weak = [_vector(1.0)] * 4
    assert identity.match_counts(same, templates)[0] == 3
    assert identity.match_counts(same[:2], templates)[0] == 2
    assert identity.match_counts(weak, templates)[0] == 0
    assert identity.select_matching_track([(4, same)], templates) == (4, False)
    assert identity.select_matching_track([(2, weak)], templates) == (None, False)
    assert identity.select_matching_track([(2, same), (7, same)], templates) == (None, True)
    assert identity.select_matching_track([(2, same[:2])], templates) == (None, False)


def test_sface_execution_errors_raise_but_invalid_face_is_unusable():
    identity = _identity()
    import numpy as np

    class BrokenAlign:
        def alignCrop(self, *_args): raise RuntimeError("internal detail")
    valid = np.zeros(15, np.float32)
    valid[14] = .99
    with pytest.raises(identity.IdentityError, match="alignment or feature extraction failed") as caught:
        identity.embed(BrokenAlign(), np.zeros((8, 8, 3), np.uint8), valid)
    assert isinstance(caught.value.__cause__, RuntimeError)

    class BrokenFeature:
        def alignCrop(self, *_args): return np.zeros((8, 8, 3), np.uint8)
        def feature(self, _crop): raise RuntimeError("internal detail")
    with pytest.raises(identity.IdentityError, match="alignment or feature extraction failed"):
        identity.embed(BrokenFeature(), np.zeros((8, 8, 3), np.uint8), valid)
    assert identity.embed(BrokenAlign(), np.zeros((8, 8, 3), np.uint8), np.zeros(14)) is None


def test_identity_cache_is_invalidated_only_when_enabled():
    import math
    nodes = _module("_saochep_nodes_test", "saochep_headswap/nodes.py")
    assert nodes.HSHeadSwapTarget.IS_CHANGED(identity_lock=False) is None
    assert math.isnan(nodes.HSHeadSwapTarget.IS_CHANGED(identity_lock=True))
