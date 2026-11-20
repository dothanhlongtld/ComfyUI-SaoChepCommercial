"""Attempt-local, bounded SFace identity lock for selected SAM body tracks."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import secrets
from pathlib import Path
from typing import Any

MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
MODEL_BYTES = 38696353
FEATURES = 128
MAX_TEMPLATES = 8
MIN_SIMILARITY = .363
MIN_MATCHES = 3
_PREFIX = re.compile(r"aiworkergrid/[0-9a-f]{32}\Z")


class IdentityError(RuntimeError):
    """Identity evidence or its persistent attempt-local state is unsafe."""


def _safe_dir(path: Path, *, create: bool = False, private: bool = False, boundary: Path | None = None) -> Path:
    path = path.absolute()
    part = path
    while True:
        if part.is_symlink():
            raise IdentityError("Identity state path contains a symlink")
        if boundary is not None and part == boundary:
            break
        if part.parent == part:
            if boundary is not None:
                raise IdentityError("Identity path escaped its output root")
            break
        part = part.parent
    if create:
        try:
            path.mkdir(mode=0o700, parents=True)
        except FileExistsError:
            pass
    if not path.is_dir() or path.resolve() != path:
        raise IdentityError("Identity state directory is unavailable")
    if private and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise IdentityError("Identity state directory permissions are too broad")
    return path


def template_path(output_root: str | os.PathLike[str], prefix: str) -> Path:
    if not isinstance(prefix, str) or not _PREFIX.fullmatch(prefix):
        raise IdentityError("Invalid attempt-owned identity prefix")
    configured_root = Path(output_root).absolute()
    if configured_root.is_symlink():
        raise IdentityError("Identity output root cannot be a symlink")
    root = configured_root.resolve()
    _safe_dir(root)
    attempt = _safe_dir(root / prefix, create=True, boundary=root)
    private = _safe_dir(attempt / ".identity", create=True, private=True, boundary=root)
    return private / "source.json"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, item in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = item
    return result


def _validate_templates(value: Any) -> list[list[float]]:
    if not isinstance(value, dict) or set(value) != {"version", "model_sha256", "dimension", "templates"}:
        raise IdentityError("Invalid identity template schema")
    if value["version"] != 1 or type(value["version"]) is not int:
        raise IdentityError("Unsupported identity template version")
    if value["model_sha256"] != MODEL_SHA256:
        raise IdentityError("Identity template model or dimensions mismatch")
    if type(value["dimension"]) is not int or value["dimension"] != FEATURES:
        raise IdentityError("Identity template model or dimensions mismatch")
    templates = value["templates"]
    if not isinstance(templates, list) or not 1 <= len(templates) <= MAX_TEMPLATES:
        raise IdentityError("Invalid identity template count")
    normalized = []
    for vector in templates:
        if not isinstance(vector, list) or len(vector) != FEATURES:
            raise IdentityError("Invalid identity feature vector")
        if any(type(item) not in (int, float) or not math.isfinite(item) for item in vector):
            raise IdentityError("Non-finite identity feature")
        norm = math.sqrt(sum(float(item) ** 2 for item in vector))
        if not .999 <= norm <= 1.001:
            raise IdentityError("Identity feature is not normalized")
        normalized.append([float(item) for item in vector])
    return normalized


def load_templates(path: Path) -> list[list[float]] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as error:
        raise IdentityError("Cannot safely open identity template") from error
    try:
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > 32768:
                raise IdentityError("Identity template file is unsafe or oversized")
            raw = stream.read(32769)
    except OSError as error:
        raise IdentityError("Cannot read identity template") from error
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise IdentityError("Malformed identity template") from error
    return _validate_templates(data)


def save_templates_once(path: Path, templates: list[list[float]]) -> None:
    value = {"version": 1, "model_sha256": MODEL_SHA256, "dimension": FEATURES, "templates": templates[:MAX_TEMPLATES]}
    _validate_templates(value)
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > 32768:
        raise IdentityError("Identity template exceeds size limit")
    temporary = path.parent / ("." + secrets.token_hex(16) + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        # link() is exclusive: a concurrent seed cannot replace the first identity.
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise IdentityError("Identity template was concurrently seeded") from error
    except OSError as error:
        raise IdentityError("Cannot create attempt identity template") from error
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass


_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


def create_recognizer(model_path: str | os.PathLike[str]):
    path = Path(model_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != MODEL_BYTES:
        raise IdentityError("Pinned SFace model is unavailable or unsafe")
    model_stat = path.stat()
    key = (str(path.resolve()), model_stat.st_ino, model_stat.st_size,
           model_stat.st_mtime_ns, model_stat.st_ctime_ns)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise IdentityError("Pinned SFace model digest mismatch")
    import cv2
    try:
        result = cv2.FaceRecognizerSF.create(str(path), "")
    except Exception as error:
        raise IdentityError("Cannot initialize verified SFace model") from error
    _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = result
    return result


def embed(recognizer: Any, bgr: Any, face: Any) -> list[float] | None:
    import numpy as np
    try:
        face = np.asarray(face)
        valid_face = (face.ndim == 1 and face.shape[0] >= 15
                      and np.isfinite(face[:15]).all() and face[14] >= .9)
    except (TypeError, ValueError):
        return None
    if not valid_face:
        return None
    try:
        crop = recognizer.alignCrop(bgr, face.reshape(1, -1))
        feature = np.asarray(recognizer.feature(crop), dtype=np.float64).reshape(-1)
    except Exception as error:
        raise IdentityError("SFace alignment or feature extraction failed") from error
    if feature.shape != (FEATURES,) or not np.isfinite(feature).all():
        raise IdentityError("SFace returned an invalid feature vector")
    norm = float(np.linalg.norm(feature))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise IdentityError("SFace returned an unnormalizable feature vector")
    feature /= norm
    return feature.tolist()


def match_counts(features: list[list[float]], templates: list[list[float]]) -> tuple[int, float]:
    import numpy as np
    refs = np.asarray(templates, dtype=np.float64)
    scores = [float(np.max(refs @ np.asarray(feature, dtype=np.float64))) for feature in features]
    matches = [score for score in scores if score >= MIN_SIMILARITY]
    return len(matches), (max(matches) if matches else 0.0)


def select_matching_track(candidates: list[tuple[int, list[list[float]]]], templates: list[list[float]]) -> tuple[int | None, bool]:
    eligible = [track for track, features in candidates if match_counts(features, templates)[0] >= MIN_MATCHES]
    if len(eligible) > 1:
        return None, True
    return (eligible[0], False) if eligible else (None, False)
