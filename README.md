# SaoChepCommercial — isolated candidate nodes

Based on the user-supplied ComfyUI-OpticalStudioEnhance reference_chroma_light_node.py snapshot; source hash in migration_manifest.json at release root.

Commercial revision:
- Existing image transform retained; opt-in and YuNet engine pinned.
- Multi-face bypass; unique sampled indices; valid preview tensor.
- No YuNet download during jobs; bundled model required.
- GPU explicitly requested must have CUDA; no silent 49-second CPU fallback.
- CPU parity only in debug. Updated parity thresholds match acceptance request.
- Out-of-gamut telemetry measured before clipping (previous zero reading was unreliable).
- Node execution serialized around mutable YuNet detector.
- Node class names are distinct from upstream; no source package replacement.

Inherited MediaPipe implementation is unreachable in this release. This avoids re-enabling the old self-intersecting polygon on a host that happens to have MediaPipe installed.
Color offsets apply globally, not skin-only; pixel identity/texture/lighting are not guaranteed bit-exact after gamut conversion or encode.
YuNet model licensing and all generation model licenses must be reviewed by the deployer before public sale; this bundle is not a license attestation.
The deployment uses ComfyUI's existing torch, numpy, Pillow and OpenCV; do not blindly pip-upgrade the GPU runtime.

## Commercial master export metadata

`SaoChepCommercialExport` adds optional BOOLEAN `device_profile_enabled` (default
`True`, including when omitted). Set it to `False` to skip synthetic make/model,
`software` and fabricated creation-time injection. When enabled, MP4 uses
`-movflags +faststart+use_metadata_tags` to persist make/model/software as mdta
metadata. Software uses the explicit `software` key, not `encoder`: FFmpeg's
bitexact muxer initialization deletes that reserved key. When disabled, movflags
remain `+faststart` only. Source metadata and chapter stripping, bitexact flags,
encoding, FPS and normalization remain enabled; this does not guarantee the
absence of all encoder/container tags.

See [FFmpeg's MOV/MP4 muxer options](https://ffmpeg.org/ffmpeg-formats.html#MOV_002fMPEG_002d4_002fISOMBFF-muxers)
and [bitexact muxer initialization (9.0.1)](https://github.com/FFmpeg/FFmpeg/blob/n9.0.1/libavformat/mux.c).
Metadata readers may differ in mdta support. FFprobe 9 may report the same
fabricated `creation_time` twice (header and mdta, separated by `;`); both
represent the same instant.

`device_brand` adds explicit `iPhone 17`: fixed make `Apple`, model `iPhone 17`,
software `26.0` (iOS 26). [Apple's iPhone 17 announcement](https://www.apple.com/newsroom/2025/09/apple-debuts-iphone-17/)
confirms the device features iOS 26. Existing choices and `AUTO_IPHONE` sampling
are unchanged; iPhone 17 is not added to the legacy automatic pool.

The existing `iso_sensor_noise` (default `0.0`), `subvisual_sin_jitter` (default
`False`) and `audio_fingerprint_disrupt` (default `False`) remain independent of
this metadata toggle. These synthetic effects do not guarantee bypassing AI
labels or copyright/audio matching, and do not establish authentic device provenance.

CPU-safe checks: `python3 -m unittest discover -s tests -v`. New export-command
unit tests mock subprocess, filesystem and ComfyUI boundaries; existing handoff
checks run when their local dependencies are available, without downloads.
