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
