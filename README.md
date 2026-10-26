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

## Source-aspect sizing and timeline fitting

`SaoChepSourceAspectSizing` accepts a non-empty ComfyUI `IMAGE` batch plus a
native pixel budget and delivery short side. It returns `native_width`,
`native_height`, `delivery_width`, and `delivery_height`. Native dimensions
uniformly scale toward the supplied pixel budget, uniformly scale down if an
axis would exceed 1280, then round to 32-pixel steps within 256–1280. Extreme
source ratios are accepted; the minimum native canvas may require downstream
padding. Delivery keeps the source ratio, fixes the shorter edge to an even
720 or 1080 pixels, and rounds the longer edge to an even pixel count. Callers
choose the budget (Lite 576×1024, Pro 640×1152, Pro++ 704×1280) and short side.

`SaoChepAdaptiveTimelineMerger` contain-fits original B-roll frames to the
rendered delivery canvas with centered black padding instead of stretching.
Explicit empty `actor_indices` metadata returns original footage fitted to the
same delivery canvas. The scene router’s existing render-all fallback when no
actor shots are detected is unchanged; this is not an all-B-roll detection fix.
Existing actor placement and crossfade behavior is unchanged. Shot geometry also scales reference dimensions together to fit its
canvas; thin images are no longer stretched to an artificial 64-pixel minimum.

CPU-safe checks: `python3 -m unittest discover -s tests -v`. New source sizing
tests use only the standard library and prove dimension bounds, alignment,
source-ratio delivery, registration order, and contain-fit geometry. Existing
ComfyUI tensor checks run only when their local dependencies are available;
no downloads are performed.

## Adaptive clarity enhancer (Shadow Toe Guard & Chroma Decoupling)

`SaoChepAdaptiveClarityEnhancer` implements:
- **Shadow Toe Guard ($Y < 0.20$)**: Limits luminance drop in dark regions to $< 5\%$ using a smoothstep toe transition curve ($[0.0, 0.20]$), preventing crushed blacks.
- **Chroma Decoupling ($YC_bC_r$)**: Micro-contrast and sharpening apply strictly to the Luma channel ($Y$), keeping original $C_b, C_r$ chromaticity intact to prevent oversaturation and color burn.
- **Scene-Adaptive Scaling**: When average scene luminance drops below $0.35$ (low-key scenes), sharpening and clarity strengths automatically scale down to avoid noise amplification.

## Smart face crop (`head_torso_clip`)

`SaoChepSmartFaceCrop` includes a dedicated `head_torso_clip` mode in `target_aspect`. It uses the bundled YuNet detector to locate the face, adds 15% headroom above the head top, and extends $2.5\times$ face height downwards to frame head and torso. Images that are already close-up portraits ($f_h / H \ge 0.35$) automatically bypass. This provides 6x face token density when feeding CLIP-Vision (Node 76).

## Motion interpolation export (Optical Flow MCI)

`SaoChepCommercialExport` adds optional BOOLEAN `motion_interpolation` (default `False`). When enabled, FFmpeg uses `minterpolate=fps={output_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1` for motion-compensated frame rate conversion (e.g. 24fps -> 30fps) with an extended 600s timeout. When disabled, standard near-frame resampling runs fast with a 180s timeout. Full timeline and audio sync are strictly preserved (no frame drops).

## Subtitle purge (Zero-Copy On-Flow)

`SaoChepSubtitlePurge` scans the lower bottom zone ($Y \ge 0.75$) for hardcoded subtitles or watermarks:
- Clean video without subtitles: **100% Zero-Copy Bypass** (0ms, 0 pixels altered).
- Subtitled video: Localized Telea inpainting on detected text glyphs to prevent DiT hallucination.

## Dynamic background composite

`SaoChepDynamicBackgroundComposite` composites rendered AI character frames onto original camera background plates using soft-feathered alpha blending (Gaussian blur radius 2–4px) to eliminate edge halos while fully preserving background camera motion, foliage, and traffic dynamics.
