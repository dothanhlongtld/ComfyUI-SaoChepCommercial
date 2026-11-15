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
native pixel budget and delivery short side. Outputs 0–3 remain
`native_width`, `native_height`, `delivery_width`, and `delivery_height`; outputs
4–5 append `relay_width` and `relay_height` (final dimensions divided by four).
Native dimensions uniformly scale toward the supplied pixel budget, uniformly
scale down if an axis would exceed 1280, then round to 32-pixel steps within
256–1280. Extreme source ratios are accepted; the minimum native canvas may
require downstream padding. Delivery keeps the source ratio, fixes the shorter
edge to 720 or 1080 pixels, and rounds the longer edge to the nearest multiple
of four. Outputs 4–5 remain a backward-compatible quarter-delivery sizing helper;
callers are not required to downsample before RelayVSR. Native generation budgets,
upscale routing, and final delivery sizes belong to the consuming worker profile.
The node remains generic and accepts a caller-supplied budget.

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

## Bounded-video worker nodes (candidate API v2)

`frame_store.py` and `bounded_video_nodes.py` are the canonical implementation for
`AWGMemoryGenerate`, `AWGMemoryChroma`, `AWGMemoryResize`, `AWGMemoryClarity`, and
`AWGMemoryEncode`. AIWorkerGrid uses these provisioned nodes; its notebook does
not generate another source copy. Ordinary node schema inspection does not
require the optional SCAIL/KJ/VHS plugins. Execution requires the worker's explicit
compatibility validation after all external nodes register. Unsupported helpers
or H.264 format changes fail readiness, without silently reverting to a full-memory
path. The API version must change if its worker contract changes incompatibly. API v2
updates the RelayVSR bridge inputs to the upstream `RELAYVSR_MODEL` handle; workers
must require v2.

`AWGMemoryRelayVSR` accepts `images: AWG_FRAMES` and `model: RELAYVSR_MODEL`
from the upstream RelayVSR model loader, streams frames through the loader-owned
engine, and writes scaled RGB frames to a new bounded stage. The optional
`sr_ratio` input accepts 2 or 4 and defaults to 4 for existing workflows. Explicit
2 requires the loader-owned engine to report native X2 geometry, so original
4× checkpoints cannot silently substitute for the X2 model. Provision the upstream
ComfyUI RelayVSR custom node and its model bundle before execution; the loader
owns model loading, residency, and offload through ComfyUI. This bridge neither
loads checkpoints itself nor unloads the shared model. Finite float RGB frames
are clamped to `[0,1]` and rounded to uint8 at the RelayVSR input boundary,
matching the upstream pixel conversion. This accepts normal bicubic/Lanczos
resize overshoot without modifying stored source frames; NaN/Infinity and
unsupported integer pixel types remain errors.
Callers may feed native generated frames directly into Relay, then apply clarity;
no pre-resize or fixed final delivery size is imposed by this bridge. The node performs no installs, downloads, or inference CLI calls.
CPU-mocked tests do not prove real ComfyUI/GPU activation, checkpoint runtime
compatibility, memory/cancellation behavior, or output quality. Checkpoint
commercial licensing must be verified by the deployer before activation or
distribution; weights are not bundled and no license claim is made here.

The bounded-video integration now covers all four AIWorkerGrid workflows; it is
not limited to single-character replacement.
Generation writes lossless NumPy chunk files under the attempt output directory;
readers materialize bounded ranges, retaining dtype and frame order. Chroma keeps
its global sampling/bypass behavior and adaptive OOM batch reduction. Clarity
keeps global head/tail corrections and original 32-frame adaptive-brightness
groups, while processing smaller GPU batches. The existing shader's optional
`scene_luma` argument supports those groups; omission retains existing behavior.
Resizing and VHS encoding reuse compatible external helpers. The final commercial
exporter remains authoritative.

Chunk files consume local ephemeral disk and add I/O; disk exhaustion is an error,
not permission to lower quality. Handles must not be reused after their final
consumer discards the files. The worker owns interruption, execution-group shutdown,
and attempt cleanup. Input loading/SAM3 are unchanged, so this is not a guarantee of
constant total memory, lower cost, or a particular GPU fit.

CPU-only business checks (requires NumPy and pytest, not Torch or model downloads):

```sh
python -m pytest -q --confcutdir=tests tests/test_bounded_frame_store.py tests/test_frame_store_contract.py tests/test_bounded_video_nodes.py tests/test_bounded_clarity_numeric.py tests/test_relayvsr_node.py tests/test_source_aspect_sizing.py
```

The numerical tests execute the real shader formulas through a NumPy facade;
they do not prove PyTorch CUDA reduction/kernel parity. Existing Torch tests remain
available in a suitable ComfyUI environment. Owner GPU/output acceptance and real
encoder-shutdown validation remain pending. Publishing these nodes and updating
live workers require separate authorization; local edits alone do not reach Colab.

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

## Head-swap nodes

The namespaced `SaoChepHeadSwap*` nodes are maintained in
[`saochep_headswap/`](saochep_headswap/README.md). See that document for their
conditioning/frame-fit contracts, canonical external-asset destinations,
dependency validation, and CPU verification limits. AIWorkerGrid owns workload
profiles, graph wiring, provisioning, job lifecycle, and final export; no second
custom-node source tree is embedded in its worker artifact.
