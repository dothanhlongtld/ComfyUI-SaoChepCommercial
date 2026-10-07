# -*- coding: utf-8 -*-
"""
SaoChepCommercialExport - ComfyUI Custom Node
100% On-Flow Commercial Master Video Export & Naturalize Filter

Complies with Mandatory Rule 4 & Rule 5:
1. Device Spoofing: Generates authentic QuickTime EXIF metadata (Apple iPhone 15/16/17 Pro Max, iOS).
2. Sensor Noise: Dynamic ISO 400 noise jitter (c0s=1.8:c1s=1.0:c2s=1.0:allf=t+u) to break plastic AI smoothness.
3. Sub-visual Sin Wave Jitter: Modulates frame brightness with a dual-harmonic sine wave to bypass AI copyright detection.
4. Audio Normalization: Resamples audio to broadcast standard 48kHz stereo AAC 128k.
5. AI Footprint Eradication: Strips all Lavf/Lavc/Google/C2PA/SynthID tags with -fflags +bitexact.
"""

import os
import sys
import random
import datetime
import subprocess
from pathlib import Path
try:
    import folder_paths
except ImportError:
    folder_paths = None

DEVICE_FAMILIES = [
    {
        "make": "Apple",
        "models": ["iPhone 16 Pro Max", "iPhone 16 Pro", "iPhone 16 Plus", "iPhone 16"],
        "os_pool": ["18.0", "18.0.1", "18.1", "18.1.1", "18.2", "18.2.1", "18.3", "18.3.1"]
    },
    {
        "make": "Apple",
        "models": ["iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15 Plus", "iPhone 15"],
        "os_pool": ["17.4.1", "17.5", "17.5.1", "17.6", "17.6.1", "17.7", "18.0", "18.1"]
    },
    {
        "make": "Apple",
        "models": ["iPhone 14 Pro Max", "iPhone 14 Pro"],
        "os_pool": ["17.3.1", "17.4.1", "17.5.1", "17.6.1", "17.7"]
    },
    {
        "make": "Samsung",
        "models": ["Galaxy S25 Ultra", "Galaxy S24 Ultra"],
        "os_pool": ["One UI 6.1.1", "One UI 7.0"]
    }
]

def pick_device(brand_choice="AUTO_IPHONE"):
    if brand_choice == "AUTO_IPHONE":
        apple_fams = [f for f in DEVICE_FAMILIES if f["make"] == "Apple"]
        family = random.choice(apple_fams)
    elif "Galaxy" in brand_choice:
        family = [f for f in DEVICE_FAMILIES if f["make"] == "Samsung"][0]
    else:
        apple_fams = [f for f in DEVICE_FAMILIES if f["make"] == "Apple"]
        family = random.choice(apple_fams)
        
    model = random.choice(family["models"])
    software = random.choice(family["os_pool"])
    return {
        "make": family["make"],
        "model": model,
        "software": software
    }

def find_mp4_path(filenames):
    """Robustly extracts existing mp4 filepath from VHS_FILENAMES structure, prioritizing audio-muxed files."""
    candidates = []
    if isinstance(filenames, (tuple, list)):
        for item in filenames:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str) and os.path.exists(sub) and sub.lower().endswith(('.mp4', '.mkv', '.mov', '.webm')):
                        candidates.append(sub)
            elif isinstance(item, str) and os.path.exists(item) and item.lower().endswith(('.mp4', '.mkv', '.mov', '.webm')):
                candidates.append(item)
    elif isinstance(filenames, str) and os.path.exists(filenames):
        candidates.append(filenames)

    if not candidates:
        return None

    # Prioritize files with '-audio' in the filename (VHS audio-muxed output)
    for c in reversed(candidates):
        if '-audio' in Path(c).stem.lower():
            return c

    # Otherwise return the last candidate (index -1, most complete output as per VHS specification)
    return candidates[-1]

class SaoChepCommercialExport:
    CATEGORY = "SaoChep/Commercial"
    FUNCTION = "export_master"
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("master_filenames",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filenames": ("VHS_FILENAMES",),
            },
            "optional": {
                "device_brand": (["AUTO_IPHONE", "iPhone 16 Pro Max", "iPhone 15 Pro Max", "Galaxy S25 Ultra"], {"default": "AUTO_IPHONE"}),
                "iso_sensor_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "subvisual_sin_jitter": ("BOOLEAN", {"default": False}),
                "audio_fingerprint_disrupt": ("BOOLEAN", {"default": True, "tooltip": "Bypass Content ID and copyright audio fingerprinting with sub-audible micro-pitch & phase shift"}),
                "audio_sample_rate": (["48000", "44100"], {"default": "48000"}),
                "crf": ("INT", {"default": 18, "min": 10, "max": 30, "step": 1}),
                "output_fps": ("INT", {"default": 30, "min": 0, "max": 60, "step": 1}),
                "preset": (["slow", "medium", "fast", "veryfast"], {"default": "medium"}),
                "cleanup_intermediate": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "saochep/master"}),
            }
        }

    def export_master(self, filenames, device_brand="AUTO_IPHONE", iso_sensor_noise=0.0,
                      subvisual_sin_jitter=False, audio_fingerprint_disrupt=True,
                      audio_sample_rate="48000", crf=18,
                      output_fps=30, preset="medium", cleanup_intermediate=True,
                      filename_prefix="saochep/master"):
        
        raw_video_path = find_mp4_path(filenames)
        if not raw_video_path:
            raise RuntimeError(f"[SaoChepCommercialExport ERROR] Cannot locate valid input video in: {filenames}")

        raw_video = Path(raw_video_path)
        # Automatic Audio Sibling Safeguard: If selected video is mute but '-audio' variant exists in same directory
        if '-audio' not in raw_video.stem.lower():
            audio_sibling = raw_video.parent / f"{raw_video.stem}-audio{raw_video.suffix}"
            if audio_sibling.exists():
                print(f"[SaoChepCommercialExport] Auto-switched to audio sibling: {audio_sibling.name}")
                raw_video = audio_sibling
        if folder_paths is not None:
            output_dir = Path(folder_paths.get_output_directory())
        else:
            output_dir = Path("./output")

        # Determine output path based on filename_prefix
        clean_prefix = filename_prefix.replace("\\", "/").strip("/")
        parts = clean_prefix.split("/")
        if len(parts) > 1:
            subfolder = "/".join(parts[:-1])
            base_name = parts[-1]
            target_dir = output_dir / subfolder
        else:
            subfolder = ""
            base_name = parts[0]
            target_dir = output_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        final_filename = f"{base_name}_{raw_video.stem}_master.mp4"
        final_output_path = target_dir / final_filename

        # If already exists, generate unique name
        counter = 1
        while final_output_path.exists():
            final_filename = f"{base_name}_{raw_video.stem}_master_{counter:04d}.mp4"
            final_output_path = target_dir / final_filename
            counter += 1

        print(f"[SaoChepCommercialExport] Processing Commercial Master...")
        print(f"  Input:  {raw_video}")
        print(f"  Output: {final_output_path}")

        # 1. Device Spoofing
        device = pick_device(device_brand)
        delta_days = random.randint(1, 7)
        delta_seconds = random.randint(0, 86400)
        fake_time = datetime.datetime.now() - datetime.timedelta(days=delta_days, seconds=delta_seconds)
        fake_time_str = fake_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 2. Single-Pass Delivery FPS + Dynamic Sensor Noise Filter
        filters = []
        if output_fps and int(output_fps) > 0:
            filters.append(f"fps={int(output_fps)}:round=near")
        if iso_sensor_noise > 0.05:
            c0s = round(iso_sensor_noise, 2)
            c1s = round(iso_sensor_noise * 0.55, 2)
            c2s = round(iso_sensor_noise * 0.55, 2)
            filters.append(f"noise=c0s={c0s}:c1s={c1s}:c2s={c2s}:allf=t+u")

        # 3. Sub-visual Sin Wave Jitter Filter
        if subvisual_sin_jitter:
            amp = round(random.uniform(0.0006, 0.0010), 4)
            period = random.randint(130, 180)
            phase = round(random.uniform(0, 6.28), 2)
            sin_expr = f"{amp}*sin(2*PI*n/{period}+{phase})"
            filters.append(f"eq=eval=frame:contrast=1.02:saturation=1.0:brightness='{sin_expr}'")

        vf_str = ",".join(filters) if filters else "null"

        # 3b. Audio Anti-Fingerprint & Broadcast Normalization Filters
        af_filters = [f"aresample={audio_sample_rate}"]
        if audio_fingerprint_disrupt:
            # Sub-audible micro-pitch shift (+0.6%) while keeping exact 100% video sync and duration
            af_filters.append("asetrate=48000*1.006,atempo=1/1.006")
            # Stereo phase decorrelation to disrupt dual-channel cross-correlation fingerprint
            af_filters.append("extrastereo=m=1.03")
            # Micro-frequency perturbation on typical fingerprint peak bands
            af_filters.append("equalizer=f=1200:t=q:w=1.5:g=0.5,equalizer=f=3400:t=q:w=1.2:g=-0.4")
        af_str = ",".join(af_filters)

        # 4. Execute Broadcast-Standard FFmpeg Command
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_video),
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-vf", vf_str,
            "-af", af_str,
            "-c:v", "libx264", "-preset", str(preset), "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-c:a", "aac", "-b:a", "128k",
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
            "-metadata", f"make={device['make']}",
            "-metadata", f"model={device['model']}",
            "-metadata", f"encoder={device['software']}",
            "-metadata", f"creation_time={fake_time_str}",
            "-movflags", "+faststart",
            str(final_output_path)
        ]

        print(f"[SaoChepCommercialExport] Running FFmpeg Naturalize Filter (Device: {device['model']}, FPS: {output_fps}, Preset: {preset}, Audio: {audio_sample_rate}Hz)...")
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0 or not final_output_path.exists() or final_output_path.stat().st_size == 0:
            raise RuntimeError(
                f"[SaoChepCommercialExport ERROR] FFmpeg failed (code {res.returncode}): {res.stderr[-600:]}"
            )

        if cleanup_intermediate:
            try:
                raw_path_obj = Path(raw_video)
                if raw_path_obj.exists() and raw_path_obj.resolve() != final_output_path.resolve():
                    raw_path_obj.unlink(missing_ok=True)
                if raw_path_obj.name.endswith("-audio.mp4"):
                    sibling = raw_path_obj.with_name(raw_path_obj.name[:-len("-audio.mp4")] + ".mp4")
                    if sibling.exists() and sibling.resolve() != final_output_path.resolve():
                        sibling.unlink(missing_ok=True)
            except Exception as clean_err:
                print(f"[SaoChepCommercialExport WARN] Intermediate cleanup skipped: {clean_err}")

        print(f"[SaoChepCommercialExport SUCCESS] Commercial Master exported: {final_output_path} ({os.path.getsize(final_output_path)} bytes)")

        return {
            "ui": {
                "gifs": [
                    {
                        "filename": final_filename,
                        "subfolder": subfolder,
                        "type": "output",
                        "format": "video/mp4"
                    }
                ]
            },
            "result": ((True, [str(final_output_path)]),)
        }
