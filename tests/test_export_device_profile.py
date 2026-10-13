"""CPU-only export command tests; no ComfyUI, FFmpeg or real files required."""
import datetime
import inspect
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import commercial_naturalize_export_node as export


def metadata(command):
    return [command[i + 1] for i, arg in enumerate(command) if arg == "-metadata"]


class ExportDeviceProfileTests(unittest.TestCase):
    def setUp(self):
        self.source = Path("/virtual/input/source.mp4")
        self.output_root = Path("/virtual/output")
        self.now = datetime.datetime(2026, 3, 10, 12, 0, 0)
        folders = types.SimpleNamespace(get_output_directory=lambda: str(self.output_root))
        self.start_patch("folder_paths", folders)
        self.start_patch("get_ffmpeg_binary", return_value="mock-ffmpeg")
        self.start_patch("os.path.exists", return_value=True)
        self.start_patch("os.path.getsize", return_value=2048)
        self.exists = self.start_patch("Path.exists", return_value=True)
        self.start_patch("Path.resolve", autospec=True, side_effect=lambda path: Path(os.path.normpath(str(path))))
        self.mkdir = self.start_patch("Path.mkdir")
        self.start_patch("Path.stat", return_value=types.SimpleNamespace(st_size=2048))
        self.unlink = self.start_patch("Path.unlink")
        self.run = self.start_patch(
            "subprocess.run", return_value=types.SimpleNamespace(returncode=0, stderr="")
        )
        self.choice = self.start_patch("random.choice", side_effect=lambda values: values[0])
        self.randint = self.start_patch("random.randint", side_effect=[2, 3600])
        self.clock = self.start_patch("datetime.datetime")
        self.clock.now.return_value = self.now
        job = patch("uuid.uuid4", return_value=types.SimpleNamespace(hex="123456789abcdef"))
        job.start()
        self.addCleanup(job.stop)

    def start_patch(self, name, *args, **kwargs):
        patcher = patch.object(export, name, *args, **kwargs) if "." not in name else patch(
            f"commercial_naturalize_export_node.{name}", *args, **kwargs
        )
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def invoke(self, **kwargs):
        result = export.SaoChepCommercialExport().export_master(str(self.source), **kwargs)
        return self.run.call_args.args[0], result

    def assert_privacy_and_encoding(self, command):
        for flag, value in (
            ("-map_metadata", "-1"), ("-map_chapters", "-1"),
            ("-fflags", "+bitexact"), ("-flags:v", "+bitexact"),
            ("-flags:a", "+bitexact"), ("-movflags", "+faststart"),
            ("-c:v", "libx264"), ("-pix_fmt", "yuv420p"),
            ("-c:a", "aac"), ("-b:a", "128k"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(
            [command[i + 1] for i, flag in enumerate(command) if flag == "-map"],
            ["0:v:0", "0:a:0?"],
        )
        for flag in ("-r", "-shortest", "-t", "-frames:v"):
            self.assertNotIn(flag, command)

    def test_optional_schema_defaults_and_positional_compatibility(self):
        inputs = export.SaoChepCommercialExport.INPUT_TYPES()["optional"]
        self.assertEqual(inputs["device_profile_enabled"][0], "BOOLEAN")
        self.assertIs(inputs["device_profile_enabled"][1]["default"], True)
        self.assertEqual(inputs["device_brand"], (
            ["AUTO_IPHONE", "iPhone 16 Pro Max", "iPhone 15 Pro Max", "Galaxy S25 Ultra", "iPhone 17"],
            {"default": "AUTO_IPHONE"},
        ))
        self.assertEqual(inputs["iso_sensor_noise"][1]["default"], 0.0)
        self.assertIs(inputs["subvisual_sin_jitter"][1]["default"], False)
        self.assertIs(inputs["audio_fingerprint_disrupt"][1]["default"], False)
        parameters = inspect.signature(export.SaoChepCommercialExport.export_master).parameters
        self.assertEqual(list(parameters)[-2:], ["filename_prefix", "device_profile_enabled"])
        self.assertIs(parameters["device_profile_enabled"].default, True)
        export.SaoChepCommercialExport().export_master(
            str(self.source), "iPhone 15 Pro Max", 0.0, False, False,
            "48000", 20, 24, "fast", False, "nested/master",
        )
        command = self.run.call_args.args[0]
        self.assertIn("model=iPhone 15 Pro Max", metadata(command))
        self.assertEqual(command[command.index("-vf") + 1], "fps=24:round=near")
        self.assertEqual(command[command.index("-crf") + 1], "20")
        self.assertEqual(command[command.index("-preset") + 1], "fast")
        self.unlink.assert_not_called()

    def test_false_suppresses_synthetic_tags_and_generation_only(self):
        self.clock.now.side_effect = AssertionError("must not fabricate time")
        self.randint.side_effect = AssertionError("must not sample fake time")
        with patch.object(export, "pick_device", side_effect=AssertionError("must not pick device")):
            command, _ = self.invoke(device_profile_enabled=False, device_brand="iPhone 17")
        self.assertEqual(metadata(command), [])
        self.choice.assert_not_called()
        self.assert_privacy_and_encoding(command)
        self.assertEqual(command[command.index("-vf") + 1], "fps=30:round=near")
        self.assertEqual(command[command.index("-af") + 1], "aresample=48000")

    def test_true_injects_fixed_iphone17_and_time_with_privacy_flags(self):
        command, _ = self.invoke(device_profile_enabled=True, device_brand="iPhone 17")
        self.assertEqual(metadata(command), [
            "make=Apple", "model=iPhone 17", "encoder=26.0",
            "creation_time=2026-03-08T11:00:00Z",
        ])
        self.choice.assert_not_called()
        self.assert_privacy_and_encoding(command)

    def test_fixed_profile_is_case_insensitive_and_never_samples(self):
        for choice in ("iPhone 17", "IPHONE 17"):
            self.assertEqual(export.pick_device(choice), {
                "make": "Apple", "model": "iPhone 17", "software": "26.0",
            })
        self.choice.assert_not_called()

    def test_omitted_flag_keeps_legacy_auto_pool_and_injection(self):
        command, _ = self.invoke()
        self.assertEqual(metadata(command)[:3], [
            "make=Apple", "model=iPhone 16 Pro Max", "encoder=18.0",
        ])
        apple_pool = self.choice.call_args_list[0].args[0]
        self.assertEqual([family["models"] for family in apple_pool], [
            ["iPhone 16 Pro Max", "iPhone 16 Pro", "iPhone 16 Plus", "iPhone 16"],
            ["iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15 Plus", "iPhone 15"],
            ["iPhone 14 Pro Max", "iPhone 14 Pro"],
        ])
        self.assertEqual([family["os_pool"] for family in apple_pool], [
            ["18.0", "18.0.1", "18.1", "18.1.1", "18.2", "18.2.1", "18.3", "18.3.1"],
            ["17.4.1", "17.5", "17.5.1", "17.6", "17.6.1", "17.7", "18.0", "18.1"],
            ["17.3.1", "17.4.1", "17.5.1", "17.6.1", "17.7"],
        ])
        self.assertEqual(self.choice.call_count, 3)
        self.assertEqual(command[command.index("-af") + 1], "aresample=48000")

    def test_old_explicit_choices_preserve_model_and_os_sampling(self):
        for model, make, software in (
            ("iPhone 16 Pro Max", "Apple", "18.0"),
            ("iPhone 15 Pro Max", "Apple", "17.4.1"),
            ("Galaxy S25 Ultra", "Samsung", "One UI 6.1.1"),
        ):
            with self.subTest(model=model):
                self.randint.side_effect = [2, 3600]
                self.choice.reset_mock()
                command, _ = self.invoke(device_brand=model)
                self.assertEqual(metadata(command)[:3], [
                    f"make={make}", f"model={model}", f"encoder={software}",
                ])
                self.assertEqual(self.choice.call_count, 1)

    def test_effects_are_independent_of_metadata_toggle(self):
        commands = []
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.randint.side_effect = [2, 3600, 150] if enabled else [150]
                with patch.object(export.random, "uniform", side_effect=[0.0008, 1.25]):
                    command, _ = self.invoke(
                        device_profile_enabled=enabled, device_brand="iPhone 17",
                        iso_sensor_noise=1.8, subvisual_sin_jitter=True,
                        audio_fingerprint_disrupt=True, cleanup_intermediate=False,
                    )
                self.assertEqual(command[command.index("-vf") + 1],
                    "fps=30:round=near,noise=c0s=1.8:c1s=0.99:c2s=0.99:allf=t+u,"
                    "eq=eval=frame:contrast=1.02:saturation=1.0:brightness='0.0008*sin(2*PI*n/150+1.25)'")
                self.assertEqual(command[command.index("-af") + 1],
                    "aresample=48000,asetrate=48000*1.006,atempo=1/1.006,extrastereo=m=1.03,"
                    "equalizer=f=1200:t=q:w=1.5:g=0.5,equalizer=f=3400:t=q:w=1.2:g=-0.4")
                commands.append([arg for i, arg in enumerate(command)
                                 if arg != "-metadata" and (i == 0 or command[i - 1] != "-metadata")])
        self.assertEqual(commands[0], commands[1])

    def test_disabled_effects_keep_configured_resampling_and_native_fps(self):
        command, _ = self.invoke(device_profile_enabled=False, output_fps=0, audio_sample_rate="44100")
        self.assertEqual(command[command.index("-vf") + 1], "null")
        self.assertEqual(command[command.index("-af") + 1], "aresample=44100")

    def test_audio_sibling_output_schema_and_cleanup_are_preserved(self):
        command, result = self.invoke(device_profile_enabled=False)
        self.assertEqual(command[command.index("-i") + 1], "/virtual/input/source-audio.mp4")
        filename = "master_source-audio_123456789abc_master.mp4"
        self.assertEqual(result, {
            "ui": {"gifs": [{"filename": filename, "subfolder": "saochep",
                             "type": "output", "format": "video/mp4"}]},
            "result": ((True, [f"/virtual/output/saochep/{filename}"]),),
        })
        self.assertEqual(export.SaoChepCommercialExport.RETURN_TYPES, ("VHS_FILENAMES",))
        self.assertEqual(export.SaoChepCommercialExport.RETURN_NAMES, ("master_filenames",))
        self.assertEqual(self.unlink.call_count, 2)
        self.assertEqual(self.run.call_args.kwargs, {
            "capture_output": True, "text": True, "encoding": "utf-8", "timeout": 180,
        })

    def test_failed_export_does_not_clean_intermediates(self):
        self.run.return_value = types.SimpleNamespace(returncode=1, stderr="failed")
        with self.assertRaisesRegex(RuntimeError, "FFmpeg failed"):
            self.invoke(device_profile_enabled=False)
        self.unlink.assert_not_called()

    def test_timeout_does_not_clean_intermediates(self):
        self.run.side_effect = subprocess.TimeoutExpired("mock-ffmpeg", 180)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.invoke(device_profile_enabled=False)
        self.unlink.assert_not_called()

    def test_output_containment_remains_enforced(self):
        with self.assertRaisesRegex(ValueError, "outside output directory"):
            self.invoke(device_profile_enabled=False, filename_prefix="../escape/master")
        self.run.assert_not_called()
        self.unlink.assert_not_called()


if __name__ == "__main__":
    unittest.main()
