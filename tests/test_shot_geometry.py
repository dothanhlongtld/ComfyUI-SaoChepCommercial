"""Test SaoChepShotGeometry memory safety and session caching."""
import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "shot_geometry_node.py"


class TestShotGeometryMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        cls.geometry_class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SaoChepShotGeometry"
        )

    def _instantiate_geometry(self, mock_cv2=None, mock_np=None, mock_torch=None):
        ns = {
            "cv2": mock_cv2 or MagicMock(),
            "np": mock_np or MagicMock(),
            "torch": mock_torch or MagicMock(),
            "os": MagicMock(),
            "tempfile": MagicMock(),
            "_REMBG_SESSION": None,
            "_get_rembg_session": Mock(return_value=None),
        }
        mod = ast.Module(body=[self.geometry_class_node], type_ignores=[])
        ast.fix_missing_locations(mod)
        code = compile(mod, filename="shot_geometry_node.py", mode="exec")
        exec(code, ns)
        return ns["SaoChepShotGeometry"]()

    def test_temp_centers_not_retained_when_empty_or_none(self):
        geom = self._instantiate_geometry()
        geom._temp_centers = [10.0, 20.0, 30.0]

        ambient, med_w, med_y, med_cx, shape = geom._analyze_driving_video(None)
        self.assertEqual(geom._temp_centers, [])
        self.assertEqual(med_cx, 288.0)

        geom._temp_centers = [99.0]
        ambient, med_w, med_y, med_cx, shape = geom._analyze_driving_video([])
        self.assertEqual(geom._temp_centers, [])
        self.assertEqual(med_cx, 288.0)

    def test_temp_centers_reset_even_on_exception(self):
        geom = self._instantiate_geometry()
        geom._temp_centers = [42.0]

        mock_frame = MagicMock()
        mock_frame.detach.side_effect = RuntimeError("GPU Out of Memory simulated")
        driving_video = [mock_frame]

        with self.assertRaises(RuntimeError):
            geom._analyze_driving_video(driving_video)

        self.assertEqual(geom._temp_centers, [])

    def test_temp_centers_cleared_after_successful_analysis(self):
        mock_cv2 = MagicMock()
        mock_cv2.COLOR_RGB2BGR = 1
        frame_bgr = MagicMock()
        frame_bgr.shape = (1024, 576, 3)
        mock_cv2.cvtColor.return_value = frame_bgr

        mock_np = MagicMock()
        mock_np.concatenate.return_value = MagicMock()
        mock_np.mean.return_value = 50.0
        mock_np.max.return_value = 60.0
        mock_np.median.return_value = MagicMock(astype=Mock(return_value=[128, 128, 128]))

        geom = self._instantiate_geometry(mock_cv2=mock_cv2, mock_np=mock_np)
        geom._temp_centers = [123.0]
        geom._detect_faces_yunet = Mock(return_value=[[100, 50, 40, 40]])

        mock_frame = MagicMock()
        mock_frame.detach().cpu().numpy().clip().astype.return_value = MagicMock()
        driving_video = [mock_frame]

        ambient, med_w, med_y, med_cx, shape = geom._analyze_driving_video(driving_video)
        self.assertEqual(geom._temp_centers, [])

    def test_rembg_session_caching(self):
        # Test module-level _get_rembg_session caching
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        session_func = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_get_rembg_session"
        )
        mod = ast.Module(body=[session_func], type_ignores=[])
        ast.fix_missing_locations(mod)

        mock_rembg = MagicMock()
        mock_session = MagicMock(name="mock_birefnet_session")
        mock_rembg.new_session.return_value = mock_session

        with patch.dict("sys.modules", {"rembg": mock_rembg}):
            ns = {
                "_REMBG_SESSION": None,
            }
            code = compile(mod, filename="shot_geometry_node.py", mode="exec")
            exec(code, ns)
            get_session = ns["_get_rembg_session"]

            # First call creates session
            s1 = get_session()
            self.assertIs(s1, mock_session)
            mock_rembg.new_session.assert_called_once_with("birefnet-portrait")

            # Second call reuses cached session without calling new_session again
            s2 = get_session()
            self.assertIs(s2, mock_session)
            self.assertEqual(mock_rembg.new_session.call_count, 1)

    def test_apply_geometry_closes_pil_and_rgba_handles(self):
        mock_cv2 = MagicMock()
        mock_cv2.COLOR_RGB2BGR = 1
        mock_cv2.COLOR_BGR2RGB = 2
        char_bgr = MagicMock()
        char_bgr.shape = (800, 600, 3)
        mock_cv2.cvtColor.return_value = char_bgr

        mock_np = MagicMock()
        mock_np.array.return_value = MagicMock(shape=(800, 600, 4))
        mock_np.full.return_value = MagicMock()
        mock_np.zeros.return_value = MagicMock()

        geom = self._instantiate_geometry(mock_cv2=mock_cv2, mock_np=mock_np)
        geom._analyze_driving_video = Mock(return_value=(
            MagicMock(), None, None, 288.0, (1024, 576)
        ))
        geom._detect_faces_yunet = Mock(return_value=[])

        char_tensor = MagicMock()
        char_tensor.__len__.return_value = 1
        char_tensor[0].detach().cpu().numpy().clip().astype.return_value = MagicMock()

        mock_pil_char = MagicMock()
        mock_res_rgba = MagicMock()

        with patch.dict("sys.modules", {"rembg": MagicMock(), "PIL": MagicMock(), "PIL.Image": MagicMock()}):
            import PIL.Image
            PIL.Image.fromarray.return_value = mock_pil_char

            import rembg
            rembg.remove.return_value = mock_res_rgba

            geom.apply_geometry(char_tensor, 576, 1024, half_body_floor_inpaint=False, ambient_autofill=True)

            mock_pil_char.close.assert_called_once()
            mock_res_rgba.close.assert_called_once()
