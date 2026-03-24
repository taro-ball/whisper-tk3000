import shutil
import unittest
import uuid
from pathlib import Path

from whisper_tk3000.settings import SettingsStore


def _make_temp_dir(prefix: str) -> Path:
    path = Path("tests") / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class SettingsStoreTests(unittest.TestCase):
    def test_install_id_persists_when_telemetry_stays_enabled(self) -> None:
        temp_dir = _make_temp_dir("tmp_settings")
        try:
            store = SettingsStore(temp_dir / "settings.json")

            first = store.load()
            second = store.load()

            self.assertTrue(first.telemetry_enabled)
            self.assertIsNotNone(first.install_id)
            self.assertEqual(first.install_id, second.install_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_install_id_regenerates_after_telemetry_is_reenabled(self) -> None:
        temp_dir = _make_temp_dir("tmp_settings")
        try:
            store = SettingsStore(temp_dir / "settings.json")

            first = store.load()
            disabled = store.set_telemetry_enabled(False)
            reenabled = store.set_telemetry_enabled(True)
            reloaded = store.load()

            self.assertTrue(first.telemetry_enabled)
            self.assertIsNotNone(first.install_id)
            self.assertFalse(disabled.telemetry_enabled)
            self.assertIsNone(disabled.install_id)
            self.assertTrue(reenabled.telemetry_enabled)
            self.assertIsNotNone(reenabled.install_id)
            self.assertNotEqual(first.install_id, reenabled.install_id)
            self.assertEqual(reenabled.install_id, reloaded.install_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
