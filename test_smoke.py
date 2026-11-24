from __future__ import annotations

import subprocess
import sys
import unittest
import json
from pathlib import Path

from autoscan import list_images


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / ".test_artifacts"


class AutoScanSmokeTest(unittest.TestCase):
    def test_pipeline_creates_report_and_annotated_image(self) -> None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        sample = ARTIFACTS / "smoke_sample.png"
        results = ARTIFACTS / "smoke_results"

        subprocess.run(
            [sys.executable, str(ROOT / "create_synthetic_example.py"), "--output", str(sample)],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "autoscan.py"),
                "--input",
                str(sample),
                "--output",
                str(results),
                "--save-debug",
            ],
            check=True,
        )

        self.assertTrue((results / "report.json").exists())
        self.assertTrue((results / "smoke_sample_annotated.jpg").exists())
        self.assertTrue((results / "smoke_sample_mask.png").exists())
        self.assertTrue((results / "smoke_sample_debug.jpg").exists())
        report = json.loads((results / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["prototype_version"], "0.1.0")
        self.assertEqual(len(report["images"]), 1)
        self.assertGreaterEqual(report["images"][0]["candidate_count"], 1)
        self.assertTrue(report["images"][0]["diagnostic_only"])

    def test_unsupported_extension_is_rejected(self) -> None:
        unsupported = ROOT / "README.md"
        with self.assertRaisesRegex(ValueError, "Неподдерживаемое расширение"):
            list_images(unsupported)


if __name__ == "__main__":
    unittest.main()

