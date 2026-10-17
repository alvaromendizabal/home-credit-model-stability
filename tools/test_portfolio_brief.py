"""Contract tests for the credential-free employer brief and its output boundary."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "portfolio_brief", SCRIPTS / "review_portfolio_brief.py"
)
assert SPEC is not None and SPEC.loader is not None
brief = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(brief)


class SafetyParser(HTMLParser):
    """The brief intentionally has no executable browser code or remote assets."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.anchors: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        fields = dict(attrs)
        if fields.get("id"):
            self.ids.append(str(fields["id"]))
        if tag == "a" and str(fields.get("href", "")).startswith("#"):
            self.anchors.append(str(fields["href"])[1:])
        if tag in {"script", "iframe", "object", "embed", "form", "base"}:
            self.errors.append(tag)
        if any(name.lower().startswith("on") for name in fields):
            self.errors.append("event handler")
        if tag in {"img", "link", "video", "audio", "source"}:
            self.errors.append("runtime asset")


class PortfolioBriefTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = brief.evidence()
        cls.bundle = brief.build_bundle()

    def test_reuses_seven_pinned_inputs(self) -> None:
        self.assertEqual(len(self.doc["source_sha256"]), 7)
        self.assertEqual(self.doc["verification"]["weekly_stability_recomputations"], 25)
        self.assertEqual(self.doc["verification"]["ablation_component_checks"], 20)

    def test_separate_populations(self) -> None:
        self.assertEqual(self.doc["population"]["release_evaluation_rows"], 203345)
        self.assertEqual(self.doc["population"]["aligned_oof_rows"], 727187)
        self.assertEqual(self.doc["population"]["labeled_applications"], 1526659)
        self.assertIn("Separate all-label refit", self.doc["original_external_evaluation"]["scope"])

    def test_preserves_metric_identity(self) -> None:
        c = self.doc["historical_release"]["stability_components"]
        expected = (
            c["mean_gini"] + 88 * min(0, c["temporal_slope"]) - 0.5 * c["residual_std"]
        )
        self.assertAlmostEqual(c["stability_score"], expected, places=10)

    def test_deterministic_bundle(self) -> None:
        self.assertEqual(self.bundle, brief.build_bundle())
        self.assertEqual(
            set(self.bundle), {"index.html", "cover.svg", "evidence.json", "manifest.json"}
        )

    def test_manifest_binds_each_payload(self) -> None:
        manifest = json.loads(self.bundle["manifest.json"])
        for name, record in manifest["outputs"].items():
            self.assertEqual(record["sha256"], brief.digest(self.bundle[name]))
            self.assertEqual(record["bytes"], len(self.bundle[name]))

    def test_html_has_no_runtime_network_or_active_code(self) -> None:
        parser = SafetyParser()
        parser.feed(self.bundle["index.html"].decode())
        self.assertEqual(parser.errors, [])
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertTrue(set(parser.anchors).issubset(parser.ids))

    def test_accessibility_and_static_fallback(self) -> None:
        text = self.bundle["index.html"].decode()
        for required in ('lang="en"', 'name="viewport"', "Skip to project evidence", "<summary>"):
            self.assertIn(required, text)
        self.assertIn("@media(max-width:760px)", text)
        self.assertIn("@media print", text)
        self.assertIn("focus-visible", text)

    def test_svg_has_accessible_actual_evidence(self) -> None:
        node = ET.fromstring(self.bundle["cover.svg"])
        self.assertEqual(node.tag, "{http://www.w3.org/2000/svg}svg")
        text = " ".join(node.itertext())
        self.assertIn("actual weekly Gini", text)
        self.assertIn("0.729674", text)
        self.assertIn("not a Kaggle result", text)

    def test_public_export_allowlist(self) -> None:
        text = b"\n".join(self.bundle.values()).decode()
        forbidden = ("s3://", "arn:aws", "/home/sagemaker-user", ".pickle", "denselight_model.pkl")
        for private in forbidden:
            self.assertNotIn(private, text)
        self.assertNotIn("intent", self.doc["historical_release"])
        self.assertNotIn("environment", self.doc["original_external_evaluation"])
        self.assertNotIn("weights", self.doc)

    def test_links_bind_evidence_revision(self) -> None:
        for path in brief.SOURCE_NAMES:
            self.assertIn(brief.EVIDENCE_COMMIT, brief.source_link(path))
        with self.assertRaises(ValueError):
            brief.source_link("../../secret")

    def test_unknown_template_slots_fail(self) -> None:
        template = (brief.ROOT / brief.TEMPLATE).read_text()
        with self.assertRaises(ValueError):
            brief.render(self.doc, template + "@@UNKNOWN@@")
        with self.assertRaises(ValueError):
            brief.render(self.doc, template.replace("@@AUC@@", "hardcoded"))

    def test_table_content_is_escaped(self) -> None:
        doc = copy.deepcopy(self.doc)
        doc["feature_ablation"][1]["experiment"] = "<img src=x onerror=alert(1)>"
        page = brief.render(doc, (brief.ROOT / brief.TEMPLATE).read_text())
        self.assertNotIn("<img src=x", page)
        self.assertIn("&lt;img", page)

    def test_atomic_publish_and_idempotent_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "brief"
            brief.publish(output)
            mtimes = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
            brief.publish(output)
            brief.publish(output, check_only=True)
            self.assertEqual(mtimes, {p.name: p.stat().st_mtime_ns for p in output.iterdir()})
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {"brief"})

    def test_existing_destination_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "brief"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserve")
            with self.assertRaises(ValueError):
                brief.publish(output)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_check_missing_output_never_creates_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing"
            with self.assertRaises(ValueError):
                brief.publish(output, check_only=True)
            self.assertFalse(output.exists())

    def test_changed_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "brief"
            brief.publish(output)
            page = output / "index.html"
            page.write_text("tampered")
            with self.assertRaises(ValueError):
                brief.publish(output)
            self.assertEqual(page.read_text(), "tampered")

    def test_extra_output_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "brief"
            brief.publish(output)
            (output / "unexpected.txt").write_text("extra")
            with self.assertRaises(ValueError):
                brief.publish(output, check_only=True)

    def test_symlink_and_parent_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / "alias"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                brief.publish(link / "brief")
            with self.assertRaises(ValueError):
                brief.publish(root / ".." / "brief")

    def test_failed_write_does_not_publish_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "brief"
            with patch.object(brief.os, "fsync", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    brief.publish(output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_mutated_source_rejected_before_presentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in {**brief.SOURCE_HASHES, **brief.EXTRA_SOURCES}:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(brief.ROOT / path, target)
            source = root / "configs/validation_protocol.json"
            source.write_bytes(source.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                brief.evidence(root)

    def test_closeout_receipt_does_not_claim_promotion(self) -> None:
        doc = json.loads((brief.ROOT / "reports/portfolio/closeout.json").read_bytes())
        run = doc["latest_inspected_execution"]
        self.assertEqual(run["status"], "SUCCESS")
        self.assertEqual(run["stages_completed"], run["stages_total"])
        self.assertFalse(run["new_champion_promoted"])
        self.assertFalse(run["new_external_score"])
        self.assertFalse(doc["source_archive_published"])
        self.assertEqual(run["submission_readiness"], "NO_BETTER_CHALLENGER")
        self.assertEqual(len(doc["source_archive_sha256"]), 64)
        self.assertIn("exploratory", run["evaluation_scope"])

    def test_committed_bundle_matches_generator(self) -> None:
        brief.verify_bundle(brief.ROOT / "reports/portfolio/brief", self.bundle)


if __name__ == "__main__":
    unittest.main()
