"""Negative fixtures for the static demo packaging boundary; no project dependencies."""

import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

from check_public_demo import verify


class DemoBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.write("index.html", '<h1>Synthetic demo</h1><script src="app.mjs"></script>')
        self.write("app.mjs", "import {run} from './model.mjs'; run();")
        self.write("model.mjs", "export const run=()=>({evidence_type:'SYNTHETIC_ONLY'});")
        self.write("styles.css", "body{color:#123}")
        self.write("model.mjs", "globalThis.DEMO={evidence_type:'SYNTHETIC_ONLY'};")

    def write(self, name: str, text: str) -> None:
        (self.root / name).write_text(text)

    def test_self_contained_demo_has_content_pins(self) -> None:
        report = verify(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["files"]), 4)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in report["files"]))

    def test_remote_script_is_rejected(self) -> None:
        self.write(
            "index.html", '<h1>Synthetic</h1><script src="https://example.org/sdk.js"></script>'
        )
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_canonical_identity_metadata_is_not_a_runtime_dependency(self) -> None:
        self.write(
            "index.html",
            '<h1>Synthetic demo</h1><link rel="canonical" href="https://example.org/demo/">',
        )
        self.assertEqual(verify(self.root)["status"], "PASS")

    def test_canonical_label_cannot_hide_a_remote_stylesheet(self) -> None:
        self.write(
            "index.html",
            '<h1>Synthetic demo</h1><link rel="canonical stylesheet" href="https://example.org/a.css">',
        )
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_escaped_import_is_rejected(self) -> None:
        self.write("app.mjs", "import '../private.js';")
        with self.assertRaisesRegex(ValueError, "escapes"):
            verify(self.root)

    def test_missing_import_is_rejected(self) -> None:
        self.write("app.mjs", "import './missing.js';")
        with self.assertRaisesRegex(ValueError, "Unpublished runtime|Missing runtime"):
            verify(self.root)

    def test_remote_css_is_rejected(self) -> None:
        self.write("styles.css", "body{background:url(https://example.org/track.png)}")
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_network_calls_are_rejected(self) -> None:
        self.write("app.mjs", "fetch('https://example.org/input');")
        with self.assertRaisesRegex(ValueError, "Runtime network"):
            verify(self.root)

    def test_remote_reexport_is_rejected(self) -> None:
        self.write("app.mjs", "export {run} from 'https://example.org/code.js';")
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_svg_cannot_pull_a_remote_image(self) -> None:
        self.write(
            "index.html",
            '<h1>Synthetic</h1><svg><image href="https://example.org/a.png"/></svg>',
        )
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_credential_material_is_rejected(self) -> None:
        self.write("app.mjs", "const key='" + "AKIA" + "A" * 16 + "';")
        with self.assertRaisesRegex(ValueError, "Credential-like"):
            verify(self.root)

    def test_kaggle_credential_material_is_rejected(self) -> None:
        self.write("app.mjs", "const key='" + "KGAT_" + "a" * 25 + "';")
        with self.assertRaisesRegex(ValueError, "Credential-like"):
            verify(self.root)

    def test_signed_download_url_is_rejected_even_as_text(self) -> None:
        self.write("app.mjs", "const location='https://example.org/file?X-Amz-Signature=abc';")
        with self.assertRaisesRegex(ValueError, "Credential-like"):
            verify(self.root)

    def test_svg_active_content_is_rejected(self) -> None:
        for active in ["<foreignObject/>", '<path onload="run()"/>']:
            with self.subTest(active=active):
                self.write("index.html", "<h1>Synthetic</h1><svg>" + active + "</svg>")
                with self.assertRaisesRegex(ValueError, "Active SVG|Inline event"):
                    verify(self.root)

    def test_private_blob_extension_is_rejected(self) -> None:
        self.write("weights.npz", "private")
        with self.assertRaisesRegex(ValueError, "Unreviewed public"):
            verify(self.root)

    def test_operational_location_is_rejected(self) -> None:
        self.write("app.mjs", "const path='s3://example-private/inputs';")
        with self.assertRaisesRegex(ValueError, "Private operational"):
            verify(self.root)

    def test_missing_synthetic_marker_is_rejected(self) -> None:
        self.write("model.mjs", "globalThis.DEMO={};")
        with self.assertRaisesRegex(ValueError, "machine-readable synthetic"):
            verify(self.root)

    def test_symlink_cannot_import_outside_artifact(self) -> None:
        (self.root / "copy.js").symlink_to(self.root / "app.mjs")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            verify(self.root)

    def test_unreviewed_extra_code_is_rejected(self) -> None:
        self.write("model.js", "const weights=[1,2,3];")
        with self.assertRaisesRegex(ValueError, "Unreviewed public"):
            verify(self.root)

    def test_missing_required_fixture_is_rejected(self) -> None:
        (self.root / "model.mjs").unlink()
        with self.assertRaisesRegex(ValueError, "Missing demo entry point"):
            verify(self.root)

    def test_remote_fixture_import_is_rejected(self) -> None:
        self.write("model.mjs", "import 'https://example.org/games.js'; // SYNTHETIC_ONLY")
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_external_links_are_allowed(self) -> None:
        self.write(
            "index.html",
            '<h1>Synthetic demo</h1><a href="https://github.com/example/repo">Source</a>',
        )
        self.assertEqual(verify(self.root)["status"], "PASS")

    def test_missing_visible_scope_is_rejected(self) -> None:
        self.write("index.html", "<h1>Tournament probabilities</h1>")
        with self.assertRaisesRegex(ValueError, "visible synthetic"):
            verify(self.root)

    def test_oversized_fixture_is_rejected(self) -> None:
        self.write("model.mjs", "// SYNTHETIC_ONLY\n" + " " * 1_000_000)
        with self.assertRaisesRegex(ValueError, "Oversized"):
            verify(self.root)

    def test_repository_tests_are_explicitly_excluded(self) -> None:
        for name in ("model.test.mjs", "app.test.mjs"):
            self.write(name, "import assert from 'node:assert/strict';")
        report = verify(self.root)
        self.assertEqual(len(report["files"]), 4)
        self.assertEqual(report["excluded_repository_tests"], ["app.test.mjs", "model.test.mjs"])

    def test_browser_cannot_import_an_excluded_test(self) -> None:
        self.write("model.test.mjs", "export const privateFixture=[];")
        self.write("app.mjs", "import './model.test.mjs';")
        with self.assertRaisesRegex(ValueError, "Unpublished runtime"):
            verify(self.root)

    def test_multiline_remote_import_is_rejected(self) -> None:
        self.write("app.mjs", "import {\n  remoteModel\n} from 'https://example.org/private.mjs';")
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_multiline_local_import_is_accepted(self) -> None:
        self.write("app.mjs", "import {\n  run\n} from './model.mjs';")
        self.assertEqual(verify(self.root)["status"], "PASS")

    def test_literal_dynamic_local_module_import_is_accepted(self) -> None:
        self.write("app.mjs", "const loadModel=()=>import('./model.mjs');")
        self.assertEqual(verify(self.root)["status"], "PASS")

    def test_dynamic_remote_module_import_is_rejected(self) -> None:
        self.write("app.mjs", "const loadModel=()=>import('https://example.org/model.mjs');")
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_nonliteral_dynamic_module_import_is_rejected(self) -> None:
        self.write("app.mjs", "const loadModel=path=>import(path);")
        with self.assertRaisesRegex(ValueError, "Nonliteral dynamic import"):
            verify(self.root)

    def test_small_shape_only_inline_favicon_is_accepted(self) -> None:
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L2 2"/></svg>'
        self.write(
            "index.html",
            '<h1>Synthetic</h1><link rel="icon" href="data:image/svg+xml,' + quote(svg) + '">',
        )
        self.assertEqual(verify(self.root)["status"], "PASS")

    def test_inline_favicon_cannot_contain_active_or_remote_content(self) -> None:
        for content in (
            "<script>run()</script>",
            '<path onload="run()"/>',
            '<image href="https://example.org/a.png"/>',
            '<path fill="url(https://example.org/style)"/>',
        ):
            with self.subTest(content=content):
                svg = '<svg xmlns="http://www.w3.org/2000/svg">' + content + "</svg>"
                self.write(
                    "index.html",
                    '<h1>Synthetic</h1><link rel="icon" href="data:image/svg+xml,'
                    + quote(svg)
                    + '">',
                )
                with self.assertRaisesRegex(ValueError, "inline favicon|favicon runtime"):
                    verify(self.root)

    def test_inline_favicon_cannot_hide_external_html(self) -> None:
        self.write(
            "index.html",
            '<h1>Synthetic</h1><link rel="icon" '
            'href="data:text/html,%3Cscript%3Ealert(1)%3C/script%3E">',
        )
        with self.assertRaisesRegex(ValueError, "favicon format"):
            verify(self.root)


if __name__ == "__main__":
    unittest.main()
