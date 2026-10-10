"""Check the reviewed static demo boundary without executing its JavaScript.

This focused packaging check complements the behavioral browser-demo tests;
it is not a security sandbox or a validation of the historical research model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

REQUIRED = {"index.html", "styles.css", "model.mjs", "app.mjs"}
REPOSITORY_TESTS = {"model.test.mjs", "app.test.mjs"}
SECRET = re.compile(
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|KGAT_[A-Za-z0-9_-]{25,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"[?&](?:X-Amz-Signature|X-Amz-Credential|X-Goog-Signature)="
)
PRIVATE = re.compile(r"s3://|arn:aws(?:-[a-z]+)?:|/(?:home|workspace|mnt)/", re.IGNORECASE)
NETWORK = re.compile(
    r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|importScripts)\s*\("
    r"|\bsendBeacon\s*\("
)
IMPORT = re.compile(r"\b(?:import|export)\s+(?:[^;\"']*?\s+from\s+)?[\"']([^\"']+)[\"']")
DYNAMIC_IMPORT = re.compile(r"\bimport\s*\(\s*([\"'])([^\"']+)\1\s*\)")
CSS_URL = re.compile(r"url\(\s*[\"']?([^\s)\"']+)[\"']?\s*\)", re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inline_favicon(reference: str) -> None:
    """Permit one small self-contained shape-only SVG favicon, never active data URLs."""
    prefix = "data:image/svg+xml,"
    require(reference.startswith(prefix), "Unreviewed inline favicon format")
    text = unquote(reference[len(prefix) :])
    require(len(text.encode("utf-8")) <= 4096, "Oversized inline favicon")
    require("<!" not in text, "Unreviewed inline favicon declaration")
    allowed = {"svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
    for element in ET.fromstring(text).iter():
        require(element.tag.rsplit("}", 1)[-1] in allowed, "Active inline favicon content")
        for key, value in element.attrib.items():
            name = key.rsplit("}", 1)[-1].lower()
            require(
                not name.startswith("on") and name not in {"href", "style"},
                "Active inline favicon attribute",
            )
            require("url(" not in value.lower(), "Inline favicon runtime reference")


def local_asset(root: Path, source: Path, reference: str) -> None:
    target = urlsplit(reference)
    require(not target.scheme and not target.netloc, "Remote runtime asset: " + reference)
    if not target.path and target.fragment:
        return
    decoded = unquote(target.path)
    require(bool(decoded) and not decoded.startswith(("/", "\\")), "Invalid asset path")
    path = (source.parent / decoded).resolve()
    require(path.is_relative_to(root.resolve()), "Asset escapes demo: " + reference)
    require(
        path.relative_to(root.resolve()).as_posix() in REQUIRED,
        "Unpublished runtime asset: " + reference,
    )
    require(path.is_file(), "Missing runtime asset: " + reference)


class RuntimeAssets(HTMLParser):
    def __init__(self, root: Path, source: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.root, self.source = root, source

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        require(tag not in {"iframe", "object", "embed", "base"}, "External document container")
        if tag == "meta":
            require(attributes.get("http-equiv", "").lower() != "refresh", "Automatic redirect")
        require(not any(key.lower().startswith("on") for key in attributes), "Inline event handler")
        if (
            tag == "link"
            and attributes.get("rel", "").lower() == "icon"
            and attributes.get("href", "").startswith("data:")
        ):
            inline_favicon(attributes["href"])
            return
        # A canonical URL is identity metadata, not a downloaded runtime asset.
        if tag == "link" and attributes.get("rel", "").lower() == "canonical":
            reference = urlsplit(attributes.get("href") or "")
            require(reference.scheme == "https" and bool(reference.netloc), "Invalid canonical URL")
            return
        require(tag not in {"foreignobject"}, "Active SVG content")
        if tag in {"script", "img", "image", "use", "source", "audio", "video", "link"}:
            for key in ("src", "href", "xlink:href", "poster"):
                if attributes.get(key):
                    local_asset(self.root, self.source, attributes[key])
            require("srcset" not in attributes, "Unreviewed responsive runtime asset")


def verify(root: Path) -> dict[str, Any]:
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Missing or symlinked demo directory")
    for name in REQUIRED:
        require((root / name).is_file(), "Missing demo entry point: " + name)
    files = []
    excluded_tests = []
    text_by_name = {}
    total = 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Symlink in public demo")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in REPOSITORY_TESTS:
            excluded_tests.append(relative)
            continue
        require(relative in REQUIRED, "Unreviewed public demo file: " + relative)
        raw = path.read_bytes()
        total += len(raw)
        require(len(raw) <= 1_000_000 and total <= 2_000_000, "Oversized static demo")
        text = raw.decode("utf-8")
        require(not SECRET.search(text), "Credential-like material in " + relative)
        require(not PRIVATE.search(text), "Private operational locator in " + relative)
        require(not NETWORK.search(text), "Runtime network in " + relative)
        if path.suffix == ".html":
            RuntimeAssets(root, path).feed(text)
        if path.suffix == ".mjs":
            for reference in IMPORT.findall(text):
                local_asset(root, path, reference)
            for _, reference in DYNAMIC_IMPORT.findall(text):
                local_asset(root, path, reference)
            require(
                not re.search(r"\bimport\s*\(", DYNAMIC_IMPORT.sub("", text)),
                "Nonliteral dynamic import in " + relative,
            )
        if path.suffix in {".css", ".html"}:
            require("@import" not in text.lower(), "Unreviewed CSS import")
            for reference in CSS_URL.findall(text):
                local_asset(root, path, reference)
        text_by_name[relative] = text
        files.append(
            {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    require(len(files) == len(REQUIRED), "Unexpected static demo file count")
    require("synthetic" in text_by_name["index.html"].lower(), "Missing visible synthetic scope")
    require(
        "SYNTHETIC_ONLY" in text_by_name["model.mjs"],
        "Missing machine-readable synthetic evidence marker",
    )
    return {
        "status": "PASS",
        "scope": "reviewed_static_demo_only",
        "files": files,
        "bytes": total,
        "excluded_repository_tests": excluded_tests,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "demo")
    print(json.dumps(verify(parser.parse_args().root), indent=2))
