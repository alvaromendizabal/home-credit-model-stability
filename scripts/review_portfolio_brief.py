#!/usr/bin/env python3
"""Build a portable employer brief from verified public evidence, never private models.

The existing public-review engine owns metric replay. This module only projects
allowlisted fields into a static presentation and reuses its filesystem guards.
No model loading, network, credentials, JavaScript or third-party Python is used.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from review_public_evidence import (
    SOURCE_HASHES,
    build_review,
    reject_duplicate_keys,
    require,
    safe_output_path,
    serialize,
    table,
    verify_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
EXTRA_SOURCES = {
    "configs/validation_protocol.json": (
        "cea658652838e09ac57d27ec96d166781f5176036ef1e4bc2cd0fbc114c00c04"
    ),
    "reports/kaggle_submission/execution.json": (
        "72c401cdb21cc299b482c13dc7f5e92940292c13362a3e5e84935651ae33baa1"
    ),
}
TEMPLATE = "docs/portfolio_brief.html"
REPOSITORY = "https://github.com/alvaromendizabal/home-credit-model-stability"
EVIDENCE_COMMIT = "1de46633c44071689915015c89a3eb42bd77e301"
SOURCE_NAMES = {
    "reports/benchmark/acceptance.json": "Accepted benchmark population",
    "reports/benchmark/metrics.json": "Model-family results",
    "reports/feature_ablation/comparison.json": "Controlled feature removals",
    "reports/model_release/evaluation.json": "Frozen later-period evaluation",
    "reports/model_release/holdout_intent.json": "Pre-evaluation release intent",
    "configs/validation_protocol.json": "Data and temporal protocol",
    "reports/kaggle_submission/execution.json": "Original external execution receipt",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_link(path: str) -> str:
    require(path in SOURCE_NAMES, "Source link is not allowlisted")
    return f"{REPOSITORY}/blob/{EVIDENCE_COMMIT}/{path}"


def evidence(root: Path = ROOT) -> dict[str, Any]:
    """Recompute existing public aggregates and project only approved extra fields."""
    for relative in {**SOURCE_HASHES, **EXTRA_SOURCES}:
        path = root / relative
        require(not path.is_symlink(), f"Symlinked evidence: {relative}")
    review = build_review(root)
    extra: dict[str, Any] = {}
    for relative, expected in EXTRA_SOURCES.items():
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()), "Evidence escaped root")
        raw = path.read_bytes()
        require(digest(raw) == expected, f"SHA256 mismatch: {relative}")
        extra[relative] = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    protocol = extra["configs/validation_protocol.json"]
    receipt = extra["reports/kaggle_submission/execution.json"]
    require(receipt["kaggle_submitted"] is True, "External submission was not accepted")
    require(receipt["hidden_test_executed"] is True, "External execution is not evidenced")
    require(receipt["job_status"] == "Completed", "External execution is incomplete")
    require(protocol["inner_temporal_cv"]["n_splits"] == 5, "Unexpected validation design")
    return {
        "schema_version": 1,
        "evidence_commit": EVIDENCE_COMMIT,
        "population": {
            "labeled_applications": protocol["observed_temporal_structure"]["train_rows"],
            "development_training_rows": protocol["outer_holdout"]["development_rows"],
            "aligned_oof_rows": review["verification"]["development_rows"],
            "release_evaluation_rows": review["historical_release"]["rows"],
        },
        "historical_release": review["historical_release"],
        "benchmark": review["benchmark"],
        "feature_ablation": review["feature_ablation"],
        "verification": review["verification"],
        "original_external_evaluation": {
            "scope": "Separate all-label refit; recorded receipt, not rescored here",
            "date": receipt["kaggle_submission"]["submitted_date_utc"],
            "public": receipt["kaggle_score"]["public"],
            "private": receipt["kaggle_score"]["private"],
            "status": receipt["kaggle_submission"]["observed_status"],
        },
        "source_sha256": {**SOURCE_HASHES, **EXTRA_SOURCES},
        "metric_contract": review["metric_contract"],
        "limitations": review["limitations"],
        "disclosure": "Historical public aggregates only; no borrower rows or active recipes",
    }


def cover(doc: dict[str, Any]) -> str:
    """Render an accessible README cover using the actual frozen weekly Gini curve."""
    release = doc["historical_release"]
    points = " ".join(
        f"{735 + (row['week'] - 73) * 21:.1f},{269 - (row['gini'] - 0.6) * 460:.1f}"
        for row in release["weekly"]
    )
    score = release["stability_components"]["stability_score"]
    rows = doc["population"]["labeled_applications"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 430"
role="img" aria-labelledby="title desc">
<title id="title">Credit risk. Measured through time.</title>
<desc id="desc">Alvaro Mendizabal's research and inference system. {rows:,} labeled
applications. The chart shows actual weekly Gini on the frozen historical evaluation,
weeks 73 to 91. Stability {score:.6f} is a local score, not a Kaggle result.</desc>
<rect width="1200" height="430" rx="24" fill="#102a33"/>
<g font-family="Arial,Helvetica,sans-serif">
<text x="54" y="57" fill="#a3d8be" font-size="14" letter-spacing="3">
ALVARO MENDIZABAL / ML ENGINEERING</text>
<text x="54" y="134" fill="#f8f5eb" font-size="57" font-weight="700">Credit risk.</text>
<text x="54" y="202" fill="#f8f5eb" font-size="57" font-weight="700">Measured through time.</text>
<text x="57" y="248" fill="#c8d7db" font-size="19">
Relational features. Temporal evidence. Recoverable inference.</text>
<line x1="56" y1="292" x2="661" y2="292" stroke="#42616b"/>
<text x="56" y="342" fill="#f8f5eb" font-size="32" font-weight="700">{rows:,}</text>
<text x="56" y="370" fill="#c8d7db" font-size="14">LABELED APPLICATIONS</text>
<text x="365" y="342" fill="#f8f5eb" font-size="32" font-weight="700">5 temporal folds</text>
<text x="365" y="370" fill="#c8d7db" font-size="14">MATCHED MODEL COMPARISONS</text>
<rect x="707" y="81" width="440" height="298" rx="16" fill="#1b3b44"/>
<text x="735" y="119" fill="#d3e6dc" font-size="13" letter-spacing="1">
FROZEN LATER-PERIOD EVALUATION</text>
<text x="735" y="165" fill="#f8f5eb" font-size="31" font-weight="700">{score:.6f}</text>
<text x="942" y="162" fill="#c8d7db" font-size="13">local stability</text>
<path d="M735 186H1115 M735 225H1115 M735 269H1115" stroke="#35545d" fill="none"/>
<polyline points="{points}" stroke="#a3d8be" fill="none" stroke-width="3"/>
<text x="735" y="292" fill="#c8d7db" font-size="12">Week 73</text>
<text x="1054" y="292" fill="#c8d7db" font-size="12">Week 91</text>
<text x="735" y="331" fill="#c8d7db" font-size="13">
Actual weekly Gini · {release['rows']:,} applications</text>
<text x="735" y="354" fill="#a3d8be" font-size="13">
Public aggregates. Private training artifacts.</text>
</g></svg>"""


def render(doc: dict[str, Any], template: str) -> str:
    """Render no-script, local-file-safe HTML. Every inserted string is escaped."""
    release = doc["historical_release"]
    metrics = release["recorded_prediction_metrics"]
    components = release["stability_components"]
    aliases = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "linear_logistic": "Logistic regression",
    }
    benchmark = table(
        ["Model family", "Mean fold stability", "Worst fold", "Pooled ROC AUC"],
        [
            [
                aliases[row["model"]],
                f"{row['mean_fold_stability']:.6f}",
                f"{row['worst_fold_stability']:.6f}",
                f"{row['recorded_prediction_metrics']['auc']:.6f}",
            ]
            for row in doc["benchmark"]
        ],
    )
    ablation = table(
        ["Controlled removal", "Mean stability change"],
        [
            [
                row["experiment"].replace("without_", "").replace("_", " "),
                f"{row['delta_vs_control']:+.6f}",
            ]
            for row in doc["feature_ablation"]
            if row["experiment"] != "control"
        ],
    )
    weekly = table(
        ["Week", "Gini"],
        [[str(row["week"]), f"{row['gini']:.6f}"] for row in release["weekly"]],
    )
    sources = "".join(
        f'<li><a href="{html.escape(source_link(path))}">{html.escape(SOURCE_NAMES[path])}'
        f'</a><code>{html.escape(value)}</code></li>'
        for path, value in sorted(doc["source_sha256"].items())
    )
    values = {
        "COVER": cover(doc),
        "APPLICATIONS": f"{doc['population']['labeled_applications']:,}",
        "RELEASE_ROWS": f"{release['rows']:,}",
        "OOF_ROWS": f"{doc['population']['aligned_oof_rows']:,}",
        "STABILITY": f"{components['stability_score']:.6f}",
        "AUC": f"{metrics['auc']:.6f}",
        "AP": f"{metrics['pr_auc']:.6f}",
        "BRIER": f"{metrics['brier_score']:.6f}",
        "LOG_LOSS": f"{metrics['log_loss']:.6f}",
        "MEAN_GINI": f"{components['mean_gini']:.6f}",
        "RESIDUAL": f"{-0.5 * components['residual_std']:.6f}",
        "EXTERNAL_PUBLIC": f"{doc['original_external_evaluation']['public']:.5f}",
        "EXTERNAL_PRIVATE": f"{doc['original_external_evaluation']['private']:.5f}",
        "BENCHMARK": benchmark,
        "ABLATION": ablation,
        "WEEKLY": weekly,
        "SOURCES": sources,
        "REPOSITORY": REPOSITORY,
    }
    for key, value in values.items():
        require(f"@@{key}@@" in template, f"Missing template slot {key}")
        template = template.replace(f"@@{key}@@", value)
    require("@@" not in template, "Unknown template slot")
    return template


def build_bundle(root: Path = ROOT) -> dict[str, bytes]:
    payload = serialize(evidence(root))
    doc = json.loads(payload)
    template_path = root / TEMPLATE
    require(not template_path.is_symlink(), "Symlinked template")
    template = template_path.read_bytes()
    files = {
        "index.html": render(doc, template.decode()).encode(),
        "cover.svg": cover(doc).encode(),
        "evidence.json": payload,
    }
    files["manifest.json"] = serialize(
        {
            "schema": "home-credit-employer-brief-v1",
            "inputs": doc["source_sha256"],
            "generator_sha256": digest(Path(__file__).read_bytes()),
            "aggregate_engine_sha256": digest(
                (root / "scripts/review_public_evidence.py").read_bytes()
            ),
            "template_sha256": digest(template),
            "outputs": {
                name: {"bytes": len(raw), "sha256": digest(raw)}
                for name, raw in sorted(files.items())
            },
            "authentication_scope": (
                "Repository byte consistency; not independent training attestation"
            ),
            "privacy": "No private paths, borrower rows, weights or active recipes exported",
        }
    )
    return files


def publish(output: Path, *, check_only: bool = False) -> None:
    """Reuse existing exact-bundle guards; never overwrite a populated destination."""
    output = safe_output_path(output)
    expected = build_bundle()
    if check_only or output.exists():
        verify_bundle(output, expected)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.portfolio-lock")
    lock.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
            stage = Path(raw)
            for name, content in expected.items():
                with (stage / name).open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            verify_bundle(stage, expected)
            safe_output_path(output)
            require(not output.exists(), "Output appeared during staging")
            stage.rename(output)
    finally:
        lock.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/portfolio-brief")
    parser.add_argument("--check", action="store_true", help="Verify bytes without any writes")
    args = parser.parse_args()
    try:
        publish(args.output, check_only=args.check)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"PORTFOLIO_BRIEF_FAILED: {exc}\n")
    print("PORTFOLIO_BRIEF_VERIFIED: 7 pinned inputs; reused public metric engine; no training")
    print(f"REPORT={args.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
