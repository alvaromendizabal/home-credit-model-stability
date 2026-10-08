#!/usr/bin/env python3
"""Reproduce a public aggregate evidence review using Python's standard library only.

No training, borrower data, model deserialization, credentials or network are used.
The pinned inputs are historical public reports. Prediction-level discrimination
and calibration metrics are displayed as recorded, not recomputed from aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    "reports/benchmark/acceptance.json": (
        "d76a0588901f07f80b94254348c7598545f35adb53ca1d5faae7d28314f99798"
    ),
    "reports/benchmark/metrics.json": (
        "e02e6d30c6559e029402e6b3d6180f759d7d2d5baeb60f1c09dbe377f237f9e3"
    ),
    "reports/feature_ablation/comparison.json": (
        "d035dc6993bf83dc9f386c23c7f6612275a13c47053758325e7ad1efcdd853ef"
    ),
    "reports/model_release/evaluation.json": (
        "42fbba1bfa0ee8b15991e85aa5a7b8171d5c69a3d8b29e53c19e24bf1bbf8152"
    ),
    "reports/model_release/holdout_intent.json": (
        "2c9b7101cae4602862009f0652fad9efc350c21f14e3bf61a68cf4060e90c5be"
    ),
}
MODELS = ("lightgbm", "xgboost", "catboost", "linear_logistic")
EXPERIMENTS = (
    "control",
    "without_depth2",
    "without_previous_applications",
    "without_credit_bureau_a",
)
COMPONENTS = ("stability_score", "mean_gini", "temporal_slope", "residual_std")
PREDICTION_METRICS = ("auc", "pr_auc", "brier_score", "log_loss")


def require(condition: bool, message: str) -> None:
    """Reject ambiguous or inconsistent evidence before publishing a review."""
    if not condition:
        raise ValueError(message)


def number(value: Any, name: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name}: numeric")
    result = float(value)
    require(math.isfinite(result), f"{name}: finite value required")
    return result


def integer(value: Any, name: str) -> int:
    require(type(value) is int, f"{name}: integer required")
    return int(value)


def close(actual: Any, expected: Any, name: str) -> None:
    require(
        math.isclose(number(actual, name), number(expected, name), rel_tol=1e-10, abs_tol=1e-10),
        f"{name}: aggregate mismatch",
    )


def stability_components(weeks: list[int], ginis: list[float]) -> dict[str, float]:
    """Fit weekly Gini on week index; apply the frozen 88 / 0.5 penalties.

    Residual dispersion is population standard deviation (ddof=0). The penalty
    makes pooled stability different from the arithmetic mean of fold stability.
    """
    require(len(weeks) == len(ginis) >= 2, "at least two aligned weekly aggregates required")
    require(weeks == sorted(set(weeks)), "weeks must be unique and increasing")
    require(all(type(week) is int for week in weeks), "weeks must be integers")
    values = [number(value, "weekly Gini") for value in ginis]
    require(all(-1 <= value <= 1 for value in values), "weekly Gini outside [-1, 1]")
    mean_week, mean_gini = statistics.fmean(weeks), statistics.fmean(values)
    slope = math.fsum(
        (week - mean_week) * (gini - mean_gini) for week, gini in zip(weeks, values, strict=True)
    ) / math.fsum((week - mean_week) ** 2 for week in weeks)
    residuals = [
        gini - (mean_gini + slope * (week - mean_week))
        for week, gini in zip(weeks, values, strict=True)
    ]
    residual_std = statistics.pstdev(residuals)
    return {
        "stability_score": mean_gini + 88.0 * min(slope, 0.0) - 0.5 * residual_std,
        "mean_gini": mean_gini,
        "temporal_slope": slope,
        "residual_std": residual_std,
    }


def rows_by_key(rows: Any, key: str, expected: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    require(isinstance(rows, list), f"{key}: expected records")
    require(all(isinstance(row, dict) for row in rows), f"{key}: expected object records")
    require(len(rows) == len(expected), f"{key}: unexpected record count")
    require({row[key] for row in rows} == set(expected), f"{key}: missing or duplicate records")
    return {row[key]: row for row in rows}


def check_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics = {key: number(row[key], key) for key in PREDICTION_METRICS}
    require(all(0 <= metrics[key] <= 1 for key in PREDICTION_METRICS[:3]), "metric range")
    require(metrics["log_loss"] >= 0, "negative log loss")
    return metrics


def check_components(actual: dict[str, float], expected: dict[str, Any], name: str) -> None:
    for key in COMPONENTS:
        close(actual[key], expected[key], f"{name}/{key}")


def component_identity(row: dict[str, Any], name: str) -> None:
    values = {key: number(row[key], key) for key in COMPONENTS}
    require(-1 <= values["mean_gini"] <= 1, f"{name}: invalid mean Gini")
    require(values["residual_std"] >= 0, f"{name}: negative residual standard deviation")
    score = (
        values["mean_gini"] + 88 * min(values["temporal_slope"], 0) - 0.5 * values["residual_std"]
    )
    close(score, values["stability_score"], name)


def population(weekly: list[dict[str, Any]], key: str) -> list[tuple[int, int, int]]:
    result = []
    for row in weekly:
        week, count, positives = (integer(row[k], k) for k in (key, "rows", "positives"))
        require(0 < positives < count, f"week {week}: empty or single-class population")
        if "positive_rate" in row:
            close(row["positive_rate"], positives / count, f"week {week}/positive rate")
        result.append((week, count, positives))
    return result


def review_documents(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Validate aggregate relationships independently of the file-hash boundary."""
    acceptance = documents["reports/benchmark/acceptance.json"]
    metrics = documents["reports/benchmark/metrics.json"]
    ablation = documents["reports/feature_ablation/comparison.json"]
    release = documents["reports/model_release/evaluation.json"]
    intent = documents["reports/model_release/holdout_intent.json"]
    for name, doc in documents.items():
        if not name.endswith("holdout_intent.json"):
            require(doc["schema_version"] == 1, f"{name}: unsupported schema")
    require(acceptance["status"] == "accepted", "benchmark evidence is not accepted")
    require(acceptance["summary_sha256"] == metrics["summary_sha256"], "benchmark lineage mismatch")
    require(acceptance["holdout_predictions_present"] is False, "benchmark includes holdout data")
    require(acceptance["model_folds"] == 20, "benchmark fold count mismatch")
    require(acceptance["oof_rows_per_model"] == 727187, "benchmark population changed")
    require((acceptance["oof_week_min"], acceptance["oof_week_max"]) == (33, 72), "week range")
    models = rows_by_key(metrics["models"], "model", MODELS)
    archived_models = rows_by_key(acceptance["models"], "model", MODELS)
    require(len(metrics["folds"]) == 20, "rescored fold count mismatch")
    require(len(acceptance["weekly_metrics"]) == 160, "weekly aggregate count mismatch")
    benchmark_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    common_population: list[tuple[int, int, int]] | None = None
    for name in MODELS:
        require(models[name]["oof_sha256"] == archived_models[name]["oof_sha256"], "OOF lineage")
        weekly = sorted(
            (row for row in acceptance["weekly_metrics"] if row["model"] == name),
            key=lambda row: row["week_num"],
        )
        require([row["week_num"] for row in weekly] == list(range(33, 73)), "weekly coverage")
        counts = population(weekly, "week_num")
        require(sum(row[1] for row in counts) == 727187, "weekly row total mismatch")
        require(common_population is None or common_population == counts, "cross-model population")
        common_population = counts
        pooled = stability_components([row[0] for row in counts], [row["gini"] for row in weekly])
        check_components(pooled, models[name], f"{name}/pooled")
        recorded_folds = [row for row in metrics["folds"] if row["model"] == name]
        require(len(recorded_folds) == 5, f"{name}: expected five folds")
        require({row["fold"] for row in recorded_folds} == set(range(1, 6)), "fold coverage")
        scores = []
        for fold in range(1, 6):
            subset = weekly[(fold - 1) * 8 : fold * 8]
            calculated = stability_components(
                [row["week_num"] for row in subset], [row["gini"] for row in subset]
            )
            expected = next(row for row in recorded_folds if row["fold"] == fold)
            check_components(calculated, expected, f"{name}/fold {fold}")
            check_metrics(expected)
            fold_rows.append({"model": name, "fold": fold, **calculated})
            scores.append(calculated["stability_score"])
        benchmark_rows.append(
            {
                "model": name,
                "mean_fold_stability": statistics.fmean(scores),
                "worst_fold_stability": min(scores),
                "pooled_stability": pooled["stability_score"],
                "recorded_prediction_metrics": check_metrics(models[name]),
            }
        )
        weekly_rows.extend(
            {"model": name, "week": row["week_num"], "gini": row["gini"]} for row in weekly
        )
    require(ablation["smoke"] is False, "ablation must be an executed full study")
    require(ablation["outer_holdout_touched"] is False, "ablation includes holdout")
    experiments = rows_by_key(ablation["rows"], "experiment", EXPERIMENTS)
    require(len(ablation["folds"]) == 20, "ablation fold count mismatch")
    ablation_rows = []
    for name in EXPERIMENTS:
        folds = [row for row in ablation["folds"] if row["experiment"] == name]
        require(
            len(folds) == 5 and {row["fold"] for row in folds} == set(range(1, 6)),
            "ablation fold coverage",
        )
        for row in folds:
            component_identity(row, f"ablation/{name}/fold {row['fold']}")
            check_metrics(row)
        scores = [row["stability_score"] for row in folds]
        mean_score = statistics.fmean(scores)
        close(mean_score, experiments[name]["mean_fold_stability"], f"{name}/fold mean")
        close(min(scores), experiments[name]["worst_fold_stability"], f"{name}/worst fold")
        delta = mean_score - experiments["control"]["mean_fold_stability"]
        close(delta, experiments[name]["delta_vs_control"], f"{name}/ablation delta")
        ablation_rows.append(
            {
                "experiment": name,
                "mean_fold_stability": mean_score,
                "worst_fold_stability": min(scores),
                "delta_vs_control": delta,
            }
        )
    for fold in range(1, 6):
        control = next(
            row
            for row in ablation["folds"]
            if row["experiment"] == "control" and row["fold"] == fold
        )
        baseline = next(
            row for row in metrics["folds"] if row["model"] == "lightgbm" and row["fold"] == fold
        )
        for key in (*COMPONENTS, *PREDICTION_METRICS):
            close(control[key], baseline[key], f"ablation matched control/{fold}/{key}")
    require(release["intent"] == intent, "frozen release intent mismatch")
    require(
        release["fit_weeks"] == [0, 72] and release["evaluation_weeks"] == [73, 91],
        "release temporal boundary mismatch",
    )
    require(release["evaluated_model_phase"] == "development", "release model phase mismatch")
    require(release["all_label_model_evaluated"] is False, "all-label fit is not a holdout test")
    weekly = release["weekly"]
    require([row["week"] for row in weekly] == list(range(73, 92)), "release weekly coverage")
    release_population = population(weekly, "week")
    require(
        sum(row[1] for row in release_population) == release["rows"] == 203345,
        "release row total mismatch",
    )
    close(
        sum(row[2] for row in release_population) / release["rows"],
        release["positive_rate"],
        "release positive rate",
    )
    release_components = stability_components(
        [row["week"] for row in weekly], [row["gini"] for row in weekly]
    )
    check_components(release_components, release["metrics"], "historical release")
    return {
        "schema_version": 1,
        "scope": "Offline reproduction of published aggregate evidence; no model training.",
        "verification": {
            "input_files": len(SOURCE_HASHES),
            "benchmark_models": 4,
            "benchmark_folds": 20,
            "benchmark_weekly_records": 160,
            "weekly_stability_recomputations": 25,
            "ablation_component_checks": 20,
            "development_rows": 727187,
            "release_rows": 203345,
            "release_weeks": 19,
        },
        "source_sha256": SOURCE_HASHES,
        "metric_contract": {
            "formula": "mean(weekly Gini) + 88 * min(0, slope) - 0.5 * residual_std",
            "residual_ddof": 0,
            "numeric_tolerance": 1e-10,
            "recomputed": "Weekly stability components, fold means, worst folds, ablation deltas.",
            "recorded_only": "ROC AUC, average precision, Brier and log loss need row predictions.",
            "prediction_policy": metrics["ranking_policy"] + "; " + metrics["probability_policy"],
        },
        "benchmark": benchmark_rows,
        "benchmark_folds": fold_rows,
        "benchmark_weekly": weekly_rows,
        "feature_ablation": ablation_rows,
        "historical_release": {
            "evaluated_utc": release["evaluated_utc"],
            "fit_weeks": [0, 72],
            "evaluation_weeks": [73, 91],
            "rows": release["rows"],
            "stability_components": release_components,
            "recorded_prediction_metrics": check_metrics(release["metrics"]),
            "weekly": [{"week": row["week"], "gini": row["gini"]} for row in weekly],
        },
        "limitations": [
            "Development folds informed early stopping and model selection; estimates are not "
            "unbiased nested validation.",
            "The September 8, 2026 frozen release observed weeks 73-91. Those weeks are no "
            "longer an untouched holdout and must not be reused for model selection.",
            "The historical release and development scores have different populations and "
            "must not be interpreted as a direct improvement comparison.",
            "Linear logistic results use the published raw-rank correction; archived clipped "
            "aggregate summaries differ. This review uses the corrected metrics.",
            "Hashes establish consistency with this repository snapshot, not independent "
            "verification of private training data, artifacts, or execution.",
            "This review does not reproduce model fitting or prediction-level metrics.",
        ],
    }


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def build_review(root: Path = ROOT) -> dict[str, Any]:
    documents = {}
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()), f"evidence path escaped: {relative}")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, f"SHA256 mismatch: {relative}")
        doc = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
        require(isinstance(doc, dict), f"{relative}: JSON object required")
        documents[relative] = doc
    return review_documents(documents)


def serialize(review: dict[str, Any]) -> bytes:
    """Round only presentation floats for deterministic cross-platform output."""

    def normalize(value: Any) -> Any:
        if isinstance(value, float):
            result = round(value, 12)
            return 0.0 if result == 0 else result
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return (
        json.dumps(normalize(review), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def table(headers: list[str], records: list[list[str]]) -> str:
    head = "".join(f'<th scope="col">{html.escape(value)}</th>' for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in records
    )
    return (
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def weekly_chart(review: dict[str, Any]) -> str:
    colors = ("#146b75", "#2563a6", "#966331", "#877291")
    lines = []
    for name, color in zip(MODELS, colors, strict=True):
        rows = [row for row in review["benchmark_weekly"] if row["model"] == name]
        points = " ".join(
            f"{60 + (row['week'] - 33) * 20:.2f},{245 - row['gini'] * 250:.2f}" for row in rows
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2.2"><title>{name}</title></polyline>'
        )
    grid = "".join(
        f'<line x1="60" y1="{245 - g * 250:.0f}" x2="840" '
        f'y2="{245 - g * 250:.0f}" stroke="#dde5e9"/><text x="18" '
        f'y="{249 - g * 250:.0f}">{g:.1f}</text>'
        for g in (0.2, 0.4, 0.6, 0.8)
    )
    labels = "".join(
        f'<text x="{60 + (w - 33) * 20}" y="275">{w}</text>' for w in (33, 41, 49, 57, 65, 72)
    )
    legend = " ".join(
        f'<span style="color:{color}">● {name.replace("_", " ")}</span>'
        for name, color in zip(MODELS, colors, strict=True)
    )
    return (
        '<figure><svg viewBox="0 0 880 300" role="img" '
        'aria-label="Weekly Gini for four development models across weeks 33 to 72">'
        f"<title>Development weekly Gini</title>{grid}{''.join(lines)}{labels}</svg>"
        f"<figcaption>{legend}</figcaption></figure>"
    )


def render_html(review: dict[str, Any]) -> str:
    benchmark = table(
        [
            "Model",
            "Mean fold stability",
            "Worst fold",
            "Pooled stability",
            "ROC AUC",
            "AP",
            "Brier",
            "Log loss",
        ],
        [
            [
                row["model"].replace("_", " "),
                f"{row['mean_fold_stability']:.6f}",
                f"{row['worst_fold_stability']:.6f}",
                f"{row['pooled_stability']:.6f}",
                *[f"{row['recorded_prediction_metrics'][key]:.6f}" for key in PREDICTION_METRICS],
            ]
            for row in review["benchmark"]
        ],
    )
    ablation = table(
        ["Controlled experiment", "Mean fold stability", "Worst fold", "Change from control"],
        [
            [
                row["experiment"].replace("_", " "),
                f"{row['mean_fold_stability']:.6f}",
                f"{row['worst_fold_stability']:.6f}",
                f"{row['delta_vs_control']:+.6f}",
            ]
            for row in review["feature_ablation"]
        ],
    )
    release = review["historical_release"]
    release_metrics = release["recorded_prediction_metrics"]
    release_table = table(
        ["Stability", "ROC AUC", "Average precision", "Brier", "Log loss"],
        [
            [
                f"{release['stability_components']['stability_score']:.6f}",
                *[f"{release_metrics[key]:.6f}" for key in PREDICTION_METRICS],
            ]
        ],
    )
    limitations = "".join(f"<li>{html.escape(value)}</li>" for value in review["limitations"])
    hashes = "".join(
        f"<tr><td>{html.escape(path)}</td><td><code>{digest}</code></td></tr>"
        for path, digest in sorted(SOURCE_HASHES.items())
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,
initial-scale=1">
<meta name="description" content="Reproducible aggregate evidence for temporal credit-risk
modeling.">
<title>Home Credit · Public evidence review</title>
<style>
:root{{color-scheme:light;
--ink:#152d3c;
--muted:#526674;
--teal:#146b75;
--line:#dce6eb}}

*{{box-sizing:border-box}}
body{{margin:0;
background:#f4f7f9;
color:var(--ink);
font:16px/1.6 system-ui,sans-serif}}

main{{max-width:1140px;
margin:auto;
padding:44px 24px 64px}}
header{{padding:34px;
border-radius:18px;
background:#152d3c;
color:white}}

.eyebrow{{color:#9ed8d3;
text-transform:uppercase;
letter-spacing:.14em;
font-size:12px;
font-weight:700}}

h1{{font-size:clamp(30px,4vw,48px);
line-height:1.15;
margin:14px 0}}
header p{{max-width:790px;
color:#d2e3ec}}

h2{{font-size:23px;
line-height:1.3;
margin:0 0 12px}}
h3{{font-size:16px}}
p{{margin:10px 0 18px}}

.cards{{display:grid;
grid-template-columns:repeat(4,1fr);
gap:14px;
margin:24px 0}}
.card,section{{background:white;
border:1px solid var(--line);
border-radius:12px}}

.card{{padding:20px}}
.card strong{{display:block;
font-size:28px;
color:var(--teal)}}
.card span{{font-size:13px;
color:var(--muted)}}

section{{padding:28px;
margin:18px 0}}
.note{{color:var(--muted);
font-size:14px}}
.formula{{padding:15px;
background:#edf5f5;
border-left:4px solid var(--teal);
font-family:ui-monospace,monospace;
font-size:14px}}

.table-scroll{{overflow-x:auto}}
table{{width:100%;
border-collapse:collapse;
font-size:13px}}
th{{background:#edf3f6;
text-align:left;
font-size:12px}}
th,td{{padding:12px;
border-bottom:1px solid var(--line)}}
td:first-child{{font-weight:600}}
tr:last-child td{{border-bottom:0}}
svg{{width:100%;
height:auto;
font-family:system-ui;
font-size:12px;
fill:var(--muted)}}
figure{{margin:20px 0 0}}
figcaption{{display:flex;
flex-wrap:wrap;
gap:16px;
font-size:13px}}
li{{margin:9px 0}}
code{{font-size:11px;
overflow-wrap:anywhere}}
a{{color:var(--teal)}}
footer{{font-size:13px;
color:var(--muted);
margin-top:24px}}

@media(max-width:720px){{main{{padding:18px 12px}}
header,section{{padding:22px}}
.cards{{grid-template-columns:repeat(2,1fr)}}
}}

@media print{{body{{background:white}}
main{{padding:0}}
section{{break-inside:avoid}}
}}

</style></head><body><main>
<header><div class="eyebrow">Machine learning engineering · Evidence you can inspect</div>
<h1>Credit risk, evaluated through time</h1>
<p>A reproducible review of model stability, controlled feature experiments and a frozen
historical release. Every result below is tied to committed aggregate evidence.</p>
<p class="note" style="color:#c6dce5">Python standard library only · Offline · No credentials · No
training</p></header>
<div class="cards"><div class="card"><strong>727,187</strong><span>development predictions per
model</span></div>
<div class="card"><strong>4 &times; 5</strong><span>model families &times; temporal
folds</span></div>
<div class="card"><strong>25</strong><span>stability scores recomputed from weekly Gini</span></div>
<div class="card"><strong>5</strong><span>hash-pinned public evidence files</span></div></div>
<section><h2>01 · Evaluate performance through time</h2>
<p>Discrimination alone can hide deterioration. This review fits the weekly Gini trend and applies
the frozen stability objective independently, then checks every result against the recorded
evidence.</p>
<p class="formula">stability = mean(weekly Gini) + 88 &times; min(0, slope) &minus; 0.5 &times;
residual standard deviation</p>
{benchmark}{weekly_chart(review)}
<p class="note">Development weeks 33&ndash;72. Mean fold and pooled stability are different
quantities because the objective is nonlinear. ROC AUC, AP, Brier and log loss are recorded
prediction-level metrics; weekly aggregates cannot reproduce them. Logistic results use the
published raw-rank correction.</p></section>
<section><h2>02 · Test whether feature groups earn their place</h2>
<p>Controlled removal experiments share the same five temporal folds. This review rechecks all 20
fold component identities, recomputes mean and worst-fold stability, and verifies the control
against the benchmark.</p>
{ablation}<p class="note">Negative deltas support retaining the removed feature group in this
development study. They do not establish causal effects or performance outside the evaluated
population.</p></section>
<section><h2>03 · Keep the release boundary explicit</h2>
<p>The frozen development model trained through week 72 and was evaluated once on weeks
73&ndash;91: <strong>203,345 observations</strong>. The review rechecks all 19 weekly aggregates
against the published frozen intent.</p>
{release_table}<p class="note">Historical evaluation: September 8, 2026. This holdout is already
observed. Its population differs from development, so these scores are not a direct
before-and-after comparison.</p></section>
<section><h2>04 · Know exactly what this reproduction establishes</h2><ul>{limitations}</ul>
<details><summary>Inspect input SHA-256 identities</summary><div
class="table-scroll"><table><thead><tr><th>Public
report</th><th>SHA-256</th></tr></thead><tbody>{hashes}</tbody></table></div></details></section>
<footer>Generated deterministically by <code>scripts/review_public_evidence.py</code>. <a
href="review.json">Machine-readable review</a>. No external assets, scripts, trackers or
requests.</footer>
</main></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/public_review")
    args = parser.parse_args()
    try:
        review = build_review()
        payload = serialize(review)
        # Render the normalized payload so rerenders match across supported Python platforms.
        report = render_html(json.loads(payload)).encode()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "review.json").write_bytes(payload)
        (args.output / "index.html").write_bytes(report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"PUBLIC_EVIDENCE_FAILED: {exc}\n")
    print(
        "PUBLIC_EVIDENCE_VERIFIED: 5 pinned files; 25 weekly stability recomputations; "
        "20 ablation component checks; no training or network"
    )
    print(f"REPORT={args.output / 'index.html'}")
    print(f"JSON={args.output / 'review.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
