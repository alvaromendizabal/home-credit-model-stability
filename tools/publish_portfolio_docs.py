"""One-time public documentation migration; removed before the closeout PR is merged."""
from pathlib import Path
import json
import subprocess

r = Path.cwd()
base = '1de46633c44071689915015c89a3eb42bd77e301'
paths = ['README.md', 'docs/employer_review.md', 'docs/project_status.md', 'docs/completion.md', 'docs/public_reproducibility.md', 'docs/case_study.md', 'MODEL_CARD.md', 'Makefile', 'scripts/check.sh']
for name in paths:
    expected = subprocess.check_output(['git', 'show', f'{base}:{name}'])
    if (r / name).is_symlink() or (r / name).read_bytes() != expected:
        raise RuntimeError('Documentation base changed: ' + name)

p=r/'README.md';s=p.read_text();s=s.replace('# Home Credit Model Stability\n', '# Home Credit Model Stability\n\n![Credit-risk engineering: actual frozen weekly Gini, source-bound evidence and 1.53 million labeled applications](reports/portfolio/brief/cover.svg)\n',1)
s=s.replace('**[Explore the interactive demo]', '**[Read the visual engineering brief](reports/portfolio/brief/index.html)** · [Verify its source-bound build](scripts/review_portfolio_brief.py)\n\n**[Explore the interactive demo]',1)
pos=s.index('## Three engineering results')
s=s[:pos]+'''## Choose your review depth

| Time | Review path | What you can verify |
|---|---|---|
| **30 seconds** | Architecture and measured results below | The problem, my ownership and the evidence scope |
| **5 minutes** | [Visual brief](reports/portfolio/brief/index.html) · [Engineering review](docs/employer_review.md) | Design trade-offs, feature value, evaluation discipline and inference delivery |
| **Hands-on** | `python3 scripts/review_portfolio_brief.py` | Seven pinned public inputs; the existing metric replay engine; deterministic HTML/SVG/JSON |
| **Implementation depth** | [Case study](docs/case_study.md) · [Executed notebooks](notebooks) · [Source](src/home_credit) | Training-only encoders, temporal comparisons, checkpoint recovery and release contracts |

The visual brief opens locally after generation. GitHub displays HTML as source;
use the existing live demo for an immediate browser experience. No installation,
account or private data is required for the standard-library review command.

'''+s[pos:]
old='The latest inspected **E98 execution completed five model fits**, but its selected candidate failed later-period confirmation and was rejected. It produced no new submission or external score. The historical release remains frozen; [current status](docs/project_status.md) records the decision and separate evidence scopes.'
new='**Portfolio closeout — 17 October 2026.** The evaluated release, public evidence and interactive review are complete within the documented research scope. A subsequent matched-control neural study completed successfully but remained exploratory; it did not promote a new model or create a new external score. No further training is required to review this project. [Current status](docs/project_status.md) and the [sanitized closeout receipt](reports/portfolio/closeout.json) separate the delivered portfolio from optional future research.'
assert old in s;s=s.replace(old,new);p.write_text(s)

p=r/'docs/employer_review.md';s=p.read_text();s=s.replace('## Start with the interactive explanation','''## Pick a role-specific route

| Reviewer | Inspect first | Evidence of ownership |
|---|---|---|
| ML engineer | [Checkpoint code](../src/home_credit/modeling/checkpoints.py), [inference tests](../tests/unit/test_inference.py) | Resumability, artifact identity, feature/inference parity and delivery safeguards |
| Data scientist | [Benchmark](../notebooks/05_benchmark_review.ipynb), [ablations](../reports/feature_ablation/comparison.json) | Matched controls, temporal generalization, probability diagnostics and rejected hypotheses |
| Applied AI engineer | [Pipeline case study](case_study.md), [runtime](../src/home_credit/runtime) | End-to-end model integration and operational reliability; this is tabular ML, not a claim of an LLM application |

For a visual walkthrough, generate the [engineering brief](../reports/portfolio/brief/index.html)
with `python3 scripts/review_portfolio_brief.py`. It reuses the established public
metric verifier and needs only Python's standard library. The checked-in HTML is a
portable document, not a newly deployed website.

## Start with the interactive explanation''',1)
s=s.replace('## Frozen evidence and ongoing research','## Completed portfolio and separately scoped research',1);p.write_text(s)

p=r/'docs/project_status.md';s=p.read_text();s=s.replace('# Project status and optional monitoring','# Portfolio release status',1).replace('## Latest inspected AWS execution — E98','## Historical AWS inspection — E98 (10 October 2026)',1)
pos=s.index('## Historical AWS inspection')
s=s[:pos]+'''## Portfolio closeout — 17 October 2026

**The public portfolio is closed within its research-and-inference scope.** The
September release, later accepted external evaluation, executed notebooks, public
metric verifier and interactive demo remain preserved. The employer-facing brief
adds a seven-source, credential-free review path without replacing the scientific
pipeline. No new training or submission is needed to demonstrate the completed work.

The latest inspected owner-return archive completed **13/13 stages and one neural
model fit**. Its comparison used a previously observed development fold, so the
result remains exploratory. It did not promote a new champion or generate a new
Kaggle submission. The [sanitized receipt](../reports/portfolio/closeout.json) binds
this statement to the inspected archive digest. The source archive and model
artifacts remain private; public checks do not independently rerun that training.

The historical **0.56035 public / 0.47652 private** result remains sourced to the
[accepted post-release report](../reports/post_release_frontier/october_2026.md).
It was not reauthenticated through a new Kaggle execution during portfolio closeout.
The frozen **0.729674** local stability still belongs to a different model artifact
and population, and is not interchangeable with that external score.

Further experiments and production-readiness studies are optional, separately
scoped work. Presentation closeout does not certify production lending readiness,
resolve all research questions or change any AWS resource's billing state.

'''+s[pos:];p.write_text(s)

p=r/'docs/completion.md';s=p.read_text();pos=s.index('## Closeout checks — 9 September 2026');s=s[:pos]+'''## Portfolio publication closeout — 17 October 2026

The public portfolio now has a deterministic, self-contained engineering brief,
source-linked metric scopes, an accessible README cover, role-specific review
routes and a tested no-overwrite publication contract. These are presentation and
public-evidence improvements, not a new model release or external evaluation.

The original credential-free verifier, synthetic stability lab, training source,
executed notebooks and accepted research artifacts are preserved. The brief reuses
that verifier rather than introducing a competing score implementation.

| Public review contract | How to inspect it |
|---|---|
| Rebuild the portable brief | `python3 scripts/review_portfolio_brief.py` |
| Verify the same bundle without writes | Add `--check` |
| Verify the committed publication | `python3 scripts/review_portfolio_brief.py --output reports/portfolio/brief --check` |
| Test failure and disclosure boundaries | `python3 -m unittest discover -s tools -p 'test_portfolio_brief.py'` |
| Inspect historical research quality gates | [Existing CI](../.github/workflows/ci.yml) |
| Inspect presentation-specific CI | [Portfolio workflow](../.github/workflows/portfolio.yml) |

The [current status](project_status.md) and [sanitized receipt](../reports/portfolio/closeout.json)
record the last inspected exploratory run. No source winner-equivalence, new
champion promotion, production adoption or financial lift is claimed.
No further model training is required for this portfolio's completed public scope.

'''+s[pos:];p.write_text(s)

p=r/'docs/public_reproducibility.md';s=p.read_text();pos=s.index('## Interactive demo');s=s[:pos]+'''## Portable engineering brief

```bash
python3 scripts/review_portfolio_brief.py
python3 scripts/review_portfolio_brief.py --check
```

Open `artifacts/portfolio-brief/index.html`. The same directory contains an accessible
`cover.svg`, the allowlisted `evidence.json` and a source/output `manifest.json`.
There are no runtime assets, JavaScript, analytics, remote fonts or background
requests. Source links open the corresponding published GitHub evidence only when
selected. Native disclosure controls work without a script or web server.

The brief **imports the established public metric engine**; it does not introduce
another scoring implementation. It authenticates the original five aggregate inputs
and two additional public files: the temporal protocol and original external
execution receipt. Only approved population fields and external score/status fields
are exported. The complete operational receipts, model settings and cloud paths are
not copied into the brief.

The generation command stages a complete bundle and publishes by directory rename.
An existing destination is verified, never overwritten. A mismatch, extra member,
unsafe path, missing source or changed source digest exits nonzero. `--check` performs
no writes. Choose a new destination when deliberately reviewing a new source version;
never delete accepted evidence just to clear a mismatch.

The committed example can be checked with:

```bash
python3 scripts/review_portfolio_brief.py --output reports/portfolio/brief --check
```

[Portfolio CI](../.github/workflows/portfolio.yml) exercises the safety contracts and
reproduces all four output files. The browser layout was additionally reviewed at
desktop and mobile sizes; those local visual checks are separate from CI's automated
source, HTML and numeric contract checks.

The [closeout receipt](../reports/portfolio/closeout.json) is a separate owner-return
attestation. It documents an exploratory run, not an independently reproduced public
training result. No private return ZIP or training recipe is required or published.

'''+s[pos:];p.write_text(s)

p=r/'docs/case_study.md';s=p.read_text();pos=s.index('## Limits and production follow-through');s=s[:pos]+'''## Delivery decisions and failure containment

| Risk | Engineering decision | Public evidence |
|---|---|---|
| An interrupted run silently repeats expensive training | Reuse only hash-consistent accepted checkpoints; stop on incompatible identity | [Checkpoint implementation](../src/home_credit/modeling/checkpoints.py) and its tests |
| An attractive mean conceals a weak period | Keep per-fold results, worst fold and stability components next to pooled AUC | [Executed benchmark](../notebooks/05_benchmark_review.ipynb) |
| A presentation refresh mutates historical results | Reuse the accepted public verifier, pin source bytes and compare generated output exactly | [Brief renderer](../scripts/review_portfolio_brief.py), [contract tests](../tools/test_portfolio_brief.py) |
| Publishing reproducibility exposes competitive assets | Export only allowlisted public aggregates; keep row-level data and active artifacts private | [Disclosure boundary](public_reproducibility.md#disclosure-boundary) |
| A polished demo is mistaken for empirical evidence | Label synthetic interaction separately from actual source-bound experiments | [Interactive demo contract](public_reproducibility.md#interactive-demo) |

The finished portfolio demonstrates evaluated modeling and reliable research
engineering. Additional model experiments are not a dependency of its review path.
The [visual brief](../reports/portfolio/brief/index.html) makes these decisions
accessible before a reviewer opens the deeper implementation.

'''+s[pos:];p.write_text(s)

p=r/'MODEL_CARD.md';s=p.read_text();lines=s.splitlines(True);lines.insert(2, '''
> **Portfolio closeout: 17 October 2026.** Public presentation now includes a
> [source-bound engineering brief](reports/portfolio/brief/index.html). This does
> not change the evaluated model, establish a new external score or certify
> production lending readiness. Later exploratory research did not promote a new
> release; see [current status](docs/project_status.md).

''');p.write_text(''.join(lines))

p=r/'Makefile';s=p.read_text().replace('connectivity public-review','connectivity public-review portfolio',1);s+='\nportfolio:\n\tpython3 scripts/review_portfolio_brief.py\n';p.write_text(s)

summary={
 'schema_version':1,'publication_date':'2026-10-17',
 'portfolio_status':'CLOSED_WITHIN_DOCUMENTED_RESEARCH_SCOPE',
 'attestation_scope':'Sanitized owner-return archive inspection; not public retraining or independent score authentication',
 'source_archive_sha256':'a0c455ce943ac9e7530c998a5f36f8183e7f3d3453ef38c184b444302ba6972f',
 'source_archive_published':False,
 'latest_inspected_execution':{'status':'SUCCESS','stages_completed':13,'stages_total':13,'new_model_fits':1,
 'evaluation_scope':'Previously observed development fold; exploratory only',
 'new_champion_promoted':False,'new_external_score':False,'submission_readiness':'NO_BETTER_CHALLENGER'},
 'preservation':{'historical_release_unchanged':True,'private_models_not_published':True,
 'private_feature_recipes_not_published':True,'borrower_rows_not_published':True},
 'cloud_billing_state':'NOT_CHANGED_OR_CERTIFIED_BY_PORTFOLIO_PUBLICATION',
 'future_work':'Optional, separately scoped research and production validation'
}
(r/'reports/portfolio/closeout.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
p=r/'scripts/check.sh'
s=p.read_text()
needle='scripts/review_public_evidence.py'
assert needle in s
s=s.replace(needle,needle+' scripts/review_portfolio_brief.py',1)
p.write_text(s)
print('Public documentation migration complete; private science unchanged.')
