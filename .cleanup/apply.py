"""Remove unpromoted follow-on work; preserve the selected release pipeline."""
from pathlib import Path
import ast,json,re,shutil,subprocess
R=Path.cwd()
if subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()!='c2b36f2531002543fae7aaa12ebce3ec99879ad4':raise RuntimeError('Main advanced')
def remove(n):
 p=R/n
 if p.is_dir():shutil.rmtree(p)
 elif p.exists():p.unlink()
def write(n,t):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf-8')
for n in ['AGENTS.md','PROJECT_PLAN.md','src/home_credit/observability/project.py','src/home_credit/features/research.py','src/home_credit/features/distributions.py']:
 remove(n)
for n in ['calibration_research','calibration_report','feature_research','feature_research_report','history_research','history_report','feature_interpretation']:
 remove('src/home_credit/modeling/'+n+'.py')
for n in ['project_status','run_history_research','review_feature_interpretation','run_feature_research','verify_calibration_research','run_feature_research_shards','run_calibration_research','review_calibration','verify_feature_research','verify_history_research','review_feature_research','review_post_release_frontier']:
 remove('scripts/'+n+'.py')
for n in ['test_feature_research','test_history_research','test_feature_shards','test_history_report','test_feature_research_report','test_calibration_report','test_project_status','test_calibration_research','test_post_release_frontier']:
 remove('tests/unit/'+n+'.py')
for p in (R/'configs').iterdir():
 if any(s in p.name for s in ['calibration','feature_research','history_research','denselight','categorical_identity','applprev_histogram']):remove(p.relative_to(R))
for p in (R/'notebooks').iterdir():
 if p.name.startswith(('11_','12_','13_')):remove(p.relative_to(R))
for n in ['denselight_frontier','feature_research','calibration','applprev_histogram','latest_execution','categorical_identity','history_research']:
 remove('reports/'+n)
for p in (R/'reports/post_release_frontier').iterdir():
 if p.name!='october_2026.md':remove(p.relative_to(R))
for n in ['calibration.md','project_status.md','history_research.md','feature_research.md','post_release_research.md','completion.md']:
 remove('docs/'+n)
# Remove stale gallery prose from the generator of the retained release notebooks.
p=R/'src/home_credit/modeling/portfolio_report.py';t=p.read_text()
lines=t.splitlines(keepends=True);changes=[]
for node in ast.walk(ast.parse(t)):
 if isinstance(node,ast.Constant) and isinstance(node.value,str) and ('notebook 11' in node.value.lower() or 'notebook 12' in node.value.lower()):
  old=node.value;blocks=old.split('\n\n')
  new='\n\n'.join(b for b in blocks if 'notebook 11' not in b.lower() and 'notebook 12' not in b.lower())
  if new!=old:
   begin=sum(len(s) for s in lines[:node.lineno-1])+node.col_offset
   end=sum(len(s) for s in lines[:node.end_lineno-1])+node.end_col_offset
   parts=[repr(line[i:i+68]) for line in new.splitlines(keepends=True) for i in range(0,len(line),68)]
   changes.append((begin,end,'(\n'+''.join('        '+part+'\n' for part in parts)+'    )'))
for a,b,new in sorted(changes,reverse=True):t=t[:a]+new+t[b:]
p.write_text(t)
write('README.md','''# Home Credit · Temporal risk modeling

![Actual frozen weekly Gini and source-linked credit-risk evidence](reports/portfolio/brief/cover.svg)

**Alvaro Mendizabal · ML engineering · Relational data · Temporal validation**

[![CI](https://github.com/alvaromendizabal/home-credit-model-stability/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaromendizabal/home-credit-model-stability/actions/workflows/ci.yml)
[![Public demo](https://github.com/alvaromendizabal/home-credit-model-stability/actions/workflows/public-demo.yml/badge.svg)](https://github.com/alvaromendizabal/home-credit-model-stability/actions/workflows/public-demo.yml)

I built a credit-risk research and inference system that joins relational histories, selects features using training data only, compares models across time and verifies the complete prediction handoff. The maintained release shows the selected pipeline and its necessary controls—not unpromoted follow-on studies.

[**Explore Stability Lab ↗**](https://home-credit-stability-lab.tartmacaw2.chatgpt.site) · [Visual engineering brief](reports/portfolio/brief/index.html) · [Case study](docs/case_study.md) · [Model card](MODEL_CARD.md)

## Scale and engineering

| Scope | Recorded result | Implementation |
|---|---|---|
| Relational data | **1,526,659 applications · 17 source groups** | [Schema and joins](src/home_credit/data) |
| Selected representation | **2,508 candidates → 700 features** | [Feature construction](src/home_credit/features), [training-only screening](src/home_credit/modeling/screening.py) |
| Controlled benchmark | **Four model families × five expanding folds · 727,187 OOF cases** | [Temporal protocol](src/home_credit/validation), [model runner](src/home_credit/modeling/runner.py) |
| Release delivery | Native-model reload, exact case order and verified reuse | [Release workflow](src/home_credit/modeling/release_workflow.py), [inference](src/home_credit/modeling/inference.py) |

![Data, modeling, validation and inference architecture](reports/portfolio/system_architecture.svg)

## Final measured results

| Artifact and population | Recorded metrics |
|---|---|
| Original frozen release · 203,345 later-period applications | **0.875759 ROC AUC · 0.729674 weekly-Gini stability** |
| Same population · probability diagnostics | **0.193755 average precision · 0.019282 Brier · 0.080695 log loss** |
| Original all-label inference refit · external evaluation | **0.56062 public · 0.47429 private** |
| Later tree + histogram + neural ensemble · separate external evaluation | **0.56035 public · 0.47652 private** |

The local **0.729674** stability result is not an external competition score. The later all-label ensemble is a different private artifact, not an identical rerun of the original public reference. Weeks 73–91 were opened after the original selection and are now observed; later development cannot treat them as a fresh holdout. [Original release](reports/model_release/evaluation.json) · [Original external receipt](reports/kaggle_submission/execution.json) · [Later final result](reports/post_release_frontier/october_2026.md)

## Evidence for the selected features

Removing bureau A, previous applications or depth-two history reduced mean development stability by **0.093813**, **0.031440** and **0.015762**, respectively. These are predictive removal effects, not causal effects on borrowers. [Controlled ablations](reports/feature_ablation/README.md)

![Matched model-family and feature-removal comparisons](reports/portfolio/overview.svg)

## Run a public review

With Python 3.12, no credentials or private data:

```bash
python3 scripts/review_public_evidence.py --output artifacts/public-review
python3 scripts/review_public_evidence.py --output artifacts/public-review --check
python3 scripts/review_portfolio_brief.py
```

Open `artifacts/public-review/index.html`. The standard-library implementation checks pinned original inputs, recalculates weekly-Gini stability and distinguishes source-reported AUC from recomputed aggregate statistics. Repeating a completed run checks its outputs before reuse. It does not claim to reconstruct unavailable borrower predictions.

The locked environment also exercises real small-data model fits, train-only encoders, temporal leakage tests, checkpoint corruption/recovery, exact-ID inference and canonical executed benchmark, selection and release notebooks. [Full commands](docs/public_reproducibility.md)

## A focused publication

The retained project includes the feature pipeline, necessary model controls, frozen selection/release/inference code, original result records, six canonical notebooks and an interactive stability explanation. Unpromoted calibration, categorical/history extensions, proposed neural experiments, their launchers and redundant notebooks are retired.

Private borrower records, fitted research weights, active feature identities and current ensemble specifications are not distributed. The browser operates on explicitly synthetic weekly aggregates; it is not a deployed credit decision system. No lending-policy impact, fairness certification or production readiness is claimed. [Disclosure](docs/public_reproducibility.md#disclosure-boundary)
''')
for n in ['docs/case_study.md','docs/employer_review.md','docs/research_engineering.md','docs/public_reproducibility.md','MODEL_CARD.md']:
 p=R/n;t=p.read_text();blocks=t.split('\n\n')
 t='\n\n'.join(b for b in blocks if not any(s in b for s in ['11_feature_research','12_calibration','13_categorical','review_feature_research.py','review_calibration.py','project_status.py','run_history_research.py','DenseLight','denselight_frontier']))
 t=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:m.group(0) if m.group(2).startswith(('http','#','mailto:')) or (p.parent/m.group(2).split('#')[0]).exists() else m.group(1),t)
 p.write_text(t)
p=R/'scripts/check.sh';t=p.read_text();lines=[]
for line in t.splitlines():
 if line.startswith('run post_release_frontier'):continue
 if line.startswith('run mypy '):
  line=' '.join(q for q in line.split() if not q.startswith('scripts/') or (R/q).exists())
 lines.append(line)
p.write_text('\n'.join(lines)+'\n')
for p in (R/'docs').glob('*.md'):
 t=p.read_text();t=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:m.group(0) if m.group(2).startswith(('http','#','mailto:')) or (p.parent/m.group(2).split('#')[0]).exists() else m.group(1),t);p.write_text(t)
print('Unpromoted extensions retired; selected training, release and inference contracts preserved.')
