"""Fast compatibility smoke tests for the core data and model stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


@dataclass(frozen=True, slots=True)
class ModelSmokeResult:
    """A tiny-model result used only to prove fit/predict compatibility."""

    model: str
    auc: float


def dataframe_roundtrip() -> tuple[int, int]:
    """Exercise NumPy -> pandas -> Arrow -> Polars interoperability."""
    pandas_frame = pd.DataFrame({"x": np.arange(10), "y": np.linspace(0.0, 1.0, 10)})
    arrow_table = pa.Table.from_pandas(pandas_frame, preserve_index=False)
    polars_frame = pl.from_arrow(arrow_table)
    if not isinstance(polars_frame, pl.DataFrame):
        raise TypeError("expected Arrow table conversion to return a Polars DataFrame")
    result = polars_frame.with_columns((pl.col("x") * pl.col("y")).alias("xy"))
    return result.shape


def model_smoke() -> list[ModelSmokeResult]:
    """Fit and predict with each primary GBDT library on a tiny CPU dataset."""
    x, y = make_classification(
        n_samples=320,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        random_state=20260904,
    )
    x_train, x_test = x[:240], x[240:]
    y_train, y_test = y[:240], y[240:]

    models = {
        "lightgbm": LGBMClassifier(
            n_estimators=25,
            learning_rate=0.08,
            num_leaves=15,
            random_state=20260904,
            n_jobs=2,
            verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=25,
            depth=4,
            learning_rate=0.08,
            random_seed=20260904,
            verbose=False,
            thread_count=2,
            allow_writing_files=False,
        ),
        "xgboost": XGBClassifier(
            n_estimators=25,
            max_depth=4,
            learning_rate=0.08,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=20260904,
            n_jobs=2,
            tree_method="hist",
        ),
    }

    results: list[ModelSmokeResult] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict_proba(x_test)[:, 1]
        if name == "lightgbm":
            contributions = np.asarray(model.booster_.predict(x_test, pred_contrib=True))
            raw_scores = model.booster_.predict(x_test, raw_score=True)
            if contributions.shape != (len(x_test), x_test.shape[1] + 1):
                raise RuntimeError("LightGBM native SHAP contribution shape mismatch")
            if not np.allclose(contributions.sum(axis=1), raw_scores, rtol=1e-7, atol=1e-8):
                raise RuntimeError("LightGBM native SHAP values do not reconstruct raw scores")
        auc = float(roc_auc_score(y_test, prediction))
        if not 0.5 <= auc <= 1.0:
            raise RuntimeError(f"{name} produced an implausible smoke-test AUC: {auc}")
        results.append(ModelSmokeResult(model=name, auc=auc))
    return results
