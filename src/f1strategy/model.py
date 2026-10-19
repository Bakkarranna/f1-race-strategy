"""Modelling helpers: training, validation and the saved artifact."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import COMPOUNDS, FEATURES, TARGET, feature_frame

LOGGER = logging.getLogger(__name__)

# Races kept out of training for the final report. They cover a range of
# circuit types: a permanent track, a street circuit, a high degradation track
# and a fast circuit.
HOLDOUT_ROUNDS = [11, 16, 17, 23]


@dataclass
class Metrics:
    model: str
    mae: float
    rmse: float
    r2: float
    n: int

    def as_dict(self) -> dict[str, float | str]:
        return {
            "model": self.model,
            "mae_s": round(self.mae, 4),
            "rmse_s": round(self.rmse, 4),
            "r2": round(self.r2, 4),
            "laps": self.n,
        }


def make_model(name: str) -> object:
    """Return an unfitted estimator by short name."""
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=25,
            max_features=0.8,
            n_jobs=-1,
            random_state=7,
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=40,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=7,
            n_jobs=4,
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate(model: object, frame: pd.DataFrame, name: str) -> Metrics:
    predictions = model.predict(feature_frame(frame))
    truth = frame[TARGET].to_numpy()
    return Metrics(
        model=name,
        mae=mean_absolute_error(truth, predictions),
        rmse=float(np.sqrt(mean_squared_error(truth, predictions))),
        r2=r2_score(truth, predictions),
        n=len(frame),
    )


def cross_validate_by_race(
    frame: pd.DataFrame,
    name: str = "xgboost",
    folds: int = 5,
) -> Metrics:
    """Grouped cross validation where a whole race is held out each fold."""
    races = np.array(sorted(frame["session_key"].unique()))
    rng = np.random.default_rng(11)
    rng.shuffle(races)
    chunks = np.array_split(races, folds)
    truths: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for held_out in chunks:
        train = frame[~frame["session_key"].isin(held_out)]
        test = frame[frame["session_key"].isin(held_out)]
        model = make_model(name)
        model.fit(feature_frame(train), train[TARGET])
        preds.append(model.predict(feature_frame(test)))
        truths.append(test[TARGET].to_numpy())
    truth = np.concatenate(truths)
    pred = np.concatenate(preds)
    return Metrics(
        model=name,
        mae=mean_absolute_error(truth, pred),
        rmse=float(np.sqrt(mean_squared_error(truth, pred))),
        r2=r2_score(truth, pred),
        n=len(truth),
    )


def train_final(frame: pd.DataFrame, name: str = "xgboost", holdout: bool = True) -> tuple[object, Metrics]:
    """Fit on the training races and score on the holdout races."""
    if holdout:
        train = frame[~frame["round"].isin(HOLDOUT_ROUNDS)]
        test = frame[frame["round"].isin(HOLDOUT_ROUNDS)]
    else:
        train = test = frame
    model = make_model(name)
    model.fit(feature_frame(train), train[TARGET])
    metrics = evaluate(model, test, name)
    LOGGER.info("%s holdout MAE %.3fs over %d laps", name, metrics.mae, metrics.n)
    return model, metrics


def save(model: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, path)


def load(path: str | Path) -> object:
    payload = joblib.load(path)
    return payload["model"]


def degradation_curve(model: object, frame: pd.DataFrame, compound: str = "MEDIUM") -> pd.DataFrame:
    """Predict the delta as a function of tyre age at average race conditions."""
    base = frame.loc[:, FEATURES].median(numeric_only=True)
    rows = []
    for age in range(0, 41):
        row = base.copy()
        row["tyre_age"] = age
        row["tyre_age_sq"] = age**2
        for name in ["compound_soft", "compound_medium", "compound_hard"]:
            row[name] = int(name == f"compound_{compound.lower()}")
        rows.append(row)
    curve = pd.DataFrame(rows)
    curve["compound"] = compound
    curve["predicted_delta"] = model.predict(feature_frame(curve))
    return curve[["tyre_age", "compound", "predicted_delta"]]


def binned_metrics(model: object, frame: pd.DataFrame) -> dict[str, float]:
    """Error on binned mean pace, where per lap traffic noise averages out."""
    predictions = model.predict(feature_frame(frame))
    data = frame[["compound", "tyre_age", TARGET]].copy()
    data["predicted"] = predictions
    data["age_bin"] = pd.cut(
        data["tyre_age"], bins=[-1, 2, 5, 8, 12, 16, 20, 25, 30, 60]
    )
    grouped = data.groupby(["compound", "age_bin"], observed=True).agg(
        observed=(TARGET, "mean"), predicted=("predicted", "mean"), n=(TARGET, "size")
    )
    error = (grouped["predicted"] - grouped["observed"]).abs()
    weights = grouped["n"] / grouped["n"].sum()
    return {
        "binned_mae_s": round(float((error * weights).sum()), 4),
        "bins": float(len(grouped)),
    }


def per_compound_metrics(
    model: object, frame: pd.DataFrame
) -> list[dict[str, float | str]]:
    """Holdout error split by compound, which shows where the model is weak."""
    rows = []
    predictions = model.predict(feature_frame(frame))
    for compound in COMPOUNDS:
        mask = frame["compound"].to_numpy() == compound
        if mask.sum() < 20:
            continue
        error = predictions[mask] - frame.loc[mask, TARGET].to_numpy()
        rows.append(
            {
                "compound": compound,
                "laps": int(mask.sum()),
                "mae_s": round(float(np.mean(np.abs(error))), 3),
                "bias_s": round(float(np.mean(error)), 3),
            }
        )
    return rows
