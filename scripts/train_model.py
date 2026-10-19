"""Train the lap delta model and write the evaluation artifacts.

Usage:
    python scripts/train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1strategy import model as modelling  # noqa: E402
from f1strategy import plots  # noqa: E402
from f1strategy.features import (  # noqa: E402
    FEATURES,
    TARGET,
    add_features,
    add_stint_delta,
    clean_laps,
    estimate_fuel_effect,
    feature_frame,
    summarise,
)

DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA / "laps_2024.csv")
    clean = clean_laps(raw)
    fuel_effect = estimate_fuel_effect(clean)
    print(f"Fuel sensitivity: {fuel_effect:.4f} s/kg ({fuel_effect * 110:.2f} s over a full tank)")
    frame = add_features(add_stint_delta(clean, fuel_effect))
    frame.to_csv(DATA / "laps_features.csv", index=False)

    print("Modelling sample")
    for key, value in summarise(frame).items():
        print(f"  {key:16s} {value:9.1f}")

    print("\nGrouped cross validation by race")
    cv_results = []
    for name in ["ridge", "forest", "xgboost"]:
        metrics = modelling.cross_validate_by_race(frame, name)
        cv_results.append(metrics.as_dict())
        print(f"  {name:8s} MAE {metrics.mae:.3f}s  R2 {metrics.r2:.3f}")

    evaluation_model, holdout = modelling.train_final(frame, "xgboost")
    per_compound = modelling.per_compound_metrics(evaluation_model, frame[frame["round"].isin(modelling.HOLDOUT_ROUNDS)])
    print(f"\nHoldout MAE {holdout.mae:.3f}s over {holdout.n} laps")
    train_rows = frame[~frame["round"].isin(modelling.HOLDOUT_ROUNDS)]
    binned_train = modelling.binned_metrics(evaluation_model, train_rows)
    binned = modelling.binned_metrics(evaluation_model, frame[frame["round"].isin(modelling.HOLDOUT_ROUNDS)])
    print(
        f"Binned pace error {binned['binned_mae_s']:.3f}s on the holdout races "
        f"({binned_train['binned_mae_s']:.3f}s on the training races)"
    )
    for row in per_compound:
        print(f"  {row['compound']:7s} MAE {row['mae_s']:.3f}s  bias {row['bias_s']:+.3f}s  n={row['laps']}")

    holdout_rows = frame[frame["round"].isin(modelling.HOLDOUT_ROUNDS)]
    truth = holdout_rows[TARGET].to_numpy()
    predicted = evaluation_model.predict(feature_frame(holdout_rows))
    plots.plot_predictions(
        truth,
        predicted,
        FIGURES / "holdout_predictions.png",
        f"Holdout races, MAE {holdout.mae:.2f}s",
    )

    curves = pd.concat(
        [modelling.degradation_curve(evaluation_model, frame, compound) for compound in ["SOFT", "MEDIUM", "HARD"]],
        ignore_index=True,
    )
    curves.to_csv(ARTIFACTS / "degradation_curves.csv", index=False)
    plots.plot_degradation(frame, curves, FIGURES / "degradation.png")

    # The shipped artifact is fitted on the training races only, so a query on
    # a holdout race remains genuinely out of sample.
    modelling.save(evaluation_model, ARTIFACTS / "lap_delta_model.joblib")

    importance = pd.Series(
        evaluation_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print("\nFeature importance")
    print(importance.head(8).to_string())

    metrics = {
        "fuel_effect_s_per_kg": round(fuel_effect, 5),
        "sample": summarise(frame),
        "cross_validation": cv_results,
        "holdout": holdout.as_dict(),
        "binned": binned,
        "binned_train": binned_train,
        "per_compound": per_compound,
        "holdout_rounds": modelling.HOLDOUT_ROUNDS,
        "feature_importance": {key: round(float(value), 4) for key, value in importance.items()},
        "residual_std_s": round(float(np.std(predicted - truth)), 4),
    }
    with (ARTIFACTS / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\nWritten {ARTIFACTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
