from __future__ import annotations

import numpy as np

from f1strategy.features import FEATURES
from f1strategy.predictor import Conditions, DeltaPredictor


class EchoModel:
    def __init__(self):
        self.last = None

    def predict(self, matrix):
        self.last = matrix
        return np.array([matrix.sum(axis=1)[0]])


def test_feature_vector_matches_training_layout():
    model = EchoModel()
    predictor = DeltaPredictor(
        model,
        Conditions(total_laps=50, track_temp_delta=1.5, air_temp_delta=-0.5),
    )
    vector = predictor.features("SOFT", age=7, lap=20)
    assert vector.shape == (1, len(FEATURES))
    values = dict(zip(FEATURES, vector[0], strict=False))
    assert values["tyre_age"] == 7
    assert values["tyre_age_sq"] == 49
    assert values["compound_soft"] == 1.0
    assert values["compound_medium"] == 0.0
    assert values["compound_hard"] == 0.0
    assert values["track_temp_delta"] == 1.5
    assert values["air_temp_delta"] == -0.5


def test_predictor_is_callable():
    predictor = DeltaPredictor(EchoModel(), Conditions(total_laps=50))
    assert predictor("HARD", 10, 40) > 0
