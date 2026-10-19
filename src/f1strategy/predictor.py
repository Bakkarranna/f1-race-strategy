"""Prediction adapter between the model and the strategy simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import COMPOUNDS, FEATURES


@dataclass
class Conditions:
    total_laps: int
    track_temp_delta: float = 0.0
    air_temp_delta: float = 0.0


class DeltaPredictor:
    """Callable that predicts the degradation delta for a tyre state.

    The simulator needs a prediction for every combination of compound, tyre
    age and lap. Building the feature matrix in one go and calling the model
    once is roughly a thousand times faster than predicting row by row, which
    matters when a whole season is replayed.
    """

    def __init__(self, model, conditions: Conditions) -> None:
        self.model = model
        self.conditions = conditions
        self._index = {name: index for index, name in enumerate(FEATURES)}
        self._table: np.ndarray | None = None
        self._table_key: tuple[int, int] | None = None

    def features(self, compound: str, age: int, lap: int) -> np.ndarray:
        vector = np.zeros((1, len(FEATURES)), dtype=float)
        vector[0, self._index["tyre_age"]] = age
        vector[0, self._index["tyre_age_sq"]] = age**2
        vector[0, self._index["track_temp_delta"]] = self.conditions.track_temp_delta
        vector[0, self._index["air_temp_delta"]] = self.conditions.air_temp_delta
        for name in COMPOUNDS:
            vector[0, self._index[f"compound_{name.lower()}"]] = float(
                name == compound.upper()
            )
        return vector

    def table(self, max_age: int, total_laps: int) -> np.ndarray:
        """Return an array of shape (compounds, max_age + 1, total_laps)."""
        key = (max_age, total_laps)
        if self._table is not None and self._table_key == key:
            return self._table

        rows = []
        for compound in COMPOUNDS:
            for age in range(max_age + 1):
                for lap in range(1, total_laps + 1):
                    rows.append(self.features(compound, age, lap)[0])
        predictions = self.model.predict(np.vstack(rows))
        self._table = predictions.reshape(len(COMPOUNDS), max_age + 1, total_laps)
        self._table_key = key
        return self._table

    def __call__(self, compound: str, age: int, lap: int) -> float:
        return float(self.model.predict(self.features(compound, age, lap))[0])
