from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1strategy.features import (
    FEATURES,
    TARGET,
    add_features,
    add_stint_delta,
    clean_laps,
    estimate_fuel_effect,
    feature_frame,
    fuel_mass,
)


def lap_row(**overrides):
    row = {
        "year": 2024,
        "round": 1,
        "event_name": "Test",
        "circuit": "Test",
        "country": "Test",
        "session_key": 1,
        "driver_number": 1,
        "driver": "TST",
        "team": "Test",
        "lap_number": 5,
        "total_laps": 50,
        "lap_duration": 90.0,
        "sector_1": 30.0,
        "sector_2": 30.0,
        "sector_3": 30.0,
        "speed_trap": 300,
        "compound": "MEDIUM",
        "tyre_age": 4,
        "stint_number": 1,
        "is_pit_out_lap": False,
        "is_pit_in_lap": False,
        "pit_lane_duration": np.nan,
        "is_safety_car_lap": False,
        "is_slow_lap": False,
        "air_temp": 25.0,
        "track_temp": 35.0,
        "humidity": 40.0,
        "rainfall": 0.0,
        "wind_speed": 2.0,
    }
    row.update(overrides)
    return row


def frame(rows):
    return pd.DataFrame(rows)


def stint_rows(start=2, end=20, base=90.0, degradation=0.06, compound="MEDIUM"):
    rows = []
    for lap in range(start, end + 1):
        age = lap - start
        rows.append(
            lap_row(
                lap_number=lap,
                tyre_age=age,
                compound=compound,
                lap_duration=base + degradation * age + 0.035 * 110 * (50 - lap + 1) / 50,
            )
        )
    return rows


def test_clean_laps_drops_pit_and_street_laps():
    rows = frame(
        [
            lap_row(lap_number=1),
            lap_row(lap_number=5),
            lap_row(lap_number=6, is_pit_in_lap=True),
            lap_row(lap_number=7, is_pit_out_lap=True),
            lap_row(lap_number=8, is_safety_car_lap=True),
            lap_row(lap_number=10, compound="UNKNOWN"),
        ]
    )
    cleaned = clean_laps(rows)
    assert cleaned["lap_number"].tolist() == [5]


def test_a_wet_session_is_dropped_entirely():
    rows = [lap_row(lap_number=lap, rainfall=0.0) for lap in range(2, 10)]
    rows.append(lap_row(lap_number=10, rainfall=1.0))
    rows.append(lap_row(lap_number=11, rainfall=1.0))
    # Two wet laps out of ten is above the ten percent threshold.
    assert clean_laps(frame(rows)).empty


def test_clean_laps_requires_tyre_age():
    rows = frame([lap_row(lap_number=5, tyre_age=None)])
    assert clean_laps(rows).empty


def test_fuel_mass_decreases_through_the_race():
    rows = frame([lap_row(lap_number=lap) for lap in [1, 25, 50]])
    masses = fuel_mass(rows).to_numpy()
    assert masses[0] == pytest.approx(110.0)
    assert masses[1] < masses[0]
    assert masses[2] < masses[1]
    assert masses[2] > 0


def test_fuel_effect_is_recovered_from_a_synthetic_race():
    rows = []
    for lap in range(2, 51):
        fuel = 110 * (50 - lap + 1) / 50
        rows.append(
            lap_row(lap_number=lap, tyre_age=min(lap - 1, 4), lap_duration=80.0 + 0.04 * fuel)
        )
    estimate = estimate_fuel_effect(clean_laps(frame(rows)))
    assert estimate == pytest.approx(0.04, abs=0.002)


def test_stint_delta_is_zero_on_fresh_tyres_and_grows_with_age():
    prepared = add_stint_delta(clean_laps(frame(stint_rows())))
    fresh = prepared[prepared["tyre_age"] <= 2][TARGET].mean()
    old = prepared[prepared["tyre_age"] >= 12][TARGET].mean()
    assert abs(fresh) < 0.05
    assert old > 0.5


def test_features_are_complete_and_finite():
    prepared = add_features(add_stint_delta(clean_laps(frame(stint_rows()))))
    matrix = feature_frame(prepared)
    assert list(matrix.columns) == FEATURES
    assert np.isfinite(matrix.to_numpy()).all()
    assert prepared["compound_medium"].eq(1).all()
    assert prepared["compound_soft"].eq(0).all()
