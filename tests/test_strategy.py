from __future__ import annotations

import numpy as np
import pytest

from f1strategy.strategy import (
    COMPOUNDS,
    TYRE_LIFE,
    RaceContext,
    StrategySimulator,
    plan_table,
    validate_stops,
    within_tyre_life,
)


def dummy_predictor(compound: str, age: int, lap: int) -> float:
    compound_offset = {"SOFT": 0.0, "MEDIUM": 0.2, "HARD": 0.4}[compound]
    return compound_offset + 0.01 * age + 0.001 * lap


def context(pit_loss: float = 20.0, total_laps: int = 50) -> RaceContext:
    return RaceContext(
        total_laps=total_laps,
        baseline=90.0,
        track_temp=35.0,
        air_temp=25.0,
        pit_loss=pit_loss,
    )


def test_plan_cost_matches_manual_sum():
    simulator = StrategySimulator(dummy_predictor, context())
    stops = ((20, "HARD"),)
    expected = 0.0
    for lap, age in zip(range(11, 21), range(6, 16), strict=False):
        expected += dummy_predictor("MEDIUM", age, lap)
    expected += 20.0
    for lap, age in zip(range(21, 51), range(0, 30), strict=False):
        expected += dummy_predictor("HARD", age, lap)
    assert simulator.plan_cost(10, "MEDIUM", 5, stops) == pytest.approx(expected)


def test_no_stop_plan_runs_to_the_end():
    simulator = StrategySimulator(dummy_predictor, context(pit_loss=0.0))
    cost = simulator.plan_cost(0, "SOFT", 0, tuple())
    # With age 0 at lap 0, lap 1 is the first lap on one lap old rubber.
    expected = sum(dummy_predictor("SOFT", age, lap) for lap, age in zip(range(1, 51), range(1, 51), strict=False))
    assert cost == pytest.approx(expected)


def test_validate_stops_rejects_illegal_plans():
    assert validate_stops(((20, "HARD"),), 10, "MEDIUM", 50)
    assert not validate_stops(((20, "MEDIUM"),), 10, "MEDIUM", 50)
    assert not validate_stops(((12, "HARD"),), 10, "MEDIUM", 50)  # too short
    assert not validate_stops(((48, "HARD"),), 10, "MEDIUM", 50)  # too near the end
    assert not validate_stops(((20, "SOFT"), (40, "SOFT")), 10, "MEDIUM", 50)


def test_enumeration_is_sorted_and_respects_limits():
    simulator = StrategySimulator(dummy_predictor, context(total_laps=20))
    plans = simulator.enumerate_plans(4, "MEDIUM", 5, max_stops=2)
    costs = [plan.cost for plan in plans]
    assert costs == sorted(costs)
    assert max(len(plan.stops) for plan in plans) == 2
    assert min(len(plan.stops) for plan in plans) == 0
    for plan in plans:
        stops = plan.stops
        assert all(compound in COMPOUNDS for _, compound in stops)
        assert within_tyre_life(4, "MEDIUM", 5, stops, 20)


def test_tyre_life_constraint_rejects_long_stints():
    assert within_tyre_life(10, "MEDIUM", 5, ((30, "HARD"),), 50)
    assert not within_tyre_life(10, "SOFT", 5, tuple(), 50)
    assert within_tyre_life(10, "MEDIUM", 5, ((20, "HARD"),), 50)
    assert not within_tyre_life(10, "MEDIUM", 5, ((20, "SOFT"),), 50)
    assert TYRE_LIFE["HARD"] > TYRE_LIFE["SOFT"]


def test_pit_loss_shifts_every_plan_by_its_stop_count():
    cheap = StrategySimulator(dummy_predictor, context(pit_loss=0.0))
    expensive = StrategySimulator(dummy_predictor, context(pit_loss=5.0))
    plans = ((20, "HARD"), (35, "SOFT"))
    difference = expensive.plan_cost(10, "MEDIUM", 5, plans) - cheap.plan_cost(10, "MEDIUM", 5, plans)
    assert difference == pytest.approx(10.0)


def test_plan_table_reports_gap_to_best():
    simulator = StrategySimulator(dummy_predictor, context())
    plans = simulator.enumerate_plans(10, "MEDIUM", 5)
    table = plan_table(plans, top=3)
    assert table.iloc[0]["gap_to_best_s"] == pytest.approx(0.0)
    assert table["cost_s"].is_monotonic_increasing
    assert np.all(table["gap_to_best_s"] >= 0)
