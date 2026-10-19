"""Replay historical races and compare the optimal plan with what happened.

This is not a claim that the model would have beaten every strategist. It
measures one thing: given the pace model fitted on other races, how much time
did the actual plan leave on the table relative to the best plan the model can
see at the same point in the race.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import COMPOUNDS, FUEL_EFFECT_S_PER_KG, fuel_mass
from .strategy import Plan, RaceContext, StrategySimulator, validate_stops, within_tyre_life

LOGGER = logging.getLogger(__name__)


@dataclass
class ReplayRow:
    round: int
    circuit: str
    driver: str
    lap: int
    compound: str
    tyre_age: int
    best_cost: float
    actual_cost: float
    best_plan: str
    actual_plan: str
    best_stops: int
    actual_stops: int
    valid_actual: bool
    clean_after: bool


def state_at(raw: pd.DataFrame, session_key: int, driver_number: int, lap: int) -> dict | None:
    """Return the tyre state of a driver at the end of a given lap."""
    rows = raw[
        (raw["session_key"] == session_key)
        & (raw["driver_number"] == driver_number)
        & (raw["lap_number"] <= lap)
    ].sort_values("lap_number")
    rows = rows[rows["compound"].isin(COMPOUNDS)]
    if rows.empty:
        return None
    last = rows.iloc[-1]
    clean = rows[
        ~rows["is_pit_in_lap"]
        & ~rows["is_pit_out_lap"]
        & ~rows["is_slow_lap"]
        & ~rows["is_safety_car_lap"]
    ]
    if clean.empty:
        clean = rows
    corrected = clean["lap_duration"] - FUEL_EFFECT_S_PER_KG * fuel_mass(clean)
    return {
        "lap": int(last["lap_number"]),
        "compound": last["compound"],
        "tyre_age": int(last["tyre_age"]),
        "baseline": float(corrected.quantile(0.02)),
        "total_laps": int(last["total_laps"]),
    }


def actual_plan(
    raw: pd.DataFrame,
    session_key: int,
    driver_number: int,
    from_lap: int,
    total_laps: int,
) -> tuple[tuple[int, str], ...]:
    """Reconstruct the stops a driver made after a given lap."""
    rows = raw[
        (raw["session_key"] == session_key)
        & (raw["driver_number"] == driver_number)
        & (raw["lap_number"] > from_lap)
        & raw["is_pit_in_lap"]
    ].sort_values("lap_number")
    stops = []
    for _, row in rows.iterrows():
        lap = int(row["lap_number"])
        next_lap = raw[
            (raw["session_key"] == session_key)
            & (raw["driver_number"] == driver_number)
            & (raw["lap_number"] == lap + 1)
        ]
        if next_lap.empty or pd.isna(next_lap["compound"].iloc[0]):
            continue
        new_compound = str(next_lap["compound"].iloc[0])
        if new_compound not in COMPOUNDS:
            return None
        if lap >= total_laps - 1:
            continue
        stops.append((lap, new_compound))
    return tuple(stops)


def replay_race(
    raw: pd.DataFrame,
    session_key: int,
    predictor,
    context: RaceContext,
    lap: int,
    min_stint: int = 4,
) -> list[ReplayRow]:
    """Evaluate every driver in one race at a fixed decision lap."""
    rows = raw[raw["session_key"] == session_key]
    results: list[ReplayRow] = []
    simulator = StrategySimulator(predictor, context)
    for driver_number in sorted(rows["driver_number"].unique()):
        state = state_at(raw, session_key, driver_number, lap)
        if state is None or state["compound"] is None:
            continue
        if state["lap"] >= context.total_laps - 3:
            continue

        plans = simulator.enumerate_plans(
            state["lap"], state["compound"], state["tyre_age"], min_stint=min_stint
        )
        stops = actual_plan(raw, session_key, driver_number, state["lap"], context.total_laps)
        if stops is None:
            continue
        valid = validate_stops(
            stops, state["lap"], state["compound"], context.total_laps, min_stint
        ) and within_tyre_life(
            state["lap"], state["compound"], state["tyre_age"], stops, context.total_laps
        )
        actual_cost = simulator.plan_cost(
            state["lap"], state["compound"], state["tyre_age"], stops
        )
        best = plans[0] if plans else None
        if best is None:
            continue
        driver_name = str(rows[rows["driver_number"] == driver_number]["driver"].iloc[0])
        later = rows[(rows["driver_number"] == driver_number) & (rows["lap_number"] > state["lap"])]
        clean_after = not bool(later["is_safety_car_lap"].any())
        results.append(
            ReplayRow(
                round=int(rows["round"].iloc[0]),
                circuit=str(rows["circuit"].iloc[0]),
                driver=driver_name,
                lap=state["lap"],
                compound=state["compound"],
                tyre_age=state["tyre_age"],
                best_cost=best.cost,
                actual_cost=actual_cost,
                best_plan=best.describe(),
                actual_plan=Plan(stops=stops, cost=actual_cost).describe(),
                best_stops=len(best.stops),
                actual_stops=len(stops),
                valid_actual=valid,
                clean_after=clean_after,
            )
        )
    return results


def summarise(results: list[ReplayRow]) -> dict[str, float]:
    if not results:
        return {}
    valid = [row for row in results if row.valid_actual]
    all_regret = np.array([row.actual_cost - row.best_cost for row in valid])

    # Safety car stops cost far less than a green flag stop, and the simulator
    # only has a green flag pit loss. The subset that stayed green is reported
    # separately instead of fudging a second pit loss model.
    clean = [row for row in valid if row.clean_after]
    clean_regret = np.array([row.actual_cost - row.best_cost for row in clean])
    agree = np.mean([row.best_stops == row.actual_stops for row in clean]) if clean else 0.0
    return {
        "drivers_evaluated": float(len(results)),
        "valid_plans": float(len(valid)),
        "median_regret_all_s": float(np.median(all_regret)) if len(all_regret) else 0.0,
        "mean_regret_all_s": float(np.mean(all_regret)) if len(all_regret) else 0.0,
        "green_flag_plans": float(len(clean)),
        "median_regret_green_s": float(np.median(clean_regret)) if len(clean_regret) else 0.0,
        "mean_regret_green_s": float(np.mean(clean_regret)) if len(clean_regret) else 0.0,
        "p90_regret_green_s": float(np.percentile(clean_regret, 90)) if len(clean_regret) else 0.0,
        "stop_count_agreement": float(agree),
    }
