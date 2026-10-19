"""Pit stop strategy enumeration.

The simulator works on degradation deltas rather than absolute times. For a
fixed remaining number of laps, fuel, track evolution and the car's base pace
are the same for every plan, so they cancel out of the comparison. A plan is
scored on its summed tyre degradation plus the pit losses it incurs.

Cost lookups are precomputed. A full season replay would otherwise make
millions of single row predictions.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import COMPOUNDS

MAX_TYRE_AGE = 55

# The 99th percentile of observed stint lengths in the 2024 season. Plans that
# run a tyre beyond this are not offered, because the model has almost no data
# there and no team would plan for it.
TYRE_LIFE = {"SOFT": 23, "MEDIUM": 31, "HARD": 41}


@dataclass(frozen=True)
class RaceContext:
    total_laps: int
    baseline: float
    track_temp: float
    air_temp: float
    pit_loss: float


@dataclass(frozen=True)
class Plan:
    stops: tuple[tuple[int, str], ...]
    cost: float
    stints: tuple[tuple[str, int, int], ...] = field(default=())

    @property
    def stop_laps(self) -> tuple[int, ...]:
        return tuple(lap for lap, _ in self.stops)

    def describe(self) -> str:
        if not self.stops:
            return "no further stops"
        parts = [f"lap {lap} to {compound.title()}" for lap, compound in self.stops]
        return ", ".join(parts)


def within_tyre_life(
    current_lap: int,
    current_compound: str,
    current_age: int,
    stops: Iterable[tuple[int, str]],
    total_laps: int,
    life: dict[str, int] | None = None,
) -> bool:
    """Check that no stint exceeds the observed tyre life for its compound."""
    life = life or TYRE_LIFE
    stops = list(stops)
    compound = current_compound
    age = current_age + 1
    start = current_lap + 1
    for lap, new_compound in stops:
        if age + (lap - start) > life.get(compound, MAX_TYRE_AGE):
            return False
        compound, age, start = new_compound, 0, lap + 1
    return age + (total_laps - start) <= life.get(compound, MAX_TYRE_AGE)


def validate_stops(
    stops: Iterable[tuple[int, str]],
    current_lap: int,
    current_compound: str,
    total_laps: int,
    min_stint: int = 4,
) -> bool:
    """Check a plan against basic sporting and mechanical constraints."""
    stops = list(stops)
    previous_lap = current_lap
    previous_compound = current_compound
    for lap, compound in stops:
        if compound not in COMPOUNDS or compound == previous_compound:
            return False
        if lap - previous_lap < min_stint:
            return False
        if total_laps - lap < min_stint:
            return False
        previous_lap = lap
        previous_compound = compound
    return True


class StrategySimulator:
    """Enumerate strategies using precomputed degradation costs."""

    def __init__(
        self,
        predictor: Callable[[str, int, int], float],
        context: RaceContext,
        max_age: int = MAX_TYRE_AGE,
    ) -> None:
        self.context = context
        self.max_age = max_age
        total = context.total_laps
        if hasattr(predictor, "table"):
            self._table = predictor.table(max_age, total)
        else:
            self._table = np.zeros((len(COMPOUNDS), max_age + 1, total))
            for c_index, compound in enumerate(COMPOUNDS):
                for age in range(max_age + 1):
                    for lap in range(1, total + 1):
                        self._table[c_index, age, lap - 1] = predictor(compound, age, lap)

        # Cumulative cost of a stint that starts on fresh tyres, indexed by
        # compound, first lap and last lap.
        self._fresh = np.zeros((len(COMPOUNDS), total + 2, total + 2))
        for c_index in range(len(COMPOUNDS)):
            for start in range(1, total + 1):
                ages = np.minimum(np.arange(0, total + 1 - start), max_age)
                laps = np.arange(start, total + 1)
                values = self._table[c_index, ages, laps - 1]
                self._fresh[c_index, start, start : total + 1] = np.cumsum(values)

    def delta(self, compound: str, age: int, lap: int) -> float:
        c_index = COMPOUNDS.index(compound)
        return float(self._table[c_index, min(age, self.max_age), lap - 1])

    def stint_cost(self, compound: str, start_age: int, first_lap: int, last_lap: int) -> float:
        if last_lap < first_lap:
            return 0.0
        c_index = COMPOUNDS.index(compound)
        if start_age == 0:
            return float(self._fresh[c_index, first_lap, last_lap])
        count = last_lap - first_lap + 1
        ages = np.minimum(start_age + np.arange(count), self.max_age)
        return float(self._table[c_index, ages, np.arange(first_lap, last_lap + 1) - 1].sum())

    def plan_cost(
        self,
        current_lap: int,
        current_compound: str,
        current_age: int,
        stops: tuple[tuple[int, str], ...],
    ) -> float:
        total = 0.0
        # The next lap is one lap older than the current state.
        compound, age, first_lap = current_compound, current_age + 1, current_lap + 1
        for lap, new_compound in stops:
            total += self.stint_cost(compound, age, first_lap, lap)
            total += self.context.pit_loss
            compound, age, first_lap = new_compound, 0, lap + 1
        total += self.stint_cost(compound, age, first_lap, self.context.total_laps)
        return total

    def enumerate_plans(
        self,
        current_lap: int,
        current_compound: str,
        current_age: int,
        max_stops: int = 2,
        min_stint: int = 4,
        require_two_compounds: bool = True,
        compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    ) -> list[Plan]:
        """Return every legal plan, cheapest first."""
        plans: list[Plan] = []
        horizon = self.context.total_laps

        zero = tuple()
        if within_tyre_life(current_lap, current_compound, current_age, zero, horizon):
            cost = self.plan_cost(current_lap, current_compound, current_age, zero)
            plans.append(Plan(stops=zero, cost=cost))

        for n_stops in range(1, max_stops + 1):
            for laps in itertools.combinations(range(current_lap + 1, horizon), n_stops):
                for chosen in itertools.product(compounds, repeat=n_stops):
                    stops = tuple(zip(laps, chosen, strict=False))
                    if not validate_stops(
                        stops, current_lap, current_compound, horizon, min_stint
                    ):
                        continue
                    if require_two_compounds and n_stops > 0:
                        used = {current_compound, *chosen}
                        if len(used) < 2:
                            continue
                    if not within_tyre_life(
                        current_lap, current_compound, current_age, stops, horizon
                    ):
                        continue
                    cost = self.plan_cost(current_lap, current_compound, current_age, stops)
                    plans.append(Plan(stops=stops, cost=cost))

        plans.sort(key=lambda plan: plan.cost)
        return plans

    def replay(
        self,
        current_lap: int,
        current_compound: str,
        current_age: int,
        stops: tuple[tuple[int, str], ...],
    ) -> pd.DataFrame:
        """Return a per lap breakdown for plotting."""
        rows = []
        compound, age = current_compound, current_age + 1
        stop_map = dict(stops)
        for lap in range(current_lap + 1, self.context.total_laps + 1):
            rows.append(
                {
                    "lap": lap,
                    "compound": compound,
                    "tyre_age": age,
                    "delta": self.delta(compound, age, lap),
                }
            )
            if lap in stop_map:
                rows[-1]["pit_loss"] = self.context.pit_loss
                compound, age = stop_map[lap], 0
            else:
                age += 1
        return pd.DataFrame(rows)


def plan_table(plans: list[Plan], top: int = 8) -> pd.DataFrame:
    """Format the best plans for printing or saving."""
    best = plans[0].cost if plans else 0.0
    rows = []
    for index, plan in enumerate(plans[:top], start=1):
        rows.append(
            {
                "rank": index,
                "plan": plan.describe(),
                "stops": len(plan.stops),
                "cost_s": round(plan.cost, 2),
                "gap_to_best_s": round(plan.cost - best, 2),
            }
        )
    return pd.DataFrame(rows)
