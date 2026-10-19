"""Command line interface.

Examples
--------
    f1strategy strategy --race 12 --driver VER --lap 20
    f1strategy strategy --race Silverstone --driver HAM --lap 30 --pit-loss 21
    f1strategy backtest --lap 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import backtest, plots
from .features import FUEL_CAPACITY_KG, FUEL_EFFECT_S_PER_KG
from .predictor import Conditions, DeltaPredictor
from .runtime import (
    ARTIFACT_DIR,
    build_simulator,
    default_model,
    find_race,
    read_laps,
    read_pit_losses,
    state_at,
)
from .strategy import plan_table


def _strategy_command(args: argparse.Namespace) -> int:
    laps = read_laps()
    pit_losses = read_pit_losses()
    race = find_race(laps, pit_losses, args.race)
    state = state_at(laps, race.session_key, args.driver, args.lap)
    model = default_model()
    simulator = build_simulator(state, race, model, pit_loss=args.pit_loss)
    plans = simulator.enumerate_plans(
        state["lap"],
        state["compound"],
        state["tyre_age"],
        max_stops=args.max_stops,
        min_stint=args.min_stint,
    )
    if not plans:
        print("No legal plan found from this state.")
        return 1

    print(f"Round {race.round}: {race.event_name} ({race.circuit}), {race.total_laps} laps")
    print(
        f"{state['driver']} on lap {state['lap']}, {state['compound'].title()} "
        f"at {state['tyre_age']} laps old"
    )
    print(
        f"Baseline pace {state['baseline']:.3f}s, pit loss {simulator.context.pit_loss:.1f}s, "
        f"track {state['track_temp']:.1f}C"
    )
    print()
    table = plan_table(plans, top=args.top)
    print(table.to_string(index=False))

    best = plans[0]
    future = range(state["lap"] + 1, race.total_laps + 1)
    fuel_time = FUEL_EFFECT_S_PER_KG * sum(
        FUEL_CAPACITY_KG * (race.total_laps - lap + 1) / race.total_laps for lap in future
    )
    estimated = state["baseline"] * (race.total_laps - state["lap"]) + fuel_time + best.cost
    print()
    print(f"Best plan: {best.describe()}")
    print(f"Estimated remaining race time: {estimated:.1f}s")

    if args.figure:
        path = Path(args.figure)
        plots.plot_strategy_map(
            simulator, state["lap"], state["compound"], state["tyre_age"], path
        )
        plots.plot_plan(
            simulator, plans, state["lap"], state["compound"], state["tyre_age"],
            path.with_name(path.stem + "_plans" + path.suffix),
        )
        print(f"Figures written to {path.parent}")
    return 0


def _backtest_command(args: argparse.Namespace) -> int:
    laps = read_laps()
    pit_losses = read_pit_losses()
    model = default_model()
    races = (
        laps[["session_key", "round", "circuit", "event_name", "total_laps"]]
        .drop_duplicates("session_key")
        .sort_values("round")
    )
    rows = []
    for _, race_row in races.iterrows():
        race = find_race(laps, pit_losses, str(int(race_row["round"])))
        sample = laps[laps["session_key"] == race.session_key].iloc[0]
        context = backtest.RaceContext(
            total_laps=race.total_laps,
            baseline=100.0,
            track_temp=float(sample["track_temp"]),
            air_temp=float(sample["air_temp"]),
            pit_loss=race.pit_loss,
        )
        predictor = DeltaPredictor(model, Conditions(total_laps=race.total_laps))
        rows.extend(
            backtest.replay_race(laps, race.session_key, predictor, context, lap=args.lap)
        )
    summary = backtest.summarise(rows)
    frame = pd.DataFrame([row.__dict__ for row in rows])
    output = ARTIFACT_DIR / f"backtest_lap{args.lap}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"Backtest at lap {args.lap}, {len(frame)} driver races")
    for key, value in summary.items():
        print(f"  {key:24s} {value:8.3f}")
    print(f"Details written to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1strategy",
        description="Lap time and pit strategy model for Formula 1 races",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    strategy = sub.add_parser("strategy", help="enumerate pit plans from a race state")
    strategy.add_argument("--race", required=True, help="round number, circuit or event name")
    strategy.add_argument("--driver", required=True, help="driver acronym or number")
    strategy.add_argument("--lap", type=int, required=True, help="current lap")
    strategy.add_argument("--max-stops", type=int, default=2)
    strategy.add_argument("--min-stint", type=int, default=4)
    strategy.add_argument("--pit-loss", type=float, default=None, help="override pit loss in seconds")
    strategy.add_argument("--top", type=int, default=8)
    strategy.add_argument("--figure", default=None, help="write strategy figures to this path")
    strategy.set_defaults(func=_strategy_command)

    bt = sub.add_parser("backtest", help="replay a season and measure strategy regret")
    bt.add_argument("--lap", type=int, default=30)
    bt.set_defaults(func=_backtest_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
