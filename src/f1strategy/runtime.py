"""Loading helpers shared by the CLI and the analysis scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import FUEL_EFFECT_S_PER_KG, fuel_mass
from .model import load as load_model
from .predictor import Conditions, DeltaPredictor
from .strategy import RaceContext

DATA_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts")


@dataclass
class RaceView:
    session_key: int
    round: int
    circuit: str
    event_name: str
    total_laps: int
    pit_loss: float


def read_laps(path: str | Path = DATA_DIR / "laps_2024.csv") -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["compound"] = frame["compound"].fillna("")
    return frame


def read_features(path: str | Path = DATA_DIR / "laps_features.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def read_pit_losses(path: str | Path = DATA_DIR / "pit_losses.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def race_list(laps: pd.DataFrame, pit_losses: pd.DataFrame) -> pd.DataFrame:
    races = (
        laps.groupby(["session_key", "round", "circuit", "event_name"])
        .agg(total_laps=("total_laps", "max"))
        .reset_index()
        .sort_values("round")
    )
    races = races.merge(pit_losses[["circuit", "pit_loss"]], on="circuit", how="left")
    races["pit_loss"] = races["pit_loss"].fillna(pit_losses["pit_loss"].median())
    return races


def find_race(laps: pd.DataFrame, pit_losses: pd.DataFrame, query: str) -> RaceView:
    races = race_list(laps, pit_losses)
    text = query.strip().lower()
    if text.isdigit():
        matches = races[races["round"] == int(text)]
    else:
        mask = races["circuit"].str.lower().str.contains(text) | races[
            "event_name"
        ].str.lower().str.contains(text)
        matches = races[mask]
    if matches.empty:
        available = ", ".join(f"{row['round']}:{row['circuit']}" for _, row in races.iterrows())
        raise SystemExit(f"No race matched {query!r}. Available: {available}")
    row = matches.iloc[0]
    return RaceView(
        session_key=int(row["session_key"]),
        round=int(row["round"]),
        circuit=str(row["circuit"]),
        event_name=str(row["event_name"]),
        total_laps=int(row["total_laps"]),
        pit_loss=float(row["pit_loss"]),
    )


def state_at(laps: pd.DataFrame, session_key: int, driver: str, lap: int) -> dict:
    rows = laps[(laps["session_key"] == session_key) & (laps["lap_number"] <= lap)]
    if driver.isdigit():
        rows = rows[rows["driver_number"] == int(driver)]
    else:
        rows = rows[rows["driver"].str.upper() == driver.upper()]
    rows = rows[rows["compound"].isin(["SOFT", "MEDIUM", "HARD"])].sort_values("lap_number")
    if rows.empty:
        raise SystemExit(f"No tyre state found for {driver} at lap {lap}")
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
    baseline = float(corrected.quantile(0.02))
    session = laps[laps["session_key"] == session_key]
    return {
        "driver": str(last["driver"]),
        "driver_number": int(last["driver_number"]),
        "lap": int(last["lap_number"]),
        "compound": str(last["compound"]),
        "tyre_age": int(last["tyre_age"]),
        "baseline": baseline,
        "track_temp": float(last["track_temp"]),
        "air_temp": float(last["air_temp"]),
        "track_temp_delta": float(last["track_temp"] - session["track_temp"].median()),
        "air_temp_delta": float(last["air_temp"] - session["air_temp"].median()),
        "total_laps": int(last["total_laps"]),
    }


def build_simulator(state: dict, race: RaceView, model, pit_loss: float | None = None):
    from .strategy import StrategySimulator

    conditions = Conditions(
        total_laps=state["total_laps"],
        track_temp_delta=state.get("track_temp_delta", 0.0),
        air_temp_delta=state.get("air_temp_delta", 0.0),
    )
    context = RaceContext(
        total_laps=state["total_laps"],
        baseline=state["baseline"],
        track_temp=state["track_temp"],
        air_temp=state["air_temp"],
        pit_loss=race.pit_loss if pit_loss is None else pit_loss,
    )
    return StrategySimulator(DeltaPredictor(model, conditions), context)


def load_metrics(path: str | Path = ARTIFACT_DIR / "metrics.json") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_model():
    return load_model(ARTIFACT_DIR / "lap_delta_model.joblib")
