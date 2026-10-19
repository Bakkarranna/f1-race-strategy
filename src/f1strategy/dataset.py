"""Build a lap level dataset from OpenF1 session data.

Each row is one lap by one driver. Pit in and pit out laps are flagged rather
than dropped so the raw file stays useful for pit loss analysis. Filtering for
model training lives in :mod:`f1strategy.features`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from .openf1 import OpenF1Client

LOGGER = logging.getLogger(__name__)

# A lap this much slower than the driver's median is treated as an outlier
# (safety car, red flag, spin, off track, heavy traffic).
SLOW_LAP_RATIO = 1.05


@dataclass(frozen=True)
class RaceInfo:
    session_key: int
    year: int
    round_number: int
    event_name: str
    circuit: str
    country: str
    date_start: str


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _weather_at(weather: list[dict[str, Any]], when: datetime | None) -> dict[str, Any]:
    """Return the most recent weather observation before a lap start."""
    if not weather:
        return {}
    if when is None:
        return weather[0]
    best = weather[0]
    best_time = _parse_time(best["date"])
    for row in weather[1:]:
        row_time = _parse_time(row["date"])
        if row_time is None:
            continue
        if row_time <= when and (best_time is None or row_time >= best_time):
            best, best_time = row, row_time
    return best


def _safety_car_laps(messages: list[dict[str, Any]]) -> set[int]:
    """Return lap numbers affected by a safety car, virtual safety car or red flag."""
    affected: set[int] = set()
    active_from: int | None = None
    for row in sorted(messages, key=lambda item: item.get("date", "")):
        text = (row.get("message") or "").upper()
        lap = row.get("lap_number")
        if lap is None:
            continue
        starts = (
            "SAFETY CAR DEPLOYED" in text
            or "VIRTUAL SAFETY CAR DEPLOYED" in text
            or "RED FLAG" in text
        )
        if starts:
            active_from = lap if active_from is None else min(active_from, lap)
            continue
        ends = (
            "TRACK CLEAR" in text
            or "SAFETY CAR IN THIS LAP" in text
            or "VIRTUAL SAFETY CAR ENDING" in text
            or "GREEN LIGHT" in text
        )
        if ends and active_from is not None:
            affected.update(range(active_from, lap + 1))
            active_from = None
    if active_from is not None:
        affected.update(range(active_from, active_from + 5))
    return affected


def build_race_frame(
    session_key: int,
    payload: dict[str, list[dict[str, Any]]],
    race: RaceInfo,
) -> pd.DataFrame:
    """Turn the seven OpenF1 payloads for one race into a lap frame."""
    laps = pd.DataFrame(payload["laps"])
    if laps.empty:
        return pd.DataFrame()

    stints = pd.DataFrame(payload["stints"])
    pits = pd.DataFrame(payload["pit"])
    drivers = pd.DataFrame(payload["drivers"])
    weather = payload["weather"]
    result = pd.DataFrame(payload["result"])

    laps["date_start"] = pd.to_datetime(laps["date_start"], utc=True, format="ISO8601")
    laps = laps.dropna(subset=["lap_duration"])

    team_by_driver = dict(zip(drivers["driver_number"], drivers["team_name"], strict=False))
    acronym_by_driver = dict(zip(drivers["driver_number"], drivers["name_acronym"], strict=False))
    if not result.empty and "number_of_laps" in result.columns:
        total_laps = int(result["number_of_laps"].max())
    else:
        total_laps = int(laps["lap_number"].max())

    pit_in_laps = set(zip(pits.get("driver_number", []), pits.get("lap_number", []), strict=False))
    pit_lane = dict(zip(zip(pits.get("driver_number", []), pits.get("lap_number", []), strict=False),
                        pits.get("lane_duration", []), strict=False))
    sc_laps = _safety_car_laps(payload["race_control"])

    rows: list[dict[str, Any]] = []
    for driver_number, driver_laps in laps.groupby("driver_number"):
        driver_laps = driver_laps.sort_values("lap_number")
        median = driver_laps["lap_duration"].median()
        driver_stints = stints[stints["driver_number"] == driver_number]

        for _, lap in driver_laps.iterrows():
            lap_number = int(lap["lap_number"])
            stint = driver_stints[
                (driver_stints["lap_start"] <= lap_number)
                & (driver_stints["lap_end"] >= lap_number)
            ]
            stint = stint.iloc[0] if len(stint) else None
            compound = stint["compound"] if stint is not None else None
            tyre_age = None
            stint_number = None
            if stint is not None:
                tyre_age = lap_number - int(stint["lap_start"]) + int(stint["tyre_age_at_start"])
                stint_number = int(stint["stint_number"])

            observation = _weather_at(weather, lap["date_start"].to_pydatetime())
            is_slow = bool(median > 0 and lap["lap_duration"] > SLOW_LAP_RATIO * median)

            rows.append(
                {
                    "year": race.year,
                    "round": race.round_number,
                    "event_name": race.event_name,
                    "circuit": race.circuit,
                    "country": race.country,
                    "session_key": session_key,
                    "driver_number": int(driver_number),
                    "driver": acronym_by_driver.get(int(driver_number), str(driver_number)),
                    "team": team_by_driver.get(int(driver_number)),
                    "lap_number": lap_number,
                    "total_laps": total_laps,
                    "lap_duration": float(lap["lap_duration"]),
                    "sector_1": lap.get("duration_sector_1"),
                    "sector_2": lap.get("duration_sector_2"),
                    "sector_3": lap.get("duration_sector_3"),
                    "speed_trap": lap.get("st_speed"),
                    "compound": compound,
                    "tyre_age": tyre_age,
                    "stint_number": stint_number,
                    "is_pit_out_lap": bool(lap.get("is_pit_out_lap", False)),
                    "is_pit_in_lap": (int(driver_number), lap_number) in pit_in_laps,
                    "pit_lane_duration": pit_lane.get((int(driver_number), lap_number)),
                    "is_safety_car_lap": lap_number in sc_laps,
                    "is_slow_lap": is_slow,
                    "air_temp": observation.get("air_temperature"),
                    "track_temp": observation.get("track_temperature"),
                    "humidity": observation.get("humidity"),
                    "rainfall": observation.get("rainfall"),
                    "wind_speed": observation.get("wind_speed"),
                    "lap_start": lap["date_start"],
                }
            )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    return frame


def build_season(
    client: OpenF1Client,
    year: int,
    rounds: list[int] | None = None,
) -> pd.DataFrame:
    """Download a season and concatenate the lap frames."""
    races = client.races(year)
    frames: list[pd.DataFrame] = []
    for index, session in enumerate(races, start=1):
        if rounds is not None and index not in rounds:
            continue
        info = RaceInfo(
            session_key=int(session["session_key"]),
            year=year,
            round_number=index,
            event_name=session.get("location", ""),
            circuit=session.get("circuit_short_name", ""),
            country=session.get("country_name", ""),
            date_start=session.get("date_start", ""),
        )
        LOGGER.info("Fetching %s %s (session %s)", year, info.event_name, info.session_key)
        payload = client.session_payload(info.session_key)
        frame = build_race_frame(info.session_key, payload, info)
        if frame.empty:
            LOGGER.warning("No usable laps for session %s", info.session_key)
            continue
        frames.append(frame)

    if not frames:
        raise RuntimeError("No laps were collected")
    return pd.concat(frames, ignore_index=True)
