"""Client for the OpenF1 API.

The public API serves session metadata, lap times, stint information, pit stops,
weather and race control messages. Responses are cached as JSON on disk so that
building the dataset again does not hit the network.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.openf1.org/v1"
LOGGER = logging.getLogger(__name__)


class OpenF1Error(RuntimeError):
    """Raised when the API returns an error that retries cannot fix."""


class OpenF1Client:
    """Small wrapper around the OpenF1 REST endpoints.

    Parameters
    ----------
    cache_dir:
        Directory for cached JSON responses. A cache file is keyed by endpoint
        and query parameters.
    pause:
        Seconds to sleep between requests. Keeps the public API happy when a
        full season is being downloaded.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/openf1",
        pause: float = 0.35,
        timeout: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "f1-race-strategy/0.3"})

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        parts = [endpoint]
        for key in sorted(params):
            parts.append(f"{key}-{params[key]}")
        name = "_".join(parts).replace("/", "-")
        return self.cache_dir / f"{name}.json"

    def get(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        """Return a list of records for one endpoint."""
        cache_path = self._cache_path(endpoint, params)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        url = f"{BASE_URL}/{endpoint}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
                continue

            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, list):
                    raise OpenF1Error(f"Expected a list from {url}, got {type(payload).__name__}")
                tmp_path = cache_path.with_suffix(".tmp")
                with tmp_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                tmp_path.replace(cache_path)
                time.sleep(self.pause)
                return payload

            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", 5))
                LOGGER.warning("Rate limited on %s, waiting %.1fs", endpoint, wait)
                time.sleep(wait)
                continue

            if response.status_code == 422:
                raise OpenF1Error(
                    f"{url} rejected the query: {response.text[:200]}. "
                    "Narrow the request (for example by driver number)."
                )

            last_error = OpenF1Error(f"HTTP {response.status_code} for {url}")
            time.sleep(1.5 * (attempt + 1))

        raise OpenF1Error(f"Could not fetch {url} after {self.max_retries} attempts") from last_error

    def races(self, year: int) -> list[dict[str, Any]]:
        """Return race sessions for a season, ordered by start date."""
        sessions = self.get("sessions", year=year, session_name="Race")
        sessions = [row for row in sessions if not row.get("is_cancelled")]
        return sorted(sessions, key=lambda row: row["date_start"])

    def session_payload(self, session_key: int) -> dict[str, list[dict[str, Any]]]:
        """Download the endpoints needed to describe a race."""
        return {
            "laps": self.get("laps", session_key=session_key),
            "stints": self.get("stints", session_key=session_key),
            "pit": self.get("pit", session_key=session_key),
            "weather": self.get("weather", session_key=session_key),
            "drivers": self.get("drivers", session_key=session_key),
            "result": self.get("session_result", session_key=session_key),
            "race_control": self.get("race_control", session_key=session_key),
        }
