from __future__ import annotations
import time
import requests

BASE = "https://statsapi.mlb.com/api/v1"


class MLBStatsAPI:
    def __init__(self, timeout=30, pause=0.08):
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "mlb-probability-model/3.0"})

    def get(self, path, params=None, retries=4):
        url = f"{BASE}/{path.lstrip('/')}"
        last = None
        for i in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                if self.pause:
                    time.sleep(self.pause)
                return r.json()
            except requests.RequestException as e:
                last = e
                time.sleep(1.2 * (i + 1))
        raise RuntimeError(f"MLB API request failed: {url}: {last}")

    def season_schedule(self, season, game_types="R"):
        return self.get("schedule", {
            "sportId": 1,
            "season": season,
            "gameType": game_types,
            "hydrate": "probablePitcher,venue",
        })

    def schedule_by_date(self, date):
        return self.get("schedule", {
            "sportId": 1,
            "date": date,
            "gameType": "R",
            "hydrate": "probablePitcher,venue",
        })

    def boxscore(self, game_pk: int):
        return self.get(f"game/{int(game_pk)}/boxscore")

    def game_feed(self, game_pk: int):
        return self.get(f"game/{int(game_pk)}/feed/live")

    def person(self, person_id: int):
        return self.get(f"people/{int(person_id)}")

    def venue(self, venue_id: int):
        return self.get(f"venues/{int(venue_id)}", {"hydrate": "location,fieldInfo,timezone"})
