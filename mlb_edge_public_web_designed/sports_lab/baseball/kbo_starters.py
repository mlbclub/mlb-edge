from __future__ import annotations

import json
from typing import Any

import requests

from sports_lab.baseball.kbo import _team_name

GAME_LIST_URL = "https://www.koreabaseball.com/ws/Main.asmx/GetKboGameList"


def parse_first_json(text: str) -> Any:
    """Parse the first valid JSON value from a response that may contain trailing HTML."""
    raw = (text or "").lstrip("\ufeff \t\r\n")
    if not raw:
        return {}

    # Fast path after trimming a common KBO trailing error page.
    lowered = raw.lower()
    cut_points = [i for token in ("<!doctype", "<html") if (i := lowered.find(token)) >= 0]
    trimmed = raw[: min(cut_points)] if cut_points else raw
    trimmed = trimmed.strip()
    if trimmed:
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            pass

    # KBO occasionally returns a valid JSON document followed by unrelated bytes.
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(raw)
    return value


def unwrap_payload(value: Any) -> Any:
    """Unwrap ASP.NET-style ``d`` payloads, including JSON encoded strings."""
    if isinstance(value, dict) and "d" in value:
        value = value["d"]
    if isinstance(value, str):
        try:
            return parse_first_json(value)
        except Exception:
            return {}
    return value


def _walk_game_dicts(value: Any):
    if isinstance(value, dict):
        keys = {str(k).upper() for k in value}
        if ({"AWAY_NM", "HOME_NM"} <= keys) or ({"AWAY", "HOME"} <= keys):
            yield value
        for child in value.values():
            yield from _walk_game_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_game_dicts(child)


def extract_starters(payload: Any) -> dict[tuple[str, str], tuple[str, str]]:
    """Extract matchup -> (away starter, home starter) without fabricating missing names."""
    payload = unwrap_payload(payload)
    out: dict[tuple[str, str], tuple[str, str]] = {}

    for game in _walk_game_dicts(payload):
        # Existing endpoint keys first, then conservative fallbacks seen on KBO payloads.
        away_raw = game.get("AWAY_NM") or game.get("away_nm") or game.get("AWAY") or game.get("away") or ""
        home_raw = game.get("HOME_NM") or game.get("home_nm") or game.get("HOME") or game.get("home") or ""
        away = _team_name(str(away_raw))
        home = _team_name(str(home_raw))
        if not away or not home:
            continue

        away_starter = str(
            game.get("T_PIT_P_NM")
            or game.get("AWAY_PIT_P_NM")
            or game.get("away_starter")
            or ""
        ).strip()
        home_starter = str(
            game.get("B_PIT_P_NM")
            or game.get("HOME_PIT_P_NM")
            or game.get("home_starter")
            or ""
        ).strip()

        # Do not inject a half-known matchup into starter features.
        if away_starter and home_starter:
            out[(away, home)] = (away_starter, home_starter)
    return out


def gamecenter_starters(date_yyyymmdd: str) -> dict[tuple[str, str], tuple[str, str]]:
    """Fetch current-day KBO starters with lenient response parsing."""
    try:
        response = requests.post(
            GAME_LIST_URL,
            json={"leId": 1, "srId": 0, "date": date_yyyymmdd},
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "Referer": "https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx",
                "User-Agent": "Mozilla/5.0 SPORTS-LAB/2.1",
            },
            timeout=15,
        )
        response.raise_for_status()
        parsed = parse_first_json(response.text)
        starters = extract_starters(parsed)
        print(f"[KBO V2 starters] found {len(starters)} matchup(s)")
        for (away, home), (away_sp, home_sp) in sorted(starters.items()):
            print(f"[KBO V2 starters] {away}({away_sp}) @ {home}({home_sp})")
        return starters
    except Exception as exc:
        print(f"[KBO V2 starters] lookup failed: {exc}")
        return {}
