from __future__ import annotations

import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .api import MLBStatsAPI
from .config import PITCHER_META, VENUE_META, PARK_FACTORS, GAME_CONTEXT

SAVANT_PARK_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
PREVIOUS_WEATHER_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]


def _norm_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def get_pitcher_meta(api: MLBStatsAPI, pitcher_ids) -> pd.DataFrame:
    """Cache pitcher handedness from MLB Stats API.

    Handedness is static player metadata and is safe to use in historical
    pre-game features. Missing/new pitchers are appended to the cache.
    """
    ids = sorted({int(float(x)) for x in pitcher_ids if x is not None and not pd.isna(x)})
    cache = _read_csv(PITCHER_META)
    if len(cache):
        cache["pitcher_id"] = pd.to_numeric(cache["pitcher_id"], errors="coerce")
    known = set(pd.to_numeric(cache.get("pitcher_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    rows = []
    for pid in ids:
        if pid in known:
            continue
        try:
            payload = api.person(pid)
            person = (payload.get("people") or [{}])[0]
            hand = ((person.get("pitchHand") or {}).get("code") or "").upper()
            rows.append({
                "pitcher_id": pid,
                "pitcher_name": person.get("fullName"),
                "pitch_hand": hand if hand in {"R", "L"} else None,
            })
        except Exception as e:
            print(f"[pitcher meta warning] {pid}: {e}")
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache = cache.drop_duplicates("pitcher_id", keep="last").sort_values("pitcher_id")
        PITCHER_META.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(PITCHER_META, index=False)
    return cache


def get_venue_meta(api: MLBStatsAPI, venue_ids) -> pd.DataFrame:
    ids = sorted({int(float(x)) for x in venue_ids if x is not None and not pd.isna(x)})
    cache = _read_csv(VENUE_META)
    if len(cache):
        cache["venue_id"] = pd.to_numeric(cache["venue_id"], errors="coerce")
    known = set(pd.to_numeric(cache.get("venue_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    rows = []
    for vid in ids:
        if vid in known:
            continue
        try:
            payload = api.venue(vid)
            venue = (payload.get("venues") or [{}])[0]
            loc = venue.get("location") or {}
            coords = loc.get("defaultCoordinates") or {}
            tz = venue.get("timeZone") or venue.get("timezone") or {}
            field = venue.get("fieldInfo") or {}
            rows.append({
                "venue_id": vid,
                "venue": venue.get("name"),
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "timezone": tz.get("id") or tz.get("name") or "UTC",
                "roof_type": field.get("roofType"),
                "turf_type": field.get("turfType"),
            })
        except Exception as e:
            print(f"[venue meta warning] {vid}: {e}")
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache = cache.drop_duplicates("venue_id", keep="last").sort_values("venue_id")
        VENUE_META.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(VENUE_META, index=False)
    return cache


def _extract_savant_json(text: str):
    patterns = [
        r"\bdata\s*=\s*(\[\{.*?\}\])\s*;",
        r"\bvar\s+data\s*=\s*(\[\{.*?\}\])\s*;",
        r"\blet\s+data\s*=\s*(\[\{.*?\}\])\s*;",
        r"\bconst\s+data\s*=\s*(\[\{.*?\}\])\s*;",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return []


def fetch_park_factors(year: int) -> pd.DataFrame:
    """Fetch Baseball Savant 3-year rolling park factors for one completed year."""
    r = requests.get(
        SAVANT_PARK_URL,
        params={
            "type": "year",
            "year": int(year),
            "batSide": "",
            "stat": "index_wOBA",
            "condition": "All",
            "rolling": 3,
            "parks": "mlb",
        },
        headers={"User-Agent": "sports-lab-pregame-context/1.0"},
        timeout=30,
    )
    r.raise_for_status()
    data = _extract_savant_json(r.text)
    rows = []
    for x in data:
        venue = x.get("venue_name") or x.get("venue") or x.get("name")
        if not venue:
            continue
        woba = x.get("index_woba") or x.get("index_wOBA") or x.get("index")
        run = x.get("index_runs") or x.get("index_run") or x.get("index_R") or x.get("index_r")
        try:
            woba = float(woba)
        except Exception:
            woba = np.nan
        try:
            run = float(run)
        except Exception:
            run = np.nan
        rows.append({
            "factor_year": int(year),
            "venue": venue,
            "venue_key": _norm_name(venue),
            "park_factor": woba,
            "park_run_factor": run,
        })
    return pd.DataFrame(rows)


def get_park_factor_cache(years) -> pd.DataFrame:
    years = sorted({int(y) for y in years if int(y) >= 2015})
    cache = _read_csv(PARK_FACTORS)
    have = set(pd.to_numeric(cache.get("factor_year", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    new = []
    for year in years:
        if year in have:
            continue
        try:
            df = fetch_park_factors(year)
            if len(df):
                new.append(df)
            else:
                print(f"[park factor warning] no rows for {year}")
        except Exception as e:
            print(f"[park factor warning] {year}: {e}")
    if new:
        cache = pd.concat([cache] + new, ignore_index=True)
        cache = cache.drop_duplicates(["factor_year", "venue_key"], keep="last")
        PARK_FACTORS.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(PARK_FACTORS, index=False)
    return cache


def _weather_request(url: str, lat: float, lon: float, start_date: str, end_date: str, previous_day1=False):
    suffix = "_previous_day1" if previous_day1 else ""
    hourly = [f"{v}{suffix}" for v in WEATHER_VARS]
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly),
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=45, headers={"User-Agent": "sports-lab-pregame-context/1.0"})
    r.raise_for_status()
    payload = r.json()
    times = (payload.get("hourly") or {}).get("time") or []
    out = pd.DataFrame({"time": pd.to_datetime(times, utc=True, errors="coerce")})
    for base, key in zip(WEATHER_VARS, hourly):
        vals = (payload.get("hourly") or {}).get(key) or [np.nan] * len(out)
        out[base] = pd.to_numeric(pd.Series(vals), errors="coerce")
    return out.dropna(subset=["time"]).sort_values("time")


def _nearest_weather(hourly: pd.DataFrame, game_dt: pd.Timestamp) -> dict:
    if hourly is None or not len(hourly):
        return {}
    dt = pd.Timestamp(game_dt)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    idx = (hourly["time"] - dt).abs().idxmin()
    row = hourly.loc[idx]
    return {
        "weather_temp_c": row.get("temperature_2m"),
        "weather_humidity": row.get("relative_humidity_2m"),
        "weather_precip_mm": row.get("precipitation"),
        "weather_wind_kmh": row.get("wind_speed_10m"),
        "weather_wind_dir": row.get("wind_direction_10m"),
    }


def _day_flag(game_dt: pd.Timestamp, timezone_name: str | None) -> float:
    dt = pd.Timestamp(game_dt)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    try:
        local = dt.tz_convert(ZoneInfo(timezone_name or "UTC"))
    except Exception:
        local = dt
    return float(10 <= local.hour < 17)


def _park_for_game(cache: pd.DataFrame, venue_name: str, factor_year: int) -> dict:
    if cache is None or not len(cache):
        return {"park_factor": 100.0, "park_run_factor": 100.0}
    key = _norm_name(venue_name)
    x = cache[(pd.to_numeric(cache["factor_year"], errors="coerce").eq(int(factor_year))) & cache["venue_key"].eq(key)]
    if not len(x):
        # Venue naming sometimes changes slightly; use conservative token match.
        toks = set(key.split())
        if toks:
            score = cache["venue_key"].fillna("").map(lambda s: len(toks.intersection(set(str(s).split()))))
            yy = pd.to_numeric(cache["factor_year"], errors="coerce").eq(int(factor_year))
            if (score[yy] >= max(1, len(toks) - 1)).any():
                x = cache[yy & score.eq(score[yy].max())]
    if not len(x):
        return {"park_factor": 100.0, "park_run_factor": 100.0}
    row = x.iloc[0]
    pf = pd.to_numeric(pd.Series([row.get("park_factor")]), errors="coerce").iloc[0]
    rf = pd.to_numeric(pd.Series([row.get("park_run_factor")]), errors="coerce").iloc[0]
    return {
        "park_factor": float(pf) if np.isfinite(pf) else 100.0,
        "park_run_factor": float(rf) if np.isfinite(rf) else (float(pf) if np.isfinite(pf) else 100.0),
    }


def build_game_context(games: pd.DataFrame, api: MLBStatsAPI | None = None) -> pd.DataFrame:
    """Build leakage-resistant historical pregame weather/park context.

    Weather uses Open-Meteo's previous_day1 series: the forecast that existed
    24 hours before each valid game time. Park factor uses the prior completed
    season's three-year rolling Savant factor.
    """
    api = api or MLBStatsAPI()
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"], utc=True, errors="coerce")
    g = g[g["game_pk"].notna() & g["game_date"].notna()].copy()

    cache = _read_csv(GAME_CONTEXT)
    done = set(pd.to_numeric(cache.get("game_pk", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    missing = g[~g["game_pk"].astype(int).isin(done)].copy()
    if not len(missing):
        return cache

    venues = get_venue_meta(api, missing["venue_id"].dropna().tolist())
    venue_map = venues.set_index("venue_id").to_dict("index") if len(venues) else {}
    park_cache = get_park_factor_cache({int(s) - 1 for s in missing["season"].dropna().astype(int)})

    rows = []
    for (venue_id, season), grp in missing.groupby(["venue_id", "season"], dropna=True):
        vid = int(venue_id)
        meta = venue_map.get(vid, {})
        lat, lon = meta.get("latitude"), meta.get("longitude")
        hourly = pd.DataFrame()
        if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
            start = grp["game_date"].dt.date.min().isoformat()
            end = grp["game_date"].dt.date.max().isoformat()
            try:
                hourly = _weather_request(PREVIOUS_WEATHER_URL, float(lat), float(lon), start, end, previous_day1=True)
            except Exception as e:
                print(f"[weather history warning] venue={vid} season={season}: {e}")

        for r in grp.itertuples(index=False):
            venue_name = getattr(r, "venue", None) or meta.get("venue") or ""
            rec = {
                "game_pk": int(r.game_pk),
                "venue_id": vid,
                "context_source": "previous_day1",
                "is_day_game": _day_flag(r.game_date, meta.get("timezone")),
            }
            rec.update(_park_for_game(park_cache, venue_name, int(r.season) - 1))
            rec.update(_nearest_weather(hourly, r.game_date))
            rows.append(rec)

    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache = cache.drop_duplicates("game_pk", keep="last").sort_values("game_pk")
        GAME_CONTEXT.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(GAME_CONTEXT, index=False)
    return cache


def live_game_context(api: MLBStatsAPI, game: dict, game_dt) -> dict:
    venue = game.get("venue") or {}
    vid = venue.get("id")
    season = pd.Timestamp(game_dt).year
    meta_df = get_venue_meta(api, [vid] if vid else [])
    meta = {}
    if vid and len(meta_df):
        x = meta_df[pd.to_numeric(meta_df["venue_id"], errors="coerce").eq(int(vid))]
        if len(x):
            meta = x.iloc[0].to_dict()

    park_cache = get_park_factor_cache([season - 1])
    venue_name = venue.get("name") or meta.get("venue") or ""
    out = _park_for_game(park_cache, venue_name, season - 1)
    out["is_day_game"] = _day_flag(pd.Timestamp(game_dt), meta.get("timezone"))

    lat, lon = meta.get("latitude"), meta.get("longitude")
    if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
        dt = pd.Timestamp(game_dt)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        try:
            hourly = _weather_request(
                FORECAST_WEATHER_URL,
                float(lat), float(lon),
                dt.date().isoformat(), dt.date().isoformat(),
                previous_day1=False,
            )
            out.update(_nearest_weather(hourly, dt))
        except Exception as e:
            print(f"[weather live warning] venue={vid}: {e}")
    return out
