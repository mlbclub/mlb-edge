from __future__ import annotations
import os
import re
import statistics
from collections import Counter, defaultdict
import requests

ODDS_BASE = "https://api.the-odds-api.com/v4"


def _norm_team(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    aliases = {
        "oakland athletics": "athletics",
        "a s": "athletics",
        "athletics": "athletics",
        "la dodgers": "los angeles dodgers",
        "la angels": "los angeles angels",
    }
    return aliases.get(s, s)


def decimal_from_price(price, odds_format: str = "decimal") -> float:
    x = float(price)
    if odds_format == "decimal":
        return x
    if x > 0:
        return 1.0 + x / 100.0
    return 1.0 + 100.0 / abs(x)


def american_from_decimal(decimal_odds: float | None):
    if decimal_odds is None:
        return None
    d = float(decimal_odds)
    if d <= 1:
        return None
    if d >= 2.0:
        return int(round((d - 1.0) * 100))
    return int(round(-100 / (d - 1.0)))


def no_vig_two_way(a_decimal: float, b_decimal: float):
    ai, bi = 1.0 / float(a_decimal), 1.0 / float(b_decimal)
    z = ai + bi
    return ai / z, bi / z, z - 1.0


def no_vig_many(decimal_prices: list[float]):
    implied = [1.0 / float(x) for x in decimal_prices]
    z = sum(implied)
    if z <= 0:
        return [None for _ in implied], None
    return [x / z for x in implied], z - 1.0


class OddsAPI:
    def __init__(self, api_key: str | None = None, timeout=30):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ODDS_API_KEY가 없습니다. setup_odds_key_windows.bat 또는 환경변수 ODDS_API_KEY를 설정하세요."
            )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "sports-lab-probability-model/1.0"})

    def current_sport(self, sport_key: str, regions="us", markets="h2h,spreads,totals", odds_format="decimal"):
        url = f"{ODDS_BASE}/sports/{sport_key}/odds"
        r = self.session.get(url, params={
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        }, timeout=self.timeout)
        r.raise_for_status()
        return r.json(), self._quota(r)

    def historical_sport(self, sport_key: str, snapshot_iso: str, regions="us", markets="h2h,spreads,totals", odds_format="decimal"):
        url = f"{ODDS_BASE}/historical/sports/{sport_key}/odds"
        r = self.session.get(url, params={
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
            "date": snapshot_iso,
        }, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        events = payload.get("data", payload if isinstance(payload, list) else [])
        return events, payload, self._quota(r)

    def current_mlb(self, regions="us", markets="h2h,spreads,totals", odds_format="decimal"):
        return self.current_sport("baseball_mlb", regions=regions, markets=markets, odds_format=odds_format)

    def historical_mlb(self, snapshot_iso: str, regions="us", markets="h2h,spreads,totals", odds_format="decimal"):
        return self.historical_sport(
            "baseball_mlb", snapshot_iso,
            regions=regions, markets=markets, odds_format=odds_format,
        )

    @staticmethod
    def _quota(r):
        return {
            "remaining": r.headers.get("x-requests-remaining"),
            "used": r.headers.get("x-requests-used"),
            "last": r.headers.get("x-requests-last"),
        }


def find_event(events: list[dict], home_team: str, away_team: str):
    hk, ak = _norm_team(home_team), _norm_team(away_team)
    for e in events:
        if _norm_team(e.get("home_team")) == hk and _norm_team(e.get("away_team")) == ak:
            return e
    return None


def _market(book: dict, key: str):
    return next((m for m in book.get("markets", []) if m.get("key") == key), None)


def _best_price(rows):
    if not rows:
        return None, None
    best = max(rows, key=lambda r: r[0])
    return float(best[0]), best[1]


def summarize_event_three_way(event: dict | None, odds_format="decimal") -> dict:
    """Summarize 1X2-style h2h markets without changing MLB two-way behavior.

    Used by KBO/NPB when a draw price is quoted and by soccer. Totals/spreads are
    still obtained from ``summarize_event`` and merged into this result.
    """
    if not event:
        return {}
    base = summarize_event(event, odds_format=odds_format)
    home = event.get("home_team")
    away = event.get("away_team")
    hk, ak = _norm_team(home), _norm_team(away)
    rows = []
    hbest, dbest, abest = [], [], []
    for book in event.get("bookmakers", []):
        m = _market(book, "h2h")
        if not m:
            continue
        prices = {_norm_team(o.get("name")): decimal_from_price(o.get("price"), odds_format) for o in m.get("outcomes", [])}
        draw_key = next((k for k in prices if k in {"draw", "tie"}), None)
        if hk not in prices or ak not in prices or draw_key is None:
            continue
        hd, dd, ad = prices[hk], prices[draw_key], prices[ak]
        if min(hd, dd, ad) <= 1:
            continue
        probs, vig = no_vig_many([hd, dd, ad])
        title = book.get("title") or book.get("key") or "unknown"
        rows.append((probs[0], probs[1], probs[2], vig))
        hbest.append((hd, title)); dbest.append((dd, title)); abest.append((ad, title))
    if rows:
        base.update({
            "home_market_novig": statistics.median(r[0] for r in rows),
            "draw_market_novig": statistics.median(r[1] for r in rows),
            "away_market_novig": statistics.median(r[2] for r in rows),
            "market_vig": statistics.median(r[3] for r in rows),
            "moneyline_books": len(rows),
        })
        base["home_ml_odds"], base["home_ml_book"] = _best_price(hbest)
        base["draw_ml_odds"], base["draw_ml_book"] = _best_price(dbest)
        base["away_ml_odds"], base["away_ml_book"] = _best_price(abest)
    return base


def summarize_event(event: dict | None, odds_format="decimal") -> dict:
    if not event:
        return {}
    home = event.get("home_team")
    away = event.get("away_team")
    hk, ak = _norm_team(home), _norm_team(away)
    out = {
        "odds_event_id": event.get("id"),
        "odds_commence_time": event.get("commence_time"),
        "home_team_odds": home,
        "away_team_odds": away,
    }

    # Moneyline consensus and best price.
    ml_rows = []
    hbest, abest = [], []
    for book in event.get("bookmakers", []):
        m = _market(book, "h2h")
        if not m:
            continue
        prices = {_norm_team(o.get("name")): decimal_from_price(o.get("price"), odds_format) for o in m.get("outcomes", [])}
        if hk in prices and ak in prices and prices[hk] > 1 and prices[ak] > 1:
            hp, ap, vig = no_vig_two_way(prices[hk], prices[ak])
            title = book.get("title") or book.get("key") or "unknown"
            ml_rows.append((hp, ap, vig))
            hbest.append((prices[hk], title)); abest.append((prices[ak], title))
    if ml_rows:
        out.update({
            "home_market_novig": statistics.median(r[0] for r in ml_rows),
            "away_market_novig": statistics.median(r[1] for r in ml_rows),
            "market_vig": statistics.median(r[2] for r in ml_rows),
            "moneyline_books": len(ml_rows),
        })
        out["home_ml_odds"], out["home_ml_book"] = _best_price(hbest)
        out["away_ml_odds"], out["away_ml_book"] = _best_price(abest)

    # Totals: select the most common line, then consensus no-vig probabilities and best prices at that line.
    total_by_line = defaultdict(list)
    for book in event.get("bookmakers", []):
        m = _market(book, "totals")
        if not m:
            continue
        outcomes = m.get("outcomes", [])
        overs = [o for o in outcomes if str(o.get("name", "")).lower() == "over"]
        unders = [o for o in outcomes if str(o.get("name", "")).lower() == "under"]
        for ov in overs:
            point = ov.get("point")
            un = next((u for u in unders if u.get("point") == point), None)
            if point is None or un is None:
                continue
            od = decimal_from_price(ov.get("price"), odds_format)
            ud = decimal_from_price(un.get("price"), odds_format)
            if od <= 1 or ud <= 1:
                continue
            op, up, vig = no_vig_two_way(od, ud)
            title = book.get("title") or book.get("key") or "unknown"
            total_by_line[float(point)].append((op, up, vig, od, ud, title))
    if total_by_line:
        counts = {k: len(v) for k, v in total_by_line.items()}
        max_count = max(counts.values())
        candidates = [k for k, v in counts.items() if v == max_count]
        line = min(candidates, key=lambda x: abs(x - statistics.median(total_by_line.keys())))
        rows = total_by_line[line]
        out.update({
            "total_line": line,
            "over_market_novig": statistics.median(r[0] for r in rows),
            "under_market_novig": statistics.median(r[1] for r in rows),
            "total_vig": statistics.median(r[2] for r in rows),
            "total_books": len(rows),
        })
        out["over_odds"], out["over_book"] = _best_price([(r[3], r[5]) for r in rows])
        out["under_odds"], out["under_book"] = _best_price([(r[4], r[5]) for r in rows])

    # Standard spread consensus plus exact -1.5 side probabilities/prices if quoted.
    spread_lines = defaultdict(list)
    minus_rows = {"home": [], "away": []}
    for book in event.get("bookmakers", []):
        m = _market(book, "spreads")
        if not m:
            continue
        outcomes = m.get("outcomes", [])
        by_team = {_norm_team(o.get("name")): o for o in outcomes}
        if hk not in by_team or ak not in by_team:
            continue
        ho, ao = by_team[hk], by_team[ak]
        hpnt, apnt = ho.get("point"), ao.get("point")
        if hpnt is None or apnt is None:
            continue
        hd = decimal_from_price(ho.get("price"), odds_format)
        ad = decimal_from_price(ao.get("price"), odds_format)
        if hd <= 1 or ad <= 1:
            continue
        hp, ap, vig = no_vig_two_way(hd, ad)
        title = book.get("title") or book.get("key") or "unknown"
        spread_lines[float(hpnt)].append((hp, ap, vig, hd, ad, title, float(apnt)))
        if abs(float(hpnt) + 1.5) < 1e-9 and abs(float(apnt) - 1.5) < 1e-9:
            minus_rows["home"].append((hp, vig, hd, title))
        if abs(float(apnt) + 1.5) < 1e-9 and abs(float(hpnt) - 1.5) < 1e-9:
            minus_rows["away"].append((ap, vig, ad, title))

    if spread_lines:
        counts = {k: len(v) for k, v in spread_lines.items()}
        home_point = max(counts, key=counts.get)
        rows = spread_lines[home_point]
        out.update({
            "home_spread_line": home_point,
            "away_spread_line": -home_point,
            "home_spread_market_novig": statistics.median(r[0] for r in rows),
            "away_spread_market_novig": statistics.median(r[1] for r in rows),
            "spread_books": len(rows),
        })
        out["home_spread_odds"], out["home_spread_book"] = _best_price([(r[3], r[5]) for r in rows])
        out["away_spread_odds"], out["away_spread_book"] = _best_price([(r[4], r[5]) for r in rows])

    for side in ("home", "away"):
        rows = minus_rows[side]
        if rows:
            out[f"{side}_minus_1_5_market_novig"] = statistics.median(r[0] for r in rows)
            out[f"{side}_minus_1_5_odds"], out[f"{side}_minus_1_5_book"] = _best_price([(r[2], r[3]) for r in rows])
            out[f"{side}_minus_1_5_books"] = len(rows)
    return out
