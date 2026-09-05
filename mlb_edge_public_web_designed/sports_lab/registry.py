from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueConfig:
    key: str
    sport: str
    label: str
    label_ko: str
    odds_sport_key: str
    timezone: str
    outcome_type: str  # "two_way" or "three_way"
    primary_markets: tuple[str, ...]
    enabled: bool = False


LEAGUES: dict[str, LeagueConfig] = {
    "mlb": LeagueConfig(
        key="mlb", sport="baseball", label="MLB", label_ko="MLB",
        odds_sport_key="baseball_mlb", timezone="America/New_York",
        outcome_type="two_way", primary_markets=("h2h", "spreads", "totals"), enabled=True,
    ),
    "kbo": LeagueConfig(
        key="kbo", sport="baseball", label="KBO", label_ko="KBO",
        odds_sport_key="baseball_kbo", timezone="Asia/Seoul",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=True,
    ),
    "npb": LeagueConfig(
        key="npb", sport="baseball", label="NPB", label_ko="NPB",
        odds_sport_key="baseball_npb", timezone="Asia/Tokyo",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=True,
    ),
    "epl": LeagueConfig(
        key="epl", sport="soccer", label="Premier League", label_ko="프리미어리그",
        odds_sport_key="soccer_epl", timezone="Europe/London",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=False,
    ),
    "laliga": LeagueConfig(
        key="laliga", sport="soccer", label="La Liga", label_ko="라리가",
        odds_sport_key="soccer_spain_la_liga", timezone="Europe/Madrid",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=False,
    ),
    "bundesliga": LeagueConfig(
        key="bundesliga", sport="soccer", label="Bundesliga", label_ko="분데스리가",
        odds_sport_key="soccer_germany_bundesliga", timezone="Europe/Berlin",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=False,
    ),
    "seriea": LeagueConfig(
        key="seriea", sport="soccer", label="Serie A", label_ko="세리에 A",
        odds_sport_key="soccer_italy_serie_a", timezone="Europe/Rome",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=False,
    ),
    "ligue1": LeagueConfig(
        key="ligue1", sport="soccer", label="Ligue 1", label_ko="리그 1",
        odds_sport_key="soccer_france_ligue_one", timezone="Europe/Paris",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=False,
    ),
    "mls": LeagueConfig(
        key="mls", sport="soccer", label="MLS", label_ko="MLS",
        odds_sport_key="soccer_usa_mls", timezone="America/New_York",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=False,
    ),
    "kleague1": LeagueConfig(
        key="kleague1", sport="soccer", label="K League 1", label_ko="K리그1",
        odds_sport_key="soccer_korea_kleague1", timezone="Asia/Seoul",
        outcome_type="three_way", primary_markets=("h2h", "spreads", "totals"), enabled=False,
    ),
}


def get_league(key: str) -> LeagueConfig:
    normalized = str(key).strip().lower()
    if normalized not in LEAGUES:
        raise KeyError(f"지원하지 않는 리그입니다: {key}")
    return LEAGUES[normalized]


def league_keys(*, sport: str | None = None, enabled_only: bool = False) -> list[str]:
    rows = []
    for key, cfg in LEAGUES.items():
        if sport and cfg.sport != sport:
            continue
        if enabled_only and not cfg.enabled:
            continue
        rows.append(key)
    return rows
