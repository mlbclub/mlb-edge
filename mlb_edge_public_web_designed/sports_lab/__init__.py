"""SPORTS LAB multi-sport expansion package.

The existing ``mlb_model`` package remains the production MLB engine. New leagues
are added here first so MLB can keep running while the shared architecture is
validated league by league.
"""

from .registry import LEAGUES, LeagueConfig, get_league

__all__ = ["LEAGUES", "LeagueConfig", "get_league"]
