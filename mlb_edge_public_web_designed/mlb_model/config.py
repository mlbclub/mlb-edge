from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
BOX_DIR = DATA_DIR / "boxscores"
HIST_ODDS_DIR = DATA_DIR / "historical_odds_cache"

RAW_GAMES = DATA_DIR / "mlb_games_2024_2026.csv"
ENRICHED_GAMES = DATA_DIR / "mlb_games_enriched.csv"
TEAM_GAMES = DATA_DIR / "team_game_stats.csv"
FEATURES = DATA_DIR / "features_v3.csv"
MODEL_FILE = MODEL_DIR / "market_model_v3.joblib"
PICK_RULES_FILE = MODEL_DIR / "pick_rules.json"
HIST_ODDS_FILE = DATA_DIR / "historical_odds.csv"
CURRENT_ODDS_HISTORY = DATA_DIR / "current_odds_history.csv"
OOF_PREDICTIONS = DATA_DIR / "walkforward_predictions.csv"
BACKTEST_FILE = DATA_DIR / "backtest_results.csv"
LIVE_PREDICTIONS = DATA_DIR / "today_market_predictions.csv"

# V7 pregame-context caches. These are intentionally stored as ordinary CSVs so
# GitHub Actions can reuse them instead of repeatedly calling public APIs.
PITCHER_META = DATA_DIR / "pitcher_meta.csv"
VENUE_META = DATA_DIR / "venue_meta.csv"
PARK_FACTORS = DATA_DIR / "park_factors_context.csv"
WEATHER_CONTEXT = DATA_DIR / "weather_context.csv"
GAME_CONTEXT = DATA_DIR / "game_pregame_context.csv"

SEASONS = [2024, 2025, 2026]
SPORT_ID = 1
GAME_TYPES = "R"
WINDOWS = (5, 10, 20, 30)
STARTER_WINDOWS = (3, 5, 10)
