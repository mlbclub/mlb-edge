from mlb_model.collect import collect_games
from mlb_model.enrich import enrich_games
from mlb_model.features import build_features
from mlb_model.train import train

if __name__ == "__main__":
    collect_games()
    enrich_games()
    build_features()
    train()
