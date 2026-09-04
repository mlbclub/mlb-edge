from mlb_model.collect import collect_games
from mlb_model.enrich import enrich_games
from mlb_model.features import build_features
from mlb_model.train import train
from mlb_model.config import DATA_DIR

if __name__ == "__main__":
    collect_games()
    enrich_games()
    build_features()
    train()
    report = DATA_DIR / "robust_ablation_report.csv"
    if not report.is_file():
        raise RuntimeError("Training did not produce robust_ablation_report.csv")
    print(f"[saved robust selection diagnostics] {report}")
