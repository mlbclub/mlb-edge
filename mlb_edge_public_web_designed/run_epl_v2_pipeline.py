from __future__ import annotations

import json
from sports_lab.soccer.epl_v2 import collect_games, build_features, train_model, predict_current


def main():
    games = collect_games()
    features = build_features(games)
    bundle, report = train_model(features)
    current, top10, quota = predict_current(games=games, bundle=bundle)
    print(f"[EPL V2] current={len(current)} top={len(top10)} quota={quota}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
