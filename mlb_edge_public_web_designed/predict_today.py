from __future__ import annotations
from datetime import date
import pandas as pd
from mlb_model.live import predict_date

if __name__ == "__main__":
    games, _ = predict_date(str(date.today()), save=True)
    rows=[]
    for g in games:
        rows.append({
            "away":g["away"],"home":g["home"],
            "away_win":g.get("away_model"),"home_win":g.get("home_model"),
            "total_line":g.get("total_line"),"under":g.get("under_prob"),"over":g.get("over_prob"),
            "away_-1.5":g.get("away_minus_1_5"),"home_-1.5":g.get("home_minus_1_5"),
            "recommendation":g.get("recommendation"),"prob":g.get("recommendation_prob"),
            "edge":g.get("recommendation_edge"),"ev":g.get("recommendation_ev"),"odds":g.get("recommendation_odds"),
        })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
