"""Prospective NPB V2 vs V4 validation.

Freeze files are immutable pregame snapshots. Settlement files are separate and may be
refreshed as official results become final. V2 remains the operating model; V4 is a
challenger only. This module compares W/D/L only because V4 did not change totals.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import npb
from . import npb_v2 as v2
from . import npb_v4 as v4
from .npb_details import GAMES_V2, collect_games, save_json

ROOT = npb.DATA_DIR / "prospective"
JST = npb.JST


def _now_naive() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(JST)).tz_localize(None)


def _classes(clf) -> list[str]:
    return [str(x) for x in clf.classes_]


def _probabilities(bundle, row: pd.Series) -> dict[str, float]:
    cols = bundle["features"]
    p = bundle["clf"].predict_proba(pd.DataFrame([row[cols].to_dict()]))[0]
    return {label: float(p[i]) for i, label in enumerate(_classes(bundle["clf"]))}


def _pick(probs: dict[str, float]) -> tuple[str, float]:
    side = max(("home", "draw", "away"), key=lambda x: probs.get(x, -1.0))
    return side, float(probs.get(side, np.nan))


def _eligible_dates(frame: pd.DataFrame, now: pd.Timestamp) -> list[pd.Timestamp]:
    f = frame.copy()
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f[f.status.eq("Scheduled") & (f.game_date > now)]
    dates = []
    for day, block in f.groupby(f.game_date.dt.normalize(), sort=True):
        if len(block) and block.game_date.min() > now:
            dates.append(pd.Timestamp(day))
    return dates


def resolve_target_date(v2_features: pd.DataFrame, requested: str | None = None, now: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now or _now_naive()
    frame = v2_features.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    if requested:
        target = pd.Timestamp(requested).normalize()
        block = frame[frame.game_date.dt.normalize().eq(target)]
        if block.empty:
            raise ValueError(f"No NPB games found for {target.date()}")
        starts = block.loc[block.status.eq("Scheduled"), "game_date"]
        if starts.empty:
            raise ValueError(f"No scheduled NPB games remain for {target.date()}")
        if starts.min() <= now:
            raise ValueError("Pregame freeze refused: at least one target-date game has already reached its scheduled start time")
        return target
    dates = _eligible_dates(frame, now)
    if not dates:
        raise ValueError("No future NPB game day is available for prospective freezing")
    return dates[0]


def _model_rows(frame: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    f = frame.copy()
    f["game_date"] = pd.to_datetime(f["game_date"])
    return f[f.game_date.dt.normalize().eq(target) & f.status.eq("Scheduled")].sort_values(["game_date", "game_id"])


def build_freeze_rows(target: pd.Timestamp, v2_features: pd.DataFrame, v4_features: pd.DataFrame,
                      v2_bundle: dict, v4_bundle: dict, frozen_at: pd.Timestamp | None = None) -> pd.DataFrame:
    frozen_at = frozen_at or _now_naive()
    a = _model_rows(v2_features, target).set_index("game_id")
    b = _model_rows(v4_features, target).set_index("game_id")
    ids = sorted(set(a.index) & set(b.index))
    if not ids:
        raise ValueError("No same-game V2/V4 prospective cohort found")
    rows = []
    for gid in ids:
        r2, r4 = a.loc[gid], b.loc[gid]
        for model_name, bundle, row in (("v2", v2_bundle, r2), ("v4", v4_bundle, r4)):
            probs = _probabilities(bundle, row)
            pick, conf = _pick(probs)
            rows.append({
                "target_date": str(target.date()), "frozen_at_jst": frozen_at.isoformat(),
                "game_id": gid, "game_date": str(pd.Timestamp(r2.game_date)),
                "away": str(r2.away), "home": str(r2.home), "model": model_name,
                "model_version": str(bundle.get("model_version", model_name)),
                "home_prob": probs.get("home"), "draw_prob": probs.get("draw"), "away_prob": probs.get("away"),
                "model_pick": pick, "model_confidence": conf,
                "source_commit": os.getenv("GITHUB_SHA", "local"),
            })
    return pd.DataFrame(rows).sort_values(["game_date", "game_id", "model"])


def freeze(target_date: str | None = None) -> tuple[Path, Path]:
    ROOT.mkdir(parents=True, exist_ok=True)
    f2 = pd.read_csv(v2.FEATURES, parse_dates=["game_date"])
    f4 = pd.read_csv(v4.FEATURES, parse_dates=["game_date"])
    target = resolve_target_date(f2, target_date)
    csv_path = ROOT / f"{target.date()}_v2_vs_v4_frozen.csv"
    manifest_path = ROOT / f"{target.date()}_manifest.json"
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Prospective freeze already exists for {target.date()}; immutable snapshot will not be overwritten")
    now = _now_naive()
    rows = build_freeze_rows(target, f2, f4, joblib.load(v2.MODEL), joblib.load(v4.MODEL), now)
    first_pitch = pd.to_datetime(rows.game_date).min()
    if first_pitch <= now:
        raise ValueError("Pregame freeze refused because the first scheduled pitch is not in the future")
    rows.to_csv(csv_path, index=False)
    manifest = {
        "status": "FROZEN", "purpose": "prospective_npb_v2_vs_v4_wdl",
        "target_date": str(target.date()), "frozen_at_jst": now.isoformat(),
        "first_pitch_jst": str(first_pitch), "games": int(rows.game_id.nunique()),
        "models": {"v2": "operating", "v4": "challenger_not_promoted"},
        "comparison": "W/D/L argmax accuracy; confidence buckets 55% and 60%; same frozen cohort",
        "notes": ["2026 historical monitoring is retrospective and is not treated as a new holdout.",
                  "Freeze files are immutable. Settlement is written separately.",
                  "V4 totals are unchanged from V2, so prospective V2-vs-V4 comparison is W/D/L only."],
        "source_commit": os.getenv("GITHUB_SHA", "local"),
    }
    save_json(manifest_path, manifest)
    print(f"[NPB prospective] frozen {rows.game_id.nunique()} games for {target.date()} before {first_pitch}", flush=True)
    return csv_path, manifest_path


def _result_label(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if float(home_score) > float(away_score):
        return "home"
    if float(home_score) < float(away_score):
        return "away"
    return "draw"


def settle(target_date: str, refresh_official: bool = True) -> tuple[Path, Path]:
    ROOT.mkdir(parents=True, exist_ok=True)
    target = pd.Timestamp(target_date).normalize()
    frozen_path = ROOT / f"{target.date()}_v2_vs_v4_frozen.csv"
    if not frozen_path.exists():
        raise FileNotFoundError(f"No frozen prospective cohort for {target.date()}")
    frozen = pd.read_csv(frozen_path)
    games = collect_games() if refresh_official else pd.read_csv(GAMES_V2, parse_dates=["game_date"])
    official = games.set_index("game_id")
    settled = frozen.copy()
    statuses, actuals, hits, hscores, ascores = [], [], [], [], []
    for r in settled.itertuples(index=False):
        if r.game_id not in official.index:
            statuses.append("PENDING"); actuals.append(None); hits.append(None); hscores.append(None); ascores.append(None); continue
        g = official.loc[r.game_id]
        if isinstance(g, pd.DataFrame):
            g = g.iloc[-1]
        final = str(g.status) == "Final" and pd.notna(g.home_score) and pd.notna(g.away_score)
        hscores.append(g.home_score); ascores.append(g.away_score)
        if not final:
            statuses.append("PENDING"); actuals.append(None); hits.append(None); continue
        actual = _result_label(g.home_score, g.away_score)
        statuses.append("FINAL"); actuals.append(actual); hits.append(bool(r.model_pick == actual))
    settled["settlement_status"] = statuses
    settled["home_score"] = hscores
    settled["away_score"] = ascores
    settled["actual_result"] = actuals
    settled["hit"] = hits
    out_path = ROOT / f"{target.date()}_settled.csv"
    settled.to_csv(out_path, index=False)

    summary = {"target_date": str(target.date()), "prospective": True, "models": {}, "head_to_head": {}}
    final = settled[settled.settlement_status.eq("FINAL")].copy()
    for model in ("v2", "v4"):
        m = final[final.model.eq(model)]
        block = {"games": int(len(m)), "hits": int(m.hit.fillna(False).sum()),
                 "accuracy": float(m.hit.mean()) if len(m) else None}
        for t in (0.55, 0.60):
            x = m[m.model_confidence >= t]
            key = str(int(t*100))
            block[f"conf_{key}_games"] = int(len(x))
            block[f"conf_{key}_hits"] = int(x.hit.fillna(False).sum())
            block[f"conf_{key}_accuracy"] = float(x.hit.mean()) if len(x) else None
        summary["models"][model] = block
    if len(final):
        p = final.pivot_table(index="game_id", columns="model", values="hit", aggfunc="last")
        both = p.dropna(subset=[c for c in ("v2", "v4") if c in p.columns]) if set(("v2", "v4")).issubset(p.columns) else pd.DataFrame()
        if len(both):
            summary["head_to_head"] = {
                "same_games": int(len(both)), "v2_hits": int(both.v2.sum()), "v4_hits": int(both.v4.sum()),
                "v2_only_correct": int(((both.v2 == True) & (both.v4 == False)).sum()),
                "v4_only_correct": int(((both.v4 == True) & (both.v2 == False)).sum()),
            }
    summary["pending_rows"] = int((settled.settlement_status != "FINAL").sum())
    summary_path = ROOT / f"{target.date()}_summary.json"
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return out_path, summary_path
