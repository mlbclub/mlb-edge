"""Artifact revisions for prediction-cache invalidation."""
from pathlib import Path
from .config import MODEL_FILE, TEAM_GAMES, PICK_RULES_FILE


def prediction_revision(paths=None):
    paths = (MODEL_FILE, TEAM_GAMES, PICK_RULES_FILE) if paths is None else paths
    return tuple((str(p), Path(p).stat().st_mtime_ns, Path(p).stat().st_size)
                 if Path(p).exists() else (str(p), None, None) for p in paths)
