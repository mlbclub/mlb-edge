"""User-designated six-input moneyline baseline; no fitted feature selection."""
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = ['elo_home_prob', 'diff_win_r20', 'diff_run_diff_r20',
            'diff_bat_ops_r20', 'diff_bullpen_era_r20', 'diff_bullpen_whip_r20']
C = .01
NAME = 'strength_C0.01'


def make_model(c=C):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, random_state=42))
