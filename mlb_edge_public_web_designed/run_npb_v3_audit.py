"""Import the model under its stable module name so joblib is portable."""
from sports_lab.baseball.npb_v3 import run_audit

if __name__ == '__main__':
    run_audit()
