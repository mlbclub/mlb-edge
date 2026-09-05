import argparse

from sports_lab.baseball.npb_v2 import run_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached', action='store_true', help='Use existing official game/detail cache')
    parser.add_argument('--no-predict', action='store_true', help='Evaluate without fetching odds or replacing today files')
    args = parser.parse_args()
    run_pipeline(collect=not args.cached, predict=not args.no_predict)
