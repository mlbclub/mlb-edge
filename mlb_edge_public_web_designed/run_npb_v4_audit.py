import argparse
from sports_lab.baseball.npb_v4 import run_audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    run_audit(collect=parser.parse_args().collect)
