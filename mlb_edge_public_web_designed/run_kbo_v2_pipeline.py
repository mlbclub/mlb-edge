from sports_lab.baseball import kbo_v2
from sports_lab.baseball.kbo_starters import gamecenter_starters


# Patch the V2 runtime hook without changing the historical KBO feature builder.
kbo_v2._gamecenter_starters = gamecenter_starters


if __name__ == "__main__":
    kbo_v2.run_pipeline()
