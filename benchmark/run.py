import argparse

from profiles import Profiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the performance and data profile comparisons.")
    parser.add_argument("--cf-old", default=False, type=bool, help="compare with the penultimate version")
    args = parser.parse_args()

    profiles = Profiles(1, 10, "unconstrained")
    profiles(["cobyqa", "newuoa"], load=False)
    if args.cf_old:
        profiles(["cobyqa", "cobyqa-old"], load=False)
    del profiles

    profiles = Profiles(1, 10, "bound")
    profiles(["cobyqa", "bobyqa"], load=False)
    if args.cf_old:
        profiles(["cobyqa", "cobyqa-old"], load=False)
    del profiles

    profiles = Profiles(1, 10, "adjacency linear")
    profiles(["cobyqa", "lincoa"], load=False)
    if args.cf_old:
        profiles(["cobyqa", "cobyqa-old"], load=False)
    del profiles

    profiles = Profiles(1, 10, "quadratic other")
    profiles(["cobyqa", "cobyla"], load=False)
    if args.cf_old:
        profiles(["cobyqa", "cobyqa-old"], load=False)
    del profiles
