import argparse

from profiles import Profiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the performance and data profile comparisons.')
    parser.add_argument('--cf-old', default=False, type=bool, help='compare with the penultimate version')
    args = parser.parse_args()

    profiles = Profiles(1, 10, 'unconstrained')
    profiles(['cobyqa', 'newuoa'])
    if args.cf_old:
        profiles(['cobyqa', 'cobyqa-old'])
    del profiles

    profiles = Profiles(1, 10, 'bound')
    profiles(['cobyqa', 'bobyqa'])
    if args.cf_old:
        profiles(['cobyqa', 'cobyqa-old'])
    del profiles

    profiles = Profiles(1, 10, 'adjacency linear')
    profiles(['cobyqa', 'lincoa'])
    if args.cf_old:
        profiles(['cobyqa', 'cobyqa-old'])
    del profiles

    profiles = Profiles(1, 10, 'quadratic other')
    profiles(['cobyqa', 'cobyla'])
    if args.cf_old:
        profiles(['cobyqa', 'cobyqa-old'])
    del profiles
