from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 2, 'unconstrained')
    profiles(['cobyqa', 'newuoa'])
