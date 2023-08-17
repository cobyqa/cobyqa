from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 10, 'unconstrained')
    profiles(['cobyqa', 'newuoa'], ['COBYQA', 'NEWUOA'], load=False)
