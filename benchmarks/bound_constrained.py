from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 5, 'bound')
    profiles(['cobyqa', 'bobyqa'], ['COBYQA', 'BOBYQA'], load=False)
