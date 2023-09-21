from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 10, 'bound')
    profiles(['cobyqa-latest', 'cobyqa'], ['COBYQA Latest', 'COBYQA PyPI'])
