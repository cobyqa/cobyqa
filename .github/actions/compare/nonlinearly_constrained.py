from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 10, 'quadratic other')
    profiles(['cobyqa-latest', 'cobyqa'], ['COBYQA Latest', 'COBYQA PyPI'], [{'radius_final': 0.0}, {'rhoend': 0.0}])
