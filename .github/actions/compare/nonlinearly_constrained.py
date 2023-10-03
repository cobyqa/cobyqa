from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 10, 'quadratic other')
    profiles(['cobyqa-latest', 'cobyqa'], ['COBYQA Latest', 'COBYQA PyPI'], [{'radius_final': 1e-12}, {'rhoend': 1e-12}])
