from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 10, 'unconstrained')
    profiles(['cobyqa-latest', 'cobyqa'], ['COBYQA Latest', 'COBYQA PyPI'], load=False)
