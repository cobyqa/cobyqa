from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 5, 'quadratic other')
    profiles(['cobyqa', 'cobyla'], ['COBYQA', 'COBYLA'], load=False)
