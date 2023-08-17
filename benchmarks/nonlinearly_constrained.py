from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 10, 'quadratic other')
    profiles(['cobyqa', 'cobyla'], ['COBYQA', 'COBYLA'], load=False)
