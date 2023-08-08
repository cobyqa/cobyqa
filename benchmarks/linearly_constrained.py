from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 5, 'adjacency linear')
    profiles(['cobyqa', 'lincoa'], ['COBYQA', 'LINCOA'])
