from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 10, 'adjacency linear')
    profiles(['cobyqa', 'lincoa'], ['COBYQA', 'LINCOA'], load=False)
