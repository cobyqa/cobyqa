# import numpy as np

from cobyqa._build_utils import npy_nodepr_api  # noqa


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('linalg', parent_package, top_path)

    # config.add_extension(
    #     'nnls',
    #     sources=['nnls.pyx'],
    #     include_dirs=[np.get_include()],
    #     **npy_nodepr_api,
    # )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
