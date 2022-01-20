import numpy as np

# from .._build_utils import numpy_nodepr_api


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('linalg', parent_package, top_path)

    config.add_extension(
        '_bvcs',
        sources=['_bvcs.pyx'],
        include_dirs=[np.get_include()],
        # **numpy_nodepr_api,
    )

    config.add_extension(
        '_bvlag',
        sources=['_bvlag.pyx'],
        include_dirs=[np.get_include()],
        # **numpy_nodepr_api,
    )

    config.add_extension(
        '_bvtcg',
        sources=['_bvtcg.pyx'],
        include_dirs=[np.get_include()],
        # **numpy_nodepr_api,
    )

    config.add_extension(
        '_cpqp',
        sources=['_cpqp.pyx'],
        include_dirs=[np.get_include()],
        # **numpy_nodepr_api,
    )

    config.add_extension(
        '_lctcg',
        sources=['_lctcg.pyx'],
        include_dirs=[np.get_include()],
        # **numpy_nodepr_api,
    )

    config.add_extension(
        '_nnls',
        sources=['_nnls.pyx'],
        include_dirs=[np.get_include()],
        # **numpy_nodepr_api,
    )

    config.add_data_dir('tests')

    config.add_extension(
        '_utils',
        sources=['_utils.pyx'],
        include_dirs=[np.get_include()],
        # **numpy_nodepr_api,
    )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
