from cobyqa._build_utils import numpy_nodepr_api  # noqa


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('linalg', parent_package, top_path)

    # config.add_extension(
    #     'base',
    #     sources=['base.pyx'],
    #     include_dirs=[get_numpy_include_dirs()],
    #     **numpy_nodepr_api,
    # )

    config.add_data_dir('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
