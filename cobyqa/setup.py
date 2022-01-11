import sys

from cobyqa._build_utils import cythonize_extensions  # noqa


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cobyqa', parent_package, top_path)

    config.add_subpackage('_build_utils')
    config.add_subpackage('linalg')
    config.add_subpackage('utils')
    config.add_data_dir('tests')

    # Skip cythonization when creating a source distribution.
    if 'sdist' not in sys.argv:
        cythonize_extensions(config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
