#!/usr/bin/env python3
import importlib
import os
import shutil
import sys

from pkg_resources import parse_version

if sys.version_info < (3, 6):
    raise RuntimeError('Python version >= 3.6 required.')

import builtins
from pathlib import Path

# Remove MANIFEST before importing setuptools to prevent improper updates.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

import setuptools  # noqa
from distutils.command.clean import clean  # noqa
from distutils.command.sdist import sdist  # noqa

# This is a bit hackish: to prevent loading components that are not yet built,
# we set a global variable to endow the main __init__ with with the ability to
# detect whether it is loaded by the setup routine.
builtins.__COBYQA_SETUP__ = True

import cobyqa  # noqa
import cobyqa._min_dependencies as min_deps  # noqa

SETUPTOOLS_COMMANDS = {
    'bdist_dumb',
    'bdist_egg',
    'bdist_rpm',
    'bdist_msi',
    'bdist_wheel',
    'bdist_wininst',
    'develop',
    'easy_install',
    'egg_info',
    'install_egg_info',
    'upload',
}
if SETUPTOOLS_COMMANDS.intersection(sys.argv[1:]):
    extra_setuptools_args = dict(
        zip_safe=False,
        include_package_data=True,
        extras_require={k: min_deps.tag_to_pkgs[k] for k in ['docs', 'tests']},
    )
else:
    extra_setuptools_args = {}


class CleanCommand(clean):
    description = 'Remove build artifacts from the source tree'

    def run(self):
        super().run()

        # Remove the 'build', 'dist', '*.egg-info', '.pytest_cache', and '.tox'
        # directories from the current working directory.
        cwd = Path(__file__).resolve(strict=True).parent
        shutil.rmtree(cwd / 'build', ignore_errors=True)
        shutil.rmtree(cwd / 'dist', ignore_errors=True)
        for dirname in cwd.glob('*.egg-info'):
            shutil.rmtree(dirname)
        shutil.rmtree(cwd / '.pytest_cache', ignore_errors=True)
        shutil.rmtree(cwd / '.tox', ignore_errors=True)

        # Remove the 'MANIFEST' file.
        if Path(cwd, 'MANIFEST').is_file():
            os.unlink(cwd / 'MANIFEST')

        # Remove the generated C/C++ files outside of a source distribution.
        # rm_c_files = not Path(cwd, 'PKG-INFO').is_file()
        for dirpath, dirnames, filenames in os.walk(cwd / 'cobyqa'):
            dirpath = Path(dirpath).resolve(strict=True)
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(dirpath / dirname)
            # for filename in filenames:
            #     basename, extension = os.path.splitext(filename)
            #     if extension in ('.dll', '.pyc', '.pyd', '.so'):
            #         os.unlink(dirpath / filename)
            #     if rm_c_files and extension in ('.c', '.cpp'):
            #         pyx_file = basename + '.pyx'
            #         if Path(dirpath, pyx_file).is_file():
            #             os.unlink(dirpath / filename)


cmdclass = {'clean': CleanCommand, 'sdist': sdist}


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    # from cobyqa._build_utils import _check_cython_version  # noqa
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    # _check_cython_version()
    config.add_subpackage('cobyqa')

    return config


def check_pkg_status(pkg, min_version):
    message = '{} version >= {} required.'.format(pkg, min_version)
    try:
        module = importlib.import_module(pkg)
        pkg_version = module.__version__  # noqa
        if parse_version(pkg_version) < parse_version(min_version):
            message += ' The current version is {}.'.format(pkg_version)
            raise ValueError(message)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(message)


def setup_package():
    metadata = dict(
        name='cobyqa',
        author='Tom M. Ragonneau',
        author_email='tom.ragonneau@connect.polyu.hk',
        maintainer='Tom M. Ragonneau',
        maintainer_email='tom.ragonneau@connect.polyu.hk',
        version=cobyqa.__version__,
        description='Constrained Optimization BY Quadratic Approximation',
        long_description=open('README.rst').read().rstrip(),
        long_description_content_type='text/x-rst',
        keywords='',
        license='BSD-3-Clause',
        url='https://ragonneau.github.io/cobyqa',
        download_url='https://pypi.org/project/cobyqa/#files',
        project_urls={
            'Bug Tracker': 'https://github.com/ragonneau/cobyqa/issues',
            'Documentation': 'https://ragonneau.github.io/cobyqa',
            'Source Code': 'https://github.com/ragonneau/cobyqa',
        },
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: MacOS',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Unix',
            # 'Programming Language :: Cython',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: Implementation :: CPython',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        platforms=['Linux', 'macOS', 'Unix', 'Windows'],
        cmdclass=cmdclass,
        python_requires='>=3.6',
        install_requires=min_deps.tag_to_pkgs['install'],
        package_data={'': ['*.pxd']},
        **extra_setuptools_args
    )

    distutils_commands = {
        'check',
        'clean',
        'egg_info',
        'dist_info',
        'install_egg_info',
        'rotate',
    }
    commands = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    if all(command in distutils_commands for command in commands):
        from setuptools import setup
    else:
        check_pkg_status('numpy', min_deps.NUMPY_MIN_VERSION)

        from numpy.distutils.core import setup

        metadata['configuration'] = configuration
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
    del builtins.__COBYQA_SETUP__  # noqa
