import argparse

# Cython and NumPy versions should be in sync with pyproject.toml.
CYTHON_MIN_VERSION = '0.29.23'
NUMPY_MIN_VERSION = '1.13.3'
PYTEST_MIN_VERSION = '5.0.1'
SCIPY_MIN_VERSION = '1.1.0'

dependent_pkgs = dict(
    cython=(CYTHON_MIN_VERSION, 'build'),
    numpy=(NUMPY_MIN_VERSION, 'build, install'),
    pytest=(PYTEST_MIN_VERSION, 'tests'),
    scipy=(SCIPY_MIN_VERSION, 'build, install'),
)

# Inverse mapping for setuptools.
tag_to_pkgs = {extra: [] for extra in {'build', 'docs', 'install', 'tests'}}
for pkg, (min_version, extras) in dependent_pkgs.items():
    for extra in extras.split(', '):
        tag_to_pkgs[extra].append(f'{pkg}>={min_version}')

# Get the minimum required version of a package via the command line.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimum dependencies')

    parser.add_argument('package', choices=dependent_pkgs)
    args = parser.parse_args()
    print(dependent_pkgs[args.package][0])
