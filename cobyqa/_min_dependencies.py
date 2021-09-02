import argparse

CYTHON_MIN_VERSION = '0.29.23'
NUMPY_MIN_VERSION = '1.14.6'
PYTEST_MIN_VERSION = '5.0.1'

dependent_pkgs = dict(
    cython=(CYTHON_MIN_VERSION, 'build'),
    numpy=(NUMPY_MIN_VERSION, 'build, install'),
    pytest=(PYTEST_MIN_VERSION, 'tests'),
)

tag_to_pkgs = {extra: [] for extra in {'build', 'docs', 'install', 'tests'}}
for pkg, (min_version, extras) in dependent_pkgs.items():
    for extra in extras.split(', '):
        tag_to_pkgs[extra].append(f'{pkg}>={min_version}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimum dependencies')

    parser.add_argument('package', choices=dependent_pkgs)
    args = parser.parse_args()
    min_version = dependent_pkgs[args.package][0]
    print(min_version)
