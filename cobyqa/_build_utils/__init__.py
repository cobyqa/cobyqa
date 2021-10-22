from distutils.version import LooseVersion

# from .._min_dependencies import CYTHON_MIN_VERSION

# Don't use the deprecated NumPy C API.
numpy_nodepr_api = dict(
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_9_API_VERSION')],
)


# def _check_cython_version():
#     """
#     Guarantee that the required version of Cython is installed.
#
#     Raises
#     ------
#     ModuleNotFoundError
#         No Cython distribution has been found.
#     ValueError
#         The version of Cython does not match the minimum requirement.
#     """
#     message = f'Cython version >= {CYTHON_MIN_VERSION} required.'
#     try:
#         import Cython
#     except ModuleNotFoundError as e:
#         raise ModuleNotFoundError(message) from e
#
#     if LooseVersion(Cython.__version__) < LooseVersion(CYTHON_MIN_VERSION):
#         message += f' The current version is {Cython.__version__}.'
#         raise ValueError(message)


# def cythonize_extensions(config):
#     """
#     Cythonize extensions of a configuration.
#
#     Parameters
#     ----------
#     config : numpy.distutils.misc_util.Configuration
#         The configuration whose extensions are to be Cythonized.
#
#     Raises
#     ------
#     ModuleNotFoundError
#         No Cython distribution has been found.
#     ValueError
#         The version of Cython does not match the minimum requirement.
#     """
#     _check_cython_version()
#
#     from Cython.Build import cythonize
#
#     config.ext_modules = cythonize(config.ext_modules)
