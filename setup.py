from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True

cython_ext = Extension(name="example.cython_chi2",
                       sources=["./example/src/cython_chi2.pyx", "./example/src/chi2.c"],
                       include_dirs=[numpy.get_include()],
                       define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

capi_example = Extension(name="example.capi",
                         sources=["./example/src/capi.c",
                                  "./example/src/chi2.c",
                                  "./example/src/iph.c"],
                         include_dirs=[numpy.get_include()],
                         define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

capi_buffer_protocol_example = Extension(name="example.capi_buffer_protocol",
                                         sources=["./example/src/capi_buffer_protocol.c",
                                                  "./example/src/fib.c"])

capi_callback_example = Extension(name="example.capi_callback",
                         sources=["./example/src/capi_callback_function.c",
                                  "./example/src/optimizer.c"],
                         include_dirs=[numpy.get_include()],
                         define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

numpy_ufuncs_ext = Extension("example.ufuncs_example",
                             ["./example/src/ufunc_example.c"],
                             include_dirs=[numpy.get_include(), "./example/src",
                                           'C:\\Users\\mskocic\\LocalApps\\Miniconda3'])

setup(
    name="C Extension Demo",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[capi_example, numpy_ufuncs_ext, capi_buffer_protocol_example, capi_callback_example] + cythonize(cython_ext)
)


# pypi
# >>> python setup.py sdist bdist_wheel
# >>> python -m twine upload dist/*
# >>> python -m twine upload --repository testpypi dist/*

# anaconda in recipe folder
# >>> conda config --set anaconda_upload no
# >>> conda build .
# >>> conda build . --output
# >>> anaconda login
# >>> anaconda upload /path/to/conda-package.tar.bz2

# anaconda with pypi folders
# >>> anaconda upload dist/*.tar.gz
# >>> pip install --extra-index-url https://pypi.anaconda.org/USERNAME/PACKAGE
