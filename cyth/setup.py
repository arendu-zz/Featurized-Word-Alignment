__author__ = 'arenduchintala'
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Model1 App',
    ext_modules=cythonize('cyth_model1.pyx'),
)

setup(
    name='Common App',
    ext_modules=cythonize('cyth_common.pyx'),
)
# Build:
# python setup.py build_ext --inplace
