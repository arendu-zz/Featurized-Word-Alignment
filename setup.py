from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np  # <---- New line

ext_modules = [Extension("HybridModel1", ["HybridModel1.pyx"])]

setup(
    name='hybridmodel1',
    cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],  # <---- New line
    ext_modules=ext_modules
)
# python setup.py build_ext --inplace
