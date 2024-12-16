# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("ising_helpers", ["ising_helpers.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=["-O3"])
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()]
)
