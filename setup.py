from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "ising_helpers",
        ["monte-carlo/ising_helpers.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="ising_helpers",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
