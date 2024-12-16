from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="ising_helpers",
        sources=["ising_helpers.pyx"],
        language="c",
        extra_compile_args=["-O3", "-march=native"],  # Optimization flags
        include_dirs=[numpy.get_include()],
        # Uncomment the following lines to enable OpenMP (if desired)
        # extra_compile_args=["-O3", "-fopenmp"],
        # extra_link_args=["-fopenmp"],
    )
]

setup(
    name="ising_helpers",
    ext_modules=cythonize(extensions, language_level=3),
)
