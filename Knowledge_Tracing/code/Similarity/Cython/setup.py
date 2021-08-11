from distutils.core import setup
from Cython.Build import cythonize
import numpy


def main():
    setup(ext_modules=cythonize("Compute_Similarity_Cython.pyx"),
          include_dirs=[numpy.get_include()])


main()
