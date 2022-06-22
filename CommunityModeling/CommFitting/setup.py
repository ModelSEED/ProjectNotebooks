# -*- coding: utf-8 -*-

from setuptools import setup # Extension
from Cython.Build import cythonize
# from Cython.Compiler import Options; Options.annotate = True

setup(
    ext_modules=cythonize("mscommfitting.pyx", annotate=True),
    package_dir={'CommFitting': ''}, zip_safe=False)