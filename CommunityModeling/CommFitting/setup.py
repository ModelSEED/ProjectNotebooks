# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:08:24 2022

@author: Andrew Freiburger
"""

from setuptools import Extension, setup
from Cython.Build import cythonize
# from Cython.Compiler import Options; Options.annotate = True

setup(
    ext_modules=cythonize("mscommfitting.pyx", annotate=True),
    package_dir={'CommFitting': ''}, zip_safe=False)