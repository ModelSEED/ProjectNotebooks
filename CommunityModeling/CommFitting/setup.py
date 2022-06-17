# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:08:24 2022

@author: Andrew Freiburger
"""

from distutils.core import setup
from Cython.Build import cythonize
# from Cython.Compiler import Options; Options.annotate = True

setup(
    name='ModelSEED Community Fitting',
    ext_modules=cythonize(
        "mscommfitting.pyx", language_level=3, annotate=True),
    zip_safe=False)