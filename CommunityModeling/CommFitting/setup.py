# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:08:24 2022

@author: Andrew Freiburger
"""

from distutils.core import setup
from Cython.Build import cythonize
setup(
    name='ModelSEED Community Fitting',
    ext_modules=cythonize("mscommfitting.pyx"),
    zip_safe=False,
)