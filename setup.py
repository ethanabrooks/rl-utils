#! /usr/bin/env python

# third party
from setuptools import find_packages, setup

setup(
    name='utils',
    version='0.0.0',
    author='Ethan Brooks',
    author_email='ethanabrooks@gmail.com',
    packages=find_packages(), install_requires=['gym', 'numpy', 'tensorflow']
)
