#! /usr/bin/env python

# third party
from setuptools import find_packages, setup

setup(
    name='rl-utils',
    version='0.0.0',
    author='Ethan Brooks',
    author_email='ethanabrooks@gmail.com',
    packages=find_packages(),
    install_requires=['gym', 'numpy'],
    extra_requires=dict(tensorflow='tensorflow'),
    entry_points=dict(console_scripts=[
        'tb=utils.tb:cli',
        'crawl=utils.crawl_events:cli',
    ]),
)
