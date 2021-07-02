# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.rst', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    reqs = f.read().split('\n')

setup(name='som-pbc',
      version='1.0.1',
      description='self organizing maps with periodic boundary conditions',
      long_description=readme,
      author='Alex Müller',
      author_email='alexarnimueller@protonmail.com',
      url='https://github.com/alexarnimueller/som',
      license='MIT',
      keywords="SOM embedding machine learning chemoinformatics bioinformatics datascience descriptor similarity",
      packages=['som'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6'],
      install_requires=reqs
      )
