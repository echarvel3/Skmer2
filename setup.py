import os
import re
from setuptools import setup, find_packages

# single source of truth for the version: skmer/__init__.py
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skmer', '__init__.py')) as f:
    version = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read()).group(1)

setup(name='skmer',
      version=version,
      description='Assembly-free and alignment-free tool for estimating genomic distances between genome-skims',
      author='Eduardo Charvel',
      author_email='echarvel3@gmail.com',
      license='MIT',
      url='https://github.com/echarvel3/ReSkmer',
      packages=find_packages(),
      package_dir={'skmer': 'skmer'},
      include_package_data=True,
      package_data={'skmer': ['data/*', 'data/*/*']},
      install_requires=['numpy>=1.15.1', 'scipy>=1.1.0', 'pandas>=0.23.4', 'pyfastx'],
      provides=["skmer"],
      entry_points={
            'console_scripts': ['skmer=skmer.__main__:main']
      },
      classifiers=["Environment :: Console",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "Operating System :: Unix",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3",
                   "Topic :: Scientific/Engineering :: Bio-Informatics"],
      )

