from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='datacontrolreach',
   version='0.1',
   description='A module for the over-approximation of the reachable set'+\
                    ' and control of unknown dynamical systems using data from a'+\
                    ' single trajectory and side information on the underlying dynamics',
   license="GNU 3.0",
   long_description=long_description,
   author='Franck Djeumou',
   author_email='fdjeumou@utexas.edu',
   url="https://github.com/wuwushrek/datacontrolreach.git",
   packages=['datacontrolreach'],
   package_dir={'datacontrolreach': 'datacontrolreach/'},
   install_requires=['numpy', 'scipy', 'matplotlib', 'jax', 'jaxlib', 'tikzplotlib', 'tqdm'],
   tests_require=['pytest', 'pytest-cov'],
)
