# To make CM available: 
# sudo python setup.py install

from distutils.core import setup, Extension
import numpy

# define the extension module
MolEmb = Extension(
	'MolEmb',
	sources=['MolEmb.cpp'],
	extra_compile_args=['-std=c++0x', '-fopenmp'],
	extra_link_args=['-lgomp'],
        include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[MolEmb])
