# To make CM available: 
# sudo python setup.py install

from distutils.core import setup, Extension
import numpy

# define the extension module
import os

LLVM=os.popen('cc --version | grep clang').read().count("LLVM")
if (not LLVM): 
	MolEmb = Extension(
	'MolEmb',
	sources=['MolEmb.cpp'],
	extra_compile_args=['-std=c++0x','-fopenmp'],
	extra_link_args=['-lgomp'],
        include_dirs=[numpy.get_include()])
else: 
        MolEmb = Extension(
        'MolEmb',
        sources=['MolEmb.cpp'],
        extra_compile_args=['-std=c++0x'],
        extra_link_args=[],
        include_dirs=[numpy.get_include()])


# run the setup
setup(ext_modules=[MolEmb])
