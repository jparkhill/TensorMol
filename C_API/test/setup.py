from distutils.core import setup, Extension
import numpy

# define the extension module
Make_CM = Extension('Make_CM', sources=['Make_CM.cpp'],extra_compile_args=['-std=c++0x'],
                          include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[Make_CM])
