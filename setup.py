# To make TensorMol available.
# sudo pip install -e .
#
# to make and upload a source dist 
# python setup.py sdist
# twine upload dist/*
# And of course also be me. 
# 

from __future__ import absolute_import,print_function
from distutils.core import setup, Extension
import numpy
import os

# Todo: There is probably a way we can do this without requiring a numpy dependency. -JD
print("Numpy Include Dir: ",numpy.get_include())

LLVM=os.popen('cc --version | grep clang').read().count("LLVM")
if (not LLVM):
	MolEmb = Extension(
	'MolEmb',
	sources=['./C_API/MolEmb.cpp'],
	extra_compile_args=['-std=c++0x','-g','-fopenmp','-w'],
	extra_link_args=['-lgomp'],
        include_dirs=[numpy.get_include()]+['./C_API/'])
else:
        MolEmb = Extension(
        'MolEmb',
        sources=['./C_API/MolEmb.cpp'],
        extra_compile_args=['-std=c++0x'],
        extra_link_args=[],
        include_dirs=[numpy.get_include()]+['./C_API/'])


# run the setup
setup(name='TensorMol',
      version='0.2',
      description='TensorFlow+Molecules = TensorMol',
      url='http://github.com/jparkhill/TensorMol',
      author='john parkhill',
      author_email='john.parkhill@gmail.com',
      license='GPL3',
      packages=['TensorMol'],
      install_requires=[
        "tensorflow==1.8.0",
        "scipy==1.2.3"
        ]
      zip_safe=False,
      include_package_data=True,
      ext_modules=[MolEmb])
