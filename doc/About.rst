About TensorMol
=================

The purpose of TensorMol is to enable simulations using Neural Network models of chemistry. A secondary purpose is to provide an experimental Pythonic playground where these models can be mixed and matched to facilitate development. It relies heavily on the numerical facilities of Google's tensorflow framework, and runs efficently on either CPU's or GPU's. Many sort of chemical models are supported in the code, and interfaced with simple environments to execute simulations. There is also a socket interface to the efficient I-Pi Force engine.

Although TensorMol is Object-Oriented, inheritance is minimized throughout the code, and most functions produce physical observables such as energies and forces given simple arrays of coordinates and atomic numbers.

The code is divided into two Modules. A standard C-Python extension called MolEmb, for small routines which must be run rapidly, and the TensorMol module itself. Installation can be easily achieved with pip (see README.md). 
