''' YIfan's repeating code of the paper P.R.L 108,058301(2012) 'Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning'
'''
from TensorMol import *

b=MSet("gdb9")
b.Load()

print b.mols[-1]
# Load GDB9
# Make a coulomb MolDigester which works over the whole molecule.
# Make a Tensordata and use it to buildtraining data
# Make a TFManager to produce a KRR instance.
# Train.
# Test.
