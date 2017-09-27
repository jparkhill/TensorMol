from TensorMol import *

Taxol = MSet("Taxol").ReadXYZ("Taxol")

GeomOptimizer("EnergyForceField").Opt(Taxol, filename="OptLog", Debug=False)
