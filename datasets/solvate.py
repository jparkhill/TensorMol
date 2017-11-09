#
# This little script shows you how to make a reasonably solvated xyz 
# from a pdb file using OpenMM and SimTk. 
# Well it makes a pdb from which you can generate an xyz in VMD
# or other reasonable converters. JAP 2017
#

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from pdbfixer import PDBFixer
import numpy as np

#fixer = PDBFixer(filename='2mzx.pdb')
#fixer.findMissingResidues()
#fixer.findMissingAtoms()
#fixer.addMissingAtoms()
#fixer.addMissingHydrogens(7.0)
##fixer.addSolvent(fixer.topology.getUnitCellDimensions())
#PDBFile.writeFile(fixer.topology, fixer.positions, open('output.pdb', 'w'))


pdb = PDBFile('output.pdb')
p = 0.8
# Center the thing and set the box size. 
x = pdb.getPositions(asNumpy=True)
print x 
xmn = np.min(x[:,0])._value - p 
ymn = np.min(x[:,1])._value - p 
zmn = np.min(x[:,2])._value - p 
x = pdb.getPositions(asNumpy=True,frame=0)
x._value -= np.array([xmn,ymn,zmn])
print x
xmx = np.max(x[:,0])._value + p 
ymx = np.max(x[:,1])._value + p 
zmx = np.max(x[:,2])._value + p 
print xmx,ymx,zmx
pdb.positions = x
pdb.topology.setUnitCellDimensions([xmx,ymx,zmx])
modeller = Modeller(pdb.topology, pdb.positions)
print modeller.topology.getUnitCellDimensions()

forcefield = ForceField('amber99sb.xml', 'tip5p.xml')
print "adding Hydrogens"
modeller.addHydrogens(forcefield, pH=5.0)
modeller.addSolvent(forcefield, padding=0.4*nanometer, model='tip5p')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.05*nanometer, constraints=HBonds)

f = open("solvated.pdb","w")
pdb.writeFile(modeller.topology,modeller.positions,f)

exit(0)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(pdb.topology, system, integrator) 
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy(maxIterations=25)
simulation.reporters.append(PDBReporter('output_exercise1.pdb', 5))
simulation.step(1000)
