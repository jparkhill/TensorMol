from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from pdbfixer import PDBFixer
import numpy as np

fixer = PDBFixer(filename='rna_ac.pdb')
#fixer.findMissingResidues()
fixer.missingResidues = {}
# Pull out and save the coordinates of the desired ligand. 
#fixer.findMissingAtoms()
#fixer.addMissingAtoms()
#fixer.addMissingHydrogens(7.0)
mnx = min([p[0] for p in fixer.positions])._value
mny = min([p[1] for p in fixer.positions])._value
mnz = min([p[2] for p in fixer.positions])._value
fixer.positions._value = [p - Vec3(mnx,mny,mnz) for p in fixer.positions._value]
maxSize = max(max((pos[i] for pos in fixer.positions))-min((pos[i] for pos in fixer.positions)) for i in range(3))
boxSize = maxSize*Vec3(1, 1, 1)
boxVectors = (maxSize*Vec3(1, 0, 0),maxSize*Vec3(0, 1, 0),maxSize*Vec3(0, 0, 1))

#
# This is basically the pdbfixer code, but without the amber lines. 
#
modeller = Modeller(fixer.topology, fixer.positions)
forcefield = ForceField('amber99sb.xml', 'tip5p.xml')
system = forcefield.createSystem(fixer.topology, nonbondedMethod=PME, nonbondedCutoff=0.05*nanometer, constraints=HBonds)
modeller.addSolvent(forcefield, padding=0.05*nanometer, boxSize=None, boxVectors=None)
#modeller.addSolvent(forcefield, padding=0.4*nanometer, boxSize, boxVectors=boxVectors, model='tip5p')
# modeller.addSolvent(forcefield, padding=padding, boxSize=boxSize, boxVectors=boxVectors, positiveIon=positiveIon, negativeIon=negativeIon, ionicStrength=ionicStrength)
fixer.topology = modeller.topology
fixer.positions = modeller.positions

proatoms = [atom.element._symbol for atom in modeller.topology.atoms()]
procoords = np.array([fixer.positions[atom.index]._value for atom in modeller.topology.atoms()])

def WriteXYZfile(atoms,coords,nm_="out.xyz"):
    natom = len(atoms)
    f = open(nm_,"w")
    f.write(str(natom)+"\n"+"\n")
    for i in range(natom): 
        f.write(atoms[i]+" "+str(coords[i][0])+" "+str(coords[i][1])+" "+str(coords[i][2])+"\n")
        
#
# This will directly generate XYZ files for both the protein and the substrate. 
#
WriteXYZfile(CofactorAtoms,CofactorCoords*10.0,"cofactor.xyz")
WriteXYZfile(proatoms,procoords*10.0,"protein.xyz")
#PDBFile.writeFile(fixer.topology, fixer.positions, open('3gm0_fixed.pdb', 'w'))
