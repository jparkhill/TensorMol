"""
Routines that will create managers from trained networks,
develop energies and forces, and gather statistics
for data tables and figures
"""

from TensorMol import *
import os
import numpy as np

# Dataset retrieval and information
a = MSet("sampling_mols")
a.ReadXYZ()
TreatedAtoms = a.AtomTypes()
# m = a.mols[] # Only need this if evaluating an individual molecule

# Digester and TensorMol training data set loader
d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
tset = TensorMolData_BP_Direct_Linear(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)

# Managers for trained networks
MD_manager = TFMolManage("Mol_DavidMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1" , tset, False, RandomTData_=False, Trainable_=False)
Metadynamics_manager = TFMolManage("Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1" , tset, False, RandomTData_=False, Trainable_=False)
NM_manager = TFMolManage("Mol_DavidNM_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1" , tset, False, RandomTData_=False, Trainable_=False)
Random_manager = TFMolManage("Mol_DavidRandom_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1" , tset, False, RandomTData_=False, Trainable_=False)

# Network parameters
PARAMS["hidden1"] = 200
PARAMS["hidden2"] = 200
PARAMS["hidden3"] = 200
PARAMS["tf_prec"] = "tf.float64"
PARAMS["GradScalar"] = 1
PARAMS["NeuronType"] = "relu"
PARAMS["HiddenLayers"] = [200,200,200]

# Develop energies and forces
def GetForceEnergy():
    managers = [Metadynamics_manager, MD_manager, Random_manager] #NM_manager, ]
    for i in managers:
        for mol in a.mols:
            i.Eval_BPEnergy_Direct_Grad_Linear(Mol(mol.atoms))

# Gather statistics
def GetStats(GetForceEnergy):
