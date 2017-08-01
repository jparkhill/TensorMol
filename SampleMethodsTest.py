"""
Routines that will create managers from trained networks,
develop energies and forces, and gather statistics
for data tables and figures
"""

from TensorMol import *
import os
import numpy as np
from math import *

def GetEnergyAndForceFromManager(MName_):
	a = MSet("sampling_mols")
	a.ReadXYZ()
	TreatedAtoms = a.AtomTypes()
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
	tset = TensorMolData_BP_Direct_Linear(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	PARAMS["hidden1"] = 200
	PARAMS["hidden2"] = 200
	PARAMS["hidden3"] = 200
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["GradScalar"] = 1
	PARAMS["NeuronType"] = "relu"
	PARAMS["HiddenLayers"] = [200,200,200]
	manager = TFMolManage(MName_ , tset, False, RandomTData_=False, Trainable_=False)
	energies=[]
	gradients=[]
	for mol in a.mols:
		en,grad=manager.Eval_BPEnergy_Direct_Grad_Linear(Mol(mol.atoms))
		energies.append(en)
		gradients.append(grad)
	return energies,gradients

def GetForceEnergies():
	managers = ["Mol_DavidMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1","Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1", "Mol_DavidNM_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1","Mol_DavidRandom_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1"]
	results={}
	for i in managers:
		ens,grads = GetEnergyAndForceFromManager(i)
		results[i] = (ens,grads)
	print results

GetEnergyAndForceFromManager("Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1")
# GetForceEnergies()
