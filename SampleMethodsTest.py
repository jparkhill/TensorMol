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
	# PARAMS["hidden1"] = 200
	# PARAMS["hidden2"] = 200
	# PARAMS["hidden3"] = 200
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["GradScalar"] = 1
	PARAMS["NeuronType"] = "relu"
	PARAMS["HiddenLayers"] = [512,512,512]
	manager = TFMolManage(MName_ , tset, False, RandomTData_=False, Trainable_=False)
	energies=[]
	gradients=[]
	for mol in a.mols:
		# mol.Distort(0.01)
		en,grad=manager.Eval_BPEnergy_Direct_Grad_Linear(Mol(mol.atoms, mol.coords))
		energies.append(en)
		gradients.append(grad/-JOULEPERHARTREE)
	print energies
	print gradients
	return energies,gradients

def GetForceEnergies():
	managers = ["Mol_DavidMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1","Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1", "Mol_DavidNM_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1","Mol_DavidRandom_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1"]
	results={}
	for i in managers:
		ens,grads = GetEnergyAndForceFromManager(i)
		results[i] = (ens,grads)
	print results

def TestOptimization(MName_):
	a = MSet("sampling_mols")
	a.ReadXYZ()
	TreatedAtoms = a.AtomTypes()
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
	tset = TensorMolData_BP_Direct_Linear(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager = TFMolManage(MName_ , tset, False, RandomTData_=False, Trainable_=False)
	rms_list = []
	for mol in a.mols:
		EnergyForceField = lambda x: manager.Eval_BPEnergy_Direct_Grad_Linear(Mol(mol.atoms,x))
		mol.Distort(0.2)
		molp = GeomOptimizer(EnergyForceField).Opt(mol)
		tmp_rms = mol.rms_inv(molp)
		rms_list.append(tmp_rms)
		print "RMS:", tmp_rms
	return rms_list
# GetEnergyAndForceFromManager("Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1")
# GetForceEnergies()
print TestOptimization("Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1")
