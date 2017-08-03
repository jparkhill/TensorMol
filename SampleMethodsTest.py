"""
Routines that will create managers from trained networks,
develop energies and forces, and gather statistics
for data tables and figures
"""

from TensorMol import *
import os
import numpy as np
from math import *
from TensorMol.ElectrostaticsTF import *

def GetEnergyAndForceFromManager(MName_, set_):
	"""
	MName: name of the manager. (specifies sampling method.)
	set_: name of dataset
	"""
	# If you wanna do eq.
	# a = MSet("sampling_mols")
	# a.ReadXYZ()
	#
	a = MSet(set_)
	a.Load()
	TreatedAtoms = a.AtomTypes()
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
	tset = TensorMolData_BP_Direct_Linear(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["GradScalar"] = 1
	PARAMS["NeuronType"] = "relu"
	PARAMS["HiddenLayers"] = [512,512,512]
	manager = TFMolManage(MName_ , tset, False, RandomTData_=False, Trainable_=False)
	nmols = len(a.mols)
	EErr = np.zeros(nmols)
	FErr = np.zeros(nmols)
	for i, mol in enumerate(a.mols):
		mol.properties["atomization"] = mol.properties["energy"]
		mol.properties["gradients"] = mol.properties["forces"]
		for j in range(0, mol.NAtoms()):
			mol.properties["atomization"] -= ele_E_david[mol.atoms[j]]
		q_en = mol.properties["atomization"]
		q_f = mol.properties["gradients"]
		#qchem_energies.append(q_en)
		en,grad=manager.Eval_BPEnergy_Direct_Grad_Linear(Mol(mol.atoms, mol.coords))
		new_grad = grad/-JOULEPERHARTREE
		#print "new_grad:", new_grad
		#print "q_f:", q_f
		EErr[i] = (en - q_en)
		F_diff = new_grad - q_f
		FErr[i] = np.sqrt(np.sum(F_diff*F_diff)/mol.NAtoms())
		#print "FErr[i]", FErr[i]
	print np.sum(FErr,axis=0), nmols
	print "RMS energy error: ", np.sqrt(np.sum(EErr*EErr)/nmols)
	print "<|F_err|> error: ", (np.sum(FErr)/nmols)
	return EnergyError,ForceError

def GetForceEnergies():
	managers = ["Mol_DavidMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1","Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1"] #, "Mol_DavidNM_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1","Mol_DavidRandom_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1"]
	results={}
	for i in managers:
		ens,grads = GetEnergyAndForceFromManager(i)
		results[i] = (ens,grads)
	print results

def TestOptimization(MName_):
	"""
	Distorts a set of molecules from their equilibrium geometries,
	then optimizes the distorted geometries using a trained network.

	Args:

		MName_: Trained network

	Returns:

		np.mean(rms_list): Average value of all of the RMS errors for all molecules
	"""

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
	return np.mean(rms_list)


GetEnergyAndForceFromManager("Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1", "DavidMetaMD")
# GetForceEnergies()
# print TestOptimization("Mol_DavidMetaMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1")
