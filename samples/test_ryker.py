from TensorMol import *
import os
import numpy as np
import time

def test():
	a = MSet("water_aug_cc_pvdz")
	a.Load()
	for mol in a.mols:
		mol.properties["quadrupole"] = mol.properties["quads"]
		mol.properties["dipole"] = mol.properties["dipoles"]
		mol.properties["gradients"] = mol.properties["forces"]
		del mol.properties["quads"]
		del mol.properties["dipoles"]
		del mol.properties["forces"]
	a.Save()

def make_mini_set(filename):
	a = MSet(filename)
	a.Load()
	b = MSet("water_aug_cc_pvdz_mini")
	for i in range(10):
		b.mols.append(a.mols[i])
	b.Save()

def train_energy_symm_func(mset):
	PARAMS["train_energy_gradients"] = False
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 500
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 100
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float32"
	PARAMS["train_dipole"] = True
	PARAMS["train_quadrupole"] = True
	manager = TFMolManageDirect(mset, network_type = "BehlerParinelloDirectSymFunc")

def get_losses(filename):
	# Returns train_loss, energy_loss, grad_loss, ...
	# test_train_loss, test_energy_loss, test_grad_loss
	with open(filename,"r") as log:
		log = log.readlines()

	keep_phrase = "TensorMol - INFO - step:"
	train_loss = []
	energy_loss = []
	grad_loss = []

	test_train_loss = []
	test_energy_loss = []
	test_grad_loss = []

	for line in log:
		if (keep_phrase in line) and (line[79] == ' '):
			a = line.split()
			train_loss.append(float(a[13]))
			energy_loss.append(float(a[15]))
			grad_loss.append(float(a[17]))
		if (keep_phrase in line) and (line[79] == 't'):
			a = line.split()
			test_train_loss.append(float(a[13]))
			test_energy_loss.append(float(a[15]))
			test_grad_loss.append(float(a[17]))

	print(str(train_loss) + "\n\n" + str(energy_loss) + "\n\n" + str(grad_loss) + "\n")
	print(str(test_train_loss) + "\n\n" + str(test_energy_loss) + "\n\n" + str(test_grad_loss) + "\n")
	return train_loss, energy_loss, grad_loss, test_train_loss, test_energy_loss, test_grad_loss

def optimize_taxol():
	Taxol = MSet("Taxol")
	Taxol.ReadXYZ()
	GeomOptimizer("EnergyForceField").Opt(Taxol, filename="OptLog", Debug=False)

#test()
#make_mini_set("water_aug_cc_pvdz")
train_energy_symm_func("water_aug_cc_pvdz_mini")
#get_losses("networks/nicotine_aimd_log.txt")
#optimize_taxol()
