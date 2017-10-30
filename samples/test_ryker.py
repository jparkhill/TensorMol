from TensorMol import *
import os
import numpy as np
import time

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



get_losses("networks/nicotine_aimd_log.txt")
#optimize_taxol()
