from TensorMol import *
import time
import numpy as np
import os

def get_losses():
	# Returns train_loss, energy_loss, and grad_loss vectors
	# indexed properly for each step
	with open("nicotine_aimd_log.txt","r") as log:
		log = log.readlines()

	keep_phrase = "TensorMol - INFO - step:"
	train_loss = []
	energy_loss = []
	grad_loss = []

	i = 0

	for line in log:
		if keep_phrase in line:
			a = line.split()
			train_loss.append(float(a[13]))
			energy_loss.append(float(a[15]))
			grad_loss.append(float(a[17]))
			i += 1

	print(str(train_loss) + "\n\n" + str(energy_loss) + "\n\n" + str(grad_loss))
	return train_loss, energy_loss, grad_loss

def optimize_taxol():
	Taxol = MSet("Taxol").ReadXYZ("Taxol")
	GeomOptimizer("EnergyForceField").Opt(Taxol, filename="OptLog", Debug=False)



get_losses()
#optimize_taxol()
