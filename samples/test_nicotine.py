from __future__ import absolute_import
#import memory_util
#memory_util.vlog(1)
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from TensorMol.ForceModels.ElectrostaticsTF import *
from TensorMol.MBE.NN_MBE import *
from TensorMol.Interfaces.TMIPIinterface import *
import random

def TrainPrepare():
	if (1):
		WB97XDAtom={}
		WB97XDAtom[1]=-0.5026682866
		WB97XDAtom[6]=-37.8387398698
		WB97XDAtom[7]=-54.5806161811
		WB97XDAtom[8]=-75.0586028656
		a = MSet("nicotine_aimd_rand")
		a.Load()
		b = MSet("nicotine_aimd_rand_train")
		for mol_index, mol in enumerate(a.mols):
			print ("mol_index:", mol_index)
			mol.properties['gradients'] = -mol.properties['forces']
			mol.properties['atomization'] =  mol.properties['energy']
			for i in range (0, mol.NAtoms()):
				mol.properties['atomization'] -= WB97XDAtom[mol.atoms[i]]
				b.mols.append(mol)
				b.Save()

def Train():
	if (1):
		a = MSet("nicotine_aimd_rand_train")
		a.Load()
		print (len(a.mols))
		TreatedAtoms = a.AtomTypes()
		PARAMS["HiddenLayers"] = [200,200,200]
		PARAMS["learning_rate"] = 0.00001
		PARAMS["momentum"] = 0.95
		PARAMS["max_steps"] = 5001
		PARAMS["batch_size"] = 200
		PARAMS["test_freq"] = 10
		PARAMS["tf_prec"] = "tf.float64"
		#PARAMS["AN1_num_r_Rs"] = 16
		#PARAMS["AN1_num_a_Rs"] = 4
		#PARAMS["AN1_num_a_As"] = 4
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
		tset = TensorMolData_BP_Direct_Linear(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_Grad_Linear") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=1001)

#TrainPrepare()
Train()
