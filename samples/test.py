"""
Generates artificial data for H_3, learns a potential for it, tests it in optimizations and whatnot.
"""
from TensorMol import *

def GenerateData(model_="Huckel"):
	"""
	Generate random configurations in a reasonable range.
	and calculate their energies and forces.
	"""
	nsamp = 10000
	crds = np.random.uniform(4.0,size = (nsamp,3,3))
	st = MSet()
	MDL = None
	natom = 4
	ANS = np.array([3,1,1])
	if (model_=="Morse"):
		MDL = MorseModel()
	else:
		MDL = QuantumElectrostatic()
	for s in range(nsamp):
		if (model_=="Morse"):
			st.mols.append(Mol(np.array([1.,1.,1.]),crds[s]))
			en,f = MDL(crds[s])
			st.mols[-1].properties["dipole"] = np.array([0.,0.,0.])
		else:
			st.mols.append(Mol(np.array([3.,1.,1.]),crds[s]))
			en, f, d, q = MDL(crds[s])
			st.mols[-1].properties["dipole"] = d
			st.mols[-1].properties["charges"] = q
		st.mols[-1].properties["energy"] = en
		st.mols[-1].properties["force"] = f
		st.mols[-1].properties["gradients"] = -1.0*f
		st.mols[-1].CalculateAtomization()
	return st

def TestTraining_John():
	PARAMS["train_dipole"] = True
	tset = GenerateData()
	net = BehlerParinelloDirectGauSH(tset)
	net.train()
	return

def TestTraining():
	a = GenerateData()
	TreatedAtoms = a.AtomTypes()
	PARAMS["NetNameSuffix"] = "training_sample"
	PARAMS["learning_rate"] = 0.00001
	PARAMS["momentum"] = 0.95
	PARAMS["max_steps"] = 15 # Train for 5 epochs in total
	PARAMS["batch_size"] =  100
	PARAMS["test_freq"] = 5 # Test for every epoch
	PARAMS["tf_prec"] = "tf.float64" # double precsion
	PARAMS["EnergyScalar"] = 1.0
	PARAMS["GradScalar"] = 1.0/20.0
	PARAMS["NeuronType"] = "sigmoid_with_param" # choose activation function
	PARAMS["sigmoid_alpha"] = 100.0  # activation params
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 1.0] # each layer's keep probability for dropout
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
	tset = TensorMolData_BP_Direct_EandG_Release(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EandG_SymFunction")
	PARAMS['Profiling']=0
	manager.Train(1)


def TestOpt():
	return

def TestMD():
	return

TestTraining()
