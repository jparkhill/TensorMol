from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from TensorMol.ElectrostaticsTF import *

def TrainPrepare():
	if (0):
		a = MSet("chemspider9_force")
		dic_list = pickle.load(open("./datasets/chemspider9_force.dat", "rb"))
		for dic in dic_list:
			atoms = []
		for atom in dic['atoms']:
			atoms.append(AtomicNumber(atom))
		atoms = np.asarray(atoms, dtype=np.uint8)
		mol = Mol(atoms, dic['xyz'])
		mol.properties['charges'] = dic['charges']
		mol.properties['dipole'] = dic['dipole']
		mol.properties['quadropole'] = dic['quad']
		mol.properties['energy'] = dic['scf_energy']
		mol.properties['gradients'] = dic['gradients']
		mol.CalculateAtomization()
		a.mols.append(mol)
		a.Save()

	if (0):
		a = MSet("chemspider9_force")
		a.Load()
		rmsgrad = np.zeros((len(a.mols)))
		for i, mol in enumerate(a.mols):
			rmsgrad[i] = (np.sum(np.square(mol.properties['gradients'])))**0.5
		meangrad = np.mean(rmsgrad)
		print "mean:", meangrad, "std:", np.std(rmsgrad)
		np.savetxt("chemspider9_force_dist.dat", rmsgrad)
		for i, mol in enumerate(a.mols):
			rmsgrad = (np.sum(np.square(mol.properties['gradients'])))**0.5
			if 2 > rmsgrad > 1.5:
				mol.WriteXYZfile(fname="large_force")
				print rmsgrad

	if (0):
		a = MSet("chemspider9_force")
		a.Load()
		b = MSet("chemspider9_force_cleaned")
		for i, mol in enumerate(a.mols):
			rmsgrad = (np.sum(np.square(mol.properties['gradients'])))**0.5
			if rmsgrad <= 1.5:
				b.mols.append(mol)
		b.Save()
		c = MSet("chemspider9_force_cleaned_debug")
		c.mols = b.mols[:1000]
		c.Save()

	if (1):
		a = MSet("DavidRandom")
		a.Load()
		for mol in a.mols:
			mol.properties["gradients"] = mol.properties["forces"]
			#mol.properties["energy"] = mol.properties["energy"] * KCALPERHARTREE
			mol.properties["atomization"] = mol.properties["energy"]
			#mol.properties["energy"] = mol.properties["energy"]/KCALPERHARTREE
			for i in range(0, mol.NAtoms()):
				mol.properties["atomization"] -= ele_E_david[mol.atoms[i]]
		#	if abs(mol.properties["energy"]) > 1000:
                 #               mol.properties["energy"] = mol.properties["energy"] / KCALPERHARTREE
			print mol.properties["atomization"], mol.properties["energy"]
		a.Save()
		print a.mols[0].properties
		a.mols[0].WriteXYZfile(fname="test")

def TrainForceField(SetName_ = "DavidRandom"):
	a = MSet(SetName_)
	a.Load()
	TreatedAtoms = a.AtomTypes()
	PARAMS["hidden1"] = 200
	PARAMS["hidden2"] = 200
	PARAMS["hidden3"] = 200
	PARAMS["learning_rate"] = 0.00001
	PARAMS["momentum"] = 0.95
	PARAMS["max_steps"] = 101
	PARAMS["batch_size"] = 100
	PARAMS["test_freq"] = 2
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["GradScaler"] = 1
	PARAMS["NeuronType"] = "relu"
	PARAMS["HiddenLayers"] = [200, 200, 200]
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP_Direct_Linear(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
	manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_Grad_Linear") # Initialzie a manager than manage the training of neural network.
	manager.Train(maxstep=101)

def TestIRLinearDirect():
	"""
	Test the IR spectrum produced by a network created and trained with TrainForceField()
	Intended to be used with MolInstance_DirectBP_EE soon... 
	"""
	a = MSet("test")
	a.ReadXYZ("1_1_Ostrech")
	m = a.mols[0]
	TreatedAtoms = a.AtomTypes()
	PARAMS["hidden1"] = 200
	PARAMS["hidden2"] = 200
	PARAMS["hidden3"] = 200
	PARAMS["learning_rate"] = 0.00001
	PARAMS["momentum"] = 0.95
	PARAMS["max_steps"] = 101
	PARAMS["batch_size"] = 100
	PARAMS["test_freq"] = 2
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["GradScaler"] = 1
	PARAMS["NeuronType"] = "relu"
	PARAMS["HiddenLayers"] = [200, 200, 200]
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP_Direct_Linear(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
	manager= TFMolManage("Mol_DavidMD_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_Linear_1" , tset, False, RandomTData_=False, Trainable_=False)
	UnitedIR(manager.BPDirectGradLinearEval,mol,"UnitedIR")

#TestCoulomb()
#TrainPrepare()
TrainForceField()
#TestIRLinearDirect()
