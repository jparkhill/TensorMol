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


        if (1):
		RIMP2Atom={}
		RIMP2Atom[1]=-0.4998098112
		RIMP2Atom[8]=-74.9659650581
                a = MSet("H2O_augmented_more_cutoff5_rimp2_force_dipole")
                dic_list = pickle.load(open("./datasets/H2O_augmented_more_cutoff5_rimp2_force_dipole.dat", "rb"))
                for dic in dic_list:
                        atoms = []
                        for atom in dic['atoms']:
                                atoms.append(AtomicNumber(atom))
                        atoms = np.asarray(atoms, dtype=np.uint8)
                        mol = Mol(atoms, dic['xyz'])
                        mol.properties['mul_charges'] = dic['mul_charges']
                        mol.properties['dipole'] = dic['dipole']
			mol.properties['scf_dipole'] = dic['scf_dipole']
                        mol.properties['energy'] = dic['energy']
			mol.properties['scf_energy'] = dic['scf_energy']
                        mol.properties['gradients'] = dic['gradients']
			mol.properties['atomization'] = dic['energy']
			for i in range (0, mol.NAtoms()):
				mol.properties['atomization'] -= RIMP2Atom[mol.atoms[i]]
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
	
	if (0):
		a = MSet("chemspider9_force")
		a.Load()
		print a.mols[0].properties
		a.mols[0].WriteXYZfile(fname="test")

def TrainForceField():
	if (0):
                a = MSet("chemspider9_force_cleaned")
		a.Load()
                TreatedAtoms = a.AtomTypes()
                PARAMS["hidden1"] = 1000
                PARAMS["hidden2"] = 1000
                PARAMS["hidden3"] = 1000
                PARAMS["learning_rate"] = 0.001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 101
                PARAMS["batch_size"] = 28
                PARAMS["test_freq"] = 2
                PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 0.05
		PARAMS["NeuronType"] = "relu"
		PARAMS["HiddenLayers"] = [200, 200, 200]
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
		#tset = TensorMolData_BP_Multipole_2_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = False)
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE") # Initialzie a manager than manage the training of neural network.
		#manager=TFMolManage("",tset,False,"Dipole_BP_2_Direct")
                manager.Train(maxstep=101)


        if (1):
                a = MSet("H2O_augmented_more_cutoff5_rimp2_force_dipole")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 1200
                PARAMS["batch_size"] = 1000
                PARAMS["test_freq"] = 10
                PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
                PARAMS["NeuronType"] = "relu"
                PARAMS["HiddenLayers"] = [200, 200, 200]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 4.4
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 100
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
                #tset = TensorMolData_BP_Multipole_2_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = False)
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE") # Initialzie a manager than manage the training of neural network.
                #manager=TFMolManage("",tset,False,"Dipole_BP_2_Direct")
                manager.Train()

	if (0):
                a = MSet("H2O_augmented_more_cutoff5_rimp2_force_dipole")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                PARAMS["learning_rate"] = 0.0001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 2
                PARAMS["batch_size"] = 1000
                PARAMS["test_freq"] = 1
                PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
                PARAMS["NeuronType"] = "relu"
                PARAMS["HiddenLayers"] = [200, 200, 200]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 4.4
		PARAMS["EECutoffOff"] = 15.0
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
                #tset = TensorMolData_BP_Multipole_2_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = False)
                manager=TFMolManage("Mol_H2O_augmented_more_cutoff5_rimp2_force_dipole_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_1",tset,False,"fc_sqdiff_BP_Direct_EE") # Initialzie a manager than manage the training of neural network.
                #manager=TFMolManage("",tset,False,"Dipole_BP_2_Direct")
                manager.Continue_Training(target="All")

def EvalForceField():
	if (1):
		a=MSet("H2O_force_test", center_=False)
		a.ReadXYZ("H2O_force_test")
		TreatedAtoms = a.AtomTypes()
		PARAMS["tf_prec"] = "tf.float64"
		PARAMS["EECutoffOn"] = 4.4
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["batch_size"] = 1000
		PARAMS["learning_rate"] = 0.00001
		PARAMS["learning_rate_dipole"] = 0.00001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 200
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
		PARAMS["HiddenLayers"] = [200, 200, 200] 
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
		tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("Mol_H2O_augmented_more_cutoff5_rimp2_force_dipole_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_1",tset,False,"fc_sqdiff_BP_Direct_EE",False,False)
		print manager.EvalBPDirectEESingle(a.mols[0], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		print manager.EvalBPDirectEESingle(a.mols[1], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])



#TestCoulomb()
#TrainPrepare()
TrainForceField()
#EvalForceField()
