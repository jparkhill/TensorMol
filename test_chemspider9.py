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
		B3LYP631GstarAtom={}
		B3LYP631GstarAtom[1]=-0.5002727827
		B3LYP631GstarAtom[6]=-37.8462793509
		B3LYP631GstarAtom[7]=-54.5844908554
		B3LYP631GstarAtom[8]=-75.0606111011
                a = MSet("chemspider9_metady_force")
                dic_list = pickle.load(open("./datasets/chemspider9_metady_force.dat", "rb"))
                for dic in dic_list:
                        atoms = []
                        for atom in dic['atoms']:
                                atoms.append(AtomicNumber(atom))
                        atoms = np.asarray(atoms, dtype=np.uint8)
                        mol = Mol(atoms, dic['xyz'])
                        mol.properties['charges'] = dic['charges']
                        mol.properties['dipole'] = np.asarray(dic['dipole'])
                        mol.properties['quadropole'] = dic['quad']
                        mol.properties['energy'] = dic['scf_energy']
                        mol.properties['gradients'] = dic['gradients']
			mol.properties['atomization'] = dic['scf_energy']
			for i in range (0, mol.NAtoms()):
				mol.properties['atomization'] -= B3LYP631GstarAtom[mol.atoms[i]]
                        a.mols.append(mol)
		a.mols[100].WriteXYZfile(fname="metady_test")
		print a.mols[100].properties
                a.Save()


        if (0):
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


        if (0):
                a = MSet("H2O_augmented_more_cutoff5_rimp2_force_dipole")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 1101
                PARAMS["batch_size"] = 1000
                PARAMS["test_freq"] = 10
                PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
                PARAMS["NeuronType"] = "relu"
                PARAMS["HiddenLayers"] = [200, 200, 200]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 4.0
		PARAMS["Erf_Width"] = 0.2 
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

	#New radius 

        if (0):
                a = MSet("H2O_augmented_more_cutoff5_rimp2_force_dipole")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 901
                PARAMS["batch_size"] = 1000
                PARAMS["test_freq"] = 10
                PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
                PARAMS["NeuronType"] = "relu"
                PARAMS["HiddenLayers"] = [200, 200, 200]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 7.0
		PARAMS["AN1_r_Rc"] = 8.0
		PARAMS["AN1_num_r_Rs"] = 64
		PARAMS["Erf_Width"] = 0.4 
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


	# With Charge Encode
        if (0):
                a = MSet("H2O_augmented_more_cutoff5_rimp2_force_dipole")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 901
                PARAMS["batch_size"] = 1000
                PARAMS["test_freq"] = 10
                PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
                PARAMS["NeuronType"] = "relu"
                PARAMS["HiddenLayers"] = [200, 200, 200]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 7.0
		PARAMS["AN1_r_Rc"] = 8.0
		PARAMS["AN1_num_r_Rs"] = 64
		PARAMS["Erf_Width"] = 0.4
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 100
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
                #tset = TensorMolData_BP_Multipole_2_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = False)
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode") # Initialzie a manager than manage the training of neural network.
                #manager=TFMolManage("",tset,False,"Dipole_BP_2_Direct")
                manager.Train()


	# With Chemspider9 Metadyn
        if (1):
                a = MSet("chemspider9_metady_force")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 101
                PARAMS["batch_size"] = 35
                PARAMS["test_freq"] = 2
                PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
                PARAMS["NeuronType"] = "relu"
                PARAMS["HiddenLayers"] = [1000, 1000, 1000]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 7.0
		PARAMS["Erf_Width"] = 0.4
		#PARAMS["AN1_r_Rc"] = 8.0
		#PARAMS["AN1_num_r_Rs"] = 64
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 10
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
                #tset = TensorMolData_BP_Multipole_2_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = False)
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode") # Initialzie a manager than manage the training of neural network.
                #manager=TFMolManage("",tset,False,"Dipole_BP_2_Direct")
                manager.Train()

def EvalForceField():
	if (1):
		a=MSet("H2O_force_test", center_=False)
		a.ReadXYZ("H2O_force_test")
		TreatedAtoms = a.AtomTypes()
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 300
                PARAMS["batch_size"] = 1000
                PARAMS["test_freq"] = 10
                PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
                PARAMS["NeuronType"] = "relu"
                PARAMS["HiddenLayers"] = [200, 200, 200]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 4.6
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 100
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
		tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("Mol_H2O_augmented_more_cutoff5_rimp2_force_dipole_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_1",tset,False,"fc_sqdiff_BP_Direct_EE",False,False)
		print manager.EvalBPDirectEESet(a, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		#print manager.EvalBPDirectEESingle(a.mols[0], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		#print manager.EvalBPDirectEESingle(a.mols[1], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		#print manager.EvalBPDirectEESingle(a.mols[2], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])


#TestCoulomb()
TrainPrepare()
TrainForceField()
#EvalForceField()
