from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from TensorMol.ElectrostaticsTF import *
from TensorMol.NN_MBE import *


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

	if (1):
		RIMP2Atom={}
		RIMP2Atom[1]=-0.4998098112
		RIMP2Atom[8]=-74.9659650581
                a = MSet("H2O_augmented_more_bowl02_rimp2_force_dipole")
                dic_list_1 = pickle.load(open("./datasets/H2O_augmented_more_cutoff5_rimp2_force_dipole.dat", "rb"))
		dic_list_2 = pickle.load(open("./datasets/H2O_long_dist_pair.dat", "rb"))
		dic_list_3 = pickle.load(open("./datasets/H2O_metady_bowl02.dat", "rb"))
		dic_list = dic_list_1 + dic_list_2 + dic_list_3
		random.shuffle(dic_list)
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

		b = MSet("H2O_bowl02_rimp2_force_dipole")
                for dic in dic_list_3:
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
                        b.mols.append(mol)
		print "number of a mols:", len(a.mols)
		print "number of b mols:", len(b.mols)
                a.Save()
		b.Save()
		

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

	#New radius: 8 A

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

	#New radius: 5 A 
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
		PARAMS["AN1_r_Rc"] = 5.0
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


	#New radius: 8 A and long distance pair

        if (0):
                a = MSet("H2O_augmented_more_rimp2_force_dipole")
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
                PARAMS["HiddenLayers"] = [512, 512, 512]
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


        if (1):
		#New radius: 8 A and long distance pair and bowl potential with K=0.2
                #a = MSet("H2O_augmented_more_bowl02_rimp2_force_dipole")
		a =  MSet("H2O_bowl02_rimp2_force_dipole")
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


	# With Chemspider9 Metadyn
        if (0):
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
	if (0):
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
                PARAMS["EECutoffOn"] = 7.0
                PARAMS["AN1_r_Rc"] = 5.0
                PARAMS["AN1_num_r_Rs"] = 32
                PARAMS["Erf_Width"] = 0.4
                PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 100
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
		tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("Mol_H2O_augmented_more_cutoff5_rimp2_force_dipole_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_1",tset,False,"fc_sqdiff_BP_Direct_EE",False,False)
		#out_list = manager.EvalBPDirectEESet(a, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		#print out_list
		#print "gradient: ", out_list[-1]/BOHRPERA
		#print manager.EvalBPDirectEESingle(a.mols[0], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		m = a.mols[1]
		def EnAndForce(x_):
                        m.coords = x_
                        Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEESingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
                        energy = Etotal[0]
                        force = gradient[0]
                        return energy, force
                ForceField = lambda x: EnAndForce(x)[-1]
                EnergyForceField = lambda x: EnAndForce(x)

		PARAMS["MDdt"] = 0.2
        	PARAMS["RemoveInvariant"]=True
        	PARAMS["MDMaxStep"] = 2000
        	PARAMS["MDThermostat"] = "Nose"
        	PARAMS["MDV0"] = None
		PARAMS["MDAnnealTF"] = 0.0
                PARAMS["MDAnnealT0"] = 300.0
		PARAMS["MDAnnealSteps"] = 2000	
       	 	anneal = Annealer(EnergyForceField, None, m, "Anneal")
       	 	anneal.Prop()
       	 	m.coords = anneal.Minx.copy()
       	 	m.WriteXYZfile("./results/", "Anneal_opt")
		raise Exception("Aneal Ended")
		#Opt = GeomOptimizer(EnergyForceField)
		#Opt.Opt(m)
                #PARAMS["MDThermostat"] = "Nose"
                #PARAMS["MDTemp"] = 30
                #PARAMS["MDdt"] = 0.1
                #PARAMS["RemoveInvariant"]=True
                #PARAMS["MDV0"] = None
                #PARAMS["MDMaxStep"] = 10000
                #md = VelocityVerlet(None, m, "11OO",EnergyForceField)
                #md.Prop()

		#mset=MSet("NeigborMB_test")
		#mset.ReadXYZ("NeigborMB_test")
		#MBEterms = MBNeighbors(mset.mols[0].coords, mset.mols[0].atoms, [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]])
		#mbe =  NN_MBE_Linear(manager)
		#def EnAndForce(x_):
                #        mset.mols[0].coords = x_
		#	MBEterms.Update(mset.mols[0].coords, 10.0, 10.0)
                #        Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEESingle(b.mols[0], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		#	mbe.EnergyForceDipole(MBEterms)
                #        energy = Etotal[0]
                #        force = gradient[0]
                #        return energy, force
                #EnergyForceField = lambda x: EnAndForce(x)

		#Opt = GeomOptimizer(EnergyForceField)
		#Opt.Opt(b.mols[0])
		#MBEterms.Update(mset.mols[0].coords, 10.0, 10.0)
		#mbe =  NN_MBE_Linear(manager)
		#mbe.EnergyForceDipole(MBEterms)
		
		def EnAndForce(x_):
                        m.coords = x_
                        Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEESingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
                        energy = Etotal[0]
                        force = gradient[0]
                        return energy, force

		def EnForceCharge(x_):
                        m.coords = x_
                        Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEESingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
                        energy = Etotal[0]
                        force = gradient[0]
                        return energy, force, atom_charge

		def ChargeField(x_):
                        m.coords = x_
                        Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEESingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
                        energy = Etotal[0]
                        force = gradient[0]
                        return atom_charge[0]

                ForceField = lambda x: EnAndForce(x)[-1]
		EnergyField = lambda x: EnAndForce(x)[0]
                EnergyForceField = lambda x: EnAndForce(x)

		#PARAMS["OptMaxCycles"]=200
		#Opt = GeomOptimizer(EnergyForceField)
		#m=Opt.Opt(m)

		PARAMS["MDdt"] = 0.2
        	PARAMS["RemoveInvariant"]=True
        	PARAMS["MDMaxStep"] = 10000
        	PARAMS["MDThermostat"] = "Nose"
        	PARAMS["MDV0"] = None
		PARAMS["MDAnnealTF"] = 300.0
                PARAMS["MDAnnealT0"] = 0.0
		PARAMS["MDAnnealSteps"] = 10000	
       	 	anneal = Annealer(EnergyForceField, None, m, "Anneal")
       	 	anneal.Prop()
       	 	m.coords = anneal.x.copy()
       	 	m.WriteXYZfile("./results/", "Anneal_opt")
	        PARAMS["MDThermostat"] = None
	        PARAMS["MDTemp"] = 0
	        PARAMS["MDdt"] = 0.1
	        PARAMS["MDV0"] = None
	        PARAMS["MDMaxStep"] = 40000
	        md = IRTrajectory(EnAndForce, ChargeField, m, "IR")
	        md.Prop()
		WriteDerDipoleCorrelationFunction(md.mu_his)		



	if (1):
		a=MSet("NeigborMB_test", center_=False)
		a.ReadXYZ("NeigborMB_test")
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
                PARAMS["EECutoffOn"] = 7.0
                PARAMS["AN1_r_Rc"] = 5.0
                PARAMS["AN1_num_r_Rs"] = 32
                PARAMS["Erf_Width"] = 0.4
                PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 100
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
		tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("Mol_H2O_augmented_more_cutoff5_rimp2_force_dipole_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_1",tset,False,"fc_sqdiff_BP_Direct_EE",False,False)
		#out_list = manager.EvalBPDirectEESet(a, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		#print out_list
		#print "gradient: ", out_list[-1]/BOHRPERA
		#print manager.EvalBPDirectEESingle(a.mols[0], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		m = a.mols[5]
		#PARAMS["MDdt"] = 0.2
        	#PARAMS["RemoveInvariant"]=True
        	#PARAMS["MDMaxStep"] = 2000
        	#PARAMS["MDThermostat"] = "Nose"
        	#PARAMS["MDV0"] = None
		#PARAMS["MDAnnealTF"] = 0.0
                #PARAMS["MDAnnealT0"] = 300.0
		#PARAMS["MDAnnealSteps"] = 2000	
       	 	#anneal = Annealer(EnergyForceField, None, m, "Anneal")
       	 	#anneal.Prop()
       	 	#m.coords = anneal.Minx.copy()
       	 	#m.WriteXYZfile("./results/", "Anneal_opt")

		mono_index = []
		for i in range (0, 10):
			mono_index.append([i*3, i*3+1, i*3+2])
		MBEterms = MBNeighbors(m.coords, m.atoms, mono_index)
		mbe =  NN_MBE_Linear(manager)
		def EnAndForce(x_):
                        m.coords = x_
			MBEterms.Update(m.coords, 10, 10)
                        Etotal, gradient, charge = mbe.EnergyForceDipole(MBEterms)
                        energy = Etotal
                        force = gradient
                        return energy, force

                EnergyForceField = lambda x: EnAndForce(x)
		print EnergyForceField(m.coords)
		raise Exception("Stop here for debugging")
		#Opt = GeomOptimizer(EnergyForceField)
		#Opt.Opt(m)

		#PARAMS["MDdt"] = 0.2
        	#PARAMS["RemoveInvariant"]=True
        	#PARAMS["MDMaxStep"] = 2000
        	#PARAMS["MDThermostat"] = "Nose"
        	#PARAMS["MDV0"] = None
		#PARAMS["MDAnnealTF"] = 0.0
                #PARAMS["MDAnnealT0"] = 100.0
		#PARAMS["MDAnnealSteps"] = 200	
       	 	#anneal = Annealer(EnergyForceField, None, m, "Anneal")
       	 	#anneal.Prop()
       	 	#m.coords = anneal.Minx.copy()
       	 	#m.WriteXYZfile("./results/", "Anneal_opt")
	
		def ChargeField(x_):
			m.coords = x_
                        MBEterms.Update(m.coords, 20.0, 20.0)
                        Etotal, gradient, charge = mbe.EnergyForceDipole(MBEterms)
			return charge

		ForceField = lambda x: EnAndForce(x)[-1]
		EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)

		#PARAMS["MDdt"] = 0.1
		#PARAMS["RemoveInvariant"]=True
		#PARAMS["MDMaxStep"] = 10000
		#PARAMS["MDThermostat"] = "Nose"
		#PARAMS["MDV0"] = None
		#PARAMS["MDAnnealTF"] = 30.0
		#PARAMS["MDAnnealT0"] = 0.1
		#PARAMS["MDAnnealSteps"] = 10000
		#anneal = Annealer(EnergyForceField, None, m, "Anneal")
		#anneal.Prop()
		#m.coords = anneal.x.copy()

	
                PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDTemp"] = 30
                PARAMS["MDdt"] = 0.1
                PARAMS["RemoveInvariant"]=True
                PARAMS["MDV0"] = None
                PARAMS["MDMaxStep"] = 10000
                warm = VelocityVerlet(ForceField, m,"warm",EnergyForceField)
                warm.Prop()
                m.coords = warm.x.copy()

		PARAMS["MDThermostat"] = None
		PARAMS["MDTemp"] = 0
		PARAMS["MDdt"] = 0.1
		PARAMS["MDV0"] = None
		PARAMS["MDMaxStep"] = 40000
		md = IRTrajectory(EnAndForce, ChargeField, m, "IR", warm.v)
		md.Prop()
		WriteDerDipoleCorrelationFunction(md.mu_his)

	if (0):
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		a = MSet("chemspider9_metady_force")
		a.Load()
		b = MSet("chemspider9_IR_test")
		#b = MSet("david_test")
		b.ReadXYZ()
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
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 10
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
		tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
		manager=TFMolManage("Mol_chemspider9_metady_force_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_1",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode", False, False) # Initialzie a manager than manage the training of neural network.
		#print manager.EvalBPDirectEESet(b, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		#print manager.EvalBPDirectEESingle(a.mols[1], PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		#print a.mols[1].properties, "Dipole in a.u.:",a.mols[1].properties["dipole"]*0.393456

		m = b.mols[2]
		def EnAndForce(x_):
			m.coords = x_
			Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEESingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
			energy = Etotal[0]
			force = gradient[0]
			return energy, force

		def EnForceCharge(x_):
			m.coords = x_
			Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEESingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
			energy = Etotal[0]
			force = gradient[0]
			return energy, force, atom_charge

		def ChargeField(x_):
			m.coords = x_
			Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEESingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
			energy = Etotal[0]
			force = gradient[0]
			return atom_charge[0]

		ForceField = lambda x: EnAndForce(x)[-1]
		EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)

		PARAMS["OptMaxCycles"]=200
		Opt = GeomOptimizer(EnergyForceField)
		m=Opt.Opt(m)

		PARAMS["MDdt"] = 0.2
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDMaxStep"] = 10000
		PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDV0"] = None
		PARAMS["MDAnnealTF"] = 200.0
		PARAMS["MDAnnealT0"] = 0.1
		PARAMS["MDAnnealSteps"] = 10000
		anneal = Annealer(EnergyForceField, None, m, "Anneal")
		anneal.Prop()
		m.coords = anneal.x.copy()
		m.WriteXYZfile("./results/", "Anneal_opt")
		PARAMS["MDThermostat"] = None
		PARAMS["MDTemp"] = 0
		PARAMS["MDdt"] = 0.1
		PARAMS["MDV0"] = None
		PARAMS["MDMaxStep"] = 40000
		md = IRTrajectory(EnAndForce, ChargeField, m, "IR", anneal.v)
		md.Prop()
		WriteDerDipoleCorrelationFunction(md.mu_his)


def TestMetadynamics(mset_name_, name_, threads_):
	a = MSet(mset_name_)
	a.ReadXYZ()
	m = a.mols[0]
	#ForceField = lambda x: QchemRIMP2(Mol(m.atoms,x), jobtype_='force', filename_='H2O_Trimer_BowlP', path_='./qchem/', threads=12)
	#ForceField = lambda x: QchemDFT(Mol(m.atoms,x),basis_ = '6-311g**',xc_='wB97X-D', jobtype_='force', filename_='metady_test', path_='./qchem/', threads=12)
	ForceField = lambda x: QchemDFT(Mol(m.atoms,x),basis_ = '6-31g*',xc_='b3lyp', jobtype_='force', filename_='H2O_BowlP_'+name_, path_='./qchem/', threads=threads_)
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms))
	print "Masses:", masses
	PARAMS["MDdt"] = 2.0
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 1000
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDTemp"]= 600.0
	PARAMS["MetaBowlK"] = 0.2
	meta = MetaDynamics(ForceField, m)
	meta.Prop()

#TestCoulomb()
#TrainPrepare()
#TrainForceField()
EvalForceField()
#TestMetadynamics("H2O_Trimer")
