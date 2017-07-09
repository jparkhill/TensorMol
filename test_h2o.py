"""
Various tests of tensormol's functionality.
Many of these tests take a pretty significant amount of time and memory to complete.
"""
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

def PrepareTrain():
        if (0):
                a = MSet("H2O_400K_squeeze")
                dimers = pickle.load(open("./datasets/H2O_Dimer_400.dat","rb"))
                trimers = pickle.load(open("./datasets/H2O_Trimer_400.dat","rb"))
		monomers = pickle.load(open("./datasets/H2O_Monomer_400.dat","rb"))
                both = dimers + trimers + monomers
                for dic in both:
                        atoms = []
                        for atom in dic['atoms']:
                                atoms.append(AtomicNumber(atom))
                        atoms = np.asarray(atoms, dtype=np.uint8)
                        mol = Mol(atoms, dic['xyz'])
                        mol.properties['charges'] = dic['charges']
                        mol.properties['dipole'] = dic['dipole']
                        mol.properties['quadropole'] = dic['quad']
                        mol.properties['scf_energy'] = dic['scf_energy']
                        mol.properties['rimp2_energy'] = dic['rimp2_energy']
                        mol.properties['energy'] = mol.properties['rimp2_energy']
                        mol.CalculateAtomization()
                        a.mols.append(mol)
                a.Save()

	if (0):
		a = MSet("H2O_augmented_more_cutoff5_b3lyp_force")
		dic_list = pickle.load(open("./datasets/H2O_augmented_more_cutoff5_b3lyp_force.dat", "rb"))
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
		a = MSet("H2O_augmented_more_cutoff5_b3lyp_force")
		a.Load()
		a.mols[50000].WriteXYZfile(fname="water_example")
		print a.mols[50000].properties


	if (0):
		a = MSet("H2O_400K_squeeze")
		a.Load()
		b = MSet("H2O_augmented_more_cutoff5")
		b.Load()
		c = MSet("H2O_augmented_more_400K_squeeze_cutoff5")
		c.mols = list(a.mols+b.mols)
		c.Save()

		


def TestANI1():
	"""
	copy uneq_chemspider from kyao@zerg.chem.nd.edu:/home/kyao/TensorMol/datasets/uneq_chemspider.xyz
	"""
	if (0):
		a = MSet("H2O_augmented_more_squeeze_cutoff5")
		a.Load()
		print "Set elements: ", a.AtomTypes()
		TreatedAtoms = a.AtomTypes()
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
		tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data fo
		tset.BuildTrain("H2O_augmented_more_squeeze_cutoff5")

	if (0):
                a = MSet("H2O_augmented_more_cutoff5")
                a.Load()
                print "Set elements: ", a.AtomTypes()
                TreatedAtoms = a.AtomTypes()
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data fo
                tset.BuildTrain("H2O_augmented_more_cutoff5_float64")



	if (0):
		tset = TensorMolData_BP(MSet(),MolDigester([]),"H2O_augmented_more_cutoff5_float64_ANI1_Sym")
		#tset = TensorMolData_BP(MSet(),MolDigester([]),"H2O_augmented_more_400K_squeeze_cutoff5_ANI1_Sym")
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
		PARAMS["hidden1"] = 100
		PARAMS["hidden2"] = 100
		PARAMS["hidden3"] = 100
		PARAMS["learning_rate"] = 0.00001
		PARAMS["momentum"] = 0.95
		PARAMS["max_steps"] = 2001
		PARAMS["batch_size"] = 10000
		PARAMS["test_freq"] = 10 
		manager.Train(maxstep=2001)
		#manager= TFMolManage("Mol_H2O_400K_squeeze_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
                #manager.Continue_Training(maxsteps=2)

        if (0):
                a = MSet("H2O_augmented_more_cutoff5")
                a.Load()
                print "Set elements: ", a.AtomTypes()
                TreatedAtoms = a.AtomTypes()
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP_Update(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data fo
                tset.BuildTrain("H2O_augmented_more_cutoff5_float64")



        if (0):
                tset = TensorMolData_BP_Update(MSet(),MolDigester([]),"H2O_augmented_more_cutoff5_float64_ANI1_Sym")
                #tset = TensorMolData_BP(MSet(),MolDigester([]),"H2O_augmented_more_400K_squeeze_cutoff5_ANI1_Sym")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Update") # Initialzie a manager than manage the training of neural network.
                PARAMS["hidden1"] = 100
                PARAMS["hidden2"] = 100
                PARAMS["hidden3"] = 100
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 2001
                PARAMS["batch_size"] = 10000
                PARAMS["test_freq"] = 10
		PARAMS["tf_prec"] = "tf.float64"
                manager.Train(maxstep=2001)
                #manager= TFMolManage("Mol_H2O_400K_squeeze_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
                #manager.Continue_Training(maxsteps=2)


	if (0):
		a = MSet("H2O_augmented_more_cutoff5")
                a.Load()
		random.shuffle(a.mols)
		nmols =len(a.mols)
		ncore = 31
		f = []
		for i in range (0, ncore+1):
			f.append(open("H2O_augmented_more_cutoff5_b3lyp_"+str(i)+".in", "w+"))	
		for i, mol in enumerate(a.mols):
			file_index = int(i/(nmols/ncore))
			f[file_index].write("$molecule\n0 1\n")
			for j in range (0, mol.NAtoms()):
				atom_name =  atoi.keys()[atoi.values().index(mol.atoms[j])]
                                f[file_index].write(atom_name+"   "+str(mol.coords[j][0])+ "  "+str(mol.coords[j][1])+ "  "+str(mol.coords[j][2])+"\n")
			f[file_index].write("$end\n")
			f[file_index].write("$rem\njobtype force\nSYM_IGNORE True\nexchange b3lyp\nbasis 6-31g*\n$end\n\n\n@@@\n\n")
			

	if (0):
		#a = MSet("uneq_chemspider")
		#a = MSet("H2O_augmented_more_cutoff5")
		a = MSet("H2O_augmented_more_cutoff5_b3lyp_force")
                a.Load()
                TreatedAtoms = a.AtomTypes()
		PARAMS["hidden1"] = 100
                PARAMS["hidden2"] = 100
                PARAMS["hidden3"] = 100
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 51
                PARAMS["batch_size"] = 1000
                PARAMS["test_freq"] = 10
                PARAMS["tf_prec"] = "tf.float64"
                #PARAMS["AN1_num_r_Rs"] = 16
                #PARAMS["AN1_num_a_Rs"] = 4
                #PARAMS["AN1_num_a_As"] = 4
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP_Direct(a, d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data fo
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=51)

	if (0):
		a = MSet("H2O_dimer_opt")
                a.ReadXYZ("H2O_dimer_opt")
		manager = TFMolManage("Mol_H2O_augmented_more_cutoff5_b3lyp_force_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_1", Trainable_ = False)
		#manager= TFMolManage("Mol_H2O_augmented_more_cutoff5_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_1", Trainable_ = False)
		#manager.Continue_Training(50)
		print manager.Eval_BPEnergy_Direct(a)


        if (1):
                a = MSet("H2O_augmented_more_cutoff5_b3lyp_force")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                PARAMS["hidden1"] = 100
                PARAMS["hidden2"] = 100
                PARAMS["hidden3"] = 100
                PARAMS["learning_rate"] = 0.00001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 1001
                PARAMS["batch_size"] = 1000
                PARAMS["test_freq"] = 10
                PARAMS["tf_prec"] = "tf.float64"
                #PARAMS["AN1_num_r_Rs"] = 16
                #PARAMS["AN1_num_a_Rs"] = 4
                #PARAMS["AN1_num_a_As"] = 4
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
                tset = TensorMolData_BP_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_Grad") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=1001)

        if (0):
                a = MSet("H2O_dimer_opt")
                a.ReadXYZ("H2O_dimer_opt")
                manager = TFMolManage("Mol_H2O_augmented_more_cutoff5_b3lyp_force_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_1", Trainable_ = False)
                #manager= TFMolManage("Mol_H2O_augmented_more_cutoff5_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_1", Trainable_ = False)
                #manager.Continue_Training(50)
                print manager.Eval_BPEnergy_Direct_Grad(a)
	if (0):
                a = MSet("H2O_dimer_opt")
                a.ReadXYZ("H2O_dimer_opt")
		manager= TFMolManage("Mol_H2O_augmented_more_cutoff5_ANI1_Sym_fc_sqdiff_BP_1", None, False)
		optimizer  = Optimizer(manager)
                optimizer.OptANI1(a.mols[0])

	if (0):
		a = MSet("H2O_dimer_opt")
                a.ReadXYZ("H2O_dimer_opt")	
                mol = a.mols[0]
                max_disp  = 0.1
                DistMat = MolEmb.Make_DistMat(mol.coords) 
                dist = DistMat[1][3]
                for i in range (0, 200):
                       new_coords = np.copy(mol.coords)
                       new_coords[3] = (-max_disp + 2.0*max_disp*i/200 + dist)*(mol.coords[3]-mol.coords[1])/dist + mol.coords[1]
                       new_mol = Mol(mol.atoms, new_coords)    
                       new_mol.WriteXYZfile("./results", "H2O_dimer_dispH")    

	if (0):
		a = MSet("H2O_dimer_dispH")
                a.ReadXYZ("H2O_dimer_dispH")
                manager= TFMolManage("Mol_H2O_augmented_more_squeeze_cutoff5_ANI1_Sym_fc_sqdiff_BP_1", None, False)
		energy_list =  manager.Eval_BPEnergy(a)
		for energy in energy_list:
			print energy
		#print "learning_rate:", manager.Instances.learning_rate
		#print "momentum" ,manager.Instances.momentum
		#print "max_steps", manager.Instances.max_steps
		#print "batch_size", manager.Instances.batch_size
		#print "hidden1" , manager.Instances.hidden1
		#print "hidden2" , manager.Instances.hidden2
		#print "hidden3", manager.Instances.hidden3

	if (0):
		a = MSet("H2O_dimer_dispH")
                a.ReadXYZ("H2O_dimer_dispH")
                manager= TFMolManage("Mol_H2O_augmented_more_cutoff5_ANI1_Sym_fc_sqdiff_BP_1", None, False)
		print manager.Eval_BPForceSingle(a.mols[0])
                #for i, mol in enumerate(a.mols):
		#	print manager.Eval_BPEnergySingle(mol)
		

	if (0):
		a = MSet("H2O_dimer_dispH")
                a.ReadXYZ("H2O_dimer_dispH")
                f = open("H2O_dimer_dispH.in","w+")
                for mol in a.mols:
                        f.write("$molecule\n0 1\n")
                        for i in range (0, mol.NAtoms()):
                                atom_name =  atoi.keys()[atoi.values().index(mol.atoms[i])]
                                f.write(atom_name+"   "+str(mol.coords[i][0])+ "  "+str(mol.coords[i][1])+ "  "+str(mol.coords[i][2])+"\n")
                        f.write("$end\n\n$rem\njobtype sp\nmethod rimp2\nbasis cc-pvtz\nAUX_BASIS rimp2-cc-pvtz\nSYM_IGNORE True\n$end\n\n\n@@@\n\n")
                f.close()




	if (0):
		a = MSet("CH3OH_dimer_noHbond")
		a.ReadXYZ("CH3OH_dimer_noHbond")
		manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		optimizer  = Optimizer(manager)
		optimizer.OptANI1(a.mols[0])
	if (0):
		a = MSet("johnsonmols_noH")
		a.ReadXYZ("johnsonmols_noH")
		for mol in a.mols:
			print "mol.coords:", mol.coords
		manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		ins, grad = manager.TData.dig.EvalDigest(a.mols[0])
		#print manager.Eval_BPForce(a.mols[0], True)
		a = MSet("johnsonmols_noH_1")
		a.ReadXYZ("johnsonmols_noH_1")
		#print manager.Eval_BPForce(a.mols[0], True)
		ins1, grad1 = manager.TData.dig.EvalDigest(a.mols[0])
		gradflat =grad.reshape(-1)
		print "grad shape:", grad.shape
		for n in range (0, a.mols[0].NAtoms()):
			diff = -(ins[n] - ins1[n]) /0.001
			for i in range (0,diff.shape[0]):
				if grad[n][i][2] != 0:
					if abs((diff[i] - grad[n][i][2]) / grad[n][i][2]) >  0.01:
						#pass
						print n, i , abs((diff[i] - grad[n][i][2]) / grad[n][i][2]), diff[i],  grad[n][i][2],  grad1[n][i][2], gradflat[n*768*17*3 + i*17*3 +2], n*768*17*3+i*17*3+2, ins[n][i], ins1[n][i]
		for n in range (0, a.mols[0].NAtoms()):
                        diff = -(ins[n] - ins1[n]) /0.001
                        for i in range (0,diff.shape[0]):
                                if grad[n][i][2] != 0:
                                        if abs((grad1[n][i][2] - grad[n][i][2]) / grad[n][i][2]) >  0.01:
						# pass
                                        	print n, i , abs((grad1[n][i][2] - grad[n][i][2]) / grad[n][i][2]), diff[i],  grad[n][i][2],  grad1[n][i][2]
		#t = time.time()
		#print manager.Eval_BPForce(a.mols[0], True)
	if (0):
		a = MSet("md_test")
		a.ReadXYZ("md_test")
		m = a.mols[0]
		tfm= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		# Convert the forces from kcal/mol ang to joules/mol ang.
		ForceField = lambda x: 4183.9953*tfm.Eval_BPForce(Mol(m.atoms,x))
		PARAMS["MNHChain"] = 0
		PARAMS["MDTemp"] = 150.0
		PARAMS["MDThermostat"] = None
		PARAMS["MDV0"]=None
		md = VelocityVerlet(ForceField,m)
		velo_hist = md.Prop()
		autocorr  = AutoCorrelation(velo_hist, md.dt)
		np.savetxt("./results/AutoCorr.dat", autocorr)
	return


def TestDipole():
	if (0):
		a = MSet("H2O_augmented_more_cutoff5")
		a.Load()
		print len(a.mols)
		dimers = pickle.load(open("./datasets/H2O_Dimer_600.dat","rb"))	
		trimers = pickle.load(open("./datasets/H2O_Trimer_600.dat","rb")) 
		both = dimers + trimers
		for dic in both:
			atoms = []
                        for atom in dic['atoms']:
                                atoms.append(AtomicNumber(atom))
                        atoms = np.asarray(atoms, dtype=np.uint8)
                        mol = Mol(atoms, dic['xyz'])
                        mol.properties['charges'] = dic['charges']
			mol.properties['dipole'] = dic['dipole']
                        mol.properties['quadropole'] = dic['quad']
                        mol.properties['scf_energy'] = dic['scf_energy']
			mol.properties['rimp2_energy'] = dic['rimp2_energy']
			mol.properties['energy'] = mol.properties['rimp2_energy']
                        mol.CalculateAtomization()
			a.mols.append(mol)
		b = MSet("H2O_augmented_more_squeeze_cutoff5")
		b.mols = list(a.mols)
		print len(b.mols)
		b.Save()
		

	if (0):
		a = MSet("H2O_augmented_more_cutoff5")
		a.Load()
		TreatedAtoms = a.AtomTypes()
		for mol in a.mols:
			mol.properties["dipole"] = np.asarray(mol.properties["dipole"])
		a.Save()
		#maxOO = 0
		#for mol in a.mols:
		#	dist = MolEmb.Make_DistMat(mol.coords)
		#	OList = filter(lambda i:  mol.atoms[i]==8, range(0, mol.NAtoms()))
		#	for i in OList:
		#		for j in OList:
		#			if i != j and mol.NAtoms() == 6:
		#				OOdist =dist[i][j]
		#				if maxOO < OOdist:
		#					print mol.coords
		#					maxOO = OOdist
		#print maxOO	
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Multipole2")
		tset = TensorMolData_BP_Multipole_2(a,d, order_=1, num_indis_=1, type_="mol")
		tset.BuildTrain("H2O_agumented_more_cutoff5_float64_multipole2")
		

	if (0):
		a = MSet("H2O_augmented")
                a.Load()
		b = MSet("H2O_augmented_cutoff5")
                for mol in a.mols:
                	dist = MolEmb.Make_DistMat(mol.coords)
               		OList = filter(lambda i:  mol.atoms[i]==8, range(0, mol.NAtoms()))
			maxOO = 0
                	for i in OList:
                               for j in OList:
                                       if i != j and mol.NAtoms() == 6:
                                               OOdist =dist[i][j]
                                               if maxOO < OOdist:
                                                       maxOO = OOdist
			if maxOO < 5 :
				b.mols.append(mol)
		print len(b.mols)
		b.Save()
	
        if (0):
                a = MSet("H2O_augmented_more_squeeze_cutoff5")
                a.Load()
		for mol in a.mols:
                        mol.properties['energy'] = mol.properties['rimp2_energy']
			mol.CalculateAtomization()
                a.Save()
	
        if (0):
                a = MSet("H2O_augmented_cutoff5")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                for mol in a.mols:
                        mol.properties["dipole"] = np.asarray(mol.properties["dipole"])
                a.Save()
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Multipole")
                tset = TensorMolData_BP_Multipole(a,d, order_=1, num_indis_=1, type_="mol")
                tset.BuildTrain("H2O_agumented_cutoff5_float64_multipole")


	
        if (0):
                #tset = TensorMolData_BP_Multipole(MSet(),MolDigester([]),"H2O_multipole_ANI1_Sym")
                tset = TensorMolData_BP_Multipole(MSet(),MolDigester([]),"H2O_agumented_cutoff5_multipole_ANI1_Sym")
                manager=TFMolManage("",tset,False,"Dipole_BP")
                manager.Train()


	if (1):
		#tset = TensorMolData_BP_Multipole(MSet(),MolDigester([]),"H2O_multipole_ANI1_Sym")
		tset = TensorMolData_BP_Multipole_2(MSet(),MolDigester([]),"H2O_agumented_more_cutoff5_float64_multipole2_ANI1_Sym")
		PARAMS["tf_prec"] = "tf.float64"
		PARAMS["hidden1"] = 100
                PARAMS["hidden2"] = 100
                PARAMS["hidden3"] = 100
                PARAMS["learning_rate"] = 0.0001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 301
                PARAMS["batch_size"] = 10000
                PARAMS["test_freq"] = 10
		manager=TFMolManage("",tset,False,"Dipole_BP_2")
		manager.Train(maxstep=301)

	if (0):
		a = MSet("furan_md")
		a.ReadXYZ("furan_md")
		manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		energies = manager.EvalBPEnergy(a)
		#np.savetxt("./results/furan_md_nn_energies.dat",energies)
		b3lyp_energies = []
		for mol in a.mols:
			b3lyp_energies.append(mol.properties["atomization"])
		#np.savetxt("./results/furan_md_b3lyp_energies.dat",np.asarray(b3lyp_energies))
	if (0):
		a = MSet("furan_md")
		a.ReadXYZ("furan_md")
                manager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False)
                net, dipole, charge = manager.Eval_BPDipole(a.mols[0], True)
		#net, dipole, charge = manager.Eval_BPDipole(a.mols, True)
		print net, dipole, charge
		#np.savetxt("./results/furan_md_nn_dipole.dat", dipole)

	if (0):
		a = MSet("furan_md")
                a.ReadXYZ("furan_md")
		manager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False)
                net, dipole, charge = manager.EvalBPDipole(a.mols[0], True)
		charge = charge[0]
		fixed_charge_dipole = np.zeros((len(a.mols),3))
		for i, mol in enumerate(a.mols):
			center_ = np.average(mol.coords,axis=0)
        		fixed_charge_dipole[i] = np.einsum("ax,a", mol.coords-center_ , charge)/AUPERDEBYE
		np.savetxt("./results/furan_md_nn_fixed_charge_dipole.dat", fixed_charge_dipole)
	if (0):
		a = MSet("H2O_dimer_flip")
                a.ReadXYZ("H2O_dimer_flip")

		#b = MSet("H2O")
		#b.ReadXYZ("H2O")
		#manager= TFMolManage("Mol_H2O_multipole_ANI1_Sym_Dipole_BP_1" , None, False)
		#net, dipole, charge = manager.Eval_BPDipole(b.mols[0], True)
	
		#nn_charge = np.tile(charge[0],2)
		#mul_charge = np.tile(np.loadtxt("./results/H2O_mul.dat"), 2)
		#hir_charge = np.tile(np.loadtxt("./results/CH3OH_hir.dat"), 2)
		mul_charge = np.loadtxt("./results/H2O_dimer_flip_mul.dat")
		#print mul_charge.shape
		#hir_charge = np.loadtxt("./results/CH3OH_dimer_flip_hir.dat")
		mul_dipole = np.zeros((len(a.mols),3))
		#hir_dipole = np.zeros((len(a.mols),3))
		#nn_dipole = np.zeros((len(a.mols),3))
		for i, mol in enumerate(a.mols):
                        center_ = np.average(mol.coords,axis=0)
			print mol.coords.shape
			mul_dipole[i] = np.einsum("ax,a", mol.coords-center_ , mul_charge[i])/AUPERDEBYE
			#hir_dipole[i] = np.einsum("ax,a", mol.coords-center_ , hir_charge)/AUPERDEBYE 
			#nn_dipole[i] = np.einsum("ax,a", mol.coords-center_ , nn_charge)
                        #mul_dipole[i] = np.einsum("ax,a", mol.coords-center_ , mul_charge[i])/AUPERDEBYE
			#hir_dipole[i] = np.einsum("ax,a", mol.coords-center_ , hir_charge[i])/AUPERDEBYE			
			#nn_dipole[i] = np.einsum("ax,a", mol.coords-center_ , nn_charge[i])/AUPERDEBYE
				
                np.savetxt("./results/H2O_dimer_flip_mul_dipole.dat", mul_dipole)
		#np.savetxt("./results/CH3OH_dimer_flip_hir_dipole.dat", hir_dipole)
		#np.savetxt("./results/H2O_dimer_flip_fixed_nn_dipole.dat", nn_dipole)


	if (0):
		a = MSet("H2O_dimer_flip")
                a.ReadXYZ("H2O_dimer_flip")
		#manager= TFMolManage("Mol_H2O_agumented_more_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False)
		##manager= TFMolManage("Mol_H2O_multipole_ANI1_Sym_Dipole_BP_1" , None, False)
		#nn_dip = np.zeros((len(a.mols),3))
		#nn_charge = np.zeros((len(a.mols),a.mols[0].NAtoms()))
		#for i, mol in enumerate(a.mols):
                #	dipole, charge = manager.Eval_BPDipole_2(mol, True)
		#	print dipole, charge[0]
		#	nn_dip[i] = dipole
		#	nn_charge[i] = charge[0]
		#np.savetxt("./results/H2O_dimer_flip_nn_dip_4.dat", nn_dip)
		#np.savetxt("./results/H2O_dimer_flip_nn_charge_4.dat", nn_charge)
		f = open("H2O_dimer_flip_mp2.in","w+")
		for mol in a.mols:
			f.write("$molecule\n0 1\n")
			for i in range (0, mol.NAtoms()):
				atom_name =  atoi.keys()[atoi.values().index(mol.atoms[i])]
                		f.write(atom_name+"   "+str(mol.coords[i][0])+ "  "+str(mol.coords[i][1])+ "  "+str(mol.coords[i][2])+"\n")
			f.write("$end\n\n$rem\njobtype sp\nmethod rimp2\nAUX_BASIS rimp2-cc-pvtz\nbasis cc-pvtz\nSYM_IGNORE True\n$end\n\n\n@@@\n\n")
		f.close()	


        if (0):
                a = MSet("H2O_dimer_opt")
                a.ReadXYZ("H2O_dimer_opt")
                mol = a.mols[0]
                DistMat = MolEmb.Make_DistMat(mol.coords)
                dist = DistMat[1][3]
		v = (mol.coords[3]-mol.coords[1])/dist
                for i in range (0, 20):
                	new_coords = np.copy(mol.coords)
			new_coords[3] = i*(mol.coords[3]-mol.coords[1])/dist + mol.coords[3]
			new_coords[4] = i*(mol.coords[3]-mol.coords[1])/dist + mol.coords[4]
			new_coords[5] = i*(mol.coords[3]-mol.coords[1])/dist + mol.coords[5]
                	new_mol = Mol(mol.atoms, new_coords)
                	new_mol.WriteXYZfile("./datasets", "H2O_dimer_di_di")

        if (0):
                a = MSet("H2O_dimer_di_di")
                a.ReadXYZ("H2O_dimer_di_di")
                f = open("H2O_dimer_di_di.in","w+")
                for m in a.mols:
			m1 = Mol(m.atoms[:m.NAtoms()/2], m.coords[:m.NAtoms()/2])
			m2 = Mol(m.atoms[m.NAtoms()/2:], m.coords[m.NAtoms()/2:])
			for mol in [m, m1, m2]:
                        	f.write("$molecule\n0 1\n")
                        	for i in range (0, mol.NAtoms()):
                        	        atom_name =  atoi.keys()[atoi.values().index(mol.atoms[i])]
                        	        f.write(atom_name+"   "+str(mol.coords[i][0])+ "  "+str(mol.coords[i][1])+ "  "+str(mol.coords[i][2])+"\n")
                        	f.write("$end\n\n$rem\njobtype sp\nmethod rimp2\nbasis cc-pvtz\nAUX_BASIS rimp2-cc-pvtz\nSYM_IGNORE True\n$end\n\n\n@@@\n\n")
                f.close()


	if (0):
		a = MSet("H2O_dimer_di_di")
                a.ReadXYZ("H2O_dimer_di_di")
		manager= TFMolManage("Mol_H2O_agumented_more_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False)
		for m in a.mols:
			dipole, charge = manager.Eval_BPDipole_2(m, True)
			m.properties['atom_charges'] = np.asarray(charge[0])/1.889725989
			print m.properties['atom_charges']
			m1 = Mol(m.atoms[:m.NAtoms()/2], m.coords[:m.NAtoms()/2])
			m1.properties['atom_charges'] = m.properties['atom_charges'][:m.NAtoms()/2]
			m2 = Mol(m.atoms[m.NAtoms()/2:], m.coords[m.NAtoms()/2:])
			m2.properties['atom_charges'] = m.properties['atom_charges'][m.NAtoms()/2:]
			print  "energy:", ChargeCharge(m1, m2)



        if (0):
                a = MSet("H2O_dist_flip")
                a.ReadXYZ("H2O_dist_flip")
                f = open("H2O_dist_flip.in","w+")
                for m in a.mols:
                        m1 = Mol(m.atoms[:m.NAtoms()/2], m.coords[:m.NAtoms()/2])
                        m2 = Mol(m.atoms[m.NAtoms()/2:], m.coords[m.NAtoms()/2:])
                        for mol in [m, m1, m2]:
                                f.write("$molecule\n0 1\n")
                                for i in range (0, mol.NAtoms()):
                                        atom_name =  atoi.keys()[atoi.values().index(mol.atoms[i])]
                                        f.write(atom_name+"   "+str(mol.coords[i][0])+ "  "+str(mol.coords[i][1])+ "  "+str(mol.coords[i][2])+"\n")
                                f.write("$end\n\n$rem\njobtype sp\nmethod rimp2\nbasis cc-pvtz\nAUX_BASIS rimp2-cc-pvtz\nSYM_IGNORE True\n$end\n\n\n@@@\n\n")
                f.close()


        if (0):
                a = MSet("H2O_dist_flip")
                a.ReadXYZ("H2O_dist_flip")
                manager= TFMolManage("Mol_H2O_agumented_more_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False)
                for m in a.mols:
                        dipole, charge = manager.Eval_BPDipole_2(m, True)
                        m.properties['atom_charges'] = np.asarray(charge[0])/1.889725989

                        m1 = Mol(m.atoms[:m.NAtoms()/2], m.coords[:m.NAtoms()/2])
                        m1.properties['atom_charges'] = m.properties['atom_charges'][:m.NAtoms()/2]
                        m2 = Mol(m.atoms[m.NAtoms()/2:], m.coords[m.NAtoms()/2:])
                        m2.properties['atom_charges'] = m.properties['atom_charges'][m.NAtoms()/2:]
                        print  "energy:", ChargeCharge(m1, m2)


#PrepareTrain()
TestANI1()	
#TestDipole()
