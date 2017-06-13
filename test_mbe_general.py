from TensorMol import *
from TensorMol.NN_MBE import *
from TensorMol.MBE_Opt import *
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# step to test a BruteForce MBE model
if (1):
	if (0):
		a=MSet("H2O_cluster")
                a.ReadXYZ("H2O_cluster")
		manager= TFMolManage("Mol_H2O_augmented_more_squeeze_cutoff5_ANI1_Sym_fc_sqdiff_BP_1", None, False)	
		dipole_manager= TFMolManage("Mol_H2O_agumented_more_squeeze_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False)
		def EnAndForce(x_):
                        a.mols[0].coords = x_
                        energy, force = manager.Eval_BPForceSet(a)
                        energy = energy[0]
                        force = force[0]
                        return energy, force
                ForceField = lambda x: EnAndForce(x)[1]
                EnergyForceField = lambda x: EnAndForce(x)

                def ChargeField(x_):
                        a.mols[0].coords = x_
                        dipole, charge = dipole_manager.Eval_BPDipole_2(a.mols[0])
                        return np.asarray(charge[0])
		optimizer = Optimizer(manager)
		optimizer.OptANI1(a.mols[0])		
		PARAMS["MDdt"] = 0.2
       	 	PARAMS["RemoveInvariant"]=True
       	 	PARAMS["MDMaxStep"] = 10000
       	 	PARAMS["MDThermostat"] = "Nose"
       	 	PARAMS["MDV0"] = None
       	 	PARAMS["MDTemp"]= 1.0
       	 	#annealIndo = Annealer(EnergyForceField, ChargeField, a.mols[0], "Anneal")
       	 	#annealIndo.Prop()
       	 	#a.mols[0].coords = annealIndo.Minx.copy()
       	 	#a.mols[0].WriteXYZfile("./results/", "h2o_dimer_opt")


	if (0):
		#a=MSet("H2O_mono")
		#a.ReadXYZ("H2O_mono")
		a=MSet("H2O_4water")
                a.ReadXYZ("H2O_4water")
		manager= TFMolManage("Mol_H2O_augmented_more_cutoff5_ANI1_Sym_fc_sqdiff_BP_1", None, False)
		dipole_manager= TFMolManage("Mol_H2O_agumented_more_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False)
                def EnAndForce(x_):
                        a.mols[0].coords = x_
			energy, force = manager.Eval_BPForceSet(a)
                	energy = energy[0]
                	force = force[0]
                        return energy, force
                ForceField = lambda x: EnAndForce(x)[1]
                EnergyForceField = lambda x: EnAndForce(x)
		
		def ChargeField(x_):
                        a.mols[0].coords = x_
                        dipole, charge = dipole_manager.Eval_BPDipole_2(a.mols[0])
                        return np.asarray(charge[0])

		#PARAMS["MDThermostat"] = "Nose"
                #PARAMS["MDTemp"] = 200.0
                #PARAMS["MDdt"] = 0.2
                #md = VelocityVerlet(ForceField,a.mols[0],"H2O_udp_md",EnergyForceField)
                #md.Prop()


		#optimizer = Optimizer(manager)
        	#optimizer.OptANI1(a.mols[0])

		PARAMS["MDFieldAmp"] = 0.0 #0.00000001
		PARAMS["MDFieldTau"] = 0.4
		PARAMS["MDFieldFreq"] = 0.8
		PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
	       	PARAMS["MDThermostat"] = "Nose"
                PARAMS["MDTemp"] = 30
                PARAMS["MDdt"] = 0.1
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDV0"] = None
		PARAMS["MDMaxStep"] = 10000
		warm = VelocityVerlet(ForceField, a.mols[0],"warm",EnergyForceField)
		warm.Prop()
		a.mols[0].coords = warm.x.copy()
		PARAMS["MDMaxStep"] = 40000
		md = IRTrajectory(EnergyForceField, ChargeField, a.mols[0],"H2O_udp_IR",warm.v.copy())
                md.Prop()
		WriteDerDipoleCorrelationFunction(md.mu_his,"H2O_udp_IR.txt")

	if (0):
		a=FragableMSetBF("H2O_cluster")
		a.ReadXYZ("H2O_cluster")
		#a=FragableMSetBF("H2O_dimer")
                #a.ReadXYZ("H2O_dimer")	
	
		print "Generate_All_MBE_term_General: "
		a.Generate_All_MBE_term_General([{"atom":"HOH", "charge":0}])
		print "End of Generate_All_MBE_term_General"	

		manager= TFMolManage("Mol_H2O_augmented_more_squeeze_cutoff5_ANI1_Sym_fc_sqdiff_BP_1", None, False, Trainable_ = False)
		dipole_manager= TFMolManage("Mol_H2O_agumented_more_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False, Trainable_ = False)
		mbe = NN_MBE_BF(manager, dipole_manager)

		Opt = MBE_Optimizer(mbe)
		for mol in a.mols:
			#Opt.MBE_Opt(mol)
			#mbe.NN_Energy_Force(mol)
			#mbe.NN_Energy(mol, False)
			#mbe.NN_Dipole(mol)
			mbe.NN_Charge(mol)
			#mbe.NN_Energy(mol, True)


	if (1):
		a = MSet("H2O_udp")
		a.ReadXYZ("H2O_udp")
		#manager= TFMolManage("Mol_H2O_augmented_more_squeeze_cutoff5_ANI1_Sym_fc_sqdiff_BP_1", None, False, Trainable_ = False)
                dipole_manager= TFMolManage("Mol_H2O_agumented_more_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False, Trainable_ = False)
		#for mol in a.mols:
			#manager.Eval_BPForceSingle(mol)
		#	dipole_manager.Eval_BPDipoleGrad_2(mol)
		print dipole_manager.Eval_BPDipoleGrad_2(a)

if (0): 
                a=FragableMSetBF("H2O_cluster_opt", center_=False)
                a.ReadXYZ("H2O_cluster_opt")

                print "Generate_All_MBE_term_General: "
                a.Generate_All_MBE_term_General([{"atom":"HOH", "charge":0}])
                print "End of Generate_All_MBE_term_General"
		#manager= TFMolManage("Mol_H2O_augmented_more_400K_squeeze_cutoff5_ANI1_Sym_fc_sqdiff_BP_1",None,False)
                manager= TFMolManage("Mol_H2O_augmented_more_squeeze_cutoff5_ANI1_Sym_fc_sqdiff_BP_1", None, False)
                dipole_manager= TFMolManage("Mol_H2O_agumented_more_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False)
                mbe = NN_MBE_BF(manager, dipole_manager)
                def EnAndForce(x_):
			a.mols[0].coords = x_
			a.mols[0].Reset_Frags()
			#a.mols[0].Generate_All_MBE_term_General([{"atom":"HOH", "charge":0}])	
			return mbe.NN_Energy_Force(a.mols[0])
		ForceField = lambda x: EnAndForce(x)[1]
		EnergyForceField = lambda x: EnAndForce(x)
	
		def ChargeField(x_):
                        a.mols[0].coords = x_
                        charge =  mbe.NN_Charge(a.mols[0])
                        return charge
	
		#PARAMS["MDThermostat"] = None
		#PARAMS["MDTemp"] = 200.0
		#PARAMS["MDdt"] = 0.2
		#md = VelocityVerlet(ForceField,a.mols[0],"MBE_test_opt",EnergyForceField)
		#md.Prop()

		#PARAMS["MDdt"] = 0.2
                #PARAMS["RemoveInvariant"]=True
                #PARAMS["MDMaxStep"] = 10000
                #PARAMS["MDThermostat"] = "Nose"
                #PARAMS["MDV0"] = None
                #PARAMS["MDTemp"]= 1.0
                #annealH2O = Annealer(EnergyForceField, ChargeField, a.mols[0], "Anneal")
                #annealH2O.Prop()
		#a.mols[0].coords = annealH2O.Minx.copy()
                #a.mols[0].WriteXYZfile("./results/", "h2o_cluster_anneal_opt")

		PARAMS["MDFieldAmp"] = 0.0 #0.00000001
                PARAMS["MDFieldTau"] = 0.4
                PARAMS["MDFieldFreq"] = 0.8
                PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
                PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDTemp"] = 300
                #PARAMS["MDTemp"] = 30
                PARAMS["MDdt"] = 0.1
                PARAMS["RemoveInvariant"]=True
                PARAMS["MDV0"] = None
                PARAMS["MDMaxStep"] = 10000
                warm = VelocityVerlet(ForceField, a.mols[0],"warm",EnergyForceField)
                warm.Prop()
                a.mols[0].coords = warm.x.copy()
                PARAMS["MDMaxStep"] = 40000
                md = IRTrajectory(EnergyForceField, ChargeField, a.mols[0],"H2O_cluster_IR",warm.v.copy())
                md.Prop()
                WriteDerDipoleCorrelationFunction(md.mu_his,"H2O_cluster_IR.txt")


if (0):
                a=MSet("water_small_md")
                a.ReadXYZ("water_small_md")
		OO_dist_list = []
		OO_dist_list_trimer = []
		min_accept = 2.7
		max_accept = 5
		tri_max_accept = 4
		dimer_accept_rate = 1.0/100.0
		trimer_accept_rate = 1.0/6.0
		t = time.time()
		mono_index = 0
		mono_max = 30000
		for mol_index, mol in enumerate(a.mols):
	          if 1000 < mol_index :
			print mol_index
			dist_mat =  MolEmb.Make_DistMat(mol.coords)
			for i in range (0, mol.NAtoms()):
				if mono_index < mono_max and mol.atoms[i] == 8:
					m = Mol(mol.atoms[[i,i+1,i+2]], mol.coords[[i,i+1,i+2]])
                                	m.WriteXYZfile(fname="monomer")
					mono_index += 1
				for j in range (i+1, mol.NAtoms()):
					if dist_mat[i][j] < max_accept and mol.atoms[i] ==8 and mol.atoms[j] == 8:
						p_accept = ((dist_mat[i][j]- max_accept) / (min_accept - max_accept))**2
						if (random.random() < p_accept*dimer_accept_rate):
							m = Mol(mol.atoms[[i,i+1,i+2, j, j+1, j+2]], mol.coords[[i,i+1,i+2, j, j+1, j+2]])
							m.WriteXYZfile(fname="dimer")
							#OO_dist_list.append(dist_mat[i][j])
					if dist_mat[i][j] < tri_max_accept and mol.atoms[i] ==8 and mol.atoms[j] == 8:
						for k in range (j+1, mol.NAtoms()):
							if dist_mat[i][k] < tri_max_accept and dist_mat[j][k] < tri_max_accept and mol.atoms[k] ==8:
								p_accept_trimer = ((dist_mat[i][j]- tri_max_accept) / (min_accept - tri_max_accept))**2* ((dist_mat[i][k]- tri_max_accept) / (min_accept - tri_max_accept))**2 * ((dist_mat[j][k]- tri_max_accept) / (min_accept - tri_max_accept))**2
								if (random.random() < p_accept_trimer*trimer_accept_rate ):
									m = Mol(mol.atoms[[i,i+1,i+2, j, j+1, j+2, k,k+1, k+2]], mol.coords[[i,i+1,i+2, j, j+1, j+2, k, k+1, k+2]])
		                                                        m.WriteXYZfile(fname="trimer")
									#OO_dist_list_trimer.append(dist_mat[i][j]*dist_mat[i][k]*dist_mat[j][k])
		#print "len of list per case trimer:", len(OO_dist_list_trimer)/100.0 
		#print "len of list per case:", len(OO_dist_list)/100.0
		#print "time per case:", (time.time() - t)/ 100.0
		#OO_dist_list = np.asarray(OO_dist_list)
		#OO_dist_list_trimer = np.asarray(OO_dist_list_trimer)
		#np.savetxt("OO_dist.dat", OO_dist_list)
		#np.savetxt("OO_dist_trimer.dat", OO_dist_list_trimer)
		


# steps to train a NN-MBE model
if (0):
	#Load .xyz files.

	if (1):
		a=FragableMSet("NaClH2O")
		a.ReadXYZ("NaClH2O")
		a.Generate_All_Pairs(pair_list=[{"pair":"NaCl", "mono":["Na","Cl"], "center":[0,0]}])
                #a.Generate_All_MBE_term_General([{"atom":"OHHNa", "charge":1}, {"atom":"OHHCl", "charge":-1},{"atom":"OHH", "charge":0}], cutoff=10, center_atom=[0,0,0]) # Generate all the many-body terms with  certain radius cutoff.
                a.Generate_All_MBE_term_General([{"atom":"OHH", "charge":0}, {"atom":"NaCl", "charge":0}], cutoff=12, center_atom=[0, -1]) # Generate all the many-body terms with  certain radius cutoff.  # -1 means center of mass
                a.Save() # Save the training set, by default it is saved in ./datasets.

	if (1):
                #a=MSet("NaCl_H2O_NaH2Ogroup") 
                a=FragableMSet("NaClH2O")
                a.Load() # Load generated training set (.pdb file).
                a.Calculate_All_Frag_Energy_General(method="qchem")  # Use PySCF or Qchem to calcuate the MP2 many-body energy of each order.
                #a.Get_All_Qchem_Frag_Energy_General()
                a.Save()



	# Do the permutation if it is necessary.
	if (0):
		a=MSet("H2O_tinker_amoeba")
		a.Load()
		a.Get_Permute_Frags(indis=[1,2]) # Include all the possible permutations of each fragment.
		a.Save()

	# Prepare data for neural newtork training.
	if (0):
		a=MSet("H2O_tinker_amoeba")
                a.Load()
		TreatedAtoms = a.AtomTypes()
		print "TreatedAtoms ", TreatedAtoms 
		d = MolDigester(TreatedAtoms, name_="SymFunc")  # Initialize a digester that apply descriptor for the fragments.
		#tset = TensorMolData(a,d, order_=2, num_indis_=2) # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
		#tset.BuildTrain("H2O_tinker_amoeba") # Genearte training data with the loaded molecule set and the chosen digester, by default it is saved in ./trainsets.
		tset = TensorMolData_BP(a,d, order_=2, num_indis_=2) # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
		tset.BuildTrain("H2O_tinker_amoeba")

	# doing the KRR for the set for debug purpose.
	if (0):
		tset = TensorMolData_BP(MSet(),MolDigester([]),"H2O_tinker_amoeba_GauInv_1") # Load the generated data for training the neural network.
		tset.KRR()
	
	# testing the BP TensorMolData
	if (0):
                tset = TensorMolData_BP(MSet(),MolDigester([]),"H2O_tinker_amoeba_SymFunc_1")
		#tset.LoadDataToScratch(True)

	# Train the neural network.
	if (0):
		tset = TensorMolData_BP(MSet(),MolDigester([]),"H2O_tinker_amoeba_SymFunc_2") # Load the generated data for training the neural network.
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=20000)  # train the neural network for 500 steps, by default it trainse 10000 steps and saved in ./networks.

	# Test the neural network.
	if (0):
		manager = TFMolManage("H2O_tinker_amoebaCoulomb_fc_sqdiff_2", None, False) # Load pre-trained network.	
		manager.Test("nn_acc_pred.dat") # Save the test result of our trained network. 


# steps to evaluate the many-body energy using  NN-MBE model
if (0):
	# load molecule
	a=MSet("H2O_opt")
	a.ReadGDB9Unpacked("./H2O_opt/")
	# load pre-trained networks {many-body order: network name}
	tfm = {1:"H2O_tinker_amoebaCoulomb_fc_sqdiff_1", 2:"H2O_tinker_amoebaCoulomb_fc_sqdiff_2"}
	# launch NN-MBE model 
	nn_mbe = NN_MBE(tfm)
	# evaluate using NN-MBE model 
	for mol in a.mols:
		#mol.Generate_All_MBE_term(atom_group=3, cutoff=5, center_atom=0)   
		nn_mbe.NN_Energy(mol)

# use NN-MBE model to optimize molecule. 
if (0):
	# load molecule
        a=MSet("H2O_opt")
        a.ReadGDB9Unpacked("./H2O_opt/")
        # load pre-trained networks {many-body order: network name}
        tfm = {1:"H2O_tinker_amoebaCoulomb_fc_sqdiff_1", 2:"H2O_tinker_amoebaCoulomb_fc_sqdiff_2"}
        # launch NN-MBE model 
        nn_mbe = NN_MBE(tfm)
	# launch Optimizer
        opt=MBE_Optimizer(nn_mbe)
	# Optimize
        for mol in a.mols:
		#mol.Generate_All_MBE_term(atom_group=3, cutoff=5, center_atom=0)
		#opt.NN_Opt(mol)
        	opt.MBE_LBFGS_Opt(mol)
