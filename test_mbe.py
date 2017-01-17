from TensorMol import *
from TensorMol.NN_MBE import *
from TensorMol.MBE_Opt import *


# for testing the fragmentation of molecule 
if (1):
	if (1):
		a= MSet("CH3OH_C6H6")
		a.ReadXYZ("CH3OH_C6H6")
		#a.mols[0].Make_Mol_Graph()

		b = MSet("CH3OH_C6H6_frag")
		b.ReadXYZ("CH3OH_C6H6_frag", "frag_of_mol")
		for mol in a.mols:	
			mol.Make_Mol_Graph()
		for mol in b.mols:
			mol.Make_Mol_Graph()
		#print a.mols[0].Compare_Node(a.mols[0].atom_nodes[0], b.mols[0].atom_nodes[0])
		t = time.time()
		a.mols[-1].NoOverlapping_Partition([b.mols[0], b.mols[1], b.mols[2]])
		#a.mols[-1].Find_Frag(b.mols[0])
		#a.mols[-1].Find_Frag(b.mols[1])
		#a.mols[-1].Find_Frag(b.mols[2])
		print "time:", time.time()- t	
		#a.mols[0].DFS_recursive_all_order(a.mols[0].atom_nodes[0], [])
		#for node in b.mols[0].atom_nodes:
		#	print node.num_of_bonds
		#	print node.connected_atoms

if (0):
	#Load .xyz files.
	if (0):
		a=MSet("H2O_tinker_amoeba") # Define our set.
		a.ReadGDB9Unpacked("./H2O_tinker_amoeba/") # Load .xyz file into set and set maxinum many-body expansion order.
		a.Generate_All_MBE_term(atom_group=3, cutoff=6, center_atom=0, max_case=2000) # Generate all the many-
		a.Save()
	
	#Calculate the MP2 many-body energies.
	if (0):
		a=MSet("H2O_tinker_amoeba")  
		a.Load() # Load generated training set (.pdb file).
		#a.Calculate_All_Frag_Energy(method="qchem")  # Use PySCF or Qchem to calcuate the MP2 many-body energy of each order.
		a.Set_Qchem_Data_Path()
		a.Get_All_Qchem_Frag_Energy()
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
		d = MolDigester(TreatedAtoms, name_="Coulomb")  # Initialize a digester that apply descriptor for the fragments.
		tset = TensorMolData(a,d, order_=1, num_indis_=2) # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
		tset.BuildTrain("H2O_tinker_amoeba") # Genearte training data with the loaded molecule set and the chosen digester, by default it is saved in ./trainsets.
		#tset = TensorMolData_BP(a,d, order_=2, num_indis_=2) # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
		#tset.BuildTrain("H2O_tinker_amoeba")

	# doing the KRR for the set for debug purpose.
	if (0):
		tset = TensorMolData(MSet(),MolDigester([]),"H2O_tinker_amoeba_Coulomb_1") # Load the generated data for training the neural network.
		tset.KRR()
	
	# testing the BP TensorMolData
	if (0):
		tset = TensorMolData_BP(MSet(),MolDigester([]),"H2O_tinker_amoeba_SymFunc_1")
		#tset.LoadDataToScratch(True)

	# Train the neural network.
	if (1):
		tset = TensorMolData(MSet(),MolDigester([]),"H2O_tinker_amoeba_Coulomb_1") # Load the generated data for training the neural network.
		manager=TFMolManage("",tset,False,"fc_sqdiff") # Initialzie a manager than manage the training of neural network.
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
