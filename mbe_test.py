from Util import *
from Sets import *
from TensorMolData import *
from TFMolManage import *
from MolDigest import *
from NN_MBE import *
from NN_Opt import *


# steps to train a NN-MBE model
if (0):
	#Load .xyz files.
	if (0):
		a=MSet("He2_1") # Define our set.
		a.ReadGDB9Unpacked("./He2_1/", mbe_order=2) # Load .xyz file into set and set maxinum many-body expansion order.
		a.Generate_All_MBE_term(atom_group=1, cutoff=2, center_atom=0) # Generate all the many-body terms with  certain radius cutoff.

		# One can also load another set and combine with orginal one.
		b=MSet("He2")   
                b.ReadGDB9Unpacked("./He2/", mbe_order=2)
		b.Generate_All_MBE_term(atom_group=1, cutoff=4, center_atom=0)
                a.CombineSet(b) 

		a.Save() # Save the training set, by default it is saved in ./datasets.

	#Calculate the MP2 many-body energies.
	if (0):
		a=MSet("He2_1He2")  
		a.Load() # Load generated training set (.pdb file).
		a.Calculate_All_Frag_Energy()  # Use PySCF to calcuate the MP2 many-body energy of each order.
		a.Save() 

	#Prepare data for neural newtork training.
	if (0):
		a=MSet("He2_1He2")
		a.Load()
		a.Get_Permute_Frags() # Include all the possible permutations of each fragment.

		d = MolDigester()  # Initialize a digester that apply descriptor for the fragments.
		tset = TensorMolData(a,d, order_=2) # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
		tset.BuildTrain("He2_1He2") # Genearte training data with the loaded molecule set and the chosen digester, by default it is saved in ./trainsets.

	# Train the neural network.
	if (0):
		tset = TensorMolData(MSet(),MolDigester([]),"He2_1He2_Coulomb_2") # Load the generated data for training the neural network.
		manager=TFMolManage("",tset,False,"fc_sqdiff") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=500)  # train the neural network for 500 steps, by default it trainse 10000 steps and saved in ./networks.

	# Test the neural network.
	if (0):
		manager = TFMolManage("He2_1He2Coulomb_fc_sqdiff", None, False) # Load pre-trained network.	
		manager.Test("acc_nn_2b.dat") # Save the test result of our trained network. 


# steps to evaluate the many-body energy using  NN-MBE model
if (1):
	# load molecule
	a=MSet("He_opt")
	a.ReadGDB9Unpacked("./He_opt/", mbe_order=4)
	# load pre-trained networks {many-body order: network name}
	tfm = {2:"He2_1angsHe2Coulomb_fc_sqdiff", 3:"He3He3_1angsHe3_1angs_2cutCoulomb_fc_sqdiff", 4:"He4He4_3cutHe4_2cutCoulomb_fc_sqdiff"}
	# launch NN-MBE model 
	nn_mbe = NN_MBE(tfm)
	# evaluate using NN-MBE model 
	for mol in a.mols:
		nn_mbe.NN_Energy(mol)

# use NN-MBE model to optimize molecule. 
if (1):
	# load molecule
	a=MSet("He_opt")
	# load pre-trained networks {many-body order: network name}
        a.ReadGDB9Unpacked("./He_opt/", mbe_order=4)
        tfm = {2:"He2_1angsHe2Coulomb_fc_sqdiff", 3:"He3He3_1angsHe3_1angs_2cutCoulomb_fc_sqdiff", 4:"He4He4_3cutHe4_2cutCoulomb_fc_sqdiff"}
	# launch NN-MBE model
        nn_mbe=NN_MBE(tfm)
	# launch Optimizer
        opt=NN_Optimizer(nn_mbe)
	# Optimize
        for mol in a.mols:
        	opt.NN_Opt(mol)
