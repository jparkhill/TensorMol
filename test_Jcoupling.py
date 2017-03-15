from TensorMol import *
from TensorMol.NN_MBE import *
from TensorMol.MBE_Opt import *


# for testing the fragmentation of molecule 
if (1):
	#Load .xyz files.
	if (1):
		a=MSet("Jcoupling") # Define our set.
		a.Read_Jcoupling("./Jcoupling/") # Load .xyz file into set and set maxinum many-body expansion order.
		a.Save()
                #a.mols[0].Make_Mol_Graph()
		#a.mols[0].Bonds_Between_All()
		#a.mols[0].Bonds_Between(24, 26)	
		#a.Analysis_Jcoupling()
	if (1):
		a=MSet("Jcoupling")
		a.Load()
		for mol in a.mols:
			print mol.name
			mol.Make_Mol_Graph()
			mol.Make_Nx_Graph()
		a.Save()
