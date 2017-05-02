from TensorMol import *
from TensorMol.NN_MBE import *
from TensorMol.MBE_Opt import *

# Basic Function test of MBE
if (1):
	if (1):
		a=FragableMSet("NaClH2O")
		a.ReadXYZ("NaClH2O")
		a.Generate_All_Pairs(pair_list=[{"pair":"NaCl", "mono":["Na","Cl"], "center":[0,0]}])
                a.Generate_All_MBE_term_General([{"atom":"OHH", "charge":0}, {"atom":"NaCl", "charge":0}], cutoff=12, center_atom=[0, -1]) # Generate all the many-body terms with  certain radius cutoff.  # -1 means center of mass
                a.Save() # Save the training set, by default it is saved in ./datasets.

	if (1):
                a=FragableMSet("NaClH2O")
                a.Load() # Load generated training set (.pdb file).
                a.Calculate_All_Frag_Energy_General(method="qchem")  # Use PySCF or Qchem to calcuate the MP2 many-body energy of each order.
                a.Save()

#Basic Function test of MolGraph
if (1):
	if (1):
                a = MSet("1_1_Ostrech")
                a.ReadXYZ("1_1_Ostrech")
		g = GraphSet(a.name, a.path)
                g.graphs = a.Make_Graphs()
                print "found?", g.graphs[4].Find_Frag(g.graphs[3])
                g.graphs[4].Calculate_Bond_Type()
                print "bond type:", g.graphs[4].bond_type

		
