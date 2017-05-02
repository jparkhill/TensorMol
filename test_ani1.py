from TensorMol import *
import cProfile

# John's tests
if (1):
	if (0):
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
		tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_energy_1_6_7_8_cleaned_for_test_ConnectedBond_Angle_Bond_BP")
                manager.TData = tset
                manager.Test()

        if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_energy_1_6_7_8_cleaned")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_energy_1_6_7_8_cleaned")
        if (0):
                tset = TensorMolData_BP(MSet(),MolDigester([]),"gdb9_energy_1_6_7_8_cleaned_ANI1_Sym")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=501)
	if (0):
		manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Continue_Training(maxsteps=901)
	if (0):
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Test()

        if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_energy_1_6_7_8_cleaned")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Center_Bond_BP", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_energy_1_6_7_8_cleaned")

	if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_energy_1_6_7_8_cleaned_ANI1_Sym_Center_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=501)


        if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_energy_1_6_7_8_cleaned")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Bond_BP", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_energy_1_6_7_8_cleaned")

        if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_energy_1_6_7_8_cleaned_ANI1_Sym_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=501)

	if (0):
                manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
                manager.Continue_Training(maxsteps=1001)
        if (0):
                a = MSet("CCdihe")
                a.ReadXYZ("CCdihe")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
                manager.Eval_BP(a)


	if (0):
                a = MSet("xave")
                a.ReadXYZ("xave")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a)
	if (0):
		a = MSet("ANI1_SYM_test")
		a.ReadXYZ("ANI1_SYM_test")
                a.Make_Graphs()
		TreatedAtoms = np.asarray([1,6,7,8])
                print "TreatedAtoms ", TreatedAtoms
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Center_Bond_BP", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
		#tset.BuildTrain("SNB_bondstrength")
		for mol in a.mols:
			print mol.bonds
                        ins = tset.dig.EvalDigest(mol)
			print ins
			name = mol.name.split()[1]
			np.savetxt(name+"_center.txt" , ins)

	if (0):
                a = MSet("linear_mol")
                a.ReadXYZ("linear_mol")
                a.Make_Graphs()
                TreatedAtoms = np.asarray([1,6,7,8])
                print "TreatedAtoms ", TreatedAtoms
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                #tset.BuildTrain("SNB_bondstrength")
                for mol in a.mols:
                        ins = tset.dig.EvalDigest(mol)
			print "ins:", ins
                        #name = mol.name.split()[1]
                        #np.savetxt(name+".txt" , ins)

	if (0):
                a = MSet("uneq_chemspider")
		#a.ReadXYZ("uneq_chemspider")
		a.Load()
                #a.Make_Graphs()
		#a.Save()
		b = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond_for_freq_compare")
		b.Load()
		#b.ReadXYZ("chemspider_9heavy_tmcleaned_opt_allowedbond_for_freq_compare")
		#b.Make_Graphs()
		#b.Save()
		import re
		for a_mol_index, a_mol in enumerate(a.mols):
			#dist_mat = MolEmb.Make_DistMat(a.mols[a_mol_index].coords)
			index = a.mols[a_mol_index].name.split()[2]
			#b_mol_index = a_mol_index / 50
			match = re.search(r'(\d+)_\d+', index)
			b_mol_index = int(match.group(1))
			print a_mol_index, b_mol_index
			if not np.array_equal(a.mols[a_mol_index].atoms, b.mols[b_mol_index].atoms):
				print "a mol:", a.mols[a_mol_index].atoms
				print "b.mol:", b.mols[b_mol_index].atoms
				raise Exception("not the same mol!")
			a.mols[a_mol_index].bonds = b.mols[b_mol_index].bonds 
			for bond_index in range (0, a.mols[a_mol_index].bonds.shape[0]): 
				a.mols[a_mol_index].bonds[bond_index][1] = a.mols[a_mol_index].DistMatrix[int(a.mols[a_mol_index].bonds[bond_index][2])][int(a.mols[a_mol_index].bonds[bond_index][3])]
			#print a.mols[a_mol_index].bonds
		a.Save()	
	if (0):
                a = MSet("uneq_chemspider")
                #a.ReadXYZ("uneq_chemspider")
                a.Load()
		for mol in a.mols:
			print mol.bonds

        if (0):
                # 1 - Get molecules into memory
                a=MSet("uneq_chemspider")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("uneq_chemspider")

	if (0):
                tset = TensorMolData_BP(MSet(),MolDigester([]),"uneq_chemspider_ANI1_Sym")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=501)
		

	if (0):
                # 1 - Get molecules into memory
                a=MSet("uneq_chemspider")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Center_Bond_BP", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("uneq_chemspider")

        if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"uneq_chemspider_ANI1_Sym_Center_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=501)


	if (0):
                a = MSet("CH_bondstrength")
                a.ReadXYZ("CH_bondstrength")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a, True)

	if (0):
                a = MSet("aminoacids")
                a.ReadXYZ("aminoacids")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a, True)

	if (1):
                a = MSet("1_1_Ostrech")
                a.ReadXYZ("1_1_Ostrech")
                g = a.Make_Graphs()
		print "found?", g[4].Find_Frag(g[3])
		g[4].Calculate_Bond_Type()
		print "bond type:", g[4].bond_type
                #a.Load()
                #manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                #manager.Eval_Bond_BP(a)
	if (0):
		a = MSet("SNB_bondstrength")
                a.ReadXYZ("SNB_bondstrength")
                a.Make_Graphs()
                a.Save()
                a.Load()
		#for mol in a.mols:
		#	mol.Find_Bond_Index()
		#	mol.Define_Conjugation()
                TreatedAtoms = a.AtomTypes()
                d = MolDigester(TreatedAtoms, name_="ConnectedBond_Angle_CM_Bond_BP", OType_="Atomization")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("SNB_bondstrength")


	if (0):
		b = MSet("ethy")
                b.ReadXYZ("ethygroup")
                b.Make_Graphs()
		b.Bonds_Between_All()
		a=MSet("gdb9_energy_1_6_7_8_cleaned")
                a.Load()
		a.AppendSet(b)
		a.Save()
	if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_energy_1_6_7_8_cleaned_ethy")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ConnectedBond_Angle_Bond_BP", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_energy_1_6_7_8_cleaned_ethy")
	if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_energy_1_6_7_8_cleaned_ethy_ConnectedBond_Angle_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=501)
	if (0):
                a = MSet("SNB_bondstrength")
                a.ReadXYZ("SNB_bondstrength")
                a.Make_Graphs()
               # a.Save()
               # a.Load()
               # manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ethy_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
               # manager.Eval_Bond_BP(a, True)


