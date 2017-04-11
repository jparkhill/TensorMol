from TensorMol import *
import cProfile

# John's tests
if (1):
	# To read gdb9 xyz files and populate an Mset.
	# Because we use pickle to save. if you write new routines on Mol you need to re-execute this.
	if (0):
		a=MSet("gdb9")
		a.ReadGDB9Unpacked("/home/kyao/TensorMol/gdb9/")
		allowed_eles=[1, 6, 7, 8]
		a.CutSet(allowed_eles)
		a.Make_Graphs()
		a.Bonds_Between_All()
		a.Save()

	if (0):
                a=MSet("gdb9_energy")
                a.ReadGDB9Unpacked("/home/kyao/TensorMol/gdb9/")
                allowed_eles=[1, 6, 7, 8]
                a.CutSet(allowed_eles)
                a.Make_Graphs()
                a.Save()

        if (0):
                a=MSet("gdb9_energy_1_6_7_8")
                a.Load()
		b=MSet("gdb9_1_6_7_8_cleaned")
                b.Load()
		mol_names = []
		for mol in b.mols:
			mol_names.append(mol.name)
		new_mols = []
		for mol in a.mols:
			if mol.name in mol_names:
				new_mols.append(mol)
		a.mols = new_mols
		a.name  = a.name + "_cleaned"
                a.Save()

        if (0):
                a=MSet("gdb9_smiles")
                a.ReadGDB9Unpacked("/home/jparkhil/TensorMol_Kun/TensorMol/gdb9/")
                allowed_eles=[1, 6, 7, 8]
                a.CutSet(allowed_eles)
                a.Make_Graphs()
                a.Save()


        if (0):
		b=MSet("gdb9_energy_1_6_7_8_cleaned")
                b.Load()
                a=MSet("gdb9_smiles_1_6_7_8")
                a.Load()
                mol_names = []
                for mol in b.mols:
                        mol_names.append(mol.name)
                new_mols = []
                for mol in a.mols:
                        if mol.name in mol_names:
                                new_mols.append(mol)
                a.mols = new_mols
                a.name  = a.name + "_cleaned"
                a.Save()

	if (0):
		a=MSet("gdb9_smiles_1_6_7_8_cleaned")
		a.Load()
		a.WriteSmiles()
		

	if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_1_6_7_8_cleaned")
                a.Load()
		a.WriteXYZ()
		#s = a.Clean_GDB9()
		#s.Save()

	# To generate training data for all the atoms in the GDB 9
	if (0):
		# 1 - Get molecules into memory
		a=MSet("gdb9_1_6_7_8_cleaned")
		a.Load()
		TreatedAtoms = a.AtomTypes()
		print "TreatedAtoms ", TreatedAtoms
		d = MolDigester(TreatedAtoms, name_="ConnectedBond_CM_BP", OType_="Atomization")  # Initialize a digester that apply descriptor for the fragments.
		tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
		tset.BuildTrain("gdb9_1_6_7_8_cleaned")

	if (0):
		tset = TensorMolData_BP(MSet(),MolDigester([]),"gdb9_1_6_7_8_cleaned_ConnectedBond_CM_BP")
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=500)  # train the neural network for 500 steps, by default it trainse 10000 steps and saved in ./networks.

        # To generate training data for all the atoms in the GDB 9
        if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_1_6_7_8_cleaned")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                d = MolDigester(TreatedAtoms, name_="Coulomb_Bond_BP", OType_="Atomization")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_1_6_7_8_cleaned")


        if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_1_6_7_8_cleaned_Coulomb_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=500)  # train the neural network for 500 steps, by default it trainse 10000 steps and saved in ./networks.


	if (0):
		manager= TFMolManage("Mol_gdb9_1_6_7_8_Coulomb_Bond_BP_fc_sqdiff_BP_1" , None, False)
		manager.Test()

        if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_1_6_7_8_cleaned")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                d = MolDigester(TreatedAtoms, name_="ConnectedBond_Angle_Bond_BP", OType_="Atomization")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_1_6_7_8_cleaned")

	if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=1)  # train the neural network for 500 steps, by default it trainse 10000 steps and saved in ./networks.



	# To generate training data for all the atoms in the GDB 9
        if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_1_6_7_8_clean")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ConnectedBond_Bond_BP", OType_="Atomization")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_1_6_7_8_clean")
	if (0):
                tset = TensorMolData_BP(MSet(),MolDigester([]),"gdb9_1_6_7_8_ConnectedBond_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=500)
        if (0):
                manager= TFMolManage("Mol_gdb9_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Test()
        if (0):
                a = MSet("SNB_bondstrength")
                a.ReadXYZ("SNB_bondstrength")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a)
	if (0):
                manager= TFMolManage("Mol_gdb9_1_6_7_8_ConnectedBond_Bond_BP_fc_sqdiff_BP_1" , None, False)
		tset = TensorMolData_BP(MSet(),MolDigester([]),"gdb9_1_6_7_8_ConnectedBond_Bond_BP")
		manager.TData = tset 
                manager.Test()	


	
	if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_1_6_7_8_cleaned")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ConnectedBond_Angle_CM_Bond_BP", OType_="Atomization")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_1_6_7_8_cleaned")

	if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_1_6_7_8_cleaned_ConnectedBond_Angle_CM_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=500)
	if (0):
                manager= TFMolManage("Mol_gdb9_1_6_7_8_cleaned_ConnectedBond_Angle_CM_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Test()
	if (0):
                manager= TFMolManage("Mol_gdb9_1_6_7_8_cleaned_ConnectedBond_Angle_CM_Bond_BP_fc_sqdiff_BP_1" , None, False)
                cProfile.run('manager.Continue_Training(500)')
	if (0):
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_CM_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Continue_Training(500)
	if (0):
                a = MSet("SNB_bondstrength")
                a.ReadXYZ("SNB_bondstrength")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_CM_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a, True)



	if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_energy_1_6_7_8_cleaned")
                a.Load()
                a.name = a.name + "_for_test"
		#for num_mol, mol in enumerate (a.mols):
		#	if num_mol % 1000 == 0:
		#		print "dealing mol:", num_mol
                #        mol.Find_Bond_Index()
                #        mol.Define_Conjugation()
                a.Save()
        if (0):
                # 1 - Get molecules into memory
                a=MSet("gdb9_energy_1_6_7_8_cleaned_for_test")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ConnectedBond_Angle_Bond_BP", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_energy_1_6_7_8_cleaned_for_test")
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
                d = MolDigester(TreatedAtoms, name_="ConnectedBond_Angle_Bond_BP", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("gdb9_energy_1_6_7_8_cleaned")
        if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=501)
	if (0):
		manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Continue_Training(maxsteps=901)
	if (0):
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Test()

	if (0):
                a = MSet("xave")
                a.ReadXYZ("xave")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a)

	if (0):
                a = MSet("SNB_bondstrength")
		a.ReadXYZ("SNB_bondstrength")
                a.Make_Graphs()
		a.Save()
		a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a, True)

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

	if (0):
                a = MSet("1_1_Ostrech")
                a.ReadXYZ("1_1_Ostrech")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a)
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
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ethy_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a, True)


# Graph subsampling
if (1):
	if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
		manager.name = manager.name + "_1percent"
                manager.Train(maxstep=50001)

	if (0):
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1_1percent" , None, False)
                manager.Continue_Training(maxsteps=10001)


	if (0):
                a = MSet("chemspider_9heavy_tmcleaned_opt")
                a.ReadXYZ("chemspider_9heavy_tmcleaned_opt")
                a.Make_Graphs()
		allowed_bonds = [2,3,4,5,6,7,8,9]
		new_set = []
		for mol in a.mols:
			flag = True
			for i in range (0, mol.NAtoms()-1):
				if  None  in mol.shortest_path[i]:
					print "not connected bond:", mol.shortest_path[i]
					flag = False
					break 
			for i in range (0, mol.bonds.shape[0]):
				if mol.bonds[i][0] not in allowed_bonds:
					print "not allowed bond:", mol.bonds[i][0]
					flag = False
					break
			if flag:
				new_set.append(mol)
		a.name = a.name + "_allowedbond"
		print "number of mol in old set:", len(a.mols)
		a.mols = new_set
		print "number of mol in new set:", len(a.mols)
		print a.BondTypes()
		a.WriteXYZ()
                a.Save()
	
	if (0):
		a = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond")
                a.Load()
		flag = pickle.load(open("chemspider_in_gdb9_flag.dat","rb"))
		contained = []
		not_contained = []
		for mol_index, mol in enumerate(a.mols):
			if flag[mol_index]:
				contained.append(mol)
			else:
				not_contained.append(mol)
		old_name = a.name
		a.name = old_name + "_ingdb9"
		a.mols = contained
		a.Save()
		a.name = old_name + "_not_ingdb9"
		a.mols = not_contained
		a.Save()

	if (0):
		a = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond")
                a.Load()
		b = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond_ingdb9")
                b.Load()	
		c = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond_not_ingdb9")
                c.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                abs_all = manager.Eval_Bond_BP(a, True)
		abs_in = manager.Eval_Bond_BP(b, True)
		abs_not_in = manager.Eval_Bond_BP(c, True)
		print "MAE all:", abs_all
		print "MAE in:", abs_in
		print "MAE not in:", abs_not_in


        if (0):
                a = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond")
                a.Load()
                b = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond_ingdb9")
                b.Load()
                c = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond_not_ingdb9")
                c.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1_1percent" , None, False)
                abs_all = manager.Eval_Bond_BP(a, True)
                abs_in = manager.Eval_Bond_BP(b, True)
                abs_not_in = manager.Eval_Bond_BP(c, True)
                print "MAE all:", abs_all
                print "MAE in:", abs_in
                print "MAE not in:", abs_not_in

	if (0): # Mol_gdb9_energy_1_6_7_8_cleaned_with_GA_sample_fixparam_step2k_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1
		a = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond")
                a.Load()
                b = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond_ingdb9")
                b.Load()
                c = MSet("chemspider_9heavy_tmcleaned_opt_allowedbond_not_ingdb9")
                c.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_with_GA_sample_leastconn_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                abs_all = manager.Eval_Bond_BP(a, True)
                abs_in = manager.Eval_Bond_BP(b, True)
                abs_not_in = manager.Eval_Bond_BP(c, True)
                print "MAE all:", abs_all
                print "MAE in:", abs_in
                print "MAE not in:", abs_not_in


	if (0):
                a = MSet("chemspider_9heavy_tmcleaned_all")
                a.ReadXYZ("chemspider_9heavy_tmcleaned_all")
                a.Make_Graphs()
                allowed_bonds = [2,3,4,5,6,7,8,9]
                new_set = []
                for mol in a.mols:
                        flag = True
                        for i in range (0, mol.NAtoms()-1):
                                if  None  in mol.shortest_path[i]:
                                        print "not connected bond:", mol.shortest_path[i]
                                        flag = False
                                        break
                        for i in range (0, mol.bonds.shape[0]):
                                if mol.bonds[i][0] not in allowed_bonds:
                                        print "not allowed bond:", mol.bonds[i][0]
                                        flag = False
                                        break
                        if flag:
                                new_set.append(mol)
                a.name = a.name + "_allowedbond"
                print "number of mol in old set:", len(a.mols)
                a.mols = new_set
                print "number of mol in new set:", len(a.mols)
                print a.BondTypes()
                a.WriteXYZ()
                a.Save()
	
	if (0):
                # 1 - Get molecules into memory
                a=MSet("chemspider_9heavy_tmcleaned_all_allowedbond")
                a.Load()
                TreatedAtoms = a.AtomTypes()
                print "TreatedAtoms ", TreatedAtoms
                TreatedBonds = list(a.BondTypes())
                print "TreatedBonds ", TreatedBonds
                d = MolDigester(TreatedAtoms, name_="ConnectedBond_Angle_Bond_BP", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_Bond_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
                tset.BuildTrain("chemspider_9heavy_tmcleaned_all_allowedbond")
        if (0):
                tset = TensorMolData_Bond_BP(MSet(),MolDigester([]),"chemspider_9heavy_tmcleaned_all_allowedbond_ConnectedBond_Angle_Bond_BP")
                manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
                manager.Train(maxstep=2501)
	

 	if (0):
                a = MSet("1_1_Ostrech")
                a.ReadXYZ("1_1_Ostrech")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_chemspider_9heavy_tmcleaned_all_allowedbond_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                manager.Eval_Bond_BP(a)
	
	if (1): # Mol_gdb9_energy_1_6_7_8_cleaned_with_GA_sample_fixparam_step2k_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1
		a = MSet("chemspider_9heavy_tmcleaned_all_fortest")
                a.ReadXYZ("chemspider_9heavy_tmcleaned_all_fortest")
                a.Make_Graphs()
                manager= TFMolManage("Mol_chemspider_9heavy_tmcleaned_all_allowedbond_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1" , None, False)
                abs_all = manager.Eval_Bond_BP(a, True)
                print "MAE all:", abs_all


# Kun's tests.
if (0):
	if (0):
		a=MSet("C2H6")
		#a.ReadXYZ("CxHy_test")
		a.ReadGDB9Unpacked("./C2H6/")
		#a.Save()
	#	a=MSet("gdb9_NEQ")
	#	a.Load()
		#b=MSet("gdb9")
		#b.Load()
		#allowed_eles=[1, 6]
		#b.CutSet(allowed_eles)
		#print "length of bmols:", len(b.mols)
		a = a.DistortedClone(100000)
		a.Save()

	if (0):
		#a=MSet("CxHy_test_NEQ")
		#a.Load()
		a=MSet("gdb9_1_6_NEQ")
	  	#a=a.DistortedClone(1)	
		a.Load()	
		# Choose allowed atoms.
		TreatedAtoms = a.AtomTypes()
		#for mol in a.mols:
		#	mol.BuildDistanceMatrix()
		# 2 - Choose Digester
		#d = Digester(TreatedAtoms, name_="SymFunc",OType_ ="Force")
		#d.TrainDigestW(a.mols[0], 6)
		print "len of amols", len(a.mols)
		d = Digester(TreatedAtoms, name_="PGaussian",OType_ ="GoForce_old_version", SamplingType_="None")
		#d.Emb(a.mols[0],0, np.zeros((1,3)))
		#d.Emb(a.mols[0],0, a.mols[0].coords[0].reshape(1,-1))
		#4 - Generate training set samples.

	if (0):
		tset = TensorData(a,d)
		tset.BuildTrain("gdb9_1_6_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.

	if (1):
		tset = TensorData(MSet(),Digester([]),"gdb9_1_6_NEQ_PGaussian")
		#tset_test = TensorData(MSet(),Digester([]),"CxHy_test_SymFunc") 
		#manager=TFManage("",tset,False,"fc_sqdiff", tset_test) # True indicates train all atoms.
		manager=TFManage("",tset,False,"fc_sqdiff")
		manager.TrainElement(6)
		#tset = TensorData(MSet(),Digester([]),"gdb9_1_6_NEQ_SymFunc")
		#manager = TFManage("", tset , True)
	

