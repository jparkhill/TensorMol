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
        if (1):
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
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Energy")  # Initialize a digester that apply descriptor for the fragments.
                tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data for the neural network for certain order of many-body expansion.
		#tset.BuildTrain("SNB_bondstrength")
		for mol in a.mols:
                        ins = tset.dig.EvalDigest(mol)
			name = mol.name.split()[1]
			np.savetxt(name+".txt" , ins)

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


