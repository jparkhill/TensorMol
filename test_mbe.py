from Util import *
from Sets import *
from TensorMolData import *
from TFMolManage import *
from Opt import *
from MolDigest import *
from NN_MBE import *
from NN_Opt import *

if (1):
	if (0):
		a=MSet("He2_deri")
		#a.ReadXYZ("CxHy_test")
		a.ReadGDB9Unpacked("./He_deri/", mbe_order=2)
		d = MolDigester()
		d.make_cm(a.mols[0])
		#a.Save()
	#	a=MSet("gdb9_NEQ")
	#	a.Load()
		#b=MSet("gdb9")
		#b.Load()
		#allowed_eles=[1, 6]
		#b.CutSet(allowed_eles)
		#print "length of bmols:", len(b.mols)
		#a = a.DistortedClone(100000)
		#a.MBE( atom_group=1, cutoff=10)
		#a.mols[0].PySCF_Energy()
		#a.Generate_All_MBE_term(atom_group=1, cutoff=2, center_atom=0)
		#print a.mols[0].mbe_frags[3][0].atoms, a.mols[0].mbe_frags[3][0].coords
                #a.mols[0].Calculate_All_Frag_Energy()
                #a.mols[0].Set_MBE_Energy()
		#print a.mols[0].energy
		#a.PySCF_Energy()
		a.Save()
		#b=a.mols[0].Generate_MBE_term(order=3, atom_group=1, cutoff=6)
		#b[0].PySCF_Frag_MBE_Energy_All()
		#print b[0].Frag_MBE_Energy()
		#print b[0].PySCF_Energy()

	if (0):
		#a=MSet("CxHy_test_NEQ")
		#a.Load()
		#a=MSet("He2_1angsHe2")
	  	#a=a.DistortedClone(1)
		#a.Load()
		
                #a.Get_Permute_Frags()
		#b=MSet("He2")
		#b.Load()
		#a.CombineSet(b)
		#a.Save()
		#a.Calculate_All_Frag_Energy()
		#a.Save()
		#b=a.mols[1].mbe_frags[3][0]
		#c=b.Permute_Frag()
		#diff = a.mols[1].energy - a.mols[0].energy
		#diff_2 = a.mols[1].mbe_energy[2] - a.mols[0].mbe_energy[2]
		#diff_3 = a.mols[1].mbe_energy[3] - a.mols[0].mbe_energy[3]
		#diff_4 = a.mols[1].mbe_energy[4] - a.mols[0].mbe_energy[4]
		#print diff-diff, diff_2-diff, diff_3-diff, diff_4 - diff
		#print a.mols[0].mbe_energy
		#print a.mols[0].mbe_frags	
		# Choose allowed atoms.
		#TreatedAtoms = a.AtomTypes()
		#for mol in a.mols:
		#	mol.BuildDistanceMatrix()
		# 2 - Choose Digester
		d = MolDigester()
		tset = TensorMolData(a,d, order_=2)
		tset.BuildTrain("He2_1angsHe2")
		#cm=d.make_cm(a.mols[0])
		#print d.GetUpTri(cm)
		#d.TrainDigestW(a.mols[0], 6)
		#print "len of amols", len(a.mols)
		#d = Digester(TreatedAtoms, name_="PGaussian",OType_ ="GoForce_old_version", SamplingType_="None")
		#d.Emb(a.mols[0],0, np.zeros((1,3)))
		#d.Emb(a.mols[0],0, a.mols[0].coords[0].reshape(1,-1))
		#4 - Generate training set samples.

	if (0):
		tset = TensorData(a,d)
		tset.BuildTrain("gdb9_1_6_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.

	if (0):
		#tset = TensorMolData(MSet(),MolDigester([]),"He4He4_3cutHe4_2cut_Coulomb_4")
		#manager=TFMolManage("",tset,False,"fc_sqdiff") # True indicates train all atoms.
		#manager.Train()
		manager = TFMolManage("He2_1angsHe2Coulomb_fc_sqdiff", None, False)	
		manager.Test("gradient_test.dat")

	if (0):
		a=MSet("He_deri")
                a.ReadGDB9Unpacked("./He_deri/", mbe_order=2)
                a.Generate_All_MBE_term(atom_group=1, cutoff=3, center_atom=0)
                #a.Save()
		manager = TFMolManage("He2_1angsHe2Coulomb_fc_sqdiff", None, False)   
		for mol in a.mols:
			print manager.Eval_Mol(mol)

	if (0):
		a=MSet("He_deri")
                a.ReadGDB9Unpacked("./He_deri/", mbe_order=3)
		tfm = {2:"He2_1angsHe2Coulomb_fc_sqdiff", 3:"He3He3_1angsHe3_1angs_2cutCoulomb_fc_sqdiff"}
		nn_mbe = NN_MBE(tfm)
		for mol in a.mols:
			nn_mbe.NN_Energy(mol)


	if (1):
		a=MSet("He_opt")
                a.ReadGDB9Unpacked("./He_opt/", mbe_order=4)
		tfm = {2:"He2_1angsHe2Coulomb_fc_sqdiff", 3:"He3He3_1angsHe3_1angs_2cutCoulomb_fc_sqdiff", 4:"He4He4_3cutHe4_2cutCoulomb_fc_sqdiff"}
		nn_mbe=NN_MBE(tfm)
		opt=NN_Optimizer(nn_mbe)
		for mol in a.mols:
			opt.NN_Opt(mol)
