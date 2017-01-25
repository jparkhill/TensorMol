from TensorMol import *

#jeherr tests
if (1):
	# Takes two nearly identical crystal lattices and interpolates a core/shell structure, must be oriented identically and stoichiometric
	if (0):
		a=MSet('cspbbr3_tess')
		#a.ReadGDB9Unpacked(path='/media/sdb2/jeherr/TensorMol/datasets/cspbbr3/pb_tess_6sc/')
		#a.Save()
		a.Load()
		mol1 = a.mols[0]
		mol2 = a.mols[1]
		mol2.RotateX()
		mol1.AlignAtoms(mol2)
		optimizer = Optimizer(None)
		optimizer.Interpolate_OptForce(mol1, mol2)
		mol1.WriteXYZfile(fpath='./results/cspbbr3_tess', fname='cspbbr3_6sc_pb_tess_goopt', mode='w')
		# mol2.WriteXYZfile(fpath='./results/cspbbr3_tess', fname='cspbbr3_6sc_ortho_rot', mode='w')

	if (0):
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		mol = a.mols[1]
		mol.BuildDistanceMatrix()
		print mol.LJForce()

	if (0):
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		test_mol = a.mols[10]
		#print "Orig Coords", test_mol.coords
		test_mol.BuildDistanceMatrix()
		test_mol.Distort(.1,1.0)
		# print test_mol.NumericLJHessian()
		# print test_mol.NumericLJHessDiag()
		#print test_mol.coords
		# print test_mol.LJForce()
		# print test_mol.NumericLJForce()
		optimizer = Optimizer(None)
		#optimizer.momentum = 0.0
		#optimizer.OptGoForce(test_mol)
		optimizer.OptLJForce(test_mol)

	if (0):
		a=MSet('cspbbr3_mixed')
		a.Load()
		mol1 = a.mols[0]
		mol2 = a.mols[1]
		mol1.BuildDistanceMatrix()
		mol2.BuildDistanceMatrix()
		#t1 = time.time()
		#for i in range(0,10000):
		#	a = np.linalg.norm(mol1.DistMatrix - mol2.DistMatrix)
		#t2 = time.time()
		#print a
		#print t2-t1
		t3 = time.time()
		for i in range(0,10000):
			b = mol1.NormMatrices(mol1.DistMatrix, mol2.DistMatrix)
		t4 = time.time()
		print t4-t3
		print b

	if (1):
		"""
		# A Network trained on Go-Force
		# """
		print "Testing a Network learning Go-Atom Force..."
		# a=MSet("OptMols")
		# a.ReadXYZ("OptMols")
		# print "nmols:",len(a.mols)
		# # c=a.DistortedClone(200)
		# b=a.DistortAlongNormals(10, True, 1.2)
		# # c.Statistics()
		# b.Statistics()
		# print len(b.mols)
		# b.Save()
		# b.WriteXYZ()
		# # b=MSet("OptMols_NEQ")
		# b.Load()
		# TreatedAtoms = b.AtomTypes()
		# # 2 - Choose Digester
		# d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
		# # 4 - Generate training set samples.
		# tset = TensorData(b,d)
		# tset.BuildTrainMolwise("OptMols_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.
		# tset2 = TensorData(c,d)
		# tset2.BuildTrain("OptMols_NEQ",TreatedAtoms,True) # generates dataset numpy arrays for each atom.
		#Train
		tset = TensorData(None,None,"OptMols_NEQ_GauSH",None,6000)
		manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
		# This Tests the optimizer.
		# a=MSet("OptMols")
		# a.ReadXYZ("OptMols")
		# test_mol = a.mols[11]
		# print "Orig Coords", test_mol.coords
		# test_mol.Distort()
		# print test_mol.coords
		# print test_mol.atoms
		# manager=TFManage("OptMols_NEQ_GauSH_fc_sqdiff",None,False)
		# optimizer  = Optimizer(manager)
		# optimizer.Opt(test_mol)
