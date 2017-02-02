from TensorMol import *

#jeherr tests
if (0):
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

def TestGoForceAtom(dig_ = "GauSH", BuildTrain_=True, net_ = "fc_sqdiff"):
	"""
	A Network trained on Go-Force
	"""
	if (BuildTrain_):
		print "Testing a Network learning Go-Atom Force..."
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		print "nmols:",len(a.mols)
		c=a.DistortedClone(300,0.25) # number of distortions, displacement
		b=a.DistortAlongNormals(30, True, 0.7)
		c.Statistics()
		b.Statistics()
		print len(b.mols)
		b.Save()
		b.WriteXYZ()
		b=MSet("OptMols_NEQ")
		b.Load()
		TreatedAtoms = b.AtomTypes()
		# 2 - Choose Digester
		d = Digester(TreatedAtoms, name_=dig_,OType_ ="Force")
		# 4 - Generate training set samples.
		tset = TensorData(b,d)
		tset.BuildTrainMolwise("OptMols_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.
		tset2 = TensorData(c,d)
		tset2.BuildTrainMolwise("OptMols_NEQ",TreatedAtoms,True) # generates dataset numpy arrays for each atom.
	#Train
	tset = TensorData(None,None,"OptMols_NEQ_"+dig_,None,10000)
	manager=TFManage("",tset,True,net_) # True indicates train all atoms
	# This Tests the optimizer.
	if (net_ == "KRR_sqdiff"):
			a=MSet("OptMols")
			a.ReadXYZ("OptMols")
			test_mol = a.mols[11]
			print "Orig Coords", test_mol.coords
			test_mol.Distort()
			optimizer = Optimizer(manager)
			optimizer.Opt(test_mol)
	a=MSet("OptMols")
	a.ReadXYZ("OptMols")
	test_mol = a.mols[11]
	print "Orig Coords", test_mol.coords
	test_mol.Distort()
	print test_mol.coords
	print test_mol.atoms
	manager=TFManage("OptMols_NEQ_"+dig_+"_fc_sqdiff",None,False)
	optimizer  = Optimizer(manager)
	optimizer.Opt(test_mol)
	return

# TestGoForceAtom("GauSH", True, "KRR_sqdiff")

if (0):
	a=MSet('toluene_tmp')
	# a.ReadGDB9Unpacked(path='/media/sdb2/jeherr/TensorMol/datasets/benzene/', has_force=True)
	# a.Save()
	# a.WriteXYZ()
	a.Load()
	print "nmols:",len(a.mols)
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
	tset = TensorData(a,d)
	tset.BuildTrainMolwise("toluene_tmp",TreatedAtoms)
	tset = TensorData(None,None,"toluene_tmp_"+"GauSH",None,2000)
	manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms

if (0):
	a=MSet('OptMols')
	a.ReadXYZ("OptMols")
	test_mol=a.mols[0]
	b=MSet("test_mol")
	b.mols.append(test_mol)
	TreatedAtoms = b.AtomTypes()
	d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
	tset = TensorData(b,d)
	tset.BuildTrainMolwise("test_mol",TreatedAtoms)

#a=MSet('md_set_gdb9')
##a.ReadGDB9Unpacked(path='/data/jeherr/TensorMol/datasets/md_datasets/md_set/', has_force=True)
##a.ReadGDB9Unpacked(path='/data/jeherr/TensorMol/datasets/gdb9/', has_force=True)
##a.Save()
##a.WriteXYZ()
#a.Load()
#print "nmols:",len(a.mols)
#TreatedAtoms = a.AtomTypes()
#d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
#tset = TensorData(a,d)
#tset.BuildTrainMolwise("md_gdb9",TreatedAtoms)
##tset = TensorData(None,None,"md_gdb9_"+"GauSH",None,2000)
##manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms

a=MSet("o-xylene")
a.ReadXYZ("o-xylene")
## # a.ReadGDB9Unpacked(path='/media/sdb2/jeherr/TensorMol/datasets/benzene/', has_force=True)
## # a.Save()
## # a.WriteXYZ()
## # a.ReadXYZ("benzene")
##a.Load()
print a.mols
test_mol = a.mols[0]
test_mol.coords = test_mol.coords - np.average(test_mol.coords, axis=0)
##test_mol.Distort(0.1)
manager=TFManage("md_set_GauSH_fc_sqdiff",None,False)
optimizer = Optimizer(manager)
optimizer.OptRealForce(test_mol)
