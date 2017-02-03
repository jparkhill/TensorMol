"""
Various tests of tensormol's functionality.
Many of these tests take a pretty significant amount of time and memory to complete.
"""
from TensorMol import *

# John's tests
def TestBP(set_= "gdb9", dig_ = "Coulomb", BuildTrain_=True):
	"""
	General Behler Parinello using ab-initio energies.
	Args:
		set_: A dataset ("gdb9 or alcohol are available")
		dig_: the digester string
	"""
	print "Testing General Behler-Parrinello using ab-initio energies...."
	if (BuildTrain_):
		a=MSet(set_)
		a.ReadXYZ(set_)
		TreatedAtoms = a.AtomTypes()
		print "TreatedAtoms ", TreatedAtoms
		d = MolDigester(TreatedAtoms, name_=dig_+"_BP", OType_="Energy")
		tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol")
		tset.BuildTrain(set_)
	tset = TensorMolData_BP(MSet(),MolDigester([]),set_+"_"+dig_+"_BP")
	manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
	manager.Train(maxstep=500)  # train the neural network for 500 steps, by default it trainse 10000 steps and saved in ./networks.
	# We should try to get optimizations working too...
	return

def TestAlign():
	"""
	align two structures for maximum similarity.
	"""
	crds = MakeUniform([0.,0.,0.],1.5,5)
	a = Mol(np.array([1 for i in range(len(crds))]),crds)
	b = copy.deepcopy(a)
	b.Distort()
	b.coords = b.coords[np.random.permutation(len(crds))] # Permute the indices to make it hard.
	b.AlignAtoms(a)
	return

def TestGoForceAtom(dig_ = "GauSH", BuildTrain_=True, net_ = "fc_sqdiff", Train_=True):
	"""
	A Network trained on Go-Force
	"""
	if (BuildTrain_):
		print "Testing a Network learning Go-Atom Force..."
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		b = a.RotatedClone(3)
		print "nmols:",len(b.mols)
		c=b.DistortedClone(300,0.25) # number of distortions, displacement
		d=b.DistortAlongNormals(30, True, 0.7)
		c.AppendSet(d)
		c.Statistics()
		TreatedAtoms = c.AtomTypes()
		# 2 - Choose Digester
		d = Digester(TreatedAtoms, name_=dig_,OType_ ="GoForce")
		# 4 - Generate training set samples.
		tset = TensorData(c,d)
		tset.BuildTrainMolwise("OptMols_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.
	#Train
	if (Train_):
		tset = TensorData(None,None,"OptMols_NEQ_"+dig_,None,10000)
		manager=TFManage("",tset,True, net_) # True indicates train all atoms
	# This Tests the optimizer.
	if (net_ == "KRR_sqdiff"):
			a=MSet("OptMols")
			a.ReadXYZ("OptMols")
			test_mol = a.mols[11]
			print "Orig Coords", test_mol.coords
			test_mol.Distort()
			optimizer  = Optimizer(manager)
			optimizer.Opt(test_mol)
	a=MSet("OptMols")
	a.ReadXYZ("OptMols")
	test_mol = a.mols[11]
	print "Orig Coords", test_mol.coords
	test_mol.Distort()
	print test_mol.coords
	print test_mol.atoms
	manager=TFManage("OptMols_NEQ_"+dig_+"_"+net_,None,False)
	optimizer  = Optimizer(manager)
	optimizer.Opt(test_mol)
	return

# Tests to run.
#TestBP(set_="gdb9", dig_="GauInv", BuildTrain_=False)
TestGoForceAtom(dig_ = "GauSH", BuildTrain_=False, net_ = "fc_sqdiff", Train_=True)

# Kun's tests.
if (0):
	if (0):
		#a=MSet("CxHy_test")
		#a.ReadXYZ("CxHy_test")
		#a.Save()
	#	a=MSet("gdb9_NEQ")
	#	a.Load()
		b=MSet("gdb9")
		b.Load()
		allowed_eles=[1, 6]
		b.CutSet(allowed_eles)
		print "length of dmols:", len(b.mols)
		#b = a.DistortedClone(20)
		b.Save()

	if (1):
		#a=MSet("CxHy_test")
		#a.Load()
		a=MSet("gdb9_1_6")
	  	a=a.DistortedClone(1)
		a.Load()
		# Choose allowed atoms.
		TreatedAtoms = a.AtomTypes()
		#for mol in a.mols:
		#	mol.BuildDistanceMatrix()
		# 2 - Choose Digester
		#d = Digester(TreatedAtoms, name_="SymFunc",OType_ ="Force")
		#d.TrainDigestW(a.mols[0], 6)
		d = Digester(TreatedAtoms, name_="PGaussian",OType_ ="GoForce_old_version")
		d.Emb(a.mols[0],0, np.zeros((1,3)))
		#d.Emb(a.mols[0],0, a.mols[0].coords[0].reshape(1,-1))
		#4 - Generate training set samples.

	if (0):
		tset = TensorData(a,d)
		tset.BuildTrain("CxHy_test") # generates dataset numpy arrays for each atom.

	if (0):
		tset = TensorData(MSet(),Digester([]),"gdb9_1_6_NEQ_SymFunc")
		tset_test = TensorData(MSet(),Digester([]),"CxHy_test_SymFunc")
		manager=TFManage("",tset,False,"fc_sqdiff", tset_test) # True indicates train all atoms.
		manager.TrainElement(1)
		tset = TensorData(MSet(),Digester([]),"gdb9_1_6_NEQ_SymFunc")
		manager = TFManage("", tset , True)


# This visualizes the go potential and projections on to basis vectors.
if (0):
	a=MSet("OptMols")
	a.Load()
	m = a.mols[0]
	#m.BuildDistanceMatrix()
	m.Distort(0,2.0);
	# I did this just to look at the go prob of morphine for various distortions... it looks good and optimizes.
	if (0):
		#   Try dumping these potentials onto the sensory atom, and seeing how that works...
		#   It worked poorly for atom centered basis, but a grid of gaussians was great.
		for i in range(1,m.NAtoms()):
			m.FitGoProb(i)
		samps, vol = m.SpanningGrid(150,2)
		Ps = m.POfAtomMoves(samps,0)
		for i in range(1,m.NAtoms()):
			Ps += m.POfAtomMoves(samps,i)
		Ps /= Ps.max()
		Ps *= 254.0
		GridstoRaw(Ps,150,"Morphine")
	# Attempt an optimization to check that mean-probability will work if it's perfectly predicted.
	if (0):
		optimizer  = Optimizer(None)
		optimizer.GoOptProb(m) # This works perfectly.

# This draws test volumes for Morphine
if (0):
	a=MSet("OptMols")
	a.Load()
	test_mol = a.mols[0]
	manager=TFManage("gdb9_NEQ_SymFunc",None,False)
	xyz,p = manager.EvalAllAtoms(test_mol)
	grids = test_mol.MolDots()
	grids = test_mol.AddPointstoMolDots(grids, xyz, p)
	np.savetxt("./densities/morph.xyz",test_mol.coords)
	GridstoRaw(grids,250,"Morphine")


# This tests the optimizer.
if (0):
	#a=MSet()
	a=MSet("OptMols")
	a.Load()
	test_mol = a.mols[0]
	test_mol.DistortedCopy(0.2)
	print test_mol.coords
	print test_mol.atoms
	manager=TFManage("gdb9Coulomb",None,False)
	optimizer  = Optimizer(manager)
	optimizer.Opt(test_mol)

# This is for test of c2h6, c2h4, c2h2
if (0):
	c2h6 = np.loadtxt("c2h4.xyz")
	atoms = (c2h6[:,0].reshape(c2h6.shape[0])).copy()
	atoms = np.array(atoms, dtype=np.uint8)
	coords = c2h6[:, 1:4].copy()
	test_mol =Mol(atoms, coords)
	#print  test_mol.coords, test_mol.atoms
	manager=TFManage("gdb9SymFunc",None,False)
	optimizer  = Optimizer(manager)
	optimizer.Opt(test_mol)

# This tests the GO-Model potential.
if (0):
	#a=MSet()
	a=MSet("OptMols")
	a.Load()
	test_mol = (a.mols)[1]
	print test_mol.coords
	test_mol.Distort(0.3)
	print test_mol.coords
	print test_mol.atoms
	optimizer  = Optimizer(None)
	optimizer.GoOpt(test_mol)
	#optimizer.GoOpt_ScanForce(test_mol)

# this generates uniform samples of morphine for me.
if (0):
	a=MSet("OptMols")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms)
	tset = TensorData(a,d,None)
	tset.BuildSamples("Test",[],True)


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
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		print "nmols:",len(a.mols)
		c=a.DistortedClone(200)
		# b=a.DistortAlongNormals(80, True, 1.2)
		# c.Statistics()
		# b.Statistics()
		# print len(b.mols)
		# b.Save()
		# b.WriteXYZ()
		b=MSet("OptMols_NEQ")
		b.Load()
		TreatedAtoms = b.AtomTypes()
		# 2 - Choose Digester
		d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
		# 4 - Generate training set samples.
		tset = TensorData(b,d)
		tset.BuildTrain("OptMols_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.
		tset2 = TensorData(c,d)
		tset2.BuildTrain("OptMols_NEQ",TreatedAtoms,True) # generates dataset numpy arrays for each atom.
		#Train
		tset = TensorData(None,None,"OptMols_NEQ_GauSH",None,6000)
		manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
		# This Tests the optimizer.
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		test_mol = a.mols[11]
		print "Orig Coords", test_mol.coords
		test_mol.Distort()
		print test_mol.coords
		print test_mol.atoms
		manager=TFManage("OptMols_NEQ_GauSH_fc_sqdiff",None,False)
		optimizer  = Optimizer(manager)
		optimizer.Opt(test_mol)
