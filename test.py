from Util import *
from Sets import *
from TensorData import *
from TFManage import *
from Opt import *

# John's tests
if (1):
	# Whole sequence just for morphine to debug.
	if (0):
		a=MSet("h2o")
		a.ReadXYZ("h2o")
		print "nmols:",len(a.mols)
		b=a.DistortAlongNormals()
	if (0):
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		print "nmols:",len(a.mols)
		c=a.DistortedClone(60)
		b=a.DistortAlongNormals()
		c.Statistics()
		b.Statistics()
		print len(b.mols)
		b.Save()
		b.WriteXYZ()
		TreatedAtoms = b.AtomTypes()
		# 2 - Choose Digester
		d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
		# 4 - Generate training set samples.
		tset = TensorData(b,d)
		tset.BuildTrain("OptMols_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.
		tset2 = TensorData(c,d)
		tset2.BuildTrain("OptMols_NEQ",TreatedAtoms,True) # generates dataset numpy arrays for each atom.
	if (1):
		tset = TensorData(None,None,"OptMols_NEQ_GauSH",None,6000)
		manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
	# This Tests the optimizer.
	if (1):
		a=MSet("OptMols_NEQ")
		a.Load()
		test_mol = a.mols[0]
		print "Orig Coords", test_mol.coords
		test_mol.Distort()
		print test_mol.coords
		print test_mol.atoms
		manager=TFManage("OptMols_NEQ_GauSH_fc_sqdiff",None,False)
		optimizer  = Optimizer(manager)
		optimizer.Opt(test_mol)
	exit(0) 
	if (1):
		# To read gdb9 xyz files and populate an Mset.
		# Because we use pickle to save. if you write new routines on Mol you need to re-execute this.
		if (1):
			a=MSet("gdb9")
			#a.ReadGDB9Unpacked()
			#a.Save()
			a.Load()
			c=a.DistortedClone(1)
			c.Save()
		# To generate training data for all the atoms in the GDB 9
		if (1):
			# 1 - Get molecules into memory
			a=MSet("gdb9_NEQ")
			a.Load()
			# Choose allowed atoms.
			TreatedAtoms = a.AtomTypes()
			# 2 - Choose Digester
			d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
			# 4 - Generate training set samples.
			tset = TensorData(a,d)
			tset.BuildTrain("gdb9_NEQ",TreatedAtoms,True) #fourth arg. generates debug data.
		#Merges two training datas...
		if (0):
			tset1 = TensorData(None,None,"gdb9_NEQ_SensoryBasis")
			tset2 = TensorData(None,None,"gdb92_NEQ_SensoryBasis")
			tset2.name="gdb92_NEQ"
			tset1.MergeWith(tset2)
		# This Trains the networks.
		if (1):
			tset = TensorData(None,None,"gdb9_NEQ_GauSH",None,6000)
			manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
		# This Tests the optimizer.
		if (1):
			a=MSet("OptMols")
			a.Load()
			test_mol = a.mols[0]
			print "Orig Coords", test_mol.coords
			test_mol.Distort()
			if (0):
				optimizer  = Optimizer(None)
				optimizer.OptGoForce(test_mol) # This works perfectly.
			print test_mol.coords
			print test_mol.atoms
			manager=TFManage("gdb9_NEQ_GauSH_fc_sqdiff",None,False)
			optimizer  = Optimizer(manager)
			optimizer.Opt(test_mol)

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
#	a=MSet()
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
#        print  test_mol.coords, test_mol.atoms
	manager=TFManage("gdb9SymFunc",None,False)
	optimizer  = Optimizer(manager)
	optimizer.Opt(test_mol)

# This tests the GO-Model potential.
if (0):
#	a=MSet()
	a=MSet("OptMols")
	a.Load()
	test_mol = (a.mols)[1]
	print test_mol.coords
	test_mol.Distort(0.3)
	print test_mol.coords
	print test_mol.atoms
	optimizer  = Optimizer(None)
	optimizer.GoOpt(test_mol)
#	optimizer.GoOpt_ScanForce(test_mol)

# this generates uniform samples of morphine for me. 
if (0):
	a=MSet("OptMols")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms)
	tset = TensorData(a,d,None)
	tset.BuildSamples("Test",[],True)
