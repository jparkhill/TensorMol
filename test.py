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
	PARAMS["NormalizeOutputs"] = True
	if (BuildTrain_):
		a=MSet(set_)
		a.ReadXYZ(set_)
		TreatedAtoms = a.AtomTypes()
		print "TreatedAtoms ", TreatedAtoms
		d = MolDigester(TreatedAtoms, name_=dig_+"_BP", OType_="AtomizationEnergy")
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
	Args:
		dig_ : type of digester to be used (GauSH, etc.)
	"""
	if (BuildTrain_):
		print "Testing a Network learning Go-Atom Force..."
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		if (PARAMS["RotateSet"]):
			b = a.RotatedClone(2)
		if (PARAMS["TransformSet"]):
			b = a.TransformedClone(OctahedralOperations())
		print "nmols:",len(b.mols)
		c=b.DistortedClone(PARAMS["NDistorts"],0.25) # number of distortions, displacement
		d=b.DistortAlongNormals(PARAMS["NModePts"], True, 0.7)
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
		tset = TensorData(None,None,"OptMols_NEQ_"+dig_)
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

def TestPotential():
	"""
	Makes volumetric data for looking at how potentials behave near and far from equilibrium.
	"""
	PARAMS["KAYBEETEE"] = 5000.0*0.000950048 # At 10*300K
	a=MSet("OptMols")
	a.ReadXYZ("OptMols")
	m = a.mols[5]
	m.Distort(0.1,0.1)
	n = 230
	ns = 35 # number of points to do around the atom.
	na1 = 1 # number of points to do as the atom.
	na2 = 2 # number of points to do as the atom.
	grid, volume = m.SpanningGrid(n,3.0, Flatten=True, Cubic=True)
	l0 = grid[0]
	dl = (grid[1]-grid[0])[2]
	vol = np.zeros((n,n,n))
	cgrid = grid.copy()
	cgrid = cgrid.reshape((n,n,n,3))
	for i in range(len(m.atoms)):
		#print m.coords[i]
		ic = np.array((m.coords[i]-l0)/dl,dtype=np.int) # Get indices in cubic grid.
		#print ic, cgrid[ic[0],ic[1],ic[2]]
		subgrid = cgrid[ic[0]-ns:ic[0]+ns,ic[1]-ns:ic[1]+ns,ic[2]-ns:ic[2]+ns].copy()
		fsubgrid = subgrid.reshape((8*ns*ns*ns,3))
		cvol = m.POfAtomMoves(fsubgrid-m.coords[i],i)
		#cvol -= cvol.min()
		#cvol /= cvol.max()
		cvol = cvol.reshape((2*ns,2*ns,2*ns))
		vol[ic[0]-ns:ic[0]+ns,ic[1]-ns:ic[1]+ns,ic[2]-ns:ic[2]+ns] += cvol
		vol[ic[0]-na1:ic[0]+na1,ic[1]-na1:ic[1]+na1,ic[2]-na1:ic[2]+na1] = 5.
		vol[ic[0]-na2:ic[0]+na2,ic[1]-na2:ic[1]+na2,ic[2]-na2:ic[2]+na2] = 2.
	#vol = m.AddPointstoMolDots(vol,grid,0.9)
	#ipyvol can nicely visualize [nx,nx,xz] integer volume arrays.
	vol = vol.reshape((n,n,n))
	np.save(PARAMS["dens_dir"]+"goEn",vol)
	exit(0)
	return

def TestIpecac(dig_ = "GauSH"):
	""" Tests reversal of an embedding type """
	a=MSet("OptMols")
	a.ReadXYZ("OptMols")
	#Remove half of a
	a.mols = a.mols[-1*int(len(a.mols)/6):]
	TreatedAtoms = a.AtomTypes()
	dig = Digester(TreatedAtoms, name_=dig_, OType_ ="GoForce")
	eopt = EmbeddingOptimizer(a,dig)
	eopt.PerformOptimization()

	if (0):
		a=MSet("OptMols")
		a.ReadXYZ("OptMols")
		m = a.mols[5]
		m.WriteXYZfile("./results/", "Before")
		goodcrds = m.coords.copy()
		m.BuildDistanceMatrix()
		gooddmat = m.DistMatrix
		print "Good Coordinates", goodcrds
		TreatedAtoms = m.AtomTypes()
		dig = Digester(TreatedAtoms, name_=dig_, OType_ ="GoForce")
		emb = dig.TrainDigestMolwise(m,MakeOutputs_=False)
		m.Distort()
		m.WriteXYZfile("./results/", "Distorted")
		bestfit = ReverseAtomwiseEmbedding(m.atoms, dig, emb, guess_=m.coords,GdDistMatrix=gooddmat)
		bestfit.WriteXYZfile("./results/", "BestFit")
	return

def TestHerrNet1(dig_ = "GauSH", net_ = "fc_sqdiff"):
	"""
	Test John Herr's first Optimized Force Network.
	"""
	# This Tests the optimizer.
	#test_mol = a.mols[0]
	a=MSet("OptMols")
	a.ReadXYZ("OptMols")
	test_mol = a.mols[5]
	print "Orig Coords", test_mol.coords
	#test_mol.Distort(0.25,0.2)
	print test_mol.coords
	print test_mol.atoms
	manager=TFManage("SmallMols_20rot_"+dig_+"_"+net_,None,False)
	optimizer  = Optimizer(manager)
	optimizer.OptTFRealForce(test_mol)
	return

def TestOCSDB(dig_ = "GauSH", net_ = "fc_sqdiff"):
	"""
	Test John Herr's first Optimized Force Network.
	OCSDB_test contains good crystal structures.
	- Evaluate RMS forces on them.
	- Optimize OCSDB_Dist02
	- Evaluate the relative RMS's of these two.
	"""
	tfm=TFManage("SmallMols_20rot_"+dig_+"_"+net_,None,False)
	a=MSet("OCSDB_test")
	a.ReadXYZ("OCSDB_test")
	b=MSet("OCSDB_Dist02")
	b.ReadXYZ("OCSDB_Dist02")
	print "A,B RMS (Angstrom): ",a.rms(b)
	frcs = np.zeros(shape=(1,3))
	for m in a.mols:
		frc = tfm.EvalRotAvForce(m, RotAv=PARAMS["RotAvOutputs"], Debug=False)
		frcs=np.append(frcs,frc,axis=0)
	print "RMS Force of crystal structures:",np.sqrt(np.sum(frcs*frcs,axis=(0,1))/(frcs.shape[0]-1))
	b.name = "OCSDB_Dist02_OPTd"
	optimizer  = Optimizer(tfm)
	for i,m in enumerate(b.mols):
		m = optimizer.OptTFRealForce(m,str(i))
	b.WriteXYZ()
	print "A,B (optd) RMS (Angstrom): ",a.rms(b)
	return

def TestNeb(dig_ = "GauSH", net_ = "fc_sqdiff"):
	"""
	Test NudgedElasticBand
	"""
	tfm=TFManage("SmallMols_20rot_"+dig_+"_"+net_,None,False)
	optimizer  = Optimizer(tfm)
	a=MSet("NEB_Berg")
	a.ReadXYZ("NEB_Berg")
	m0 = a.mols[0]
	m1 = a.mols[1]
	# These have to be aligned and optimized if you want a good PES.
	m0.AlignAtoms(m1)
	m0 = optimizer.OptTFRealForce(m0,"NebOptM0")
	m1 = optimizer.OptTFRealForce(m1,"NebOptM1")
	PARAMS["NebNumBeads"] = 30
	PARAMS["NebK"] = 2.0
	PARAMS["OptStepSize"] = 0.002
	PARAMS["OptMomentum"] = 0.0
	PARAMS["OptMomentumDecay"] = 1.0
	neb = NudgedElasticBand(tfm, m0, m1)
	neb.OptNeb()
	return

def TestNebGLBFGS(dig_ = "GauSH", net_ = "fc_sqdiff"):
	"""
	Test NudgedElasticBand
	"""
	tfm=TFManage("SmallMols_20rot_"+dig_+"_"+net_,None,False)
	optimizer  = Optimizer(tfm)
	a=MSet("NEB_Berg")
	a.ReadXYZ("NEB_Berg")
	m0 = a.mols[0]
	m1 = a.mols[1]
	# These have to be aligned and optimized if you want a good PES.
	m0.AlignAtoms(m1)
	m0 = optimizer.OptTFRealForce(m0,"NebOptM0")
	m1 = optimizer.OptTFRealForce(m1,"NebOptM1")
	PARAMS["NebNumBeads"] = 30
	PARAMS["NebK"] = 2.0
	PARAMS["OptStepSize"] = 0.001
	PARAMS["OptMomentum"] = 0.0
	PARAMS["OptMomentumDecay"] = 1.0
	neb = NudgedElasticBand(tfm, m0, m1)
	neb.OptNebGLBFGS()
	return

#
# Tests to run.
#

#TestBP(set_="gdb9", dig_="GauSH", BuildTrain_= True)
#TestGoForceAtom(dig_ = "GauSH", BuildTrain_=True, net_ = "fc_sqdiff", Train_=True)
#TestPotential()
#TestIpecac()
#TestHerrNet1()
TestOCSDB()
#TestNeb()
#TestNebGLBFGS() # Not working... for some reason.. I'll try DIIS next.

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
