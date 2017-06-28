from TensorMol import *
import time

#jeherr tests

# PARAMS["RBFS"] = np.array([[0.24666382, 0.37026093], [0.42773663, 0.47058503], [0.5780647, 0.47249905], [0.63062578, 0.60452219],
# 			[1.30332807, 1.2604625], [2.2, 2.4], [4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]])
# PARAMS["ANES"] = np.array([0.96763427, 1., 1., 1., 1., 2.14952757, 1.95145955, 2.01797792])
# PARAMS["RBFS"] = np.array([[0.1, 0.2], [0.2, 0.3], [0.5, 0.35], [0.9, 0.3], [1.1, 0.3], [1.3, 0.3], [1.6, 0.4], [1.9, 0.5],
# 				[4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]])
# PARAMS["ANES"] = np.array([2.20, 1., 1., 1., 1., 2.55, 3.04, 3.98])
# PARAMS["RBFS"] = np.array([[0.14281105, 0.25747465], [0.24853184, 0.38609822], [0.64242406, 0.36870154], [0.97548212, 0.39012401],
#   							[1.08681976, 0.25805578], [1.34504847, 0.16033599], [1.49612151, 0.31475267], [1.91356037, 0.52652435],
# 							[2.2, 2.4], [4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]])
PARAMS["RBFS"] = np.array([[0.14281105, 0.25747465], [0.24853184, 0.38609822], [0.64242406, 0.36870154], [0.97548212, 0.39012401],
 							[1.08681976, 0.25805578], [1.34504847, 0.16033599], [1.49612151, 0.31475267], [1.91356037, 0.52652435],
							[2.35, 0.8], [2.8, 0.8], [3.25, 0.8], [3.7, 0.8], [4.15, 0.8], [4.6, 0.8], [5.05, 0.8], [5.5, 0.8], [5.95, 0.8],
							[6.4, 0.8], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]])
PARAMS["ANES"] = np.array([[1.02539286, 1., 1., 1., 1., 2.18925953, 2.71734044, 3.03417733]])
PARAMS["SH_NRAD"] = 14
PARAMS["SH_LMAX"] = 4

S_Rad = MolEmb.Overlap_RBF(PARAMS)
S_RadOrth = MatrixPower(S_Rad,-1./2)
PARAMS["SRBF"] = S_RadOrth
PARAMS["RandomizeData"] = True
# PARAMS["InNormRoutine"] = "MeanStd"
# PARAMS["OutNormRoutine"] = "MeanStd"
PARAMS["TestRatio"] = 0.2
PARAMS["max_steps"] = 5000
PARAMS["batch_size"] = 8000
PARAMS["NeuronType"] = "elu"

# PARAMS["AN1_r_Rc"] = 6.
# PARAMS["AN1_a_Rc"] = 4.
# PARAMS["AN1_eta"] = 4.0
# PARAMS["AN1_zeta"] = 8.0
# PARAMS["AN1_num_r_Rs"] = 16
# PARAMS["AN1_num_a_Rs"] = 4
# PARAMS["AN1_num_a_As"] = 8
# PARAMS["hidden1"] = 64
# PARAMS["hidden2"] = 128
# PARAMS["hidden3"] = 64
# PARAMS["max_steps"] = 1001
# PARAMS["GradWeight"] = 1.0
# PARAMS["AN1_r_Rs"] = np.array([ PARAMS["AN1_r_Rc"]*i/PARAMS["AN1_num_r_Rs"] for i in range (0, PARAMS["AN1_num_r_Rs"])])
# PARAMS["AN1_a_Rs"] = np.array([ PARAMS["AN1_a_Rc"]*i/PARAMS["AN1_num_a_Rs"] for i in range (0, PARAMS["AN1_num_a_Rs"])])
# PARAMS["AN1_a_As"] = np.array([ 2.0*Pi*i/PARAMS["AN1_num_a_As"] for i in range (0, PARAMS["AN1_num_a_As"])])

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Takes two nearly identical crystal lattices and interpolates a core/shell structure, must be oriented identically and stoichiometric
def InterpolateGeometries():
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

def ReadSmallMols(set_="SmallMols", dir_="/media/sdb2/jeherr/TensorMol/datasets/small_mol_dataset_del/*/*/", energy=False, forces=False, mmff94=False):
	import glob
	a=MSet(set_)
	for dir in glob.iglob(dir_):
		a.ReadXYZUnpacked(dir, has_force=forces, has_energy=energy, has_mmff94=mmff94)
	print len(a.mols)
	a.Save()


def TrainKRR(set_ = "SmallMols", dig_ = "GauSH", OType_ ="Force"):
	a=MSet("SmallMols_rand")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms, name_=dig_,OType_ =OType_)
	tset = TensorData(a,d)
	tset.BuildTrainMolwise("SmallMols",TreatedAtoms)
	manager=TFManage("",tset,True,"KRR_sqdiff")
	return

def RandomSmallSet(set_, size_):
	""" Returns an MSet of random molecules chosen from a larger set """
	print "Selecting a subset of "+str(set_)+" of size "+str(size_)
	a=MSet(set_)
	a.Load()
	b=MSet(set_+"_rand")
	mols = random.sample(range(len(a.mols)), size_)
	for i in mols:
		b.mols.append(a.mols[i])
	b.Save()
	return b

def BasisOpt_KRR(method_, set_, dig_, OType = None, Elements_ = []):
	""" Optimizes a basis based on Kernel Ridge Regression """
	a=MSet(set_)
	a.Load()
	TreatedAtoms = a.AtomTypes()
	dig = Digester(TreatedAtoms, name_=dig_, OType_ = OType)
	eopt = EmbeddingOptimizer(method_, a, dig, OType, Elements_)
	eopt.PerformOptimization()
	return

def BasisOpt_Ipecac(method_, set_, dig_):
	""" Optimizes a basis based on Ipecac """
	a=MSet(set_)
	a.Load()
	print "Number of mols: ", len(a.mols)
	TreatedAtoms = a.AtomTypes()
	dig = MolDigester(TreatedAtoms, name_=dig_, OType_ ="GoForce")
	eopt = EmbeddingOptimizer("Ipecac", a, dig, "radial")
	eopt.PerformOptimization()
	return

def TestIpecac(dig_ = "GauSH"):
	""" Tests reversal of an embedding type """
	a=MSet("OptMols")
	a.ReadXYZ("OptMols")
	m = a.mols[1]
	print m.atoms
	# m.WriteXYZfile("./results/", "Before")
	goodcrds = m.coords.copy()
	m.BuildDistanceMatrix()
	gooddmat = m.DistMatrix
	print "Good Coordinates", goodcrds
	TreatedAtoms = m.AtomTypes()
	dig = MolDigester(TreatedAtoms, name_=dig_, OType_ ="GoForce")
	emb = dig.Emb(m, MakeOutputs=False)
	m.Distort()
	ip = Ipecac(a, dig, eles_=[1,6,7,8])
	# m.WriteXYZfile("./results/", "Distorted")
	bestfit = ip.ReverseAtomwiseEmbedding(emb, atoms_=m.atoms, guess_=m.coords,GdDistMatrix=gooddmat)
	# bestfit = ReverseAtomwiseEmbedding(dig, emb, atoms_=m.atoms, guess_=None,GdDistMatrix=gooddmat)
	# print bestfit.atoms
	print m.atoms
	# bestfit.WriteXYZfile("./results/", "BestFit")
	return

def TestBP(set_= "gdb9", dig_ = "Coulomb",BuildTrain_ =False):
	"""
	General Behler Parinello using ab-initio energies.
	Args:
		set_: A dataset ("gdb9 or alcohol are available")
		dig_: the digester string
	"""
	print "Testing General Behler-Parrinello using ab-initio energies...."
	PARAMS["NormalizeOutputs"] = True
	#	if (BuildTrain_): # Need to add missing parts of set to get this separated...
	a=MSet(set_)
	a.ReadXYZ(set_)
	TreatedAtoms = a.AtomTypes()
	print "TreatedAtoms ", TreatedAtoms
	d = MolDigester(TreatedAtoms, name_=dig_+"_BP", OType_="AtomizationEnergy")
	tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol")
	tset.BuildTrain(set_)
	#tset = TensorMolData_BP(MSet(),MolDigester([]),set_+"_"+dig_+"_BP")
	manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
	manager.Train(maxstep=500)  # train the neural network for 500 steps, by default it trainse 10000 steps and saved in ./networks.
	# We should try to get optimizations working too...
	return

def TestANI1():
	"""
	copy uneq_chemspider from kyao@zerg.chem.nd.edu:/home/kyao/TensorMol/datasets/uneq_chemspider.xyz
	"""
	# a = MSet("uneq_chemspider")
	# a.ReadXYZ("uneq_chemspider")
	# a.Save()
	a = MSet("uneq_chemspider")
	a.Load()
	a=a.RotatedClone(1)
	print "Set elements: ", a.AtomTypes()
	TreatedAtoms = a.AtomTypes()
	d = MolDigester(TreatedAtoms, name_="GauSH", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data fo
	tset.BuildTrain("uneq_chemspider")
	# tset = TensorMolData_BP(MSet(),MolDigester([]),"uneq_chemspider_ANI1_Sym")
	manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
	manager.Train(maxstep=2000)
	#manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
            #manager.Continue_Training(maxsteps=2)
	return

def TrainForces(set_ = "SmallMols", dig_ = "GauSH", BuildTrain_=True, numrot_=None):
	if (BuildTrain_):
		a=MSet(set_)
		a.Load()
		if numrot_ != None:
			a = a.RotatedClone(numrot_)
			a.Save(a.name+"_"+str(numrot_)+"rot")
		TreatedAtoms = a.AtomTypes()
		print "Number of Mols: ", len(a.mols)
		d = Digester(TreatedAtoms, name_=dig_, OType_="Force")
		tset = TensorData(a,d)
		tset.BuildTrainMolwise(set_,TreatedAtoms)
	else:
		tset = TensorData(None,None,set_+"_"+dig_)
	manager=TFManage("",tset,False,"fc_sqdiff")
	manager.TrainElement(1)

def OptTFForces(set_= "SmallMols", dig_ = "GauSH", mol = 0):
	a=MSet(set_)
	a.ReadXYZ()
	tmol=copy.deepcopy(a.mols[mol])
	# tmol.Distort(0.1)
	manager=TFManage("SmallMols_20rot_"+dig_+"_"+"fc_sqdiff", None, False)
	opt=Optimizer(manager)
	opt.OptTFRealForce(tmol)

def TestOCSDB(dig_ = "GauSH", net_ = "fc_sqdiff"):
	"""
	Test John Herr's first Optimized Force Network.
	OCSDB_test contains good crystal structures.
	- Evaluate RMS forces on them.
	- Optimize OCSDB_Dist02
	- Evaluate the relative RMS's of these two.
	"""
	tfm=TFManage("SmallMols_20rot_"+dig_+"_"+net_,None,False)
	a=MSet("OCSDB_Dist02_opt")
	a.ReadXYZ()
	b=MSet("OCSDB_Dist02_opt_test")
	b.mols = copy.deepcopy(a.mols)
	for m in b.mols:
		m.Distort(0.1)
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
	Test NudgedElasticBand with LBFGS... not working :(
	"""
	tfm=TFManage("SmallMols_20rot_"+dig_+"_"+net_,None,False)
	optimizer  = Optimizer(tfm)
	a=MSet("NEB_Berg")
	a.ReadXYZ("NEB_Berg")
	m0 = a.mols[0]
	m1 = a.mols[1]
	# These have to be aligned and optimized if you want a good PES.
	m0.AlignAtoms(m1)
	PARAMS["RotAvOutputs"] = 10
	PARAMS["DiisSize"] = 20
	m0 = optimizer.OptTFRealForce(m0,"NebOptM0")
	m1 = optimizer.OptTFRealForce(m1,"NebOptM1")
	PARAMS["NebNumBeads"] = 30
	PARAMS["NebK"] = 2.0
	PARAMS["OptStepSize"] = 0.001
	PARAMS["OptMomentum"] = 0.0
	PARAMS["RotAvOutputs"] = 10
	PARAMS["OptMomentumDecay"] = 1.0
	neb = NudgedElasticBand(tfm, m0, m1)
	neb.OptNebGLBFGS()
	return

def TestMD(dig_ = "GauSH", net_ = "fc_sqdiff"):
	"""
	Test MolecularDynamics
	"""
	tfm=TFManage("SmallMols_20rot_"+dig_+"_"+net_,None,False)
	a=MSet("OCSDB_test")
	a.ReadXYZ("OCSDB_test")
	m = a.mols[1]
	# Convert the forces from kcal/mol ang to joules/mol ang.
	ForceField = lambda x: 4183.9953*tfm.EvalRotAvForce(Mol(m.atoms,x), RotAv=PARAMS["RotAvOutputs"])
	PARAMS["MNHChain"] = 10
	PARAMS["MDTemp"] = 300.0
	PARAMS["MDThermostat"] = "NosePerParticle"
	PARAMS["MDdt"] = 0.1 # In fs.
	md = VelocityVerlet(ForceField,m)
	md.Prop()
	return

def TestAnneal(dig_ = "GauSH", net_ = "fc_sqdiff"):
	"""
	Test MolecularDynamics
	"""
	tfm=TFManage("SmallMols_20rot_"+dig_+"_"+net_,None,False)
	a=MSet("OCSDB_test")
	a.ReadXYZ("OCSDB_test")
	m = a.mols[1]
	# Convert the forces from kcal/mol ang to joules/mol ang.
	ForceField = lambda x: 4183.9953*tfm.EvalRotAvForce(Mol(m.atoms,x), RotAv=PARAMS["RotAvOutputs"])
	PARAMS["MNHChain"] = 10
	PARAMS["MDTemp"] = 300.0
	PARAMS["MDThermostat"] = "NosePerParticle"
	PARAMS["MDdt"] = 0.1 # In fs.
	md = NoEnergyAnnealer(ForceField,m)
	md.Prop()
	return



def TestMorphIR():
	"""
	Try to model the IR spectra of Johnson's peptides...
	Optimize, then get charges, then do an isotropic IR spectrum.
	"""
	a = MSet("johnsonmols")
	a.ReadXYZ("johnsonmols")
	manager= TFManage("SmallMols_20rot_GauSH_fc_sqdiff", None, False, RandomTData_=False, Trainable_=False)
	PARAMS["OptMomentum"] = 0.0
	PARAMS["OptMomentumDecay"] = 0.9
	PARAMS["OptStepSize"] = 0.02
	PARAMS["OptMaxCycles"]=200
	morphine = a.mols[1]
	heroin = a.mols[2]
	optimizer = Optimizer(manager)
	optimizer.OptTFRealForce(morphine)
	qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	ForceField = lambda x: manager.EvalRotAvForce(Mol.Mol(morphine.atoms,x))
	ChargeField = lambda x: qmanager.Eval_BPDipole(Mol.Mol(morphine.atoms,x),False)[2][0]
	PARAMS["MDdt"] = 0.2
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 10000
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDV0"] = None
	PARAMS["MDTemp"]= 1.0
	annealMorph = Annealer(ForceField, ChargeField, morphine, "Anneal")
	annealMorph.Prop()
	morphine.coords = annealMorph.Minx.copy()
	PARAMS["MDTemp"]= 0.0
	PARAMS["MDThermostat"] = None
	PARAMS["MDFieldAmp"] = 20.0 #0.00000001
	PARAMS["MDFieldTau"] = 0.4
	PARAMS["MDFieldFreq"] = 0.8
	PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
	md0 = IRTrajectory(ForceField, ChargeField, morphine, "MorphineIR")
	md0.Prop()
	WriteDerDipoleCorrelationFunction(md0.mu_his,"MorphineMutM0.txt")
	return
	optimizer.OptANI1(heroin)
	ForceField = lambda x: manager.Eval_BPForceSingle(Mol(heroin.atoms,x),True)
	ChargeField = lambda x: qmanager.Eval_BPDipole(Mol(heroin.atoms,x),False)[2][0]
	annealHeroin = Annealer(ForceField, ChargeField, heroin, "Anneal")
	annealHeroin.Prop()
	heroin.coords = annealHeroin.Minx.copy()
	PARAMS["MDTemp"]= 0.0
	PARAMS["MDThermostat"] = None
	PARAMS["MDFieldAmp"] = 3.0 #0.00000001
	PARAMS["MDFieldTau"] = 0.4
	PARAMS["MDFieldFreq"] = 0.8
	PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
	md1 = IRTrajectory(ForceField, ChargeField, heroin, "HeroinIR")
	md1.Prop()
	WriteDerDipoleCorrelationFunction(md1.mu_his,"HeroinMutM0.txt")
	return

def Test_ULJ():
	"""
	Create a Universal Lennard-Jones model.
	"""
	# This Tests the optimizer.
	print "Learning Best-Fit element specific LJ parameters."
	a=MSet("SmallMols")
	a.Load()
	print "Loaded data..."
	TreatedAtoms = a.AtomTypes()
	d = MolDigester(TreatedAtoms, name_="CZ", OType_ ="Force")
	tset = TensorMolData(a,d)
	PARAMS["learning_rate"]=0.0001
	PARAMS["momentum"]=0.85
	manager=TFMolManage("",tset,True,"LJForce") # True indicates train all atoms
	return

def Test_LJMD():
	"""
	Test TensorFlow LJ fluid Molecular dynamics
	"""
	a=MSet("Test")
	ParticlesPerEdge = 2
	EdgeSize = 2
	a.mols=[Mol(np.ones(ParticlesPerEdge*ParticlesPerEdge*ParticlesPerEdge,dtype=np.uint8),MakeUniform([0.0,0.0,0.0],EdgeSize,ParticlesPerEdge))]
	#a.mols=[Mol(np.ones(512),MakeUniform([0.0,0.0,0.0],4.0,8))]
	m = a.mols[0]
	TreatedAtoms = a.AtomTypes()
	d = MolDigester(TreatedAtoms, name_="CZ", OType_ ="Force")
	tset = TensorMolData(a,d)
	ins = MolInstance_DirectForce(tset,None,False,"Harm")
	ins.train_prepare()
	# Convert from hartree/ang to joules/mol ang.
	ForceField = lambda x: ins.EvalForce(Mol(m.atoms,x))[0][0]
	EnergyForceField = lambda x: ins.EvalForce(Mol(m.atoms,x))

	if (0):
		PARAMS["OptThresh"] = 0.01
		m = GeomOptimizer(EnergyForceField).Opt(m)
		anneal = Annealer(EnergyForceField, None, m, "Anneal")
		anneal.Prop()
		m.coords = anneal.Minx.copy()
		m = GeomOptimizer(EnergyForceField).Opt(m)

	PARAMS["MDTemp"] = 300.0
	PARAMS["MDThermostat"] = None
	PARAMS["MDV0"] = None
	PARAMS["MDdt"] = 0.2
	#print "TF grad:",EnergyForceField(m.coords)
	#print "Fdiff Grad: "
	#print JOULEPERHARTREE*FdiffGradient(ForceField,m.coords)
	#Ee = 0.01*np.ones((8,8))
	#Re = 1.*np.ones((8,8))
	#EnergyField = lambda x: ins.EvalForce(Mol(m.atoms,x))[0][0]
	#EnergyField = lambda x: LJEnergy_Numpy(x, m.atoms, Ee, Re)
	#ForceField = lambda x: -1.0*JOULEPERHARTREE*FdiffGradient(EnergyField,x)
	#EnergyForceField = lambda x: (EnergyField(x), ForceField(x))
	md = VelocityVerlet(ForceField,m,"LJ test", EnergyForceField)
	md.Prop()
	return

def Brute_LJParams():
	a=MSet("SmallMols_rand")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	d = MolDigester(TreatedAtoms, name_="CZ", OType_ ="Energy")
	tset = TensorMolData(a,d)
	ins = MolInstance_DirectForce_tmp(tset,None,False,"Harm")
	ins.train_prepare()
	import scipy.optimize
	rranges = (slice(-1000, 1000, 10), slice(0.5, 6, 0.25))
	resbrute = scipy.optimize.brute(ins.LJFrc, rranges, full_output=True, finish=scipy.optimize.fmin)
	print resbrute[0]
	print resbrute[1]
	# print ins.LJFrc(p)

def QueueTrainForces(set_ = "SmallMols", dig_ = "GauSH", BuildTrain_=True, numrot_=None):
	if (BuildTrain_):
		a=MSet(set_)
		a.Load()
		if numrot_ != None:
			a = a.RotatedClone(numrot_)
			a.Save(a.name+"_"+str(numrot_)+"rot")
		TreatedAtoms = a.AtomTypes()
		print "Number of Mols: ", len(a.mols)
		d = Digester(TreatedAtoms, name_=dig_, OType_="Force")
		tset = TensorData_TFRecords(a,d)
		tset.BuildTrainMolwise(set_,TreatedAtoms)
	else:
		tset = TensorData(None,None,set_+"_"+dig_)
	manager=TFManage("",tset,False,"fc_sqdiff_queue")
	manager.TrainElement(1)

def TestForces():
	a=MSet("chemspid")
	# a=MSet("SmallMols")
	a.Load()
	manager=TFManage("SmallMols_GauSH_fc_sqdiff", None, False)
	err = np.zeros((32000,3))
	ntest = 0
	for mol in a.mols:
		for i, atom in enumerate(mol.atoms):
			if atom == 7:
				pforce = manager.evaluate(mol, i)
				print "True force:", mol.properties["forces"][i], "Predicted force:", pforce
				err[ntest] = mol.properties["forces"][i] - pforce
				ntest += 1
				if ntest == 32000:
					break
		if ntest == 32000:
			break
	print "MAE:", np.mean(np.abs(err)), " Std:", np.std(np.abs(err))
	# print err

def MakeTestSet():
	a=MSet("SmallMols")
	a.Load()
	b, c = a.SplitTest()





# InterpoleGeometries()
# ReadSmallMols(set_="chemspid1", dir_="/media/sdb2/jeherr/TensorMol/datasets/chemspider1_data/*/", forces=True)
# TrainKRR(set_="SmallMols_rand", dig_ = "GauSH", OType_="Force")
# RandomSmallSet("SmallMols", 30000)
# BasisOpt_KRR("KRR", "SmallMols_rand", "GauSH", OType = "Force", Elements_ = [1,6,7,8])
# BasisOpt_Ipecac("KRR", "ammonia_rand", "GauSH")
# TestIpecac()
# TestBP()
# TestANI1()
# TrainForces(set_ = "SmallMols", BuildTrain_=False, numrot_=1)
# OptTFForces(set_ = "peptide", mol=0)
# TestOCSDB()
# TestNeb()
# TestNebGLBFGS()
# TestMD()
# TestAnneal()
# TestMorphIR()
# Brute_LJParams()
# QueueTrainForces(set_ = "SmallMols", BuildTrain_=False, numrot_=1)
# TestForces()
MakeTestSet()


# a=MSet("pentane_eq_align")
# a.ReadXYZ()
# tmol = copy.deepcopy(a.mols[0])
# # print tmol.coords
# b=MSet("pentane_stretch")
# s_list = np.linspace(1,-0.75,100).tolist()
# for i in s_list:
# 	nmol = Mol(tmol.atoms, tmol.coords)
# 	for j in range(4):
# 		nmol.coords[j] += i*(tmol.coords[1] - tmol.coords[4])
# 	# print nmol.coords
# 	b.mols.append(nmol)
# b.Save()
# b.WriteXYZ()

# a=MSet("pentane_stretch")
# a.Load()
# manager=TFManage("SmallMols_20rot_GauSH_fc_sqdiff",None,False)
# veloc = np.zeros((len(a.mols),4))
# for i, mol in enumerate(a.mols):
# 	for j in range(4):
# 		veloc[i,j] = self.tfm.evaluate(mol,j)
# print veloc

# a=MSet("pentane_eq")
# a.ReadXYZ()
# tmol = Mol(a.mols[0].atoms, a.mols[0].coords-a.mols[0].coords[4])
#
# def RodriguesRot(mol, vec, axis, angle):
# 	tmpcoords = tmol.coords.copy()
# 	vec = vec/np.linalg.norm(vec)
# 	k = 0.5*(vec+axis)/np.linalg.norm(0.5*(vec+axis))
# 	for m in range(len(tmpcoords)): #rotate so oxygen is eclipsed by carbon
# 		tmpcoords[m] = (math.cos(math.pi)*tmpcoords[m])+(numpy.cross(k,tmpcoords[m])*math.sin(math.pi))+(k*(numpy.dot(k, tmpcoords[m])*(1-math.cos(math.pi))))
# 	return tmpcoords
#
# tmol.coords = RodriguesRot(tmol, (tmol.coords[1]-tmol.coords[4]), np.array((1,0,0)), np.pi)
# tmol.WriteXYZfile(fpath="./", fname="pentane_eq_align", mode="w")

# a=MSet("pentane_stretch")
# a.Load()
# f=open("./results/pentane_stretch.in", "w")
# for i, mol in enumerate(a.mols):
# 	f.write("$molecule\n")
# 	f.write("0 1\n")
# 	for j, atom in enumerate(mol.atoms):
# 		if (atom == 1):
# 			f.write("H          ")
# 		if (atom == 6):
# 			f.write("C          ")
# 		f.write(str(mol.coords[j,0])+"        "+str(mol.coords[j,1])+"        "+str(mol.coords[j,2])+"\n")
# 	f.write("$end\n\n$rem\n")
# 	f.write("jobtype           force\n")
# 	f.write("method            wB97X-D \n")
# 	f.write("basis             6-311G**\n")
# 	f.write("sym_ignore        true\n")
# 	f.write("$end\n\n@@@\n\n")
# f.close()
