from TensorMol import *
import time, os
os.environ["CUDA_VISIBLE_DEVICES"]=""

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
PARAMS["hidden1"] = 100
PARAMS["hidden2"] = 100
PARAMS["hidden3"] = 100

S_Rad = MolEmb.Overlap_RBF(PARAMS)
S_RadOrth = MatrixPower(S_Rad,-1./2)
PARAMS["SRBF"] = S_RadOrth
PARAMS["RandomizeData"] = True
# PARAMS["InNormRoutine"] = "MeanStd"
# PARAMS["OutNormRoutine"] = "MeanStd"
PARAMS["TestRatio"] = 0.2
PARAMS["max_steps"] = 200
PARAMS["test_freq"] = 10
PARAMS["batch_size"] = 8000
PARAMS["NeuronType"] = "relu"
# PARAMS["Profiling"] = True

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

#os.environ["CUDA_VISIBLE_DEVICES"]="0"


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

def ReadSmallMols(set_="MDMols", dir_="/media/sdb1/dtoth/sampling_mols/qchem_data/*/*", energy=False, forces=False, charges=False, mmff94=False):
	import glob
	a=MSet(set_)
	for dir in glob.iglob(dir_):
		a.ReadXYZUnpacked(dir, has_force=forces, has_energy=energy, has_charge=charges, has_mmff94=mmff94)
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

def QueueTrainForces(trainset_ = "SmallMols_train", testset_ = "SmallMols_test", dig_ = "GauSH", BuildTrain_=True, numrot_=None):
	if (BuildTrain_):
		a=MSet(trainset_)
		a.Load()
		b=MSet(testset_)
		b.Load()
		if numrot_ != None:
			a = a.RotatedClone(numrot_)
			a.Save(a.name+"_"+str(numrot_)+"rot")
		TreatedAtoms = a.AtomTypes()
		print "Number of Mols: ", len(a.mols)
		d = Digester(TreatedAtoms, name_=dig_, OType_="Force")
		tset = TensorData_TFRecords(a,d)
		tset.BuildTrainMolwise(set_,TreatedAtoms)
	else:
		trainset = TensorData_TFRecords(None,None,trainset_+"_"+dig_)
		testset = TensorData_TFRecords(None, None,testset_+"_"+dig_)
	manager=TFManage_Queue("",trainset, testset, False,"fc_sqdiff_queue")
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
	b=MSet("SmallMols_train")
	b.Load()
	# c=MSet("SmallMols_test")
	# c.Load()
	TreatedAtoms = b.AtomTypes()
	print "Number of train Mols: ", len(b.mols)
	# print "Number of test Mols: ", len(c.mols)
	d = Digester(TreatedAtoms, name_="GauSH", OType_="Force")
	train_set = TensorData_TFRecords(b,d)
	train_set.BuildTrainMolwise("SmallMols_train",TreatedAtoms)
	# test_set = TensorData_TFRecords(c,d, test_=True)
	# test_set.BuildTrainMolwise("SmallMols_test",TreatedAtoms)

def TestMetadynamics():
	a = MSet("sampling_mols")
	a.ReadXYZ()
	m = a.mols[4]
	ForceField = lambda x: QchemDFT(Mol(m.atoms,x),basis_ = '6-311g**',xc_='wB97X-D', jobtype_='force', filename_='hexanol', path_='./qchem/', threads=24)
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms))
	print "Masses:", masses
	PARAMS["MDdt"] = 2.0
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 10000
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDTemp"]= 600.0
	meta = MetaDynamics(ForceField, m, 'hexanol')
	meta.Prop()

def TestTFBond():
	a=MSet("SmallMols_rand")
	a.Load()
	for mol in a.mols:
		mol.CalculateAtomization()
	a.Save()
	d = MolDigester(a.BondTypes(), name_="CZ", OType_="AtomizationEnergy")
	tset = TensorMolData_BPBond_Direct(a,d)
	# batchdata=tset.RawBatch()
	# Zxyzs = tf.Variable(batchdata[0], dtype=tf.float32)
	# BondIdxMatrix = tf.Variable(batchdata[1], dtype=tf.int64)
	# eles = [1,6,7,8]
	# eles_np = np.asarray(eles).reshape(4,1)
	# eles_pairs = []
	# for i in range (len(eles)):
	# 	for j in range(i, len(eles)):
	# 		eles_pairs.append([eles[i], eles[j]])
	# eles_pairs_np = np.asarray(eles_pairs)
	# Ele = tf.constant(eles_np, dtype = tf.int32)
	# Elep = tf.constant(eles_pairs_np, dtype = tf.int32)
	# sess=tf.Session()
	# init = tf.global_variables_initializer()
	# sess.run(init)
	# print(sess.run(TFBond(Zxyzs, BondIdxMatrix, Elep)))
	manager=TFMolManage("",tset,True,"fc_sqdiff_BPBond_DirectQueue")

# InterpoleGeometries()
# ReadSmallMols(set_="SmallMols", forces=True, energy=True)
#ReadSmallMols(set_="DavidRandom", dir_="/media/sdb1/dtoth/qchem_jobs/new/rndjobs/data/*/", energy=True, forces=True)
# TrainKRR(set_="SmallMols_rand", dig_ = "GauSH", OType_="Force")
# RandomSmallSet("SmallMols", 50000)
# BasisOpt_KRR("KRR", "SmallMols_rand", "GauSH", OType = "Force", Elements_ = [1,6,7,8])
# BasisOpt_Ipecac("KRR", "ammonia_rand", "GauSH")
# TestIpecac()
# TrainForces(set_ = "SmallMols", BuildTrain_=False, numrot_=1)
# OptTFForces(set_ = "peptide", mol=0)
# TestOCSDB()
# Brute_LJParams()
# QueueTrainForces(trainset_ = "SmallMols_train", testset_ = "SmallMols_test", BuildTrain_=False, numrot_=None)
# TestForces()
# MakeTestSet()
TestMetadynamics()
# TestMD()
# TestTFBond()

# a=MSet("OptMols")
# a.ReadXYZ()
# print QchemDFT(a.mols[0],basis_ = '6-311g**',xc_='wB97X-D', jobtype_='force', filename_='optmols0', path_='./qchem/', threads=2)
