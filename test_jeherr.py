from TensorMol import *
import time

#jeherr tests

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

if (0):
	a=MSet("md_set")
	#a.ReadGDB9Unpacked(path='/data/jeherr/TensorMol/datasets/md_datasets/md_set/', has_force=True)
	#a.Save()
	#a.WriteXYZ()
	a.Load()
	b=a.RotatedClone(3)
	b.WriteXYZ("md_set_rot")
	b.Save("md_set_rot")
	a.Load()
	print "nmols:",len(a.mols)
	b=MSet("md_set_rotated")
	b=a.RotatedClone(3)
	b.Save()
	print "nmols:",len(b.mols)
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
	tset = TensorData(a,d)
	tset.BuildTrainMolwise("md_set_rotated",TreatedAtoms)
	tset = TensorData(None,None,"md_set_rotated_"+"GauSH",None,2000)
	manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
	optimizer = Optimizer(manager)
	optimizer.OptRealForce(test_mol)

if(0):
	a=MSet("benzene")
	a.Load()
	test_mol = a.mols[0]
	test_mol.coords = test_mol.coords - np.average(test_mol.coords, axis=0)
	test_mol.Distort()
	manager=TFManage("md_set_rotated_GauSH_fc_sqdiff",None,False)
	optimizer=Optimizer(manager)
	optimizer.OptRealForce(test_mol)

if(0):
	# a=MSet("SmallMols")
	# a.Load()
	# a = a.RotatedClone(20)
	#print "nmols:",len(a.mols)
	#a.Save("SmallMols_20rot")
	#a.WriteXYZ("SmallMols_20rot")
	##a=MSet("mddataset_smallmols")
	##a.Load()
	#TreatedAtoms = a.AtomTypes()
	#d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
	#tset = TensorData(a,d)
	#tset.BuildTrainMolwise("SmallMols",TreatedAtoms)
	tset = TensorData(None,None,"SmallMols_20rot_"+"GauSH")
	manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
	# a=MSet("toluene")
	# a.Load()
	# test_mol = a.mols[0]
	# manager=TFManage("md_set_full_rot_log_GauSH_fc_sqdiff",None,False)
	# optimizer = Optimizer(manager)
	# optimizer.OptTFRealForce(test_mol)

if(0):
	#a=MSet("OptMols")
	##a.ReadXYZ('OptMols')
	##a.Save()
	##a.WriteXYZ()
	#a.Load()
        #b=a.DistortAlongNormals(80, True, 0.7)
	#print "nmols:",len(b.mols)
	#c=b.RotatedClone(6)
	#c.WriteXYZ("OptMols_NEQ_rot")
	#c.Save("OptMols_NEQ_rot")
	##b=MSet("md_set_rot")
	##b.Load()
	###print "nmols:",len(a.mols)
	#print "nmols:",len(c.mols)
	#TreatedAtoms = c.AtomTypes()
	#d = Digester(TreatedAtoms, name_="GauSH",OType_ ="GoForce")
	#tset = TensorData(c,d)
	#tset.BuildTrainMolwise("OptMols_NEQ_rot",TreatedAtoms)
	tset = TensorData(None,None,"OptMols_NEQ_rot_"+"GauSH")
	manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
	#optimizer = Optimizer(manager)
	#optimizer.OptRealForce(test_mol)

if(0):
	#a=MSet("md_OptMols_gdb9clean")
	#a.Save()
	#a.WriteXYZ()
	#print "nmols:",len(a.mols)
	#TreatedAtoms = a.AtomTypes()
	#d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
	#tset = TensorData(a,d)
	#tset.BuildTrainMolwise("md_OptMols_gdb9clean",TreatedAtoms)
	tset = TensorData(None,None,"md_OptMols_gdb9clean_"+"GauSH")
	manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
	#optimizer = Optimizer(manager)
	#optimizer.OptRealForce(test_mol)

# a=MSet("toluene_tmp")
# a.Load()
# test_mol = a.mols[0]
#TreatedAtoms = a.AtomTypes()
#d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
#tset = TensorData(a,d)
#tset.BuildTrainMolwise("benzene_NEQ",TreatedAtoms)
# tset = TensorData(None,None,"toluene_tmp_GauSH")
# manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
# manager=TFManage("toluene_tmp_GauSH_fc_sqdiff",None,False)
# optimizer = Optimizer(manager)
# optimizer.OptTFRealForce(test_mol)

# b=MSet("mixed_KRR_rand")
# b.ReadXYZUnpacked("/media/sdb2/jeherr/TensorMol/datasets/mixed_KRR/", has_force=True)
# b = b.RotatedClone(1)
# b.Save()
# # b.Load()
# TreatedAtoms = b.AtomTypes()
# d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
# tset = TensorData(b,d)
# tset.BuildTrainMolwise("mixed_KRR_rand",TreatedAtoms)
# tset = TensorData(None,None,"mixed_KRR_rand_GauSH")
# # manager=TFManage("",tset,True,"KRR_sqdiff")
# h_inst = Instance_KRR(tset, 1, None)
# mae_h = h_inst.basis_opt_run()

#import glob
#a=MSet("SmallMols")
#for dir in glob.iglob("/media/sdb1/jeherr/TensorMol/datasets/small_mol_dataset/*/*/"):
#	a.ReadXYZUnpacked(dir, has_force=True)
#print len(a.mols)
#a.Save()
#a.WriteXYZ()

#a=MSet("SmallMols")
#a.Load()
#b=MSet("SmallMols_minset")
#mols = random.sample(range(len(a.mols)), 50000)
#for i in mols:
#	b.mols.append(a.mols[i])
#b = b.RotatedClone(1)
#b.Save()
#b.WriteXYZ()
#a=MSet("SmallMols_minset")
#a.Load()
#TreatedAtoms = a.AtomTypes()
#d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
#tset = TensorData(a,d)
#tset.BuildTrainMolwise("SmallMols",TreatedAtoms)
#tset = TensorData(None,None,"SmallMols_GauSH")
#manager=TFManage("",tset,True,"KRR_sqdiff")
##h_inst = Instance_KRR(tset, 1, None)
##print h_inst.basis_opt_run()

def RandomSmallSet(set_, size_):
	""" Returns an MSet of random molecules chosen from a larger set """
	print "Selecting a subset of %s of size %i", set_, size_
	a=MSet(set_)
	a.Load()
	b=MSet(set_+"_rand")
	mols = random.sample(range(len(a.mols)), size_)
	for i in mols:
		b.mols.append(a.mols[i])
	b.Save()
	return b

# a=RandomSmallSet("md_set_full", 20000)
#b=MSet("uracil")
#b.ReadXYZUnpacked("/media/sdb2/jeherr/TensorMol/datasets/md_datasets/uracil/", has_force=True)
#b.Save()
#b.Load()
#b=RandomSmallSet("uracil", 20000)
#b.Save("uracil_rand_20k")
# a.AppendSet(b)
# a.Save()

# a=MSet("md_set_full_rand")
# a.Load()
# a=a.RotatedClone(1)
# a.Save()

def BasisOpt_KRR(method_, set_, dig_, OType = None, Elements_ = []):
	""" Optimizes a basis based on Kernel Ridge Regression """
	a=MSet(set_)
	a.Load()
	TreatedAtoms = a.AtomTypes()
	dig = Digester(TreatedAtoms, name_=dig_, OType_ = OType)
	eopt = EmbeddingOptimizer(method_, a, dig, OType, Elements_)
	eopt.PerformOptimization()
	return

# BasisOpt_KRR("KRR", "md_set_full_rand", "GauSH", OType = "Force", Elements_ = [1,6,7,8])
#BasisOpt_KRR("KRR", "uracil_rand_20k", "GauSH", OType = "Force", Elements_ = [7])

def BasisOpt_Ipecac(method_, set_, dig_):
	""" Optimizes a basis based on Ipecac """
	a=MSet(set_)
	a.Load()
	b=MSet("SmallMolsRand")
	mols = random.sample(range(len(a.mols)), 10)
	#Remove half of a
	for i in mols:
		b.mols.append(a.mols[i])
	# for i in range(len(c.mols)):
	# 	b.mols.append(c.mols[i])
	print "Number of mols: ", len(b.mols)
	TreatedAtoms = b.AtomTypes()
	dig = Digester(TreatedAtoms, name_=dig_, OType_ ="GoForce")
	eopt = EmbeddingOptimizer(b,dig)
	eopt.PerformOptimization()
	return

def TestIpecac(dig_ = "GauSH"):
	""" Tests reversal of an embedding type """
	a=MSet("OptMols")
	a.ReadXYZ("OptMols")
	m = a.mols[5]
	print m.atoms
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
	bestfit = ReverseAtomwiseEmbedding(dig, emb, atoms_=m.atoms, guess_=m.coords,GdDistMatrix=gooddmat)
	bestfit.WriteXYZfile("./results/", "BestFit")
	return

#TestIpecac()

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
		d = MolDigester(TreatedAtoms, name_=dig_+"_BP", OType_="AtomizationEnergy")
		tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol")
		tset.BuildTrain(set_)
	tset = TensorMolData_BP(MSet(),MolDigester([]),set_+"_"+dig_+"_BP")
	manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
	manager.Train(maxstep=500)  # train the neural network for 500 steps, by default it trainse 10000 steps and saved in ./networks.
	# We should try to get optimizations working too...
	return

# TestBP()

def TrainForces(set_ = "SmallMols", dig_ = "GauSH", BuildTrain_=True):
	if (BuildTrain_):
		a=MSet(set_)
		a.Load()
		a = a.RotatedClone(1)
		TreatedAtoms = a.AtomTypes()
		print "Number of Mols: ", len(a.mols)
		d = Digester(TreatedAtoms, name_=dig_, OType_="Force")
		tset = TensorData(a,d)
		tset.BuildTrainMolwise(set_,TreatedAtoms)
	else:
		tset = TensorData(None,None,set_+"_"+dig_)
	manager=TFManage("",tset,True,"fc_sqdiff")

# TrainForces(set_ = "toluene", BuildTrain_=False)

def TestForces(set_= "SmallMols", dig_ = "GauSH", mol = 0):
	a=MSet(set_)
	a.Load()
	tmol=copy.deepcopy(a.mols[mol])
	manager=TFManage(set_+"_"+dig_+"_"+"fc_sqdiff", None, False)
	opt=Optimizer(manager)
	opt.OptTFRealForce(tmol)

# TestForces(set_ = "toluene")

# a=MSet("toluene_0")
# a.Load()
# tmol = copy.deepcopy(a.mols[0])
# manager=TFManage("toluene_0_GauSH_fc_sqdiff", None, False)
# for i in range(len(tmol.atoms)):
# 	outs = manager.TData.dig.Emb(tmol, i, tmol.coords[i],False)
# 	print outs

# a=MSet("toluene_0")
# a.Load()
# b=MSet("toluene_1")
# b.mols.append(copy.deepcopy(a.mols[0]))
# TreatedAtoms = b.AtomTypes()
# d = Digester(TreatedAtoms, name_="GauSH", OType_="Force")
# tset=TensorData(b,d)
# tset.BuildTrainMolwise(b, TreatedAtoms)
# tset.BuildTrain(b, TreatedAtoms)

# a=MSet("toluene")
# a.Load()
# b=MSet("toluene_0")
# for i in range(6000):
# 	b.mols.append(copy.deepcopy(a.mols[0]))
# b.Save()

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

b = Basis_GauSH(Name_=None)
print b.rbfs
