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

if(1):
	#a=MSet("md_set_full")
	#a.Load()
	#a = a.RotatedClone(1)
	#b=MSet("SmallMols")
	#b.Load()
	#b=b.RotatedClone(10)
	#a.AppendSet(b)
	#print "nmols:",len(a.mols)
	#a.Save("mddataset_smallmols")
	#a.WriteXYZ("mddataset_smallmols")
	a=MSet("mddataset_smallmols")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
	tset = TensorData(a,d)
	tset.BuildTrainMolwise("mddataset_smallmols",TreatedAtoms)
	tset = TensorData(None,None,"mddataset_smallmols_"+"GauSH")
	manager=TFManage("",tset,True,"fc_sqdiff") # True indicates train all atoms
	#optimizer = Optimizer(manager)
	#optimizer.OptRealForce(test_mol)

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

#from scipy.optimize import minimize
#step=0
##optset = np.array([0.1, 0.156787, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 1.3, 1.3, 1., 1., 1., 1.])
#optset = np.array([ 0.60310955,  0.5236758,  0.92999626,  1.27205388,  0.60979713,  0.3985326,
#  0.62031682,  0.3728798,   1.2550871,   1.18446065,  0.84459955,  1.4283786,
#  0.83621637,  0.85743746])
#
#
#def opt_basis(rbfs):
#	global step
#	PARAMS["RBFS"] = rbfs[:PARAMS["SH_NRAD"]*2].reshape(PARAMS["SH_NRAD"],2).copy()
#	PARAMS["ANES"][0] = rbfs[PARAMS["SH_NRAD"]*2].copy()
#	PARAMS["ANES"][5] = rbfs[PARAMS["SH_NRAD"]*2+1].copy()
#	PARAMS["ANES"][6] = rbfs[PARAMS["SH_NRAD"]*2+2].copy()
#	PARAMS["ANES"][7] = rbfs[PARAMS["SH_NRAD"]*2+3].copy()
#	S_Rad = MolEmb.Overlap_RBF(PARAMS)
#	try:
#		if (np.amin(np.linalg.eigvals(S_Rad)) < 1.e-10):
#			mae = 100
#			return mae
#	except numpy.linalg.linalg.LinAlgError:
#		mae = 100
#		return mae
#	PARAMS["SRBF"] = MatrixPower(S_Rad,-1./2)
#	a=MSet("SmallMols_minset")
#	a.Load()
#	TreatedAtoms = a.AtomTypes()
#	d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
#	tset = TensorData(a,d)
#	tset.BuildTrainMolwise("SmallMols",TreatedAtoms)
#	tset = TensorData(None,None,"SmallMols_GauSH")
#	h_inst = Instance_KRR(tset, 1, None)
#	mae_h = h_inst.basis_opt_run()
#	c_inst = Instance_KRR(tset, 6, None)
#	mae_c = c_inst.basis_opt_run()
#	n_inst = Instance_KRR(tset, 7, None)
#	mae_n = n_inst.basis_opt_run()
#	o_inst = Instance_KRR(tset, 8, None)
#	mae_o = o_inst.basis_opt_run()
#	mae = mae_h + mae_c + mae_o + mae_n
#	step+=1
#	LOGGER.info("RBFS params: "+str(rbfs))
#	LOGGER.info("Minimal Overlap Eigenvalue: "+str(np.amin(np.linalg.eigvals(S_Rad))))
#	LOGGER.info("MAE: "+str(mae))
#	LOGGER.info("Step: "+str(step))
#	return mae
#
##res = minimize(opt_basis, optset, method='L-BFGS-B', bounds=((0,None),(0,None),(0,None),
##	(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(None,None),(None,None),(None,None),(None,None)), jac=False, tol=1.e-2, options={'disp':True, 'factr':0.0000001, 'maxcor':30, 'eps':0.1})
#res = minimize(opt_basis, optset, method='L-BFGS-B', jac=False, tol=1.e-2, options={'disp':True, 'factr':0.0000001, 'maxcor':30, 'eps':0.01})

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

# import glob
# a=MSet("SmallMols")
# for dir in glob.iglob("/media/sdb2/jeherr/TensorMol/datasets/small_mol_dataset/*/*/"):
# 	a.ReadXYZUnpacked(dir, has_force=True)
# print len(a.mols)
# a.Save()
# a.WriteXYZ()

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
