
from __future__ import absolute_import
from __future__ import print_function
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

def GetChemSpider12(a):
	TreatedAtoms = np.array([1,6,7,8], dtype=np.uint8)
	PARAMS["NetNameSuffix"] = "act_sigmoid100"
	PARAMS["learning_rate"] = 0.00001
	PARAMS["momentum"] = 0.95
	PARAMS["max_steps"] = 21
	PARAMS["batch_size"] =  50   # 40 the max min-batch size it can go without memory error for training
	PARAMS["test_freq"] = 1
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["EnergyScalar"] = 1.0
	PARAMS["GradScalar"] = 1.0/20.0
	PARAMS["DipoleScaler"]=1.0
	PARAMS["NeuronType"] = "sigmoid_with_param"
	PARAMS["sigmoid_alpha"] = 100.0
	PARAMS["HiddenLayers"] = [2000, 2000, 2000]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	#PARAMS["Erf_Width"] = 1.0
	#PARAMS["Poly_Width"] = 4.6
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	#PARAMS["AN1_r_Rc"] = 8.0
	#PARAMS["AN1_num_r_Rs"] = 64
	PARAMS["EECutoffOff"] = 15.0
	#PARAMS["DSFAlpha"] = 0.18
	PARAMS["DSFAlpha"] = 0.18
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]
	#PARAMS["KeepProb"] = 0.7
	PARAMS["learning_rate_dipole"] = 0.0001
	PARAMS["learning_rate_energy"] = 0.00001
	PARAMS["SwitchEpoch"] = 2
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("Mol_chemspider12_maxatom35_H2O_with_CH4_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout_act_sigmoid100_rightalpha", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	return manager

def Eval():
	a=MSet("EndiandricC", center_=False)
	a.ReadXYZ()
	# Optimize all three structures.
	manager = GetChemSpider12(a)
	def GetEnergyForceForMol(m):
		def EnAndForce(x_, DoForce=True):
			tmpm = Mol(m.atoms,x_)
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(tmpm, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			if DoForce:
				return energy, force
			else:
				return energy
		return EnAndForce
	if 0:
		# Optimize all three steps of the reaction.
		PARAMS["OptMaxCycles"]=20
		print("Optimizing ", len(a.mols), " mols")
		for i in range(6):
			F = GetEnergyForceForMol(a.mols[i])
			Opt = GeomOptimizer(F)
			a.mols[i] = Opt.Opt(a.mols[i])
			a.mols[i].WriteXYZfile("./results/", "OptMol"+str(i))

	# The set consists of PreF, PreG, G, F, B, C
	# Important transitions are 1<=>2, 2<>3, 1<>4, 3<>6, 4<>5

	# Achieve element alignment.
#	a.mols[0], a.mols[1] = a.mols[0].AlignAtoms(a.mols[1])
#	a.mols[0].WriteXYZfile("./results/", "Aligned"+str(0))

	# Finally do the NEB. between each.
	PARAMS["OptMaxCycles"]=500
	PARAMS["NebSolver"]="Verlet"
	PARAMS["SDStep"] = 0.05
	PARAMS["NebNumBeads"] = 22
	PARAMS["MaxBFGS"] = 12
	a.mols[0], a.mols[1] = a.mols[0].AlignAtoms(a.mols[1])
	a.mols[0].WriteXYZfile("./results/", "Aligned"+str(0))
	a.mols[0].WriteXYZfile("./results/", "Aligned"+str(1))
	F = GetEnergyForceForMol(a.mols[0])
	neb = NudgedElasticBand(F,a.mols[0],a.mols[1])
	Beads = neb.Opt("NebStep1")

def TestBetaHairpin():
	a=MSet("2evq", center_=False)
	a.ReadXYZ()
	a.OnlyAtoms([1, 6, 7, 8])
	# Optimize all three structures.
	manager = GetChemSpider12(a)
	def F(z_, x_, nreal_, DoForce = True):
		"""
		This is the primitive form of force routine required by PeriodicForce.
		"""
		mtmp = Mol(z_,x_)
		if (DoForce):
			en,f = manager.EvalBPDirectEEUpdateSinglePeriodic(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], nreal_,True)
			return en[0], f[0]
		else:
			en = manager.EvalBPDirectEEUpdateSinglePeriodic(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], nreal_, True, DoForce)
			return en[0]
	m = a.mols[0]
	xmn = np.amin(m.coords,axis=0)
	m.coords -= xmn
	xmx = np.amax(m.coords,axis=0)
	print("Xmn,Xmx",xmn,xmx)
	m.properties["lattice"] = np.array([[xmx[0],0.,0.],[0.,xmx[1],0.],[0.,0.,xmx[2]]])
	PF = PeriodicForce(m,m.properties["lattice"])
	PF.BindForce(F, 15.0)

	if 0:
		for i in range(4):
			print("En0:", PF(m.coords)[0])
			m.coords += (np.random.random((1,3))-0.5)*3.0
			m.coords = PF.lattice.ModuloLattice(m.coords)
			print("En:"+str(i), PF(m.coords)[0])

	PARAMS["OptMaxCycles"]=100
	POpt = PeriodicGeomOptimizer(PF)
	PF.mol0=POpt.Opt(m,"ProOpt")

	PARAMS["MDTemp"] = 300.0
	PARAMS["MDdt"] = 0.2 # In fs.
	PARAMS["MDMaxStep"]=2000
	traj = PeriodicVelocityVerlet(PF,"Protein0")
	traj.Prop()

def TestUrey():
	a = MSet("2mzx_open")
	a.ReadXYZ()
	m = a.mols[0]
	m.coords -= np.min(m.coords)
	# Optimize all three structures.
	manager = GetChemSpider12(a)
	def GetEnergyForceForMol(m):
		def EnAndForce(x_, DoForce=True):
			tmpm = Mol(m.atoms,x_)
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(tmpm, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			if DoForce:
				return energy, force
			else:
				return energy
		return EnAndForce
	F = GetEnergyForceForMol(m)
	PARAMS["OptMaxCycles"]=500
	Opt = MetaOptimizer(F,m,Box_=False)
	Opt.Opt(m)

def MetadynamicsStatistics():
	"""
	Gather statistics about the metadynamics exploration process varying bump depth, and width.
	"""
	sugarXYZ="""23
	Comment: ;;;Step 49;;;Energy -3.59693651142
	C   0.469967402303  -0.184327130198  -0.917373448369
	O   -0.860722187686  0.100808621215  -0.54490767945
	C   -1.33117697015  -0.50864531104  0.653969200715
	C   -0.43674566003  -0.0711904472758  1.80766028994
	O   -0.77002931029  -0.682382709637  3.03710172981
	C   1.00094194018  -0.472728470218  1.52537394557
	C   1.47651659856  0.082882226831  0.194808971819
	O   1.67882168598  1.46505030148  0.293016270788
	O   1.85344147164  0.0627404787338  2.51095975747
	C   -2.7831517316  -0.10480234569  0.824111204699
	O   0.624777981739  -1.51188274174  -1.31544630493
	H   -1.26892701887  -1.60055527765  0.558878833361
	H   -0.477952648862  1.02554438685  1.88903361815
	H   1.04603853328  -1.56934691498  1.51216391872
	H   2.41653181894  -0.42448501622  -0.0663941448752
	H   0.686722010962  0.49969621976  -1.74595815766
	H   -0.187352616832  -1.77441341862  -1.74973034436
	H   2.04109700394  1.59384830863  1.1769462844
	H   1.58593984973  -0.300556807915  3.35779231898
	H   -1.59337604893  -0.319220019115  3.36340560897
	H   -2.86433382659  0.961010767071  1.05113694235
	H   -3.32279481219  -0.294255086797  -0.104506847018
	H   -3.26848346524  -0.671549613462  1.61956803093
	"""
	m = Mol()
	m.FromXYZString(sugarXYZ)
	def GetEnergyForceForMol(m):
		s = MSet()
		s.mols.append(m)
		manager = GetChemSpider12(s)
		def EnAndForce(x_, DoForce=True):
			tmpm = Mol(m.atoms,x_)
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(tmpm, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			if DoForce:
				return energy, force
			else:
				return energy
		return EnAndForce
	F = GetEnergyForceForMol(m)

	Opt = GeomOptimizer(F)
	m = Opt.Opt(m)
	PARAMS["MDdt"] = 0.2 # In fs.
	PARAMS["MDMaxStep"] = 4000
	PARAMS["MetaBumpTime"] = 10.0
	PARAMS["MetaMaxBumps"] = 500
	PARAMS["MetaMDBumpHeight"] = 0.000
	PARAMS["MetaMDBumpWidth"] = 0.5
	PARAMS["MetaBowlK"] = 0.0
	PARAMS["MDThermostat"]="Langevin"
	PARAMS["MDTemp"]=300.0
	PARAMS["MDV0"]=None
	traj = MetaDynamics(None, m,"MetaMD_050_01",F)
	traj.Prop()

#Eval()
#TestBetaHairpin()
#TestUrey()
MetadynamicsStatistics()
