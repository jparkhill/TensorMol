"""
Various tests of tensormol's functionality.
Many of these tests take a pretty significant amount of time and memory to complete.
"""
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# John's tests
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
	if (1):
		#a = MSet("uneq_chemspider")
		#a.ReadXYZ("uneq_chemspider")
		#a.Save()
		#a = MSet("uneq_chemspider")
		#a.Load()
		#print "Set elements: ", a.AtomTypes()
		#TreatedAtoms = a.AtomTypes()
		#d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
		#tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data fo
		#tset.BuildTrain("uneq_chemspider")
		tset = TensorMolData_BP(MSet(),MolDigester([]),"uneq_chemspider_ANI1_Sym")
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=2000)
		#manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
                #manager.Continue_Training(maxsteps=2)
	if (0):
		a = MSet("CH3OH_dimer_noHbond")
		a.ReadXYZ("CH3OH_dimer_noHbond")
		manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		optimizer  = Optimizer(manager)
		optimizer.OptANI1(a.mols[0])
	if (0):
		a = MSet("johnsonmols_noH")
		a.ReadXYZ("johnsonmols_noH")
		for mol in a.mols:
			print "mol.coords:", mol.coords
		manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		ins, grad = manager.TData.dig.EvalDigest(a.mols[0])
		#print manager.Eval_BPForce(a.mols[0], True)
		a = MSet("johnsonmols_noH_1")
		a.ReadXYZ("johnsonmols_noH_1")
		#print manager.Eval_BPForce(a.mols[0], True)
		ins1, grad1 = manager.TData.dig.EvalDigest(a.mols[0])
		gradflat =grad.reshape(-1)
		print "grad shape:", grad.shape
		for n in range (0, a.mols[0].NAtoms()):
			diff = -(ins[n] - ins1[n]) /0.001
			for i in range (0,diff.shape[0]):
				if grad[n][i][2] != 0:
					if abs((diff[i] - grad[n][i][2]) / grad[n][i][2]) >  0.01:
						#pass
						print n, i , abs((diff[i] - grad[n][i][2]) / grad[n][i][2]), diff[i],  grad[n][i][2],  grad1[n][i][2], gradflat[n*768*17*3 + i*17*3 +2], n*768*17*3+i*17*3+2, ins[n][i], ins1[n][i]
		for n in range (0, a.mols[0].NAtoms()):
                        diff = -(ins[n] - ins1[n]) /0.001
                        for i in range (0,diff.shape[0]):
                                if grad[n][i][2] != 0:
                                        if abs((grad1[n][i][2] - grad[n][i][2]) / grad[n][i][2]) >  0.01:
						# pass
                                        	print n, i , abs((grad1[n][i][2] - grad[n][i][2]) / grad[n][i][2]), diff[i],  grad[n][i][2],  grad1[n][i][2]
		#t = time.time()
		#print manager.Eval_BPForce(a.mols[0], True)
	if (0):
		a = MSet("md_test")
		a.ReadXYZ("md_test")
		m = a.mols[0]
		tfm= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		# Convert the forces from kcal/mol ang to joules/mol ang.
		ForceField = lambda x: 4183.9953*tfm.Eval_BPForce(Mol(m.atoms,x))
		PARAMS["MNHChain"] = 0
		PARAMS["MDTemp"] = 150.0
		PARAMS["MDThermostat"] = None
		PARAMS["MDV0"]=None
		md = VelocityVerlet(ForceField,m)
		velo_hist = md.Prop()
		autocorr  = AutoCorrelation(velo_hist, md.dt)
		np.savetxt("./results/AutoCorr.dat", autocorr)
	return


def TestJohnson():
	"""
	Try to model the IR spectra of Johnson's peptides...
	Optimize, then get charges, then do an isotropic IR spectrum.
	"""
	a = MSet("johnsonmols")
	a.ReadXYZ("johnsonmols")
	manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)

	PARAMS["NeuronType"]="softplus"
	m = a.mols[1]

	qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	EnergyField = lambda x: manager.Eval_BPEnergySingle(Mol(m.atoms,x))
	ForceField = lambda x: manager.Eval_BPForceSingle(Mol(m.atoms,x),True)
	ChargeField = lambda x: qmanager.Eval_BPDipole(Mol(m.atoms,x),False)[2][0]
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms))
	print "Masses:", masses

	if (0):
		test_mat = np.zeros((6,6))
		test_mat[:3,:3] = np.array([[1.0,0.1,0.001],[0.1,1.2,0.1],[0.001,0.1,1.5]])
		test_mat[3:,3:] = np.array([[1.0,0.1,0.001],[0.1,1.2,0.1],[0.001,0.1,1.5]])
		test_mat[3:,:3] = np.array([[0.3,0.3,0.3],[0.3,0.3,0.3],[0.3,0.3,0.3]])
		test_mat[:3,3:] = np.array([[0.3,0.3,0.3],[0.3,0.3,0.3],[0.3,0.3,0.3]])
		print test_mat
		x0 = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
		TESTEN = lambda x_: 0.5*np.dot(np.dot(x_.reshape(6),test_mat),x_.reshape(6))
		masses = np.array([0.001,0.002])
		HarmonicSpectra(TESTEN,x0,masses,None,0.01)
		return

	if (0):
		PYSCFFIELD = lambda x: PyscfDft(Mol(m.atoms,x))
		QCHEMFIELD = lambda x: QchemDft(Mol(m.atoms,x))
		#CoordinateScan(PYSCFFIELD,m.coords,"Pyscf")
		#print "scan complete..."
		HarmonicSpectra(PYSCFFIELD,m.coords,masses,None,0.005)

	PARAMS["MDdt"] = 0.10
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 8000
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDV0"] = None
	PARAMS["MDTemp"]= 1.0

	PARAMS["OptMomentum"] = 0.0
	PARAMS["OptMomentumDecay"] = 0.9
	PARAMS["OptStepSize"] = 0.0002
	PARAMS["OptMaxCycles"]=200
	optimizer = Optimizer(manager)
	optimizer.OptANI1(m)
	anneal = Annealer(ForceField, ChargeField, m, "Anneal")
	anneal.Prop()
	m.coords = anneal.Minx.copy()

	CoordinateScan(EnergyField,m.coords)
	HartreeForce = lambda x: -1*manager.Eval_BPForceSingle(Mol(m.atoms,x),False)/JOULEPERHARTREE
	HarmonicSpectra(EnergyField, m.coords, masses, HartreeForce)
	return

	PARAMS["MDThermostat"] = None
	PARAMS["MDV0"] = None
	PARAMS["MDTemp"]= 0.0
	PARAMS["MDFieldAmp"] = 500.0 #0.00000001
	PARAMS["MDFieldTau"] = 0.8
	PARAMS["MDFieldFreq"] = 0.1
	PARAMS["MDUpdateCharges"] = False
	PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
	md0 = IRTrajectory(ForceField, ChargeField, m, "0")
	md0.Prop()
	if (0):
		PARAMS["MDFieldVec"] = np.array([0.0,1.0,0.0])
		md1 = IRTrajectory(ForceField, ChargeField, m, "1")
		md1.Prop()
		PARAMS["MDFieldVec"] = np.array([0.0,0.0,1.0])
		md2 = IRTrajectory(ForceField, ChargeField, m, "2")
		md2.Prop()
	#WriteDerDipoleCorrelationFunction(md0.mu_his)
	return

def TestMorphIR():
	"""
	Try to model the IR spectra of Johnson's peptides...
	Optimize, then get charges, then do an isotropic IR spectrum.
	"""
	a = MSet("johnsonmols")
	a.ReadXYZ("johnsonmols")
	manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	PARAMS["OptMomentum"] = 0.0
	PARAMS["OptMomentumDecay"] = 0.9
	PARAMS["OptStepSize"] = 0.02
	PARAMS["OptMaxCycles"]=200
	morphine = a.mols[1]
	heroin = a.mols[2]
	optimizer = Optimizer(manager)
	optimizer.OptANI1(morphine)
	qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	ForceField = lambda x: manager.Eval_BPForceSingle(Mol(morphine.atoms,x),True)
	ChargeField = lambda x: qmanager.Eval_BPDipole(Mol(morphine.atoms,x),False)[2][0]
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

def TestDipole():
	if (0):
		a = MSet("chemspider9")
		a.Load()
		TreatedAtoms = a.AtomTypes()
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Multipole2")
		tset = TensorMolData_BP_Multipole_2(a,d, order_=1, num_indis_=1, type_="mol")
		tset.BuildTrain("chemspider9_multipole2")

	if (1):
		tset = TensorMolData_BP_Multipole_2(MSet(),MolDigester([]),"chemspider9_multipole2_ANI1_Sym")
		manager=TFMolManage("",tset,False,"Dipole_BP_2")
		manager.Train()

	if (0):
		a = MSet("furan_md")
		a.ReadXYZ("furan_md")
		manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		energies = manager.EvalBPEnergy(a)
		#np.savetxt("./results/furan_md_nn_energies.dat",energies)
		b3lyp_energies = []
		for mol in a.mols:
			b3lyp_energies.append(mol.properties["atomization"])
		#np.savetxt("./results/furan_md_b3lyp_energies.dat",np.asarray(b3lyp_energies))
	if (0):
		a = MSet("furan_md")
		a.ReadXYZ("furan_md")
                manager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False)
                net, dipole, charge = manager.Eval_BPDipole(a.mols[0], True)
		#net, dipole, charge = manager.Eval_BPDipole(a.mols, True)
		print net, dipole, charge
		#np.savetxt("./results/furan_md_nn_dipole.dat", dipole)

	if (0):
		a = MSet("furan_md")
                a.ReadXYZ("furan_md")
		manager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False)
                net, dipole, charge = manager.EvalBPDipole(a.mols[0], True)
		charge = charge[0]
		fixed_charge_dipole = np.zeros((len(a.mols),3))
		for i, mol in enumerate(a.mols):
			center_ = np.average(mol.coords,axis=0)
        		fixed_charge_dipole[i] = np.einsum("ax,a", mol.coords-center_ , charge)/AUPERDEBYE
		np.savetxt("./results/furan_md_nn_fixed_charge_dipole.dat", fixed_charge_dipole)
	if (0):
		a = MSet("thf_dimer_flip")
                a.ReadXYZ("thf_dimer_flip")

		#b = MSet("CH3OH")
		#b.ReadXYZ("CH3OH")
		#manager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False)
		#net, dipole, charge = manager.Eval_BPDipole(b.mols[0], True)

		#nn_charge = np.tile(charge[0],2)
		#mul_charge = np.tile(np.loadtxt("./results/CH3OH_mul.dat"), 2)
		#hir_charge = np.tile(np.loadtxt("./results/CH3OH_hir.dat"), 2)
		mul_charge = np.loadtxt("./results/thf_dimer_flip_mul.dat")
		print mul_charge.shape
		#hir_charge = np.loadtxt("./results/CH3OH_dimer_flip_hir.dat")
		mul_dipole = np.zeros((len(a.mols),3))
		#hir_dipole = np.zeros((len(a.mols),3))
		#nn_dipole = np.zeros((len(a.mols),3))
		for i, mol in enumerate(a.mols):
                        center_ = np.average(mol.coords,axis=0)
			print mol.coords.shape
			mul_dipole[i] = np.einsum("ax,a", mol.coords-center_ , mul_charge[i])/AUPERDEBYE
			#hir_dipole[i] = np.einsum("ax,a", mol.coords-center_ , hir_charge)/AUPERDEBYE
			#nn_dipole[i] = np.einsum("ax,a", mol.coords-center_ , nn_charge)/AUPERDEBYE
                        #mul_dipole[i] = np.einsum("ax,a", mol.coords-center_ , mul_charge[i])/AUPERDEBYE
			#hir_dipole[i] = np.einsum("ax,a", mol.coords-center_ , hir_charge[i])/AUPERDEBYE
			#nn_dipole[i] = np.einsum("ax,a", mol.coords-center_ , nn_charge[i])/AUPERDEBYE

                np.savetxt("./results/thf_dimer_flip_mul_dipole.dat", mul_dipole)
		#np.savetxt("./results/CH3OH_dimer_flip_hir_dipole.dat", hir_dipole)
		#np.savetxt("./results/CH3OH_dimer_flip_fixed_nn_dipole.dat", nn_dipole)


	if (0):
		a = MSet("CH3OH_dimer_flip")
                a.ReadXYZ("CH3OH_dimer_flip")
	#	manager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False)
	#	nn_dip = np.zeros((len(a.mols),3))
	#	nn_charge = np.zeros((len(a.mols),a.mols[0].NAtoms()))
	#	for i, mol in enumerate(a.mols):
        #        	net, dipole, charge = manager.Eval_BPDipole(mol, True)
	#		nn_dip[i] = dipole
	#		nn_charge[i] = charge[0]
	#	np.savetxt("./results/thf_dimer_flip_nn_dip.dat", nn_dip)
	#	np.savetxt("./results/thf_dimer_flip_nn_charge.dat", nn_charge)
		f = open("CH3OH_dimer_flip.in","w+")
		for mol in a.mols:
			f.write("$molecule\n0 1\n")
			for i in range (0, mol.NAtoms()):
				atom_name =  atoi.keys()[atoi.values().index(mol.atoms[i])]
                		f.write(atom_name+"   "+str(mol.coords[i][0])+ "  "+str(mol.coords[i][1])+ "  "+str(mol.coords[i][2])+"\n")
			f.write("$end\n\n$rem\njobtype sp\nexchange b3lyp\nbasis 6-31g(d)\nSYM_IGNORE True\n$end\n\n\n@@@\n\n")
		f.close()




def TestGeneralMBEandMolGraph():
	a=FragableMSet("NaClH2O")
	a.ReadXYZ("NaClH2O")
	a.Generate_All_Pairs(pair_list=[{"pair":"NaCl", "mono":["Na","Cl"], "center":[0,0]}])
	a.Generate_All_MBE_term_General([{"atom":"OHH", "charge":0}, {"atom":"NaCl", "charge":0}], cutoff=12, center_atom=[0, -1]) # Generate all the many-body terms with  certain radius cutoff.  # -1 means center of mass
	a.Calculate_All_Frag_Energy_General(method="qchem")  # Use PySCF or Qchem to calcuate the MP2 many-body energy of each order.
	a.Save() # Save the training set, by default it is saved in ./datasets.
	a = MSet("1_1_Ostrech")
	a.ReadXYZ("1_1_Ostrech")
	g = GraphSet(a.name, a.path)
	g.graphs = a.Make_Graphs()
	print "found?", g.graphs[4].Find_Frag(g.graphs[3])

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
	PARAMS["MDTemp"] = 150.0
	PARAMS["MDThermostat"] = "NosePerParticle"
	md = VelocityVerlet(ForceField,m)
	md.Prop()
	return


def TestEE():
	"""
	Test an electrostatically embedded Behler-Parinello
	"""
	a = MSet("H2ONaCl")
	a.Load()
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
	if (0):
		#a = MSet("uneq_chemspider")
		#a.ReadXYZ("uneq_chemspider")
		#a.Save()
		#a = MSet("uneq_chemspider")
		#a.Load()
		#print "Set elements: ", a.AtomTypes()
		#TreatedAtoms = a.AtomTypes()
		#d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
		#tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol") # Initialize TensorMolData that contain the training data fo
		#tset.BuildTrain("uneq_chemspider")
		tset = TensorMolData_BP(MSet(),MolDigester([]),"uneq_chemspider_ANI1_Sym")
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=1500)
		#manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
                #manager.Continue_Training(maxsteps=2)
	if (0):
		a = MSet("gradient_test_0")
                a.ReadXYZ("gradient_test_0")
                manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		optimizer  = Optimizer(manager)
		optimizer.OptANI1(a.mols[0])
	if (0):
                a = MSet("gradient_test_0")
                a.ReadXYZ("gradient_test_0")
                manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
                print manager.Eval_BP(a)

                a = MSet("gradient_test_1")
                a.ReadXYZ("gradient_test_1")
		t = time.time()
                print manager.Eval_BP(a)
		print "time cost to eval:", time.time() -t

		a = MSet("gradient_test_2")
                a.ReadXYZ("gradient_test_2")
                t = time.time()
                print manager.Eval_BP(a)
                print "time cost to eval:", time.time() -t

	if (1):
		a = MSet("md_test")
		a.ReadXYZ("md_test")
		m = a.mols[0]
	        tfm= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False)
		# Convert the forces from kcal/mol ang to joules/mol ang.
		ForceField = lambda x: 4183.9953*tfm.Eval_BPForce(Mol(m.atoms,x))
		PARAMS["MNHChain"] = 0
		PARAMS["MDTemp"] = 150.0
		PARAMS["MDThermostat"] = None
		PARAMS["MDV0"]=None
		md = VelocityVerlet(ForceField,m)
		velo_hist = md.Prop()
		autocorr  = AutoCorrelation(velo_hist, md.dt)
		np.savetxt("./results/AutoCorr.dat", autocorr)
	return


#
# Tests to run.
#

#TestBP(set_="gdb9", dig_="GauSH", BuildTrain_= True)
#TestANI1()
TestDipole()
#TestJohnson()
#TestMorphIR()
#TestGeneralMBEandMolGraph()
#TestGoForceAtom(dig_ = "GauSH", BuildTrain_=True, net_ = "fc_sqdiff", Train_=True)
#TestPotential()
#TestIpecac()
#TestHerrNet1()
#TestOCSDB()
#TestNeb()
#TestMD()
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
