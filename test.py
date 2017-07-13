"""
Various tests of tensormol's functionality.
Many of these tests take a pretty significant amount of time and memory to complete.
"""
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def TestBPDirect():
	"""
	Test Behler-Parrinello with gradient learning and direct descriptor.
	"""
	a=MSet("H2O_force_test")
        a.ReadXYZ("H2O_force_test")
	#a = MSet("H2O_augmented_more_cutoff5_b3lyp_force")
	#a.Load()
	TreatedAtoms = a.AtomTypes()
	PARAMS["hidden1"] = 100
	PARAMS["hidden2"] = 100
	PARAMS["hidden3"] = 100
	PARAMS["learning_rate"] = 0.00001
	PARAMS["momentum"] = 0.95
	PARAMS["max_steps"] = 10
	PARAMS["batch_size"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["tf_prec"] = "tf.float64"
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	if (0):
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_Grad") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=10)
	# Test out some MD with the trained network.

	manager=TFMolManage("Mol_H2O_augmented_more_cutoff5_b3lyp_force_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_Grad_1",tset,False,"fc_sqdiff_BP_Direct_Grad",False,False) # Initialzie a manager than manage the training of neural network.
	dipole_manager= TFMolManage("Mol_H2O_agumented_more_cutoff5_multipole2_ANI1_Sym_Dipole_BP_2_1", None, False, Trainable_ = False)

	m = a.mols[0]
	#print manager.Eval_BPEnergy_Direct_Grad(m)
	#print manager.EvalBPDirectSingleEnergyWGrad(m)
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms))
	EnergyForceField = lambda x: manager.Eval_BPEnergy_Direct_Grad(Mol(m.atoms,x))
	EnergyField = lambda x: manager.Eval_BPEnergy_Direct_Grad(Mol(m.atoms,x), Grad=False)
	def ChargeField(x_):
		m.coords = x_
		dipole, charge = dipole_manager.Eval_BPDipole_2(m)
		return np.asarray(charge[0])


	#PARAMS["MDdt"] = 0.2
	#PARAMS["RemoveInvariant"]=True
	#PARAMS["MDMaxStep"] = 10000
	#PARAMS["MDThermostat"] = "Nose"
	#PARAMS["MDTemp"]= 600.0
	##traj = VelocityVerlet(None,m,"DirectMD", EnergyForceField)
	##traj.Prop()

	#PARAMS["MDdt"] = 0.2
        #PARAMS["RemoveInvariant"]=True
        #PARAMS["MDMaxStep"] = 10000
        #PARAMS["MDThermostat"] = "Nose"
        #PARAMS["MDV0"] = None
        #PARAMS["MDTemp"]= 1.0
	#PARAMS["MDAnnealSteps"] = 2000
	#anneal = Annealer(EnergyForceField, None, m, "Anneal")
        #anneal.Prop()
	#m.coords = anneal.Minx.copy()
	#m = GeomOptimizer(EnergyForceField).Opt(m)
	#m.WriteXYZfile("./results/", "H2O_trimer_opt")

	PARAMS["MDFieldAmp"] = 0.0 #0.00000001
	PARAMS["MDFieldTau"] = 0.4
	PARAMS["MDFieldFreq"] = 0.8
	PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDTemp"] = 30
	PARAMS["MDdt"] = 0.1
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDV0"] = None
	PARAMS["MDMaxStep"] = 10000
	warm = VelocityVerlet(None, m,"warm",EnergyForceField)
	warm.Prop()
	m.coords = warm.x.copy()
	PARAMS["MDMaxStep"] = 40000
	md = IRTrajectory(EnergyForceField, ChargeField, m,"H2O_udp_grad_IR",warm.v.copy())
	md.Prop()
	WriteDerDipoleCorrelationFunction(md.mu_his,"H2O_udp_grad_IR.txt")
	return

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
		#tset.BuildTrain("uneq_chemspider_float64")

		PARAMS["hidden1"] = 200
		PARAMS["hidden2"] = 200
		PARAMS["hidden3"] = 200
		PARAMS["learning_rate"] = 0.00001
		PARAMS["momentum"] = 0.95
		PARAMS["max_steps"] = 1001
		PARAMS["batch_size"] = 10000
		PARAMS["test_freq"] = 10
		PARAMS["tf_prec"] = "tf.float64"
		tset = TensorMolData_BP(MSet(),MolDigester([]),"uneq_chemspider_float64_ANI1_Sym")
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP") # Initialzie a manager than manage the training of neural network.
		manager.Train(maxstep=1500)
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

def TestBP_WithGrad():
	"""
	copy glymd.pdb from the google drive...
	"""
	if (0):
		# Train the atomization energy in a normal BP network to test.
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
	if (1):
		a = MSet("glymd")
		a.Load()
		a.pop(45000) # help out my puny laptop
		for mol in a.mols:
			mol.properties['force'] *= BOHRPERA
			#mol.properties['force'] /= (BOHRPERA*BOHRPERA)
			mol.CalculateAtomization()
		PARAMS["AN1_r_Rc"] = 4.6
		PARAMS["AN1_a_Rc"] = 3.1
		PARAMS["AN1_eta"] = 4.0
		PARAMS["AN1_zeta"] = 8.0
		PARAMS["AN1_num_r_Rs"] = 8
		PARAMS["AN1_num_a_Rs"] = 4
		PARAMS["AN1_num_a_As"] = 4
		PARAMS["batch_size"] = 1500
		PARAMS["hidden1"] = 64
		PARAMS["hidden2"] = 128
		PARAMS["hidden3"] = 64
		PARAMS["max_steps"] = 1001
		PARAMS["GradWeight"] = 1.0
		PARAMS["AN1_r_Rs"] = np.array([ PARAMS["AN1_r_Rc"]*i/PARAMS["AN1_num_r_Rs"] for i in range (0, PARAMS["AN1_num_r_Rs"])])
		PARAMS["AN1_a_Rs"] = np.array([ PARAMS["AN1_a_Rc"]*i/PARAMS["AN1_num_a_Rs"] for i in range (0, PARAMS["AN1_num_a_Rs"])])
		PARAMS["AN1_a_As"] = np.array([ 2.0*Pi*i/PARAMS["AN1_num_a_As"] for i in range (0, PARAMS["AN1_num_a_As"])])
		TreatedAtoms = a.AtomTypes()
		if (0):
			# Train the atomization energy in a normal BP network to test.
			d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AtomizationEnergy")  # Initialize a digester that apply descriptor for the fragme
			print "Set elements: ", a.AtomTypes()
			tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol")
			tset.BuildTrain("glymd", append=False, max_nmols_=1000000)
			manager=TFMolManage("",tset,False,"fc_sqdiff_BP")
			manager.Train(maxstep=200)
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="AEAndForce")  # Initialize a digester that apply descriptor for the fragme
		print "Set elements: ", a.AtomTypes()
		tset = TensorMolData_BP(a,d, order_=1, num_indis_=1, type_="mol", WithGrad_=True)
		tset.BuildTrain("glymd", append=False, max_nmols_=1000000, WithGrad_=True)
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_WithGrad")
		manager.Train(maxstep=2000)

	if (0):
		PARAMS["AN1_r_Rc"] = 4.6
		PARAMS["AN1_a_Rc"] = 3.1
		PARAMS["AN1_eta"] = 4.0
		PARAMS["AN1_zeta"] = 8.0
		PARAMS["AN1_num_r_Rs"] = 16
		PARAMS["AN1_num_a_Rs"] = 4
		PARAMS["AN1_num_a_As"] = 4
		PARAMS["batch_size"] = 1500
		PARAMS["hidden1"] = 64
		PARAMS["hidden2"] = 128
		PARAMS["hidden3"] = 64
		PARAMS["max_steps"] = 1001
		PARAMS["GradWeight"] = 1.0
		PARAMS["AN1_r_Rs"] = np.array([ PARAMS["AN1_r_Rc"]*i/PARAMS["AN1_num_r_Rs"] for i in range (0, PARAMS["AN1_num_r_Rs"])])
		PARAMS["AN1_a_Rs"] = np.array([ PARAMS["AN1_a_Rc"]*i/PARAMS["AN1_num_a_Rs"] for i in range (0, PARAMS["AN1_num_a_Rs"])])
		PARAMS["AN1_a_As"] = np.array([ 2.0*Pi*i/PARAMS["AN1_num_a_As"] for i in range (0, PARAMS["AN1_num_a_As"])])

		tset = TensorMolData_BP(MSet(),MolDigester([]),"glymd_ANI1_Sym")
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_WithGrad")
		manager.Train(maxstep=2000)
	return

def TestMetadynamics():
	a = MSet("johnsonmols")
	a.ReadXYZ("johnsonmols")
	manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	PARAMS["NeuronType"]="softplus"
	m = a.mols[3]
	qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	EnergyField = lambda x: manager.Eval_BPEnergySingle(Mol(m.atoms,x))
	ForceField = lambda x: manager.Eval_BPForceSingle(Mol(m.atoms,x),False)
	ChargeField = lambda x: qmanager.Eval_BPDipole(Mol(m.atoms,x),False)[2][0]
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms))
	print "Masses:", masses
	PARAMS["MDdt"] = 0.2
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 8000
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDTemp"]= 300.0
	meta = MetaDynamics(ForceField, m)
	meta.Prop()

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

def IRProtocol(mol_, ForceField_, ChargeField_, name_= "IR"):
	"""
	This is pretty much a best-practice way to get IR spectra.
	Optimize, then anneal, then warm to 30k, then propagate IR.

	Args:
		mol_: A molecule
		ForceField_: An function returning Energy (Eh), and Force (j/Ang)
		ChargeField_: A function returning charges.
	"""
	PARAMS["OptMomentum"] = 0.0
	PARAMS["OptMomentumDecay"] = 0.9
	PARAMS["OptStepSize"] = 0.02
	PARAMS["OptMaxCycles"]=100
	opt = GeomOptimizer(ForceField_)
	optmol = opt.Opt(mol_)
	PARAMS["MDdt"] = 0.2
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 400
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDV0"] = None
	PARAMS["MDTemp"]= 1.0
	anneal = Annealer(ForceField_, None, optmol,name_+"_Anneal")
	anneal.Prop()
	optmol.coords = anneal.Minx.copy()
	PARAMS["MDTemp"]= 60.0
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDMaxStep"] = 1000
	warm = VelocityVerlet(None,optmol,name_+"_Warm",ForceField_)
	warm.Prop()
	optmol.coords = warm.x.copy()
	#Finally get the IR.
	PARAMS["MDMaxStep"] = 40000
	PARAMS["MDdt"] = 0.1
	PARAMS["MDUpdateCharges"] = True
	ir = IRTrajectory(ForceField_, ChargeField_, optmol,name_+"_IR", warm.v.copy())
	ir.Prop()
	WriteDerDipoleCorrelationFunction(ir.mu_his,name_+"MutMu0.txt")
	return

def TestIndoIR():
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
        indo = a.mols[0]
	print "number of atoms in indo", indo.NAtoms()
        #optimizer = Optimizer(manager)
        #optimizer.OptANI1(indo)
        qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
        ForceField = lambda x: manager.Eval_BPForceSingle(Mol(indo.atoms,x),True)
        ChargeField = lambda x: qmanager.Eval_BPDipole(Mol(indo.atoms,x),False)[2][0]
        PARAMS["MDdt"] = 0.2
        PARAMS["RemoveInvariant"]=True
        PARAMS["MDMaxStep"] = 10000
        PARAMS["MDThermostat"] = "Nose"
        PARAMS["MDV0"] = None
        PARAMS["MDTemp"]= 1.0
        annealIndo = Annealer(ForceField, ChargeField, indo, "Anneal")
        annealIndo.Prop()
        indo.coords = annealIndo.Minx.copy()
	indo.WriteXYZfile("./results/", "indo_opt")

        PARAMS["MDFieldAmp"] = 0.0 #0.00000001
        PARAMS["MDFieldTau"] = 0.4
        PARAMS["MDFieldFreq"] = 0.8
        PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
        PARAMS["MDThermostat"] = "Nose"
        PARAMS["MDTemp"] = 30
        PARAMS["MDdt"] = 0.1
        PARAMS["RemoveInvariant"]=True
        PARAMS["MDV0"] = None

        PARAMS["MDMaxStep"] = 10000
        warm = VelocityVerlet(ForceField, indo, "warm", ForceField)
        warm.Prop()
        indo.coords = warm.x.copy()

        PARAMS["MDMaxStep"] = 40000
        md = IRTrajectory(ForceField, ChargeField, indo,"indo_IR_30K",warm.v.copy(),)
        md.Prop()
        WriteDerDipoleCorrelationFunction(md.mu_his,"indo_IR_30K.txt")


        #PARAMS["MDTemp"]= 0.0
        #PARAMS["MDThermostat"] = None
        #PARAMS["MDFieldAmp"] = 20.0 #0.00000001
        #PARAMS["MDFieldTau"] = 0.4
        #PARAMS["MDFieldFreq"] = 0.8
        #PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
        #md0 = IRTrajectory(ForceField, ChargeField, indo, "indo")
        #md0.Prop()
        #WriteDerDipoleCorrelationFunction(md0.mu_his,"indo.txt")
        return
def david_testIR():
	"""
	Try to model the IR spectra of Johnson's peptides...
	Optimize, then get charges, then do an isotropic IR spectrum.
	"""
	a = MSet("david_test")
	a.ReadXYZ("david_test")
	manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	PARAMS["OptMomentum"] = 0.0
	PARAMS["OptMomentumDecay"] = 0.9
	PARAMS["OptStepSize"] = 0.02
	PARAMS["OptMaxCycles"]=200
	indo = a.mols[11]
	print "number of atoms in indo", indo.NAtoms()
	#optimizer = Optimizer(manager)
	#optimizer.OptANI1(indo)
	qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	EnergyField = lambda x: manager.Eval_BPForceSingle(Mol(indo.atoms,x),True)[0]
	ForceField = lambda x: manager.Eval_BPForceSingle(Mol(indo.atoms,x),True)
	ChargeField = lambda x: qmanager.Eval_BPDipole(Mol(indo.atoms,x),False)[2][0]
	PARAMS["MDdt"] = 0.2
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 100
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDV0"] = None
	PARAMS["MDTemp"]= 1.0
	annealIndo = Annealer(ForceField, ChargeField, indo, "Anneal")
	annealIndo.Prop()
	indo.coords = annealIndo.Minx.copy()
	indo.WriteXYZfile("./results/", "davidIR_opt")
	# Perform a Harmonic analysis
	m=indo
	print "Harmonic Analysis"
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms))
	w,v = HarmonicSpectra(EnergyField, m.coords, masses)
	v = v.real
	print np.sign(w)*np.sqrt(KCONVERT*abs(w))*CMCONVERT
	for i in range(3*m.NAtoms()):
		print np.sign(w[i])*np.sqrt(KCONVERT*abs(w[i]))*CMCONVERT
		nm = v[:,i].reshape((m.NAtoms(),3))
		nm *= np.sqrt(np.array([map(lambda x: ATOMICMASSESAMU[x-1],m.atoms)])).T
		print nm
		for alpha in np.append(np.linspace(-.1,.1,30),np.linspace(.1,-.1,30)):
			mdisp = Mol(m.atoms,m.coords+alpha*nm)
			mdisp.WriteXYZfile("./results/","NormalMode_"+str(i))

	# PARAMS["MDFieldAmp"] = 0.0 #0.00000001
	# PARAMS["MDFieldTau"] = 0.4
	# PARAMS["MDFieldFreq"] = 0.8
	# PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
	# PARAMS["MDThermostat"] = "Nose"
	# PARAMS["MDTemp"] = 30
	# PARAMS["MDdt"] = 0.1
	# PARAMS["RemoveInvariant"]=True
	# PARAMS["MDV0"] = None

	# PARAMS["MDMaxStep"] = 10000
	# warm = VelocityVerlet(ForceField, indo, "warm", ForceField)
	# warm.Prop()
	# indo.coords = warm.x.copy()

	# PARAMS["MDMaxStep"] = 40000
	# md = IRTrajectory(ForceField, ChargeField, indo,"david_IR_30K",warm.v.copy(),)
	# md.Prop()
	# WriteDerDipoleCorrelationFunction(md.mu_his,"david_IR_30K.txt")


	#PARAMS["MDTemp"]= 0.0
	#PARAMS["MDThermostat"] = None
	#PARAMS["MDFieldAmp"] = 20.0 #0.00000001
	#PARAMS["MDFieldTau"] = 0.4
	#PARAMS["MDFieldFreq"] = 0.8
	#PARAMS["MDFieldVec"] = np.array([1.0,0.0,0.0])
	#md0 = IRTrajectory(ForceField, ChargeField, indo, "indo")
	#md0.Prop()
	#WriteDerDipoleCorrelationFunction(md0.mu_his,"indo.txt")
	return

def david_HarmonicAnalysis():
	# print "TestIR"
	a = MSet("david_test")
	a.ReadXYZ("david_test")
	manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	# PARAMS["OptMomentum"] = 0.0
	# PARAMS["OptMomentumDecay"] = 0.9
	# PARAMS["OptStepSize"] = 0.02
	# PARAMS["OptMaxCycles"]=200
	indo = a.mols[8]
	# print "number of atoms in indo", indo.NAtoms()
	# #optimizer = Optimizer(manager)
	# #optimizer.OptANI1(indo)
	qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	EnergyField = lambda x: manager.Eval_BPForceSingle(Mol(indo.atoms,x),True)[0]
	# ForceField = lambda x: manager.Eval_BPForceSingle(Mol(indo.atoms,x),True)
	# ChargeField = lambda x: qmanager.Eval_BPDipole(Mol(indo.atoms,x),False)[2][0]
	# PARAMS["MDdt"] = 0.2
	# PARAMS["RemoveInvariant"]=True
	# PARAMS["MDMaxStep"] = 100
	# PARAMS["MDThermostat"] = "Nose"
	# PARAMS["MDV0"] = None
	# PARAMS["MDTemp"]= 1.0
	# annealIndo = Annealer(ForceField, ChargeField, indo, "Anneal")
	# annealIndo.Prop()
	# indo.coords = annealIndo.Minx.copy()
	# indo.WriteXYZfile("./results/", "davidIR_opt")
	# # Perform a Harmonic analysis
	m=indo
	print "Harmonic Analysis"
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms))
	w,v = HarmonicSpectra(EnergyField, m.coords, masses)
	v = v.real
	wave = np.sign(w)*np.sqrt(KCONVERT*abs(w))*CMCONVERT
	for i in range(3*m.NAtoms()):
		np.sign(w[i])*np.sqrt(KCONVERT*abs(w[i]))*CMCONVERT
		nm = v[:,i].reshape((m.NAtoms(),3))
		nm *= np.sqrt(np.array([map(lambda x: ATOMICMASSESAMU[x-1],m.atoms)])).T
		for alpha in np.append(np.linspace(-.1,.1,30),np.linspace(.1,-.1,30)):
			mdisp = Mol(m.atoms,m.coords+alpha*nm)
			mdisp.WriteXYZfile("./results/","NormalMode_"+str(i))
	return nm

def david_AnnealHarmonic(set_ = "david_test", Anneal = True, WriteNM_ = False):
	"""
	Optionally anneals a molecule and then runs it through a finite difference normal mode analysis

	Args:
		set_: dataset from which a molecule comes
		Anneal: whether or not to perform an annealing routine before the analysis
		WriteNM_: whether or not to write normal modes to a readable file
	Returns:
		Frequencies (wavenumbers)
		Normal modes (cartesian)
	"""

	a = MSet(set_)
	a.ReadXYZ(set_)
	manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	x_ = a.mols[6] #Choose index of molecule in a given dataset
	qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	EnergyField = lambda x: manager.Eval_BPForceSingle(Mol(x_.atoms,x),True)[0]
	ForceField = lambda x: manager.Eval_BPForceSingle(Mol(x_.atoms,x),True)
	DipoleField = lambda x: qmanager.Eval_BPDipole(Mol(x_.atoms,x),False)[1]
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],x_.atoms))
	m_ = masses
	eps_ = 0.04 #finite difference step
	if Anneal == True:
		PARAMS["OptMomentum"] = 0.0
		PARAMS["OptMomentumDecay"] = 0.9
		PARAMS["OptStepSize"] = 0.02
		PARAMS["OptMaxCycles"]=200
		PARAMS["MDdt"] = 0.2
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDMaxStep"] = 100
		PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDV0"] = None
		PARAMS["MDTemp"]= 1.0
		annealx_ = Annealer(ForceField, None, x_, "Anneal")
		annealx_.Prop()
		x_.coords = annealx_.Minx.copy()
		x_.WriteXYZfile("./results/", "davidIR_opt")
	HarmonicSpectra(EnergyField, x_.coords, masses, x_.atoms, WriteNM_ = True, Mu_ = DipoleField)
def TestIR():
	"""
	Runs a ton of Infrared Spectra.
	"""
	a = MSet("johnsonmols")
	a.ReadXYZ("johnsonmols")
	manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	qmanager= TFMolManage("Mol_chemspider9_multipole_ANI1_Sym_Dipole_BP_1" , None, False, RandomTData_=False, Trainable_=False)
	for ind,mol in enumerate(a.mols):
		ForceField = lambda x: manager.Eval_BPForceSingle(Mol(mol.atoms,x),True)
		ChargeField = lambda x: qmanager.Eval_BPDipole(Mol(mol.atoms,x),False)[2][0]
		IRProtocol(mol,ForceField,ChargeField,str(ind))
	return

def TestDipole():
	if (0):
		a = MSet("chemspider9")
		a.Load()
		TreatedAtoms = a.AtomTypes()
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Multipole")
		tset = TensorMolData_BP_Multipole(a,d, order_=1, num_indis_=1, type_="mol")
		tset.BuildTrain("chemspider9_multipole")

	if (1):
		a = MSet("chemspider9")
		a.Load()
		TreatedAtoms = a.AtomTypes()
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym", OType_="Multipole2")
		tset = TensorMolData_BP_Multipole_2(a,d, order_=1, num_indis_=1, type_="mol")
		tset.BuildTrain("chemspider9_multipole3_float64")

	if (0):
		tset = TensorMolData_BP_Multipole_2(MSet(),MolDigester([]),"chemspider9_multipole3_ANI1_Sym")
		manager=TFMolManage("",tset,False,"Dipole_BP")
		manager.Train()

	if (1):
		PARAMS["hidden1"] = 100
                PARAMS["hidden2"] = 100
                PARAMS["hidden3"] = 100
                PARAMS["learning_rate"] = 0.0001
                PARAMS["momentum"] = 0.95
                PARAMS["max_steps"] = 501
                PARAMS["batch_size"] = 10000
                PARAMS["test_freq"] = 10
                PARAMS["tf_prec"] = "tf.float64"
		tset = TensorMolData_BP_Multipole_2(MSet(),MolDigester([]),"chemspider9_multipole3_float64_ANI1_Sym")
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
	ins.TrainPrepare()
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
	md = VelocityVerlet(ForceField,m,"LJ test", EnergyForceField)
	md.Prop()
	return

def Test_Periodic_LJMD():
	"""
	Test TensorFlow LJ fluid Molecular dynamics with periodic BC
	This version also tests linear-scaling-ness of the neighbor list
	Etc.
	"""
	a=MSet("Test")
	ParticlesPerEdge = 20
	EdgeSize = 18
	a.mols=[Mol(np.ones(ParticlesPerEdge*ParticlesPerEdge*ParticlesPerEdge,dtype=np.uint8),MakeUniform([0.0,0.0,0.0],EdgeSize,ParticlesPerEdge))]
	#a.mols=[Mol(np.ones(512),MakeUniform([0.0,0.0,0.0],4.0,8))]
	m = a.mols[0]
	TreatedAtoms = a.AtomTypes()
	d = MolDigester(TreatedAtoms, name_="CZ", OType_ ="Force")
	tset = TensorMolData(a,d)
	ins = MolInstance_DirectForce(tset,None,False,"LJ")
	ins.TrainPrepare()
	ForceField = lambda x: ins.EvalForceLinear(Mol(m.atoms,x))[0][0]
	EnergyForceField = lambda x: ins.EvalForceLinear(Mol(m.atoms,x))
	PARAMS["MDTemp"] = 300.0
	PARAMS["MDThermostat"] = None
	PARAMS["MDV0"] = None
	PARAMS["MDdt"] = 0.2
	md = VelocityVerlet(ForceField,m,"LJLinearTest", EnergyForceField)
	md.Prop()

	# Generate a Periodic Force field.
	PF = PeriodicForce(m, [[10.0,0.0,0.0],[0.0,10.0,0.0],[0.,0.,10.0]])
	PF.AddLocal(ins.LocalLJForce)
	PARAMS["MDTemp"] = 300.0
	PARAMS["MDThermostat"] = None
	PARAMS["MDV0"] = None
	PARAMS["MDdt"] = 0.2
	md = PeriodicVelocityVerlet(PF,m,"Periodic test")
	md.Prop()
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
	a=MSet("david_test")
	a.ReadXYZ("david_test")
	m = a.mols[3]
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

def TestRandom():
	a = MSet("sampling_mols")
	a.ReadXYZ("sampling_mols")
	mol = a.mols[4]
	f = open('/media/sdb1/dtoth/sampling_mols/rnd_mols/hexanol/hexanol_rnd' + '.in', 'w')
	for i in range(10000):
		tmp_mol = copy.deepcopy(mol)
		tmp_mol.Distort(0.05, .80)
		f.write("$molecule \n0 1 \n")
		for i in range(len(tmp_mol.atoms)):
			if tmp_mol.atoms[i] == 1:
				f.write("H  " + str(tmp_mol.coords[i, 0]) + "  " + str(tmp_mol.coords[i, 1]) + "  " + str(tmp_mol.coords[i,2]) + "\n")
			if tmp_mol.atoms[i] == 6:
				f.write("C  " + str(tmp_mol.coords[i, 0]) + "  " + str(tmp_mol.coords[i,1]) + "  " + str(tmp_mol.coords[i,2]) + "\n")
			if tmp_mol.atoms[i] == 8:
				f.write("O  " + str(tmp_mol.coords[i, 0]) + "  " + str(tmp_mol.coords[i,1]) + "  " + str(tmp_mol.coords[i,2]) + "\n")
		f.write("$end \n \n$rem \njobtype  force \nmethod   wB97X-D \nbasis  6-311G** \n$end \n@@@ \n \n")


# Tests to run.
#

#TestBP(set_="gdb9", dig_="GauSH", BuildTrain_= True)
#TestANI1()
#TestBP_WithGrad()
#Test_ULJ()
#Test_LJMD()
#TestDipole()
#TestJohnson()
#TestIR()
# TestIndoIR()
# david_testIR()
#david_HarmonicAnalysis()
# TestMetadynamics()
#TestMetadynamics()
# Test_Periodic_LJMD()
#TestGeneralMBEandMolGraph()
#TestGoForceAtom(dig_ = "GauSH", BuildTrain_=True, net_ = "fc_sqdiff", Train_=True)
#TestPotential()
#TestIpecac()
#TestHerrNet1()
#TestOCSDB()
#TestNeb()
TestMD()
# TestRandom()
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
