from __future__ import absolute_import
from __future__ import print_function
from TensorMol import *
print(dir())
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]="" # set to use CPU

# Functions that load pretrained network
def GetWaterNetwork(a):
	TreatedAtoms = a.AtomTypes()
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["NeuronType"] = "sigmoid_with_param"
	PARAMS["sigmoid_alpha"] = 100.0
	PARAMS["HiddenLayers"] = [500, 500, 500]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	PARAMS["EECutoffOff"] = 15.0
	PARAMS["DSFAlpha"] = 0.18
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 1.0]
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("water_network",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	return manager

def GetChemSpiderNetwork(a, Solvation_=False):
	TreatedAtoms = np.array([1,6,7,8], dtype=np.uint8)
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["NeuronType"] = "sigmoid_with_param"
	PARAMS["sigmoid_alpha"] = 100.0
	PARAMS["HiddenLayers"] = [2000, 2000, 2000]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	PARAMS["EECutoffOff"] = 15.0
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	if Solvation_:
		PARAMS["DSFAlpha"] = 0.18
		manager=TFMolManage("chemspider12_solvation", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	else:
		PARAMS["DSFAlpha"] = 0.18*BOHRPERA
		manager=TFMolManage("chemspider12_nosolvation", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
	return manager

a=MSet("morphine", center_=False)
a.ReadXYZ("morphine")
manager = GetChemSpiderNetwork(a, False) # load chemspider network

#Use this for testing water network
#a=MSet("water10", center_=False)
#a.ReadXYZ("water10")
#manager = GetWaterNetwork(a)  # load water network

m = a.mols[0]

# Make wrapper functions for energy, force and dipole
def EnAndForce(x_, DoForce=True):
	mtmp = Mol(m.atoms,x_)
	Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
	energy = Etotal[0]
	force = gradient[0]
	if DoForce:
		return energy, force
	else:
		return energy
EnergyForceField =  lambda x: EnAndForce(x)

def ChargeField(x_):
	mtmp = Mol(m.atoms,x_)
	Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
	energy = Etotal[0]
	force = gradient[0]
	return atom_charge[0]

def EnergyField(x_):
	return EnAndForce(x_,True)[0]

def DipoleField(x_):
	q = np.asarray(ChargeField(x_))
	dipole = np.zeros(3)
	for i in  range(0, q.shape[0]):
		dipole += q[i]*x_[i]*BOHRPERA
	return dipole

# Perform geometry optimization
if (0):
	PARAMS["OptMaxCycles"]= 2000
	PARAMS["OptThresh"] =0.00002
	Opt = GeomOptimizer(EnAndForce)
	m=Opt.Opt(a.mols[0],"morphine_tm_opt")
	m.WriteXYZfile("./results/", "optimized_morphine")

# Run molecular dynamic
if (0):
	PARAMS["MDThermostat"] = "Nose" # use None for
	PARAMS["MDTemp"] = 300
	PARAMS["MDdt"] = 0.1 # fs
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDV0"] = "Random"
	PARAMS["MDMaxStep"] = 100000
	md = VelocityVerlet(None, m, "morphine_md_300K", EnAndForce)
	md.Prop()

#Perform Harmonic frequency analysis
if (0):
	masses = np.array(list(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms)))
	w,v = HarmonicSpectra(EnergyField, m.coords, m.atoms, WriteNM_=True, Mu_ = DipoleField)

# Generate Realtime IR spectrum
if (0):
	# run an annealer to warm system to the target T
	PARAMS["MDdt"] = 0.1 #fs
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDThermostat"] = "Nose"
	PARAMS["MDV0"] = "Random"
	PARAMS["MDAnnealTF"] = 300.0
	PARAMS["MDAnnealT0"] = 300.0
	PARAMS["MDAnnealKickBack"] = 5.0
	PARAMS["MDAnnealSteps"] = 5000
	anneal = Annealer(EnAndForce, None, m, "morphine_aneal")
	anneal.Prop()
	m.coords = anneal.x.copy()

	# run an energy conserved MD trajectory for realtime IR
	PARAMS["MDThermostat"] = None
	PARAMS["MDTemp"] = 0
	PARAMS["MDdt"] = 0.1
	PARAMS["MDV0"] = None
	PARAMS["MDMaxStep"] = 100000
	md = IRTrajectory(EnAndForce, ChargeField, m, "morphine_IR_300K", anneal.v)
	md.Prop()
	WriteDerDipoleCorrelationFunction(md.mu_his) # CorrelationFunction is stored in "./results/MutMu0.txt" by default
