
from __future__ import absolute_import
from __future__ import print_function
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from TensorMol.ElectrostaticsTF import *
from TensorMol.NN_MBE import *

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
	PARAMS["DSFAlpha"] = 0.18*BOHRPERA
	PARAMS["AddEcc"] = True
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]
	#PARAMS["KeepProb"] = 0.7
	PARAMS["learning_rate_dipole"] = 0.0001
	PARAMS["learning_rate_energy"] = 0.00001
	PARAMS["SwitchEpoch"] = 2
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("Mol_chemspider12_clean_maxatom35_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout_act_sigmoid100", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
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
		PARAMS["OptMaxCycles"]=10
		print("Optimizing ", len(a.mols), " mols")
		for i in range(3):
			F = GetEnergyForceForMol(a.mols[i])
			Opt = GeomOptimizer(F)
			a.mols[i] = Opt.Opt(a.mols[i])
			a.mols[i].WriteXYZfile("./results/", "OptMol"+str(i))

	# Achieve element alignment.
	a.mols[0], a.mols[1] = a.mols[0].AlignAtoms(a.mols[1])
	a.mols[0].WriteXYZfile("./results/", "Aligned"+str(0))

	# Finally do the NEB. between each.
	PARAMS["OptMaxCycles"]=200
	PARAMS["NebSolver"]="SD"
	PARAMS["MaxBFGS"] = 12
	F = GetEnergyForceForMol(a.mols[0])
	neb = NudgedElasticBand(F,a.mols[0],a.mols[1])
	Beads = neb.Opt("NebStep1")

	a.mols[1], a.mols[2] = a.mols[1].AlignAtoms(a.mols[2])
	a.mols[1].WriteXYZfile("./results/", "Aligned"+str(1))
	a.mols[2].WriteXYZfile("./results/", "Aligned"+str(2))
	neb2 = NudgedElasticBand(F,a.mols[1],a.mols[2])
	Beads2 = neb2.Opt("NebStep2")

Eval()
