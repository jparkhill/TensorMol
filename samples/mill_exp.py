from __future__ import absolute_import
#import memory_util
#memory_util.vlog(1)
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from TensorMol.ForceModels.ElectrostaticsTF import *
from TensorMol.MBE.NN_MBE import *
from TensorMol.Interfaces.TMIPIinterface import *
import random

def Prepare():
	if (1):
		a=MSet("watercube", center_=False)
		a.ReadXYZ("watercube")
		repeat = 5
		space  = 3.0
		mol_cout = np.zeros(repeat**3,dtype=int)
		tm_coords = []
		tm_atoms = []
		portion = np.array([1.0,1.0,1.0,1,0, 9.0])/np.sum(np.array([1,1,1,1,9]))
		print ("portion:", portion)
		for i in range(0, repeat):
			for j in range(0, repeat):
				for k in range(0, repeat):
					s = random.random()
					print ("s:",s)
					if s < np.sum(portion[:1]):
						index = 0
					elif s < np.sum(portion[:2]):
						index = 1
					elif s < np.sum(portion[:3]):
						index = 2
					elif s < np.sum(portion[:4]):
						index = 3
					else:
						index = 4
					print ("index:",index)
					m = a.mols[index]
					tm_coords += list(m.coords+np.asarray([i*space,j*space,k*space]))
					tm_atoms += list(m.atoms)
		tm_coords  = np.asarray(tm_coords )
		tm_atoms = np.asarray(tm_atoms, dtype=int)
		tm = Mol(tm_atoms, tm_coords)
		tm.WriteXYZfile(fpath="./datasets", fname="reactor")
	if (0):
		a=MSet("watercube", center_=False)
		a.ReadXYZ("watercube")
		m = a.mols[0]
		m.coords = m.coords+1
		repeat = 4
		space = 4.0
		tm_coords = np.zeros((repeat**3*m.NAtoms(),3))
		tm_atoms = np.zeros(repeat**3*m.NAtoms(), dtype=int)
		p = 0
		for i in range(0, repeat):
			for j in range(0, repeat):
				for k in range(0, repeat):
					tm_coords[p*m.NAtoms():(p+1)*m.NAtoms()]=m.coords+np.asarray([i*space,j*space,k*space])
					tm_atoms[p*m.NAtoms():(p+1)*m.NAtoms()]=m.atoms
					p += 1
		tm = Mol(tm_atoms, tm_coords)
		tm.WriteXYZfile(fpath="./datasets", fname="watercube")


def Eval():
	if (1):
		a=MSet("reactor", center_=True)
		a.ReadXYZ("reactor")
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

		m = a.mols[0]
		print ("m.coords:", np.max(m.coords,axis=0), np.min(m.coords,axis=0))
		m.coords = m.coords - np.min(m.coords,axis=0)
		print ("m.coords:", np.max(m.coords,axis=0), np.min(m.coords,axis=0))
		#m.coords += np.asarray([1,1,1])
		#print manager.EvalBPDirectEEUpdateSinglePeriodic(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())
		#print manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
		#return
		#charge = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[6]
		#bp_atom = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[2]
		#for i in range (0, m.NAtoms()):
		#	print i+1, charge[0][i],bp_atom[0][i]

		def EnAndForce(x_, DoForce=True):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			if DoForce:
				return energy, force
			else:
				return energy

		def GetEnergyForceForMol(m):
			def EnAndForce(x_, DoForce=True):
				tmpm = Mol(m.atoms,x_)
				Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
				energy = Etotal[0]
				force = gradient[0]
				if DoForce:
					return energy, force
				else:
					return energy
			return EnAndForce

		def EnForceCharge(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return energy, force, atom_charge[0]

		def ChargeField(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return atom_charge[0]

		def EnergyField(x_):
			return EnAndForce(x_,True)[0]

		def DipoleField(x_):
			q = np.asarray(ChargeField(x_))
			dipole = np.zeros(3)
			for i in  range(0, q.shape[0]):
				dipole += q[i]*x_[i]
			return dipole

		def DFTForceField(x_, DoForce=True):
			if DoForce:
				return QchemDFT(Mol(m.atoms,x_),basis_ = '6-31g*',xc_='b3lyp', jobtype_='force', threads=24)
			else:
				return np.asarray([QchemDFT(Mol(m.atoms,x_),basis_ = '6-31g*',xc_='b3lyp', jobtype_='sp', threads=24)])[0]
		#DFTForceField = lambda x: np.asarray([QchemDFT(Mol(m.atoms,x),basis_ = '6-31g',xc_='b3lyp', jobtype_='sp', threads=12)])[0]
		DFTDipoleField = lambda x: QchemDFT(Mol(m.atoms,x),basis_ = '6-31g',xc_='b3lyp', jobtype_='dipole', threads=12)
		#ForceField = lambda x: EnAndForce(x)[-1]
		#EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)


		#PARAMS["OptMaxCycles"]=200
		#Opt = GeomOptimizer(EnAndForce)
		#m=Opt.Opt(m)
		#print ("m.coords:", m.coords)
		#return
		PARAMS["MDdt"] = 0.2
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDMaxStep"] = 10000
		PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDTemp"]= 300.0
		meta = BoxedMetaDynamics(EnergyForceField, m, name_="BoxMetaReactor", Box_=np.array(18.0*np.eye(3)))
		meta.Prop()
		return

	if (0):
		a=MSet("watercube", center_=False)
		a.ReadXYZ("watercube")
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

		m = a.mols[1]
		print ("m.coords:", np.max(m.coords,axis=0), np.min(m.coords,axis=0))
		m.coords += np.asarray([0,1,0])
		#print manager.EvalBPDirectEEUpdateSinglePeriodic(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())
		#print manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
		#return
		#charge = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[6]
		#bp_atom = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[2]
		#for i in range (0, m.NAtoms()):
		#	print i+1, charge[0][i],bp_atom[0][i]

		def EnAndForce(x_, DoForce=True):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			if DoForce:
				return energy, force
			else:
				return energy

		def GetEnergyForceForMol(m):
			def EnAndForce(x_, DoForce=True):
				tmpm = Mol(m.atoms,x_)
				Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
				energy = Etotal[0]
				force = gradient[0]
				if DoForce:
					return energy, force
				else:
					return energy
			return EnAndForce

		def EnForceCharge(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return energy, force, atom_charge[0]

		def ChargeField(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return atom_charge[0]

		def EnergyField(x_):
			return EnAndForce(x_,True)[0]

		def DipoleField(x_):
			q = np.asarray(ChargeField(x_))
			dipole = np.zeros(3)
			for i in  range(0, q.shape[0]):
				dipole += q[i]*x_[i]
			return dipole

		def DFTForceField(x_, DoForce=True):
			if DoForce:
				return QchemDFT(Mol(m.atoms,x_),basis_ = '6-31g*',xc_='b3lyp', jobtype_='force', threads=24)
			else:
				return np.asarray([QchemDFT(Mol(m.atoms,x_),basis_ = '6-31g*',xc_='b3lyp', jobtype_='sp', threads=24)])[0]
		#DFTForceField = lambda x: np.asarray([QchemDFT(Mol(m.atoms,x),basis_ = '6-31g',xc_='b3lyp', jobtype_='sp', threads=12)])[0]
		DFTDipoleField = lambda x: QchemDFT(Mol(m.atoms,x),basis_ = '6-31g',xc_='b3lyp', jobtype_='dipole', threads=12)
		#ForceField = lambda x: EnAndForce(x)[-1]
		#EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)


		#PARAMS["OptMaxCycles"]=200
		#Opt = GeomOptimizer(EnAndForce)
		#m=Opt.Opt(m)
		#print ("m.coords:", m.coords)
		#return
		PARAMS["MDdt"] = 0.2
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDMaxStep"] = 10000
		PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDTemp"]= 300.0
		meta = BoxedMetaDynamics(EnergyForceField, m, name_="BoxMetaTest", Box_=np.array(16.0*np.eye(3)))
		meta.Prop()
		return
#Prepare()
Eval()
