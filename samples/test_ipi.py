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

def TestIPI():
	if (1):
		#a=MSet("H2O_cluster_meta", center_=False)
		#a.ReadXYZ("H2O_cluster_meta")
		a=MSet("water_tiny", center_=False)
		a.ReadXYZ("water_tiny")
		m = a.mols[-1]
		m.coords = m.coords - np.min(m.coords) + 0.00001

		TreatedAtoms = a.AtomTypes()
		PARAMS["learning_rate"] = 0.00001
		PARAMS["momentum"] = 0.95
		PARAMS["max_steps"] = 101
		PARAMS["batch_size"] =  150   # 40 the max min-batch size it can go without memory error for training
		PARAMS["test_freq"] = 1
		PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScalar"] = 1.0/20.0
		PARAMS["DipoleScaler"]=1.0
		PARAMS["NeuronType"] = "relu"
		PARAMS["HiddenLayers"] = [500, 500, 500]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 0
		#PARAMS["Erf_Width"] = 1.0
		PARAMS["Poly_Width"] = 4.6
		#PARAMS["AN1_r_Rc"] = 8.0
		#PARAMS["AN1_num_r_Rs"] = 64
		PARAMS["EECutoffOff"] = 15.0
		#PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 15
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")

		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("Mol_H2O_wb97xd_1to21_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_1",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw",False,False)
		cellsize = 9.3215
		lat = cellsize*np.eye(3)
		PF = TFPeriodicVoxelForce(15.0,lat)
		zp = np.zeros(m.NAtoms()*PF.tess.shape[0], dtype=np.int32)
		xp = np.zeros((m.NAtoms()*PF.tess.shape[0], 3))
		for i in range(0, PF.tess.shape[0]):
			zp[i*m.NAtoms():(i+1)*m.NAtoms()] = m.atoms
			xp[i*m.NAtoms():(i+1)*m.NAtoms()] = m.coords + cellsize*PF.tess[i]

		m_periodic = Mol(zp, xp)

		def EnAndForce(x_):
			x_ = np.mod(x_, cellsize)
			xp = np.zeros((m.NAtoms()*PF.tess.shape[0], 3))
			for i in range(0, PF.tess.shape[0]):
				xp[i*m.NAtoms():(i+1)*m.NAtoms()] = x_ + cellsize*PF.tess[i]
			m_periodic.coords = xp
			m_periodic.coords[:m.NAtoms()] = x_
			Etotal, gradient  = manager.EvalBPDirectEEUpdateSinglePeriodic(m_periodic, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())
			energy = Etotal[0]
			force = gradient[0]
			print ("energy:", energy)
			return energy, force

		ForceField = lambda x: EnAndForce(x)[-1]
		EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)

		interface = TMIPIManger(EnergyForceField, TCP_IP="localhost", TCP_PORT= 31415)
		interface.md_run()
		return

TestIPI()
