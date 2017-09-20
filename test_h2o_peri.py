from __future__ import absolute_import
#import memory_util
#memory_util.vlog(1)
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from TensorMol.ElectrostaticsTF import *
from TensorMol.NN_MBE import *
from TensorMol.TMIPIinterface import *
import random

def TestPeriodicLJVoxel():
	"""
	Tests a Simple Periodic optimization.
	Trying to find the HCP minimum for the LJ crystal.
	"""
	if (1):
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
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 15
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")

		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("Mol_H2O_wb97xd_1to21_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_1",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw",False,False)
		#print np.sum(manager.EvalBPDirectEEUpdateSinglePeriodic(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())[6])
		#print manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)

		cellsize = 9.3215
		lat = cellsize*np.eye(3)
		PF = TFPeriodicVoxelForce(15.0,lat)
		#zp, xp = PF(m.atoms,m.coords,lat)  # tessilation in TFPeriodic seems broken   
		
		zp = np.zeros(m.NAtoms()*PF.tess.shape[0], dtype=np.int32)
		xp = np.zeros((m.NAtoms()*PF.tess.shape[0], 3))
		for i in range(0, PF.tess.shape[0]):
			zp[i*m.NAtoms():(i+1)*m.NAtoms()] = m.atoms
			xp[i*m.NAtoms():(i+1)*m.NAtoms()] = m.coords + cellsize*PF.tess[i]
		print (zp.shape, xp)
		m_periodic = Mol(zp, xp)
		#print ("nreal:", m.NAtoms())
		print manager.EvalBPDirectEEUpdateSinglePeriodic(m_periodic, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())
		return

	if (0):
		a=MSet("water_tiny", center_=False)
		a.ReadXYZ("water_tiny")
		m = a.mols[-1]
		m.coords = m.coords - np.min(m.coords)
		print("Original coords:", m.coords)
		# Generate a Periodic Force field.
		cellsize = 9.3215
		lat = cellsize*np.eye(3)
		PF = TFPeriodicVoxelForce(15.0,lat)
		#zp, xp = PF(m.atoms,m.coords,lat)  # tessilation in TFPeriodic seems broken   
		
		zp = np.zeros(m.NAtoms()*PF.tess.shape[0], dtype=np.int32)
		xp = np.zeros((m.NAtoms()*PF.tess.shape[0], 3))
		for i in range(0, PF.tess.shape[0]):
			zp[i*m.NAtoms():(i+1)*m.NAtoms()] = m.atoms
			xp[i*m.NAtoms():(i+1)*m.NAtoms()] = m.coords + cellsize*PF.tess[i]
		

		#print (zp, xp)
		t0 = time.time()
		NL = NeighborListSetWithImages(xp.reshape((1,-1,3)), np.array([zp.shape[0]]), np.array([m.NAtoms()]), True, True, zp.reshape((1,-1)), sort_=True)
		rad_p_ele, ang_t_elep, mil_j, mil_jk = NL.buildPairsAndTriplesWithEleIndexPeriodic(4.6, 3.1, np.array([1,8]), np.array([[1,1],[1,8],[8,8]]))
		print ("time cost:", time.time() - t0)
		NLEE = NeighborListSetWithImages(xp.reshape((1,-1,3)), np.array([zp.shape[0]]), np.array([m.NAtoms()]), False, False,  None)
		rad_eep = NLEE.buildPairs(15.0)
		print ("time cost:", time.time() - t0)
		print (mil_j[:50], mil_jk[:50])
		print (rad_p_ele.shape, rad_eep.shape,  ang_t_elep.shape)
		return
		print (ang_t_elep)
		count = 0
		for i in range (0, rad_p_ele.shape[0]):
			if rad_p_ele[i][1] == 0:
				#print (ang_t_elep[i])
				count += 1
		print ("count", count)
		mp = Mol(zp, xp)


		mp.WriteXYZfile(fname="H2O_peri_tiny")
		cutoff = 4.6

		mostatom=[]
		mostatomele=[]
		for i in range (0, m.NAtoms()):
			around = 0
			for j in range (0, mp.NAtoms()):
				dist = np.sum(np.square(m.coords[i] - mp.coords[j]))**0.5
				if i!=j and  dist <  cutoff:
					around += 1
					if i == 7:
						mostatom.append(mp.coords[j])
						mostatomele.append(mp.atoms[j])
						print (j, mp.coords[j])
			print (around)
		mostatom = np.asarray(mostatom)
		mostatomele =  np.asarray(mostatomele, dtype=np.int32)
		most_mol = Mol(mostatomele, mostatom)
		most_mol.WriteXYZfile(fname="H2O_most")
			#for j in range(0, rad_p_ele.shape[0]):
			#	print (np.sum(np.square(m.coords[i] - mp.coords[int(rad_p_ele[j][1])]))**0.5)
		return

TestPeriodicLJVoxel()
