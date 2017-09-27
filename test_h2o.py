from __future__ import absolute_import
#import memory_util
#memory_util.vlog(1)
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from TensorMol.ElectrostaticsTF import *
from TensorMol.NN_MBE import *
from TensorMol.TMIPIinterface import *
import random

def TrainPrepare():
	if (0):
		import math, random
		a = MSet("H2O_wb97xd_1to21")
		a.Load()
		random.shuffle(a.mols)
		#a=MSet("H2O_cluster_meta", center_=False)
		#a.ReadXYZ("H2O_cluster_meta")
		Hbondcut = 2.0
		Hbondangle = 20.0*math.pi/180.0
		HOcut = 1.1


		singlemax = 0.04 * len(a.mols)
		doublemax = 0.02 * len(a.mols)

		single_record = np.zeros((21,2))
		single_record[:,0] = range(1,22)
		double_record = np.zeros((21,2))
		double_record[:,0] = range(1,22)
		def MakeH3O(mol, Hbonds_set, xyzname="H2O_meta_with_H3O", doublepro = False):
			new_m = Mol(mol.atoms, mol.coords)
			for i, Hbonds in enumerate(Hbonds_set):
				O1_index = Hbonds[0]
				H_index = Hbonds[1]
				O2_index = Hbonds[2]
				O2_H_index = []
				for j in range (0, mol.NAtoms()):
					if mol.atoms[j] == 1 and len(O2_H_index) <=2 :
						dist = np.sum(np.square(mol.coords[O2_index] - mol.coords[j]))**0.5
						if dist < HOcut:
							O2_H_index.append(j)
				if len(O2_H_index) != 2:
					return
				O2H_1 = mol.coords[O2_index] - mol.coords[O2_H_index[0]]
				O2H_2 = mol.coords[O2_index] - mol.coords[O2_H_index[1]]
				y_axis = np.cross(O2H_1, O2H_2)
				y_axis = y_axis/np.sum(np.square(y_axis))**0.5


				x_axis = mol.coords[O2_index] - (mol.coords[O2_H_index[0]]+mol.coords[O2_H_index[1]])/2.0
				x_axis = x_axis/np.sum(np.square(x_axis))**0.5
				OH_vec = mol.coords[O1_index] - mol.coords[H_index]

				angle_cri = math.pi/3

				if np.dot(x_axis, OH_vec)/np.sum(np.square(OH_vec))**0.5 < math.cos(math.pi/3):
					return

				t_angle = math.pi/180.0*(180.0-131.75 + 10*(2.0*random.random() - 1.0))

				t_length = (1.0 + 0.1*(2.0*random.random() - 1.0))

				#print y_axis, x_axis
				vec1 =(math.tan(t_angle)*y_axis + x_axis)
				vec1 = vec1/np.sum(np.square(vec1))**0.5
				vec1 = vec1*t_length + mol.coords[O2_index]

				vec2 = -(math.tan(t_angle)*y_axis) + x_axis
				vec2 = vec2/np.sum(np.square(vec2))**0.5
				vec2 = vec2*t_length + mol.coords[O2_index]

				if np.sum(np.square(mol.coords[H_index] - vec1))**0.5 < np.sum(np.square(mol.coords[H_index] - vec2))**0.5:
					vec = vec1
				else:
					vec = vec2

				if (not doublepro and i==0) or i==1:
					vec =  random.random()*(vec - mol.coords[H_index]) + mol.coords[H_index]
				new_m.coords[H_index] = vec
			new_m.WriteXYZfile(fname=xyzname)
			if doublepro:
				double_record[int(mol.NAtoms())/3-1,1] += 1
			else:
				single_record[int(mol.NAtoms())/3-1,1] += 1
			return 1

		singlepro = 0
		doublepro = 0
		for mol_index, m in enumerate(a.mols):
			i_ran = random.randint(0, m.NAtoms()-1)
			for i_ini in range (0, m.NAtoms()):
				i  = i_ini + i_ran
				if i > m.NAtoms()-1:
					i = i - m.NAtoms() + 1
				if m.atoms[i] == 8: #it is a O:
					H1_index = -1
					H2_index = -1
					Hbonds = []
					for j in range (0, m.NAtoms()):
						if H1_index != -1 and H2_index != -1:
							break
						if m.atoms[j] == 1:
							dist = np.sum(np.square(m.coords[i] - m.coords[j]))**0.5
							if dist < HOcut and H1_index == -1:
								H1_index = j
							elif dist < HOcut and H1_index != -1:
								H2_index = j
							else:
								continue
					if H1_index != -1 and H2_index != -1:
						Hbondflag1 = False
						for j in range (0, m.NAtoms()):
							if j==i:
								continue
							if m.atoms[j] == 8:
								Hbonddist = np.sum(np.square(m.coords[H1_index] - m.coords[j]))**0.5
								if Hbonddist < Hbondcut:
									OOdist = np.sum(np.square(m.coords[i] - m.coords[j]))**0.5
									HOdist = np.sum(np.square(m.coords[H1_index] - m.coords[i]))**0.5
									angle = (OOdist**2 + HOdist**2 - Hbonddist**2)/(2*OOdist*HOdist)
									if angle > math.cos(Hbondangle):
										Hbondflag1 = True
										Hbonds.append([i, H1_index, j])
						Hbondflag2 = False
						for j in range (0, m.NAtoms()):
							if j==i:
								continue
							if m.atoms[j] == 8:
								Hbonddist = np.sum(np.square(m.coords[H2_index] - m.coords[j]))**0.5
								if Hbonddist < Hbondcut:
									OOdist = np.sum(np.square(m.coords[i] - m.coords[j]))**0.5
									HOdist = np.sum(np.square(m.coords[H2_index] - m.coords[i]))**0.5
									angle = (OOdist**2 + HOdist**2 - Hbonddist**2)/(2*OOdist*HOdist)
									if angle > math.cos(Hbondangle):
										Hbondflag2 = True
										Hbonds.append([i, H2_index, j])
					if len(Hbonds) == 1 and singlepro < singlemax:
						if MakeH3O(m, Hbonds, xyzname="H2O_meta_with_H3O_single", doublepro=False):
							singlepro += 1
							print (single_record)
							print ("single pronated...", singlepro, " mol_index:", mol_index)
						break
					elif len(Hbonds) == 2 and doublepro < doublemax:
						continue
						#if MakeH3O(m, Hbonds, xyzname="H2O_meta_with_H3O_double", doublepro=True):
						#	doublepro += 1
						#	print (double_record)
						#	print ("double pronated...", doublepro, " mol_index:", mol_index)
						#break
					else:
						continue


	if (0):
		WB97XDAtom={}
		WB97XDAtom[1]=-0.5026682866
		WB97XDAtom[6]=-37.8387398698
		WB97XDAtom[7]=-54.5806161811
		WB97XDAtom[8]=-75.0586028656
                a = MSet("H2O_wb97xd_1to21_with_prontonated")
                dic_list = pickle.load(open("./datasets/H2O_wbxd_1to21_with_prontonated.dat", "rb"))
                for mol_index, dic in enumerate(dic_list):
                        atoms = []
			print ("mol_index:", mol_index)
                        for atom in dic['atoms']:
                                atoms.append(AtomicNumber(atom))
                        atoms = np.asarray(atoms, dtype=np.uint8)
			#print (dic.keys())
			#print (dic['xyz'])
                        mol = Mol(atoms, dic['xyz'])
                        #mol.properties['charges'] = dic['charges']
                        mol.properties['dipole'] = np.asarray(dic['dipole'])
                        #mol.properties['quadropole'] = dic['quad']
                        mol.properties['energy'] = dic['scf_energy']
                        mol.properties['gradients'] = dic['gradients']
			mol.properties['atomization'] = dic['scf_energy']
			for i in range (0, mol.NAtoms()):
				mol.properties['atomization'] -= WB97XDAtom[mol.atoms[i]]
                        a.mols.append(mol)
		#a.mols[10000].WriteXYZfile(fname="H2O_sample.xyz")
		#print(a.mols[100].properties)
                a.Save()

	if (0):
		a = MSet("H2O_wb97xd_1to21_with_prontonated")
		a.Load()
		total_water = 0.0
		total_atomization = 0.0
		for mol in a.mols:
			total_water += mol.NAtoms()/3
			total_atomization += mol.properties['atomization']
		avg_atomization = total_atomization/total_water
		print ("avg_atomization:", avg_atomization)  # ('avg_atomization:', -0.35551059977287547)
		for mol in a.mols:
			mol.properties['atomization_old'] = mol.properties['atomization']
			mol.properties['atomization'] = mol.properties['atomization']-mol.NAtoms()/3.0*avg_atomization
			print ("mol.properties['atomization']:,mol.properties['atomization_old']", mol.properties['atomization'], mol.properties['atomization_old'])
		a.Save()


	if (1):
		WB97XDAtom={}
		WB97XDAtom[1]=-0.5026682866
		WB97XDAtom[6]=-37.8387398698
		WB97XDAtom[7]=-54.5806161811
		WB97XDAtom[8]=-75.0586028656
		ch4_min_atomization = -0.6654760227400112
		water_avg_atomization = -0.35551059977287
                a = MSet("H2O_wb97xd_1to21_with_prontonated_with_ch4")
                dic_list = pickle.load(open("./datasets/H2O_wbxd_1to21_with_prontonated_with_ch4.dat", "rb"))
                for mol_index, dic in enumerate(dic_list):
                        atoms = []
			print ("mol_index:", mol_index)
                        for atom in dic['atoms']:
                                atoms.append(AtomicNumber(atom))
                        atoms = np.asarray(atoms, dtype=np.uint8)
			#print (dic.keys())
			#print (dic['xyz'])
                        mol = Mol(atoms, dic['xyz'])
                        #mol.properties['charges'] = dic['charges']
                        mol.properties['dipole'] = np.asarray(dic['dipole'])
                        #mol.properties['quadropole'] = dic['quad']
                        mol.properties['energy'] = dic['scf_energy']
                        mol.properties['gradients'] = dic['gradients']
			mol.properties['atomization_old'] = dic['scf_energy']
			for i in range (0, mol.NAtoms()):
				mol.properties['atomization_old'] -= WB97XDAtom[mol.atoms[i]]
                        a.mols.append(mol)
			if 6 in mol.atoms: # contain one CH4
				mol.properties['atomization'] = mol.properties['atomization_old'] - (mol.NAtoms()-5)/3*water_avg_atomization - ch4_min_atomization
			else:
				mol.properties['atomization'] = mol.properties['atomization_old'] - mol.NAtoms()/3*water_avg_atomization
			print ("mol.properties['atomization']:", mol.properties['atomization'])
		#a.mols[10000].WriteXYZfile(fname="H2O_sample.xyz")
		#print(a.mols[100].properties)
                a.Save()
def Train():
	if (0):
		a = MSet("H2O_wb97xd_1to10")
		a.Load()
		#random.shuffle(a.mols)
		#for i in range(150000):
		#	a.mols.pop()
		TreatedAtoms = a.AtomTypes()
		PARAMS["learning_rate"] = 0.00001
		PARAMS["momentum"] = 0.95
		PARAMS["max_steps"] = 201
		PARAMS["batch_size"] =  300   # 40 the max min-batch size it can go without memory error for training
		PARAMS["test_freq"] = 5
		PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
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
		PARAMS["SwitchEpoch"] = 40
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		#tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
		#tset = TensorMolData_BP_Multipole_2_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = False)
		#manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode") # Initialzie a manager than manage the training of neural network.
		#manager=TFMolManage("",tset,False,"Dipole_BP_2_Direct")
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_Update")
		PARAMS['Profiling']=0
		manager.Train(1)
		#with memory_util.capture_stderr() as stderr:
		#	manager.Train(1)
		#memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)

	if (0):
		a = MSet("H2O_wb97xd_1to10")
		a.Load()
		#random.shuffle(a.mols)
		#for i in range(150000):
		#	a.mols.pop()
		TreatedAtoms = a.AtomTypes()
		PARAMS["learning_rate"] = 0.00001
		PARAMS["momentum"] = 0.95
		PARAMS["max_steps"] = 201
		PARAMS["batch_size"] =  300   # 40 the max min-batch size it can go without memory error for training
		PARAMS["test_freq"] = 5
		PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
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
		PARAMS["SwitchEpoch"] = 40
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		#tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
		#tset = TensorMolData_BP_Multipole_2_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = False)
		#manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode") # Initialzie a manager than manage the training of neural network.
		#manager=TFMolManage("",tset,False,"Dipole_BP_2_Direct")
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update")
		PARAMS['Profiling']=0
		manager.Train(1)


	if (0):
		a = MSet("H2O_wb97xd_1to21")
		a.Load()
		#random.shuffle(a.mols)
		#for i in range(300000):
		#	a.mols.pop()
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
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw")
		PARAMS['Profiling']=0
		manager.Train(1)

	if (0):
		a = MSet("H2O_wb97xd_1to21_with_prontonated")
		a.Load()
		random.shuffle(a.mols)
		#for i in range(360000):
		#	a.mols.pop()
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
		#PARAMS["Poly_Width"] = 4.6
		PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
		#PARAMS["AN1_r_Rc"] = 8.0
		#PARAMS["AN1_num_r_Rs"] = 64
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["DSFAlpha"] = 0.18
		PARAMS["AddEcc"] = True
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 15
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu")
		PARAMS['Profiling']=0
		manager.Train(1)


	if (1):
		a = MSet("H2O_wb97xd_1to21_with_prontonated_with_ch4")
		a.Load()
		random.shuffle(a.mols)
		#for i in range(680000):
		#	a.mols.pop()
		TreatedAtoms = a.AtomTypes()
		PARAMS["learning_rate"] = 0.00001
		PARAMS["momentum"] = 0.95
		PARAMS["max_steps"] = 71
		PARAMS["batch_size"] =  80   # 40 the max min-batch size it can go without memory error for training
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
		PARAMS["AddEcc"] = False
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 10
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw")
		PARAMS['Profiling']=0
		manager.Train(1)
def Eval():
	if (1):
		#a = MSet("water_tiny", center_=False)
		#a.ReadXYZ("water_tiny")
		a=MSet("H2O_cluster_meta", center_=False)
		a.ReadXYZ("H2O_cluster_meta")
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
		#PARAMS["Poly_Width"] = 4.6
		PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
		#PARAMS["AN1_r_Rc"] = 8.0
		#PARAMS["AN1_num_r_Rs"] = 64
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["DSFAlpha"] = 0.18
		PARAMS["AddEcc"] = True
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 15
		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")

		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("Mol_H2O_wb97xd_1to21_with_prontonated_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_1",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu",False,False)
		m = a.mols[-2]
		#print manager.EvalBPDirectEEUpdateSinglePeriodic(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())
		#print manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
		#return
		#charge = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[6]
		#bp_atom = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[2]
		#for i in range (0, m.NAtoms()):
		#	print i+1, charge[0][i],bp_atom[0][i]

		def EnAndForce(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return energy, force

		def EnForceCharge(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return energy, force, atom_charge

		def ChargeField(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return atom_charge[0]

		ForceField = lambda x: EnAndForce(x)[-1]
		EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)

		#PARAMS["OptMaxCycles"]=200
		#Opt = GeomOptimizer(EnergyForceField)
		#m=Opt.Opt(m)


                PARAMS["MDThermostat"] = "Nose"
                PARAMS["MDTemp"] = 300
                PARAMS["MDdt"] = 0.2
                PARAMS["RemoveInvariant"]=True
                PARAMS["MDV0"] = None
                PARAMS["MDMaxStep"] = 10000
                md = VelocityVerlet(None, m, "water_tiny_noperi",EnergyForceField)
                md.Prop()
		return


	if (0):
		a = MSet("water_tiny", center_=False)
		a.ReadXYZ("water_tiny")
		#a=MSet("H2O_cluster_meta", center_=False)
		#a.ReadXYZ("H2O_cluster_meta")
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
		m = a.mols[0]
		#print manager.EvalBPDirectEEUpdateSinglePeriodic(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())
		print manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
		return
		#charge = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[6]
		#bp_atom = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[2]
		#for i in range (0, m.NAtoms()):
		#	print i+1, charge[0][i],bp_atom[0][i]

		def EnAndForce(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return energy, force

		def EnForceCharge(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return energy, force, atom_charge

		def ChargeField(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return atom_charge[0]

		ForceField = lambda x: EnAndForce(x)[-1]
		EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)

		#PARAMS["OptMaxCycles"]=200
		#Opt = GeomOptimizer(EnergyForceField)
		#m=Opt.Opt(m)


                PARAMS["MDThermostat"] = "Nose"
                PARAMS["MDTemp"] = 1000
                PARAMS["MDdt"] = 0.2
                PARAMS["RemoveInvariant"]=True
                PARAMS["MDV0"] = None
                PARAMS["MDMaxStep"] = 10000
                md = VelocityVerlet(None, m, "water_tiny_noperi",EnergyForceField)
                md.Prop()
		return

		#PARAMS["OptMaxCycles"]=1000
		#Opt = GeomOptimizer(EnergyForceField)
		#m=Opt.Opt(m)

		PARAMS["MDdt"] = 0.2
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDMaxStep"] = 2000
		PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDV0"] = None
		PARAMS["MDAnnealTF"] = 1.0
		PARAMS["MDAnnealT0"] = 300.0
		PARAMS["MDAnnealSteps"] = 1000
		anneal = Annealer(EnergyForceField, None, m, "Anneal")
		anneal.Prop()
		m.coords = anneal.Minx.copy()
		m.WriteXYZfile("./results/", "Anneal_opt_648")
		return

		PARAMS["MDdt"] = 0.2
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDMaxStep"] = 10000
		PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDV0"] = None
		PARAMS["MDAnnealTF"] = 30.0
		PARAMS["MDAnnealT0"] = 0.1
		PARAMS["MDAnnealSteps"] = 10000
		anneal = Annealer(EnergyForceField, None, m, "Anneal")
		anneal.Prop()
		m.coords = anneal.Minx.copy()
		m.WriteXYZfile("./results/", "Anneal_opt")
		PARAMS["MDThermostat"] = None
		PARAMS["MDTemp"] = 0
		PARAMS["MDdt"] = 0.1
		PARAMS["MDV0"] = None
		PARAMS["MDMaxStep"] = 40000
		md = IRTrajectory(EnAndForce, ChargeField, m, "water_10_IR", anneal.v)
		md.Prop()
		WriteDerDipoleCorrelationFunction(md.mu_his)

		outlist = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], HasVdw=True)
		print outlist
		print np.sum(outlist[5].reshape((-1,3)),axis=-1)
		print np.sum((outlist[5].reshape((-1,1))*m.coords), axis=0)

		charge = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)[5]
		Ecc = 0.0
		for i in range (0, m.NAtoms()):
			for j in range (i+1, m.NAtoms()):
				dist = (np.sum(np.square(m.coords[i] - m.coords[j])))**0.5 *  BOHRPERA
				if dist < PARAMS["EECutoffOn"]*BOHRPERA:
					cut = 0.0
				elif dist > (PARAMS["EECutoffOn"]+PARAMS["Poly_Width"])*BOHRPERA:
					cut = 1.0
				else:
					t = (dist-PARAMS["EECutoffOn"]*BOHRPERA)/(PARAMS["Poly_Width"]*BOHRPERA)
					cut = -t*t*(2.0*t-3.0)
				Ecc +=  cut*charge[0][i]*charge[0][j]/dist
		print ("Ecc manual:", Ecc)

	if (0):  # i-pi test
		a=MSet("H2O_cluster_meta", center_=False)
		a.ReadXYZ("H2O_cluster_meta")
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
		manager=TFMolManage("Mol_H2O_wb97xd_1to21_with_prontonated_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_1",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu",False,False)
		m = a.mols[9]
		print (m.coords)
		def EnAndForce(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return energy, force

		def EnForceCharge(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return energy, force, atom_charge

		def ChargeField(x_):
			m.coords = x_
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
			energy = Etotal[0]
			force = gradient[0]
			return atom_charge[0]

		ForceField = lambda x: EnAndForce(x)[-1]
		EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)
		interface = TMIPIManger(EnergyForceField, TCP_IP="localhost", TCP_PORT= 31415)
		interface.md_run()

	if (0):
		a=MSet("H2O_cluster_meta", center_=False)
		a.ReadXYZ("H2O_cluster_meta")
		TreatedAtoms = a.AtomTypes()
		PARAMS["learning_rate"] = 0.00001
		PARAMS["momentum"] = 0.95
		PARAMS["max_steps"] = 201
		PARAMS["batch_size"] =  300   # 40 the max min-batch size it can go without memory error for training
		PARAMS["test_freq"] = 5
		PARAMS["tf_prec"] = "tf.float64"
		PARAMS["GradScaler"] = 1.0
		PARAMS["DipoleScaler"]=1.0
		PARAMS["NeuronType"] = "relu"
		PARAMS["HiddenLayers"] = [500, 500, 500]
		PARAMS["EECutoff"] = 15.0
		PARAMS["EECutoffOn"] = 0.0
		#PARAMS["EECutoffOn"] = 7.0
		#PARAMS["Erf_Width"] = 0.4
		PARAMS["Poly_Width"] = 4.6
		#PARAMS["AN1_r_Rc"] = 8.0
		#PARAMS["AN1_num_r_Rs"] = 64
		PARAMS["EECutoffOff"] = 15.0
		PARAMS["learning_rate_dipole"] = 0.0001
		PARAMS["learning_rate_energy"] = 0.00001
		PARAMS["SwitchEpoch"] = 40

		d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")

		tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
		manager=TFMolManage("Mol_H2O_wb97xd_1to10_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_Update_1",tset,False,"fc_sqdiff_BP_Direct_EE_Update",False,False)

		m = a.mols[-2]
		outlist =  manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
		print outlist[4]
		return
		print "self.Ree_on:", manager.Instances.Ree_on
		print outlist
		print np.sum(outlist[4].reshape((-1,3)),axis=-1)
		print np.sum((outlist[4].reshape((-1,1))*m.coords), axis=0)
		charge = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])[4]
		Ecc = 0.0
		for i in range (0, m.NAtoms()):
			for j in range (i+1, m.NAtoms()):
				dist = (np.sum(np.square(m.coords[i] - m.coords[j])))**0.5 *  BOHRPERA
				if dist < PARAMS["EECutoffOn"]*BOHRPERA:
					cut = 0.0
				elif dist > (PARAMS["EECutoffOn"]+PARAMS["Poly_Width"])*BOHRPERA:
					cut = 1.0
				else:
					t = (dist-PARAMS["EECutoffOn"]*BOHRPERA)/(PARAMS["Poly_Width"]*BOHRPERA)
					cut = -t*t*(2.0*t-3.0)
				Ecc +=  cut*charge[0][i]*charge[0][j]/dist
		print ("Ecc manual:", Ecc)
		return

		def EnAndForce(x_):
			m.coords = x_
			Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
			energy = Etotal[0]
			force = gradient[0]
			return energy, force

		def EnForceCharge(x_):
			m.coords = x_
			Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
			energy = Etotal[0]
			force = gradient[0]
			return energy, force, atom_charge

		def ChargeField(x_):
			m.coords = x_
			Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(m, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"])
			energy = Etotal[0]
			force = gradient[0]
			return atom_charge[0]

		ForceField = lambda x: EnAndForce(x)[-1]
		EnergyField = lambda x: EnAndForce(x)[0]
		EnergyForceField = lambda x: EnAndForce(x)

		PARAMS["OptMaxCycles"]=200
		Opt = GeomOptimizer(EnergyForceField)
		m=Opt.Opt(m)

		#PARAMS["MDdt"] = 0.2
		#PARAMS["RemoveInvariant"]=True
		#PARAMS["MDMaxStep"] = 2000
		#PARAMS["MDThermostat"] = "Nose"
		#PARAMS["MDV0"] = None
		#PARAMS["MDAnnealTF"] = 0.0
		#PARAMS["MDAnnealT0"] = 300.0
		#PARAMS["MDAnnealSteps"] = 1000
		#anneal = Annealer(EnergyForceField, None, m, "Anneal")
		#anneal.Prop()
		#m.coords = anneal.Minx.copy()
		#m.WriteXYZfile("./results/", "Anneal_opt")

		PARAMS["MDdt"] = 0.2
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDMaxStep"] = 10000
		PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDV0"] = None
		PARAMS["MDAnnealTF"] = 30.0
		PARAMS["MDAnnealT0"] = 0.1
		PARAMS["MDAnnealSteps"] = 10000
		anneal = Annealer(EnergyForceField, None, m, "Anneal")
		anneal.Prop()
		m.coords = anneal.Minx.copy()
		m.WriteXYZfile("./results/", "Anneal_opt")
		PARAMS["MDThermostat"] = None
		PARAMS["MDTemp"] = 0
		PARAMS["MDdt"] = 0.1
		PARAMS["MDV0"] = None
		PARAMS["MDMaxStep"] = 40000
		md = IRTrajectory(EnAndForce, ChargeField, m, "water_10_IR", anneal.v)
		md.Prop()
		WriteDerDipoleCorrelationFunction(md.mu_his)
def BoxAndDensity():
	# Prepare a Box of water at a desired density
	# from a rough water molecule.
	a = MSet()
	a.mols.append(Mol(np.array([1,1,8]),np.array([[0.9,0.1,0.1],[1.,0.9,1.],[0.1,0.1,0.1]])))
	TreatedAtoms = a.AtomTypes()
	m = a.mols[0]
	# Prepare the force field.
	PARAMS["batch_size"] =  150   # 40 the max min-batch size it can go without memory error for training
	PARAMS["tf_prec"] = "tf.float64"
	PARAMS["NeuronType"] = "relu"
	PARAMS["HiddenLayers"] = [500, 500, 500]
	PARAMS["EECutoff"] = 15.0
	PARAMS["EECutoffOn"] = 0
	PARAMS["Poly_Width"] = 4.6
	PARAMS["EECutoffOff"] = 15.0
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
	#PARAMS["Poly_Width"] = 4.6
	PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
	#PARAMS["AN1_r_Rc"] = 8.0
	#PARAMS["AN1_num_r_Rs"] = 64
	PARAMS["EECutoffOff"] = 15.0
	PARAMS["DSFAlpha"] = 0.18
	PARAMS["AddEcc"] = True
	PARAMS["learning_rate_dipole"] = 0.0001
	PARAMS["learning_rate_energy"] = 0.00001
	PARAMS["SwitchEpoch"] = 15

	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("Mol_H2O_wb97xd_1to21_with_prontonated_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_1",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu",False,False)

	def EnAndForceAPeriodic(x_):
		"""
		This is the primitive form of force routine required by PeriodicForce.
		"""
		mtmp = Mol(m.atoms,x_)
		en,f = manager.EvalBPDirectEEUpdateSinglePeriodic(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())
		#print("EnAndForceAPeriodic: ", en,f)
		return en[0], f[0]

	def EnAndForce(z_, x_, nreal_, DoForce = True):
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

	if 0:
		# opt the first water.
		PARAMS["OptMaxCycles"]=60
		Opt = GeomOptimizer(EnAndForceAPeriodic)
		a.mols[-1] = Opt.Opt(a.mols[-1])
		m = a.mols[-1]

		# Tesselate that water to create a box
		ntess = 4
		latv = 2.8*np.eye(3)
		# Start with a water in a ten angstrom box.
		lat = Lattice(latv)
		mc = lat.CenteredInLattice(m)
		mt = Mol(*lat.TessNTimes(mc.atoms,mc.coords,ntess))
		nreal = mt.NAtoms()
		mt.Distort(0.01)

	def EnAndForceAPeriodic(x_):
		"""
		This is the primitive form of force routine required by PeriodicForce.
		"""
		mtmp = Mol(mt.atoms,x_)
		en,f = manager.EvalBPDirectEEUpdateSinglePeriodic(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], mt.NAtoms())
		#print("EnAndForceAPeriodic: ", en,f)
		return en[0], f[0]

	if 0:
		PARAMS["OptMaxCycles"]=30
		Opt = GeomOptimizer(EnAndForceAPeriodic)
		mt = Opt.Opt(mt,"UCopt")

		# Anneal the tesselation.
		EnAndForceAPeriodic = lambda x_: EnAndForce(mt.atoms,x_,mt.NAtoms())
		PARAMS["MDAnnealT0"] = 20.0
		PARAMS["MDAnnealSteps"] = 200
		aper = Annealer(EnAndForceAPeriodic,None,mt)
		aper.Prop()
		mt.coords = aper.Minx

		# Optimize the tesselated system.
		lat0 = (np.max(mt.coords)+0.5)*np.eye(3)
		lat0[0,1] = 0.01
		lat0[1,0] -= 0.01
		lat0[0,2] = 0.01
		lat0[2,0] -= 0.01
		latp = np.eye(3)*8.0
		print(lat0,latp)
		m = Lattice(lat0).CenteredInLattice(mt)
		print(m.coords)

	s = MSet("water64")
	s.ReadXYZ()
	m = s.mols[0]

	PF = PeriodicForce(m,m.properties["Lattice"])
	PF.BindForce(EnAndForce, 20.0)

	# Test that the energy is invariant to translations of atoms through the cell.
	if 0:
		for i in range(20):
			print("En0:", PF(m.coords)[0])
			m.coords += (np.random.random((1,3))-0.5)*3.0
			m.coords = PF.lattice.ModuloLattice(m.coords)
			print("En:"+str(i), PF(m.coords)[0])
			Mol(*PF.lattice.TessLattice(m.atoms,m.coords,12.0)).WriteXYZfile("./results/", "TessCHECK")
	if 0:
		# Try optimizing that....
		PARAMS["OptMaxCycles"]=10
		POpt = PeriodicGeomOptimizer(PF)
		m = POpt.OptToDensity(m,1.0)
		PF.mol0.coords = m.coords
		PF.mol0.properties["Lattice"] = PF.lattice.lattice.copy()
		PF.mol0.WriteXYZfile("./results", "Water64", "w", wprop=True)

	PARAMS["MDAnnealT0"] = 20.0
	PARAMS["MDAnnealTF"] = 300.0
	PARAMS["MDAnnealSteps"] = 1000
	traj = PeriodicAnnealer(PF,"PeriodicWarm")
	traj.Prop()

	# Finally do thermostatted MD.
	PARAMS["MDTemp"] = 300.0
	traj = PeriodicVelocityVerlet(PF,"PeriodicWaterMD")
	traj.Prop()

#TrainPrepare()
#Train()
#Eval()
BoxAndDensity()
