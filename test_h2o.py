from __future__ import absolute_import
#import memory_util
#memory_util.vlog(1)
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from TensorMol.ElectrostaticsTF import *
from TensorMol.NN_MBE import *
import random

def TrainPrepare():
	if (1):
		import math
		a=MSet("H2O_cluster_meta", center_=False)
		a.ReadXYZ("H2O_cluster_meta")
		m=a.mols[2]
		Hbondcut = 2.0
		Hbondangle = 15.0*math.pi/180.0
		HOcut = 1.1
		for i in range (0, m.NAtoms()):
			if m.atoms[i] == 8: #it is a O:
				H1_index = -1
				H2_index = -1
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
				print H1_index, H2_index
				if H1_index != -1 and H2_index != -1:
					Hbondflag1 = False
					for j in range (0, m.NAtoms()):
						if j==i:
							continue
						if m.atoms[j] == 8:
							print m.coords[H1_index], m.coords[j], m.atoms[H1_index], m.atoms[j], H1_index, j
							Hbonddist = np.sum(np.square(m.coords[H1_index] - m.coords[j]))**0.5
							print Hbonddist
							if Hbonddist < Hbondcut:
								OOdist = np.sum(np.square(m.coords[i] - m.coords[j]))**0.5
								HOdist = np.sum(np.square(m.coords[H1_index] - m.coords[i]))**0.5
								angle = (OOdist**2 + HOdist**2 - Hbonddist**2)/(2*OOdist*HOdist)
								print angle
								if angle > math.cos(Hbondangle):
									Hbondflag1 = True
					Hbondflag2 = False
					for j in range (0, m.NAtoms()):
						if j==i:
							continue
						if m.atoms[j] == 8:
							print np.sum(np.square(m.coords[H2_index] - m.coords[j]))**0.5, m.coords[H2_index], m.coords[j], m.atoms[H2_index], m.atoms[j], H2_index, j
							Hbonddist = np.sum(np.square(m.coords[H2_index] - m.coords[j]))**0.5
							if Hbonddist < Hbondcut:
								OOdist = np.sum(np.square(m.coords[i] - m.coords[j]))**0.5
								HOdist = np.sum(np.square(m.coords[H2_index] - m.coords[i]))**0.5
								angle = (OOdist**2 + HOdist**2 - Hbonddist**2)/(2*OOdist*HOdist)
								if angle > math.cos(Hbondangle):
									Hbondflag2 = True
					print i, Hbondflag1, Hbondflag2


	if (0):
		WB97XDAtom={}
		WB97XDAtom[1]=-0.5026682866
		WB97XDAtom[6]=-37.8387398698
		WB97XDAtom[7]=-54.5806161811
		WB97XDAtom[8]=-75.0586028656
                a = MSet("H2O_wb97xd_1to21")
                dic_list = pickle.load(open("./datasets/H2O_wb97xd_1to21.dat", "rb"))
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
		a.mols[10000].WriteXYZfile(fname="H2O_sample.xyz")
		print(a.mols[100].properties)
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


	if (1):
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

def Eval():
	if (1):
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
		manager=TFMolManage("Mol_H2O_wb97xd_1to21_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_1",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw",False,False)
		m = a.mols[-1]
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
                PARAMS["MDTemp"] = 100
                PARAMS["MDdt"] = 0.2
                PARAMS["RemoveInvariant"]=True
                PARAMS["MDV0"] = None
                PARAMS["MDMaxStep"] = 10000
                md = VelocityVerlet(None, m, "water_cluster_small_opt",EnergyForceField)
                md.Prop()
		return

		PARAMS["OptMaxCycles"]=1000
		Opt = GeomOptimizer(EnergyForceField)
		m=Opt.Opt(m)

		PARAMS["MDdt"] = 0.2
		PARAMS["RemoveInvariant"]=True
		PARAMS["MDMaxStep"] = 2000
		PARAMS["MDThermostat"] = "Nose"
		PARAMS["MDV0"] = None
		PARAMS["MDAnnealTF"] = 300.0
		PARAMS["MDAnnealT0"] = 30.0
		PARAMS["MDAnnealSteps"] = 8000
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

	if (1):
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
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("Mol_H2O_wb97xd_1to21_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_1",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw",False,False)

	def EnAndForceAPeriodic(x_):
		"""
		This is the primitive form of force routine required by PeriodicForce.
		"""
		mtmp = Mol(m.atoms,x_)
		en,f = manager.EvalBPDirectEEPeriodic(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], m.NAtoms())
		print("EnAndForceAPeriodic: ", en,f)
		return en, f

	def EnAndForce(z_, x_, nreal_):
		"""
		This is the primitive form of force routine required by PeriodicForce.
		"""
		mtmp = Mol(z_,x_)
		en,f = manager.EvalBPDirectEEPeriodic(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], nreal_)
		print("EnAndForce: ", en,f)
		return en, f

	# opt the first water.
	PARAMS["OptMaxCycles"]=20
	Opt = GeomOptimizer(EnAndForceAPeriodic)
	a.mols[-1] = Opt.Opt(a.mols[-1])
	m = a.mols[-1]

	# Tesselate that water to create a box of 27
	ntess = 2
	latv = np.array([[5.0,0.,0.],[0.,5.,0.],[0.,0.,5.]])
	# Start with a water in a ten angstrom box.
	lat = Lattice(latv)
	mc = lat.CenteredInLattice(m)
	mt = Mol(*lat.TessNTimes(mc.atoms,mc.coords,ntess))
	print(mt.coords)
	mt.coords += np.min(mt.coords)
	nreal = mt.NAtoms()

	# Optimize the tesselated system.
	mindistance = 2.0
	lat0 = np.array([[np.max(mt.coords),0.,0.],[0.,np.max(mt.coords),0.],[0.,0.,np.max(mt.coords)]])*4.0
	latp = np.array([[mindistance,0.,0.],[0.,mindistance,0.],[0.,0.,mindistance]])
	print(lat0,latp)
	m = Lattice(lat0).CenteredInLattice(mt)
	print(m.coords)
	PF = PeriodicForce(mt,lat0)
	PF.BindForce(EnAndForce,10.0)

	# Try optimizing that....
	PARAMS["OptMaxCycles"]=20
	POpt = PeriodicGeomOptimizer(PF)
	mt = POpt.Opt(mt)

	# finally start boxing it up
	Box = PeriodicBoxingDynamics(PF, latp, "BoxingMD")
	Box.Prop()

#TrainPrepare()
#Train()
#Eval()
BoxAndDensity()
