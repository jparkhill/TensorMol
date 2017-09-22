#
# These work Moleculewise the versions without the mol prefix work atomwise.
# but otherwise the behavior of these is the same as TFManage etc.
#
from __future__ import absolute_import
from __future__ import print_function
from .TFManage import *
from .TensorMolData import *
from .TFMolInstance import *
from .TFMolInstanceDirect import *
from .TFMolInstanceEE import *
from .TFMolInstanceDirect import *
from .QuasiNewtonTools import *

import numpy as np
import gc

class TFMolManage(TFManage):
	"""
		A manager of tensorflow instances which perform molecule-wise predictions
		including Many Body and Behler-Parinello
	"""
	def __init__(self, Name_="", TData_=None, Train_=False, NetType_="fc_sqdiff", RandomTData_=True, Trainable_ = True):
		"""
			Args:
				Name_: If not blank, will try to load a network with that name using Prepare()
				TData_: A TensorMolData instance to provide and process data.
				Train_: Whether to train the instances raised.
				NetType_: Choices of Various network architectures.
				RandomTData_: Modifes the preparation of training batches.
		"""
		self.path = "./networks/"
		self.Trainable = Trainable_
		if (Name_!=""):
			self.name = Name_
			self.Prepare()
			return
		TFManage.__init__(self, Name_, TData_, False, NetType_, RandomTData_, Trainable_)
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.TData.order)
		self.TrainedAtoms=[] # In order of the elements in TData
		self.TrainedNetworks=[] # In order of the elements in TData
		self.Instances=None # In order of the elements in TData
		if (Train_):
			self.Train()
			return
		return

	def Train(self, maxstep=3000):
		"""
		Instantiates and trains a Molecular network.

		Args:
			maxstep: The number of training steps.
		"""
		#if (self.TData.dig.eshape==None):
		#	raise Exception("Must Have Digester Shape.")
		# It's up the TensorData to provide the batches and input output shapes.
		if (self.NetType == "fc_classify"):
			self.Instances = MolInstance_fc_classify(self.TData, None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = MolInstance_fc_sqdiff(self.TData, None)
		elif (self.NetType == "fc_sqdiff_BP"):
			self.Instances = MolInstance_fc_sqdiff_BP(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_WithGrad"):
			self.Instances = MolInstance_fc_sqdiff_BP_WithGrad(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Update"):
			self.Instances = MolInstance_fc_sqdiff_BP_Update(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct"):
			self.Instances = MolInstance_DirectBP_NoGrad(self.TData)
		elif (self.NetType == "fc_sqdiff_BPBond_Direct"):
			self.Instances = MolInstance_DirectBPBond_NoGrad(self.TData)
		elif (self.NetType == "fc_sqdiff_BPBond_DirectQueue"):
			self.Instances = MolInstance_DirectBPBond_NoGrad_Queue(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad"):
			self.Instances = MolInstance_DirectBP_Grad(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad_noGradTrain"):
			self.Instances = MolInstance_DirectBP_Grad_noGradTrain(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad_NewIndex"):
			self.Instances = MolInstance_DirectBP_Grad_NewIndex(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad_Linear"):
			self.Instances = MolInstance_DirectBP_Grad_Linear(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad_Linear_EmbOpt"):
			self.Instances = MolInstance_DirectBP_Grad_Linear_EmbOpt(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE"):
			self.Instances = MolInstance_DirectBP_EE(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_Update"):
			self.Instances = MolInstance_DirectBP_EE_Update(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_ChargeEncode"):
			self.Instances = MolInstance_DirectBP_EE_ChargeEncode(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_ChargeEncode_Update"):
			self.Instances = MolInstance_DirectBP_EE_ChargeEncode_Update(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw"):
			self.Instances = MolInstance_DirectBP_EE_ChargeEncode_Update_vdw(self.TData)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu"):
			self.Instances = MolInstance_DirectBP_EE_ChargeEncode_Update_vdw_DSF_elu(self.TData)
		elif (self.NetType == "Dipole_BP"):
			self.Instances = MolInstance_BP_Dipole(self.TData)
		elif (self.NetType == "Dipole_BP_2"):
			self.Instances = MolInstance_BP_Dipole_2(self.TData)
		elif (self.NetType == "Dipole_BP_2_Direct"):
			self.Instances = MolInstance_BP_Dipole_2_Direct(self.TData)
		elif (self.NetType == "LJForce"):
			self.Instances = MolInstance_LJForce(self.TData)
		else:
			raise Exception("Unknown Network Type!")
		if (PARAMS["Profiling"]>0):
			self.Instances.profile()
			return
		self.n_train = PARAMS["max_steps"]
		self.Instances.train(self.n_train)
		nm = self.Instances.name
		# Here we should print some summary of the pupil's progress as well, maybe.
		if self.TrainedNetworks.count(nm)==0:
			self.TrainedNetworks.append(nm)
		self.Save()
		gc.collect()
		return

	def Continue_Training(self, maxsteps=3000):   # test a pretrained network
		self.n_train = PARAMS["max_steps"]
		self.Instances.TData = self.TData
		self.Instances.TData.LoadDataToScratch(self.Instances.tformer)
		self.Instances.Prepare()
		self.Instances.continue_training(self.n_train)
		self.Save()
		return

	def Eval(self, inputs):
		if (self.Instances[mol_t.atoms[atom]].tformer.innorm != None):
			inputs = self.Instances[mol_t.atoms[atom]].tformer.NormalizeIns(inputs, train=False)
		outputs = self.Instances.evaluate(inputs)
		if (self.Instances[mol_t.atoms[atom]].tformer.outnorm != None):
			outputs = self.Instances[mol_t.atoms[atom]].tformer.UnNormalizeOuts(outputs)
		return outputs

	def Eval_BPEnergy(self, mol_set, total_energy =True):
		nmols = len(mol_set.mols)
		natoms = mol_set.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((natoms, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		t = time.time()
		for mol in mol_set.mols:
			ins = self.TData.dig.EvalDigest(mol, False)
			nat = mol.NAtoms()
			cases[casep:casep+nat] = ins
			for i in range (casep, casep+nat):
				meta[i, 0] = mols_done
				meta[i, 1] = mol.atoms[i - casep]
				meta[i, 2] = casep
				meta[i, 3] = casep + nat
			casep += nat
			mols_done += 1
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		matrices = []
		outputpointer = 0
		for i in range (0, natoms):
			sto[self.TData.eles.index(meta[i, 1])] += 1
		currentmol = 0
		for e in range (len(self.TData.eles)):
			inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
			matrices.append(np.zeros((sto[e], nmols)))
		for i in range (0, natoms):
			if currentmol != meta[i, 0]:
				outputpointer += 1
				currentmol = meta[i, 0]
			e = meta[i, 1]
			ei = self.TData.eles.index(e)
			inputs[ei][offsets[ei], :] = cases[i]
			matrices[ei][offsets[ei], outputpointer] = 1.0
			offsets[ei] += 1
		t = time.time()
		pointers = [0 for ele in self.TData.eles]
		mol_out, atom_out, nn_gradient = self.Instances.evaluate([inputs, matrices, dummy_outputs])
		total_list = []
		for i in range (0, nmols):
			total = mol_out[0][i]
			mol = mol_set.mols[i]
			if total_energy:
				for j in range (0, mol.NAtoms()):
					total += ele_U[mol.atoms[j]]
			total_list.append(total)
		return total_list

	def Eval_BPEnergySingle(self, mol):
		"""
		Args:
			mol: a Mol.
		Returns:
			Energy in Hartree
		"""
		nmols = 1
		natoms = mol.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((natoms, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		t = time.time()
		#
		ins = self.TData.dig.EvalDigest(mol, False)
		nat = mol.NAtoms()
		cases[casep:casep+nat] = ins
		for i in range (casep, casep+nat):
			meta[i, 0] = mols_done
			meta[i, 1] = mol.atoms[i - casep]
			meta[i, 2] = casep
			meta[i, 3] = casep + nat
		casep += nat
		mols_done += 1
		#
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		matrices = []
		outputpointer = 0
		for i in range (0, natoms):
			sto[self.TData.eles.index(meta[i, 1])] += 1
		currentmol = 0
		for e in range (len(self.TData.eles)):
				inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
				matrices.append(np.zeros((sto[e], nmols)))
		for i in range (0, natoms):
			if currentmol != meta[i, 0]:
				outputpointer += 1
				currentmol = meta[i, 0]
			e = meta[i, 1]
			ei = self.TData.eles.index(e)
			inputs[ei][offsets[ei], :] = cases[i]
			matrices[ei][offsets[ei], outputpointer] = 1.0
			offsets[ei] += 1
		t = time.time()
		pointers = [0 for ele in self.TData.eles]
		mol_out, atom_out = self.Instances.evaluate([inputs, matrices, dummy_outputs],IfGrad=False)
		total = mol_out[0][0]
		for j in range (0, mol.NAtoms()):
			total += ele_U[mol.atoms[j]]
		return total

	def Eval_BPForceSet(self, mol_set, total_energy = False):
		"""
		Args:
			mol_set: a MSet
			total_energy: whether to also return the energy as a first argument.
		Returns:
			(if total_energy == True): Energy in Hartree
			and Forces (J/mol)
		"""
		nmols = len(mol_set.mols)
		natoms = mol_set.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
		cases_grads = []
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((natoms, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		t = time.time()
		for mol in mol_set.mols:
			ins, grads = self.TData.dig.EvalDigest(mol,True)
			#print "ins, grads", ins.shape, grads.shape
			nat = mol.NAtoms()
			cases[casep:casep+nat] = ins
			cases_grads += list(grads)
			for i in range (casep, casep+nat):
				meta[i, 0] = mols_done
				meta[i, 1] = mol.atoms[i - casep]
				meta[i, 2] = casep
				meta[i, 3] = casep + nat
			casep += nat
			mols_done += 1
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		matrices = []
		inputs_grads = [[] for i in range (len(self.TData.eles))]
		outputpointer = 0
		for i in range (0, natoms):
			sto[self.TData.eles.index(meta[i, 1])] += 1
			currentmol = 0
			for e in range (len(self.TData.eles)):
				inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
				matrices.append(np.zeros((sto[e], nmols)))
				atom_index_in_mol = [[] for i in range (len(self.TData.eles))]
				for i in range (0, natoms):
					if currentmol != meta[i, 0]:
						outputpointer += 1
						currentmol = meta[i, 0]
					e = meta[i, 1]
					ei = self.TData.eles.index(e)
					inputs[ei][offsets[ei], :] = cases[i]
		inputs_grads[ei].append(cases_grads[i])
		#inputs_grads[ei][offsets[ei], :]  = cases_grads[i]
		matrices[ei][offsets[ei], outputpointer] = 1.0
		atom_index_in_mol[ei].append(currentmol)
		offsets[ei] += 1
		print(("data prepare cost:", time.time() -t))
		t = time.time()
		pointers = [0 for ele in self.TData.eles]
		mol_out, atom_out, nn_gradient = self.Instances.evaluate([inputs, matrices, dummy_outputs],IfGrad=True)
		print(("actual evaluation cost:", time.time() -t))
		t = time.time()
		total_gradient_list = []
		total_energy_list = []
		for i in range (0, nmols):
			total = mol_out[0][i]
			mol = mol_set.mols[i]
			total_gradient = np.zeros((mol.NAtoms()*3))
			for j, ele in enumerate(self.TData.eles):
				ele_index = [k for k, tmp_index in enumerate(atom_index_in_mol[j]) if tmp_index == i]
				ele_desp_grads = np.asarray([ tmp_array for k, tmp_array in enumerate(inputs_grads[j]) if k in ele_index])
				ele_nn_grads = np.asarray([ tmp_array for k, tmp_array in enumerate(nn_gradient[j]) if k in ele_index])
				total_gradient += np.einsum("ad,adx->x", ele_nn_grads, ele_desp_grads) # Chain rule.
		total_gradient_list.append(-JOULEPERHARTREE*total_gradient.reshape((-1,3)))
		#total_gradient_list.append(-total_gradient.reshape((-1,3)))
		if total_energy:
			for j in range (0, mol.NAtoms()):
				total += ele_U[mol.atoms[j]]
			total_energy_list.append(total)
		else:
			total_energy_list.append(total)
		print(("recombine molecule cost:", time.time() -t))
		return total_energy_list, total_gradient_list

	def Eval_BPForceSingle(self, mol, total_energy = False):
		"""
		Args:
			mol: a Mol.
			total_energy: whether to also return the energy as a first argument.
		Returns:
			(if total_energy == True): Energy in Hartree
			and Forces (J/mol)
		"""
		nmols = 1
		natoms = mol.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
		cases_grads = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)+list([3*natoms])))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((natoms, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		t = time.time()
		# Fictitious set loop.
		ins, grads = self.TData.dig.EvalDigest(mol)
		nat = mol.NAtoms()
		cases[casep:casep+nat] = ins
		cases_grads[casep:casep+nat] = grads
		for i in range (casep, casep+nat):
			meta[i, 0] = mols_done
			meta[i, 1] = mol.atoms[i - casep]
			meta[i, 2] = casep
			meta[i, 3] = casep + nat
		#casep += nat
		#mols_done += 1
		# End fictitious set loop.
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		inputs_grads = []
		matrices = []
		outputpointer = 0
		for i in range (0, natoms):
			sto[self.TData.eles.index(meta[i, 1])] += 1
		currentmol = 0
		for e in range (len(self.TData.eles)):
			inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
			inputs_grads.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape), 3*natoms)))
			matrices.append(np.zeros((sto[e], nmols)))
		for i in range (0, natoms):
			if currentmol != meta[i, 0]:
				outputpointer += 1
				currentmol = meta[i, 0]
			e = meta[i, 1]
			ei = self.TData.eles.index(e)
			inputs[ei][offsets[ei], :] = cases[i]
			inputs_grads[ei][offsets[ei], :]  = cases_grads[i]
			matrices[ei][offsets[ei], outputpointer] = 1.0
			offsets[ei] += 1
		t = time.time()
		mol_out, atom_out, nn_gradient = self.Instances.evaluate([inputs, matrices, dummy_outputs],IfGrad=True)
		total_gradient = np.zeros((natoms*3))
		#print ("atom_out\n:", atom_out)
		for i in range (0, len(nn_gradient)): # Loop over element types.
			total_gradient += np.einsum("ad,adx->x",nn_gradient[i],inputs_grads[i]) # Chain rule.
			#print "atom_grads: \n", np.einsum("ad,adx->ax",nn_gradient[i],inputs_grads[i])
		if (total_energy):
			total = mol_out[0][0]
			for j in range (0, mol.NAtoms()):
				total += ele_U[mol.atoms[j]]
			#return total,total_gradient.reshape((-1,3))
			return  total, (-JOULEPERHARTREE*total_gradient.reshape((-1,3)))
		else:
			#return total_gradient.reshape((-1,3))
			return  (-JOULEPERHARTREE*total_gradient.reshape((-1,3)))

	def Eval_BPForceHalfNumerical(self, mol, total_energy = False):
		"""
		This version uses a half-numerical gradient.
		It was written for debugging purposes.
		Args:
			mol: a Mol.
			total_energy: whether to also return the energy as a first argument.
		Returns:
			(if total_energy == True): Energy in Hartree
			and Forces (kcal/mol)
		"""
		nmols = 1
		natoms = mol.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((natoms, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		t = time.time()
		# Fictitious set loop.
		ins = self.TData.dig.EvalDigest(mol,False)
		nat = mol.NAtoms()
		cases[casep:casep+nat] = ins
		for i in range (casep, casep+nat):
			meta[i, 0] = mols_done
			meta[i, 1] = mol.atoms[i - casep]
			meta[i, 2] = casep
			meta[i, 3] = casep + nat
		casep += nat
		mols_done += 1
		# End fictitious set loop.
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		matrices = []
		outputpointer = 0
		for i in range (0, natoms):
			sto[self.TData.eles.index(meta[i, 1])] += 1
		currentmol = 0
		for e in range (len(self.TData.eles)):
			inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
			matrices.append(np.zeros((sto[e], nmols)))
		for i in range (0, natoms):
			if currentmol != meta[i, 0]:
				outputpointer += 1
				currentmol = meta[i, 0]
			e = meta[i, 1]
			ei = self.TData.eles.index(e)
			inputs[ei][offsets[ei], :] = cases[i]
			matrices[ei][offsets[ei], outputpointer] = 1.0
			offsets[ei] += 1
		t = time.time()
		mol_out, atom_out, nn_gradient = self.Instances.evaluate([inputs, matrices, dummy_outputs],IfGrad=True)
		total_gradient = np.zeros((natoms*3))
		for i in range (0, len(nn_gradient)): # Loop over element types.
			Eval_Input = lambda x_: self.Eval_Input(Mol(mol.atoms,x_.reshape((-1,3))))[i]
			input_grad = np.transpose(FdiffGradient(Eval_Input, mol.coords.flatten(), 0.00001),(1,2,0))
			total_gradient += np.einsum("ad,adx->x",nn_gradient[i],input_grad) # Chain rule.
		if (total_energy):
			total = mol_out[0][0]
			for j in range (0, mol.NAtoms()):
				total += ele_U[mol.atoms[j]]
			return  total, (-JOULEPERHARTREE*total_gradient.reshape((-1,3)))
		else:
			return  (-JOULEPERHARTREE*total_gradient.reshape((-1,3)))

	def Eval_Input(self, mol):
		"""
		Args:
			mol: a Mol.
			total_energy: whether to also return the energy as a first argument.
		Returns:
			(if total_energy == True): Energy in Hartree
			and Forces (kcal/mol)
		"""
		nmols = 1
		natoms = mol.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
		cases_grads = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)+list([3*natoms])))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((natoms, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		t = time.time()
		# Fictitious set loop.
		ins, grads = self.TData.dig.EvalDigest(mol)
		nat = mol.NAtoms()
		cases[casep:casep+nat] = ins
		cases_grads[casep:casep+nat] = grads
		for i in range (casep, casep+nat):
			meta[i, 0] = mols_done
			meta[i, 1] = mol.atoms[i - casep]
			meta[i, 2] = casep
			meta[i, 3] = casep + nat
		casep += nat
		mols_done += 1
		# End fictitious set loop.
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		inputs_grads = []
		outputpointer = 0
		for i in range (0, natoms):
			sto[self.TData.eles.index(meta[i, 1])] += 1
		currentmol = 0
		for e in range (len(self.TData.eles)):
			inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
		for i in range (0, natoms):
			if currentmol != meta[i, 0]:
				outputpointer += 1
				currentmol = meta[i, 0]
			e = meta[i, 1]
			ei = self.TData.eles.index(e)
			inputs[ei][offsets[ei], :] = cases[i]
			offsets[ei] += 1
		return inputs

	def Eval_InputGrad(self, mol):
		"""
		Args:
			mol: a Mol.
			total_energy: whether to also return the energy as a first argument.
		Returns:
			(if total_energy == True): Energy in Hartree
			and Forces (kcal/mol)
		"""
		nmols = 1
		natoms = mol.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
		cases_grads = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)+list([3*natoms])))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((natoms, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		t = time.time()
		# Fictitious set loop.
		ins, grads = self.TData.dig.EvalDigest(mol)
		nat = mol.NAtoms()
		cases[casep:casep+nat] = ins
		cases_grads[casep:casep+nat] = grads
		for i in range (casep, casep+nat):
			meta[i, 0] = mols_done
			meta[i, 1] = mol.atoms[i - casep]
			meta[i, 2] = casep
			meta[i, 3] = casep + nat
		casep += nat
		mols_done += 1
		# End fictitious set loop.
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		inputs_grads = []
		matrices = []
		outputpointer = 0
		for i in range (0, natoms):
			sto[self.TData.eles.index(meta[i, 1])] += 1
		currentmol = 0
		for e in range (len(self.TData.eles)):
			inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
			inputs_grads.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape), 3*natoms)))
			matrices.append(np.zeros((sto[e], nmols)))
		for i in range (0, natoms):
			if currentmol != meta[i, 0]:
				outputpointer += 1
				currentmol = meta[i, 0]
			e = meta[i, 1]
			ei = self.TData.eles.index(e)
			inputs[ei][offsets[ei], :] = cases[i]
			inputs_grads[ei][offsets[ei], :]  = cases_grads[i]
			matrices[ei][offsets[ei], outputpointer] = 1.0
			offsets[ei] += 1
		return inputs_grads

	def Test_BPGrad(self,mol):
		"""
		Computes the gradient a couple different ways. Compares them.
		"""
		EnergyFunction = lambda x_: JOULEPERHARTREE*self.Eval_BPEnergySingle(Mol(mol.atoms,x_))
		NumForce0 = FdiffGradient(EnergyFunction, mol.coords, 0.1)
		NumForce1 = FdiffGradient(EnergyFunction, mol.coords, 0.01)
		NumForce2 = FdiffGradient(EnergyFunction, mol.coords, 0.001)
		NumForce3 = FdiffGradient(EnergyFunction, mol.coords, 0.0001)
		print("Force Differences", RmsForce(NumForce1-NumForce0))
		print("Force Differences", RmsForce(NumForce2-NumForce1))
		print("Force Differences", RmsForce(NumForce3-NumForce2))
		print("Force Differences", RmsForce(NumForce3-NumForce1))
		AnalForce = self.Eval_BPForceSingle( mol, total_energy = False)
		HalfAnalForce = self.Eval_BPForceHalfNumerical( mol, total_energy = False)
		print("Force Differences2", RmsForce(NumForce0-AnalForce))
		print("Force Differences2", RmsForce(NumForce1-AnalForce))
		print("Force Differences2", RmsForce(NumForce2-AnalForce))
		print("Force Differences2", RmsForce(NumForce3-AnalForce))
		print("Force Differences3", RmsForce(NumForce0-HalfAnalForce))
		print("Force Differences3", RmsForce(NumForce1-HalfAnalForce))
		print("Force Differences3", RmsForce(NumForce2-HalfAnalForce))
		print("Force Differences3", RmsForce(NumForce3-HalfAnalForce))
		print("Force Differences4", RmsForce(AnalForce-HalfAnalForce))
		print("Numerical force 0 / Analytical force", NumForce0/AnalForce)
		print("Numerical force 1 / Analytical force", NumForce1/AnalForce)
		print("HalfAnalForce / Analytical force", HalfAnalForce/AnalForce)
		if (0):
			print("Testing chain rule components... ")
			tmp = self.Eval_InputGrad(mol)
			for ele in range(len(tmp)):
				Eval_Input = lambda x_: self.Eval_Input(Mol(mol.atoms,x_.reshape((-1,3))))[ele]
				Analyticaldgdr = self.Eval_InputGrad(mol)[ele]
				Numericaldgdr0 = np.transpose(FdiffGradient(Eval_Input, mol.coords.flatten(), 0.01),(1,2,0))
				Numericaldgdr1 = np.transpose(FdiffGradient(Eval_Input, mol.coords.flatten(), 0.001),(1,2,0))
				Numericaldgdr2 = np.transpose(FdiffGradient(Eval_Input, mol.coords.flatten(), 0.0001),(1,2,0))
				Numericaldgdr3 = np.transpose(FdiffGradient(Eval_Input, mol.coords.flatten(), 0.00001),(1,2,0))
				print("Shapes", Analyticaldgdr.shape, Numericaldgdr1.shape)
				for i in range(Analyticaldgdr.shape[0]):
					for j in range(Analyticaldgdr.shape[1]):
						for k in range(Analyticaldgdr.shape[2]):
							if (abs(Analyticaldgdr[i,j,k])>0.0000000001):
								if (abs((Analyticaldgdr[i,j,k]/Numericaldgdr2[i,j,k])-1.)>0.05):
									print(ele,i,j,k," :: ",Analyticaldgdr[i,j,k]," ", Numericaldgdr0[i,j,k]," ", Numericaldgdr1[i,j,k]," ", Numericaldgdr2[i,j,k])

	def Eval_BPDipole(self, mol_set,  ScaleCharge_ = False):
		"""
		can take either a single mol or mol set
		return netcharge, dipole, atomcharge
		Dipole has unit in debye
		"""
		eles = self.Instances.eles
		if isinstance(mol_set, Mol):
			tmp = MSet()
			tmp.mols = [mol_set]
			mol_set = tmp
			nmols = len(mol_set.mols)
			natoms = mol_set.NAtoms()
			cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
			dummy_outputs = np.zeros((nmols, 4))
			meta = np.zeros((natoms, 4), dtype = np.int)
			xyzmeta = np.zeros((natoms, 3))
			casep = 0
			mols_done = 0
			t = time.time()
			for mol in mol_set.mols:
				ins, grads = self.TData.dig.EvalDigest(mol)
				nat = mol.NAtoms()
				xyz_centered = mol.coords - np.average(mol.coords, axis=0)
				cases[casep:casep+nat] = ins
				for i in range (casep, casep+nat):
					meta[i, 0] = mols_done
					meta[i, 1] = mol.atoms[i - casep]
					meta[i, 2] = casep
					meta[i, 3] = casep + nat
					xyzmeta[i] = xyz_centered[i - casep]
				casep += nat
				mols_done += 1
				sto = np.zeros(len(eles),dtype = np.int32)
				offsets = np.zeros(len(eles),dtype = np.int32)
				inputs = []
				matrices = []
				xyz = []
				outputpointer = 0
				for i in range (0, natoms):
					sto[self.TData.eles.index(meta[i, 1])] += 1
				currentmol = 0
				for e in range (len(eles)):
					inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
					matrices.append(np.zeros((sto[e], nmols)))
					xyz.append(np.zeros((sto[e], 3)))
				for i in range (0, natoms):
					if currentmol != meta[i, 0]:
						outputpointer += 1
						currentmol = meta[i, 0]
					e = meta[i, 1]
					ei = eles.index(e)
					inputs[ei][offsets[ei], :] = cases[i]
					matrices[ei][offsets[ei], outputpointer] = 1.0
					xyz[ei][offsets[ei]] = xyzmeta[i]
					offsets[ei] += 1
				t = time.time()
				netcharge, dipole, atomcharge = self.Instances.evaluate([inputs, matrices, xyz, dummy_outputs])
		molatomcharge = []
		pointers = [0 for ele in eles]
		for i, mol in enumerate(mol_set.mols):
			tmp_atomcharge = np.zeros(mol.NAtoms())
			for j in range (0, mol.NAtoms()):
				atom_type = mol.atoms[j]
				atom_index = eles.index(atom_type)
				tmp_atomcharge[j] = atomcharge[atom_index][0][pointers[atom_index]]
				pointers[atom_index] +=1
			molatomcharge.append(tmp_atomcharge)
		if ScaleCharge_:
			sdipole = np.zeros((dipole.shape[0], 3))
			smolatomcharge = []
			pointers = [0 for ele in eles]
			for i, mol in enumerate(mol_set.mols):
				tmp_atomcharge = molatomcharge[i]
				tmp_atomcharge = tmp_atomcharge - netcharge[i]/mol.NAtoms()
				smolatomcharge.append(tmp_atomcharge)
				center_ = np.average(mol.coords,axis=0)
				sdipole[i] = np.einsum("ax,a", mol.coords - center_, tmp_atomcharge)/AUPERDEBYE
			return  np.zeros(nmols), sdipole, smolatomcharge
		return netcharge, dipole, molatomcharge


	def Eval_BPDipole_2(self, mol_set,  ScaleCharge_ = False):
		"""
		can take either a single mol or mol set
		return dipole, atomcharge
		Dipole has unit in debye
		"""
		eles = self.Instances.eles
		if isinstance(mol_set, Mol):
			tmp = MSet()
			tmp.mols = [mol_set]
			mol_set = tmp
			nmols = len(mol_set.mols)
			natoms = mol_set.NAtoms()
			cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
			dummy_outputs = np.zeros((nmols, 3))
			natom_in_mol = np.zeros((nmols, 1))
			natom_in_mol.fill(float('inf'))
			meta = np.zeros((natoms, 4), dtype = np.int)
			xyzmeta = np.zeros((natoms, 3))
			casep = 0
			mols_done = 0
			t = time.time()
			for mol in mol_set.mols:
				ins, grads = self.TData.dig.EvalDigest(mol)
				nat = mol.NAtoms()
				xyz_centered = mol.coords - np.average(mol.coords, axis=0)
				cases[casep:casep+nat] = ins
				for i in range (casep, casep+nat):
					meta[i, 0] = mols_done
					meta[i, 1] = mol.atoms[i - casep]
					meta[i, 2] = casep
					meta[i, 3] = casep + nat
					xyzmeta[i] = xyz_centered[i - casep]
				casep += nat
				mols_done += 1
				sto = np.zeros(len(eles),dtype = np.int32)
				offsets = np.zeros(len(eles),dtype = np.int32)
				inputs = []
				matrices = []
				xyz = []
				natom = []
				outputpointer = 0
				for i in range (0, natoms):
					sto[self.TData.eles.index(meta[i, 1])] += 1
				currentmol = 0
				for e in range (len(eles)):
					inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
					matrices.append(np.zeros((sto[e], nmols)))
					xyz.append(np.zeros((sto[e], 3)))
				for i in range (0, natoms):
					if currentmol != meta[i, 0]:
						outputpointer += 1
						currentmol = meta[i, 0]
					e = meta[i, 1]
					ei = eles.index(e)
					inputs[ei][offsets[ei], :] = cases[i]
					matrices[ei][offsets[ei], outputpointer] = 1.0
					xyz[ei][offsets[ei]] = xyzmeta[i]
					natom_in_mol[outputpointer] = meta[i,3] - meta[i,2]
					offsets[ei] += 1
				t = time.time()
				dipole, atomcharge = self.Instances.evaluate([inputs, matrices, xyz, 1.0/natom_in_mol, dummy_outputs])
		elif (mol_set, MSet):
			nmols = len(mol_set.mols)
			natoms = mol_set.NAtoms()
			print("number of molecules in the set:", nmols)
			cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
			dummy_outputs = np.zeros((nmols, 3))
			natom_in_mol = np.zeros((nmols, 1))
			natom_in_mol.fill(float('inf'))
			meta = np.zeros((natoms, 4), dtype = np.int)
			xyzmeta = np.zeros((natoms, 3))
			casep = 0
			mols_done = 0
			t = time.time()
			for mol in mol_set.mols:
				ins, grads = self.TData.dig.EvalDigest(mol)
				nat = mol.NAtoms()
				xyz_centered = mol.coords - np.average(mol.coords, axis=0)
				cases[casep:casep+nat] = ins
				for i in range (casep, casep+nat):
					meta[i, 0] = mols_done
					meta[i, 1] = mol.atoms[i - casep]
					meta[i, 2] = casep
					meta[i, 3] = casep + nat
					xyzmeta[i] = xyz_centered[i - casep]
				casep += nat
				mols_done += 1
			sto = np.zeros(len(eles),dtype = np.int32)
			offsets = np.zeros(len(eles),dtype = np.int32)
			inputs = []
			matrices = []
			xyz = []
			natom = []
			outputpointer = 0
			for i in range (0, natoms):
				sto[self.TData.eles.index(meta[i, 1])] += 1
			currentmol = 0
			for e in range (len(eles)):
				inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
				matrices.append(np.zeros((sto[e], nmols)))
				xyz.append(np.zeros((sto[e], 3)))
			for i in range (0, natoms):
				if currentmol != meta[i, 0]:
					outputpointer += 1
					currentmol = meta[i, 0]
				e = meta[i, 1]
				ei = eles.index(e)
				inputs[ei][offsets[ei], :] = cases[i]
				matrices[ei][offsets[ei], outputpointer] = 1.0
				xyz[ei][offsets[ei]] = xyzmeta[i]
				natom_in_mol[outputpointer] = meta[i,3] - meta[i,2]
				offsets[ei] += 1
			t = time.time()
			dipole, atomcharge  = self.Instances.evaluate([inputs, matrices, xyz, 1.0/natom_in_mol, dummy_outputs], False)
			#print dipole, atomcharge
			#print  atomcharge
		else:
			raise Exception("wrong input")
		molatomcharge = []
		pointers = [0 for ele in eles]
		for i, mol in enumerate(mol_set.mols):
			tmp_atomcharge = np.zeros(mol.NAtoms())
			for j in range (0, mol.NAtoms()):
				atom_type = mol.atoms[j]
				atom_index = eles.index(atom_type)
				tmp_atomcharge[j] = atomcharge[atom_index][0][pointers[atom_index]]/BOHRPERA  # hacky way to do
				pointers[atom_index] +=1
			molatomcharge.append(tmp_atomcharge)
		return dipole, molatomcharge

	def Eval_BPDipoleGrad_2(self, mol_set,  ScaleCharge_ = False):
		"""
		can take either a single mol or mol set
		return dipole, atomcharge, gradient of the atomcharge
		Dipole has unit in debye
		"""
		eles = self.Instances.eles
		if isinstance(mol_set, Mol):
			tmp = MSet()
			tmp.mols = [mol_set]
			mol_set = tmp
			nmols = len(mol_set.mols)
			natoms = mol_set.NAtoms()
			cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
			cases_grads = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)+list([3*natoms])))
			dummy_outputs = np.zeros((nmols, 3))
			natom_in_mol = np.zeros((nmols, 1))
			natom_in_mol.fill(float('inf'))
			meta = np.zeros((natoms, 4), dtype = np.int)
			xyzmeta = np.zeros((natoms, 3))
			casep = 0
			mols_done = 0
			for mol in mol_set.mols:
				ins, grads = self.TData.dig.EvalDigest(mol)
				nat = mol.NAtoms()
				xyz_centered = mol.coords - np.average(mol.coords, axis=0)
				cases[casep:casep+nat] = ins
				cases_grads[casep:casep+nat] = grads
				for i in range (casep, casep+nat):
					meta[i, 0] = mols_done
					meta[i, 1] = mol.atoms[i - casep]
					meta[i, 2] = casep
					meta[i, 3] = casep + nat
					xyzmeta[i] = xyz_centered[i - casep]
				casep += nat
				mols_done += 1
				sto = np.zeros(len(eles),dtype = np.int32)
				offsets = np.zeros(len(eles),dtype = np.int32)
				inputs = []
				inputs_grads = []
				matrices = []
				xyz = []
				natom = []
				outputpointer = 0
				for i in range (0, natoms):
					sto[self.TData.eles.index(meta[i, 1])] += 1
				currentmol = 0
				for e in range (len(eles)):
					inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
					inputs_grads.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape), 3*natoms)))
					matrices.append(np.zeros((sto[e], nmols)))
					xyz.append(np.zeros((sto[e], 3)))
				for i in range (0, natoms):
					if currentmol != meta[i, 0]:
						outputpointer += 1
						currentmol = meta[i, 0]
					e = meta[i, 1]
					ei = eles.index(e)
					inputs[ei][offsets[ei], :] = cases[i]
					inputs_grads[ei][offsets[ei], :]  = cases_grads[i]
					matrices[ei][offsets[ei], outputpointer] = 1.0
					xyz[ei][offsets[ei]] = xyzmeta[i]
					natom_in_mol[outputpointer] = meta[i,3] - meta[i,2]
					offsets[ei] += 1
				t = time.time()
				dipole, atomcharge, charge_gradients = self.Instances.evaluate([inputs, matrices, xyz, 1.0/natom_in_mol, dummy_outputs], True)
				total_unscaled_gradient = []
				for i in range (0, len(charge_gradients)): # Loop over element types.
					total_unscaled_gradient += list(np.einsum("aij,ai->aj", inputs_grads[i],  charge_gradients[i])) # Chain rule.
				total_unscaled_gradient  = np.asarray(total_unscaled_gradient)/BOHRPERA  # hacky way to do
				total_scaled_gradient =  total_unscaled_gradient - np.sum(total_unscaled_gradient, axis=0)/total_unscaled_gradient.shape[0]
				total_scaled_gradient_list = []
				ele_pointer = 0
				for i in range (0, len(charge_gradients)):
					total_scaled_gradient_list.append(total_scaled_gradient[ele_pointer:ele_pointer+charge_gradients[i].shape[0]])
					ele_pointer += charge_gradients[i].shape[0]

		elif (mol_set, MSet):
			nmols = len(mol_set.mols)
			natoms = mol_set.NAtoms()
			cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
			cases_grads = []
			dummy_outputs = np.zeros((nmols, 3))
			natom_in_mol = np.zeros((nmols, 1))
			natom_in_mol.fill(float('inf'))
			meta = np.zeros((natoms, 4), dtype = np.int)
			xyzmeta = np.zeros((natoms, 3))
			casep = 0
			mols_done = 0
			t = time.time()
			for mol in mol_set.mols:
				ins, grads = self.TData.dig.EvalDigest(mol)
				nat = mol.NAtoms()
				xyz_centered = mol.coords - np.average(mol.coords, axis=0)
				cases[casep:casep+nat] = ins
				cases_grads += list(grads)
				for i in range (casep, casep+nat):
					meta[i, 0] = mols_done
					meta[i, 1] = mol.atoms[i - casep]
					meta[i, 2] = casep
					meta[i, 3] = casep + nat
					xyzmeta[i] = xyz_centered[i - casep]
				casep += nat
				mols_done += 1
			sto = np.zeros(len(eles),dtype = np.int32)
			offsets = np.zeros(len(eles),dtype = np.int32)
			inputs = []
			inputs_grads = [[] for i in range (len(self.TData.eles))]
			matrices = []
			xyz = []
			natom = []
			outputpointer = 0
			for i in range (0, natoms):
				sto[self.TData.eles.index(meta[i, 1])] += 1
			currentmol = 0
			for e in range (len(eles)):
				inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
				matrices.append(np.zeros((sto[e], nmols)))
				xyz.append(np.zeros((sto[e], 3)))
			atom_index_in_mol = [[] for i in range (len(self.TData.eles))]
			for i in range (0, natoms):
				if currentmol != meta[i, 0]:
					outputpointer += 1
					currentmol = meta[i, 0]
				e = meta[i, 1]
				ei = eles.index(e)
				inputs[ei][offsets[ei], :] = cases[i]
				inputs_grads[ei].append(cases_grads[i])
				matrices[ei][offsets[ei], outputpointer] = 1.0
				xyz[ei][offsets[ei]] = xyzmeta[i]
				natom_in_mol[outputpointer] = meta[i,3] - meta[i,2]
				atom_index_in_mol[ei].append(currentmol)
				offsets[ei] += 1
			t = time.time()
			dipole, atomcharge, charge_gradients  = self.Instances.evaluate([inputs, matrices, xyz, 1.0/natom_in_mol, dummy_outputs], True)
			total_scaled_gradient_list = []
			for i in range (0, nmols):
				mol = mol_set.mols[i]
				total_unscaled_gradient=[]
				n_ele_in_mol = []
				for j, ele in enumerate(self.TData.eles):
					ele_index = [k for k, tmp_index in enumerate(atom_index_in_mol[j]) if tmp_index == i]
					n_ele_in_mol.append(len(ele_index))
					ele_desp_grads = np.asarray([ tmp_array for k, tmp_array in enumerate(inputs_grads[j]) if k in ele_index])
					ele_nn_grads = np.asarray([ tmp_array for k, tmp_array in enumerate(charge_gradients[j]) if k in ele_index])
					total_unscaled_gradient += list(np.einsum("ad,adx->ax", ele_nn_grads, ele_desp_grads))
					total_unscaled_gradient  = np.asarray(total_unscaled_gradient)/BOHRPERA  #hack way to do
					total_scaled_gradient =  total_unscaled_gradient - np.sum(total_unscaled_gradient, axis=0)/total_unscaled_gradient.shape[0]
					total_scaled_gradient_tmp = []
					ele_pointer = 0
				for j in range (0, len(n_ele_in_mol)):
					total_scaled_gradient_tmp.append(total_scaled_gradient[ele_pointer:ele_pointer+n_ele_in_mol[j]])
					ele_pointer += n_ele_in_mol[j]
			total_scaled_gradient_list.append(total_scaled_gradient_tmp)
			total_scaled_gradient_list_tmp = [[] for i in range (len(self.TData.eles))]
			for tmp in total_scaled_gradient_list:
				for e in range (len(eles)):
					total_scaled_gradient_list_tmp[e] += list(tmp[e])
					total_scaled_gradient_list = total_scaled_gradient_list_tmp
		else:
			raise Exception("wrong input")
		molatomcharge = []
		molatomcharge_gradient  = []
		pointers = [0 for ele in eles]
		for i, mol in enumerate(mol_set.mols):
			tmp_atomcharge = np.zeros(mol.NAtoms())
			tmp_atomcharge_gradient = np.zeros((mol.NAtoms(), mol.NAtoms(), 3))
			for j in range (0, mol.NAtoms()):
				atom_type = mol.atoms[j]
				atom_index = eles.index(atom_type)
				tmp_atomcharge[j] = atomcharge[atom_index][0][pointers[atom_index]]/BOHRPERA  #hacky way to do
				tmp_atomcharge_gradient[j] = total_scaled_gradient_list[atom_index][pointers[atom_index]].reshape((-1,3))
				pointers[atom_index] +=1
			molatomcharge.append(tmp_atomcharge)
			molatomcharge_gradient.append(tmp_atomcharge_gradient)
		return dipole, molatomcharge, molatomcharge_gradient


	def Eval_Bond_BP(self, mol_set, total_energy = False):
		nmols = len(mol_set.mols)
		nbonds = mol_set.NBonds()
		cases = np.zeros(tuple([nbonds]+list(self.TData.dig.eshape)))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((nbonds, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		for mol in mol_set.mols:
			ins = self.TData.dig.EvalDigest(mol)
			nbo = mol.NBonds()
			cases[casep:casep+nbo] = ins
			for i in range (casep, casep+nbo):
				meta[i, 0] = mols_done
				meta[i, 1] = mol.bonds[i - casep,0]
				meta[i, 2] = casep
				meta[i, 3] = casep + nbo
			casep += nbo
			mols_done += 1
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		matrices = []
		outputpointer = 0
		for i in range (0, nbonds):
			sto[self.TData.eles.index(meta[i, 1])] += 1
		currentmol = 0
		for e in range (len(self.TData.eles)):
			inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
			matrices.append(np.zeros((sto[e], nmols)))
		for i in range (0, nbonds):
			if currentmol != meta[i, 0]:
				outputpointer += 1
				currentmol = meta[i, 0]
			e = meta[i, 1]
			ei = self.TData.eles.index(e)
			inputs[ei][offsets[ei], :] = cases[i]
			matrices[ei][offsets[ei], outputpointer] = 1.0
			offsets[ei] += 1
		#print "[inputs, matrices, dummy_outputs]", [inputs, matrices, dummy_outputs]
		mol_out, atom_out = self.Instances.evaluate([inputs, matrices, dummy_outputs])

		pointers = [0 for ele in self.TData.eles]
		diff = 0
		for i in range (0, nmols):
			mol = mol_set.mols[i]
			print("for mol :", mol.name," energy:", mol.energy)
			print("total atomization energy:", mol_out[0][i])
			#diff += abs(mol.energy - mol_out[0][i])
			if total_energy:
				total = mol_out[0][i]
				for j in range (0, mol.NAtoms()):
					total += ele_U[mol.atoms[j]]
				print("total electronic energy:", total)
			for j in range (0, mol.bonds.shape[0]):
				bond_type = mol.bonds[j, 0]
				bond_index = self.TData.eles.index(bond_type)
				print("bond: ", mol.bonds[j], " energy:", atom_out[bond_index][0][pointers[bond_index]])
				pointers[bond_index] += 1
		#print "mol out:", mol_out, " atom_out", atom_out
		#return	diff / nmols
		return

	def EvalBPPairPotential(self):
		return self.Instances.Evaluate()


	def Eval_Mol(self, mol):
		total_case = len(mol.mbe_frags[self.TData.order])
		if total_case == 0:
			return 0.0
		natom = mol.mbe_frags[self.TData.order][0].NAtoms()
		cases = np.zeros((total_case, self.TData.dig.eshape))
		cases_deri = np.zeros((total_case, natom, natom, 6)) # x1,y1,z1,x2,y2,z2
		casep = 0
		for frag in mol.mbe_frags[self.TData.order]:
			ins, embed_deri =  self.TData.dig.EvalDigest(frag)
			cases[casep:casep+1] += ins
			cases_deri[casep:casep+1]=embed_deri
			casep += 1
		print("evaluating order:", self.TData.order)
		nn, nn_deri=self.Eval(cases)
		#print "nn:",nn, "nn_deri:",nn_deri, "cm_deri:", cases_deri, "cases:",cases, "coord:", mol.coords
		mol.Set_Frag_Force_with_Order(cases_deri, nn_deri, self.TData.order)
		return nn.sum()

	def Eval_BPEnergy_Direct(self, mol_set):
		nmols = len(mol_set.mols)
		dummy_outputs = np.zeros((nmols))
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
		mol_out, atom_out,gradient = self.Instances.evaluate([xyzs, Zs, dummy_outputs], True)
		return mol_out, atom_out, gradient

	def Eval_BPEnergy_Direct_Grad(self, mol, Grad=True, Energy=True):
		mol_set = MSet()
		mol_set.mols.append(mol)
		nmols = len(mol_set.mols)
		self.TData.MaxNAtoms = mol.NAtoms()
		dummy_outputs = np.zeros((nmols))
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
		mol_out, atom_out,gradient = self.Instances.evaluate([xyzs, Zs, dummy_outputs, dummy_grads])
		if Grad and Energy:
			return mol_out[0], -JOULEPERHARTREE*gradient[0][0][:mol.NAtoms()]
		elif Energy and not Grad:
			return mol_out[0]
		else:
			return -JOULEPERHARTREE*gradient[0][0][:mol.NAtoms()]

	def Eval_BPEnergy_Direct_Grad_Linear(self, mol, Grad=True, Energy=True):
		"""
		THIS IS THE only working LINEAR evaluate routine, so far.
		Also generates angular pairs, triples. etc.
		"""
		mol_set = MSet()
		mol_set.mols.append(mol)
		nmols = len(mol_set.mols)
		self.TData.MaxNAtoms = mol.NAtoms()
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
		NLs = NeighborListSet(xyzs, np.array([mol.NAtoms()]), True, True, Zs)
		NLs.Update(xyzs,PARAMS["AN1_r_Rc"],PARAMS["AN1_a_Rc"])
		#print NLs.pairs
		#print NLs.triples
		mol_out, atom_out, gradient = self.Instances.evaluate([xyzs, Zs, NLs.pairs, NLs.triples])
		if Grad and Energy:
			return mol_out[0], -JOULEPERHARTREE*gradient[0][0][:mol.NAtoms()]
		elif Energy and not Grad:
			return mol_out[0]
		else:
			return -JOULEPERHARTREE*gradient[0][0][:mol.NAtoms()]

	def EvalBPDirectSingleEnergyWGrad(self, mol):
		"""
		The energy and force routine for Kun's new direct BPs.
		"""
		mol_set=MSet()
		mol_set.mols.append(mol)
		nmols = len(mol_set.mols)
		dummy_outputs = np.zeros((nmols))
		self.TData.MaxNAtoms = mol.NAtoms()
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
		mol_out, atom_out, gradient = self.Instances.evaluate([xyzs, Zs, dummy_outputs, dummy_grads], True)
		return mol_out[0], -JOULEPERHARTREE*gradient[0][0][:mol.NAtoms()]

	def EvalBPDirectEESingle(self, mol, Rr_cut, Ra_cut, Ree_cut):
		"""
		The energy, force and dipole routine for BPs_EE.
		"""
		mol_set=MSet()
		mol_set.mols.append(mol)
		nmols = len(mol_set.mols)
		dummy_energy = np.zeros((nmols))
		dummy_dipole = np.zeros((nmols, 3))
		self.TData.MaxNAtoms = mol.NAtoms()
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((nmols), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = mol.NAtoms()
		NL = NeighborListSet(xyzs, natom, True, True, Zs)
		rad_p, ang_t = NL.buildPairsAndTriples(Rr_cut, Ra_cut)
		NLEE = NeighborListSet(xyzs, natom, False, False,  None)
		rad_eep = NLEE.buildPairs(Ree_cut)
		Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient  = self.Instances.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p, ang_t, rad_eep, 1.0/natom])
		return Etotal, Ebp, Ecc, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0]

	def EvalBPDirectEESet(self, mol_set, Rr_cut=PARAMS["AN1_r_Rc"], Ra_cut=PARAMS["AN1_a_Rc"], Ree_cut=PARAMS["EECutoffOff"]):
		"""
		The energy, force and dipole routine for BPs_EE.

		Returns:
			Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient
		"""
		nmols = len(mol_set.mols)
		dummy_energy = np.zeros((nmols))
		dummy_dipole = np.zeros((nmols, 3))
		self.TData.MaxNAtoms = mol_set.MaxNAtoms()
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((nmols), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = mol.NAtoms()
		NL = NeighborListSet(xyzs, natom, True, True, Zs)
		rad_p, ang_t = NL.buildPairsAndTriples(Rr_cut, Ra_cut)
		NLEE = NeighborListSet(xyzs, natom, False, False,  None)
		rad_eep = NLEE.buildPairs(Ree_cut)
		Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient  = self.Instances.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p, ang_t, rad_eep, 1.0/natom])
		return Etotal, Ebp, Ecc, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0]

	def EvalBPDirectEEUpdateSingle(self, mol, Rr_cut, Ra_cut, Ree_cut, HasVdw = False):
		"""
		The energy, force and dipole routine for BPs_EE.
		"""
		mol_set=MSet()
		mol_set.mols.append(mol)
		nmols = len(mol_set.mols)
		dummy_energy = np.zeros((nmols))
		dummy_dipole = np.zeros((nmols, 3))
		self.TData.MaxNAtoms = mol.NAtoms()
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((nmols), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = mol.NAtoms()
		NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
		rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(Rr_cut, Ra_cut, self.Instances.eles_np, self.Instances.eles_pairs_np)
		NLEE = NeighborListSet(xyzs, natom, False, False,  None)
		rad_eep = NLEE.buildPairs(Ree_cut)
		if not HasVdw:
			Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient  = self.Instances.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
			return Etotal, Ebp, Ecc, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0]
		else:
			Etotal, Ebp, Ebp_atom, Ecc, Evdw,  mol_dipole, atom_charge, gradient  = self.Instances.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
			return Etotal, Ebp, Ebp_atom ,Ecc, Evdw, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0]

	def EvalBPDirectEEPeriodic(self, mol, Rr_cut, Ra_cut, Ree_cut, nreal_, HasVdw = False):
		"""
		The energy, force and dipole routine for BPs_EE
		This one properly only embeds the energy for real atoms.
		Kun:
		We need to adjust how the graph is made so that only the real atoms have their BP and charge-charge
		interactions evaluated.

		Args:
			mol: a molecule
			Rr_cut: Cutoff for radial pairwise part of the embedding.
			Ra_cut: Cutoff for the angular triples.
			Ree_cut: Cutoff for the electrostatic embedding.
			nreal_: number of non-image atoms. These are the first nreal_ atoms in mol.
		"""
		t0 = time.time()
		mol_set=MSet()
		mol_set.mols.append(mol)
		nmols = len(mol_set.mols)
		dummy_energy = np.zeros((nmols))
		dummy_dipole = np.zeros((nmols, 3))
		nreal = np.zeros((nmols))
		print("NREAL NREAL NREAL", nreal_)
		nreal[0] = nreal_
		self.TData.MaxNAtoms = mol.NAtoms()
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((nmols), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = mol.NAtoms()
		t1 = time.time()
		print("Garbage time", t1-t0)
		NL = NeighborListSetWithImages(xyzs, natom, nreal,True, True, Zs, sort_=True)
		t2 = time.time()
		print("Garbage time1", t2-t1)
		rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(Rr_cut, Ra_cut, self.Instances.eles_np, self.Instances.eles_pairs_np)
		t3 = time.time()
		print("BPNLbuild time", t3-t2)
		NLEE = NeighborListSetWithImages(xyzs, natom, nreal, False, False,  None)
		rad_eep = NLEE.buildPairs(Ree_cut)
		t4 = time.time()
		print("EENLbuild time", t4-t3)
		Etotal, Ebp, Ebp_atom, Ecc, Evdw,  mol_dipole, atom_charge, gradient  = self.Instances.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
		t5 = time.time()
		print("Inference time", t5-t4)
		return Etotal[0], -JOULEPERHARTREE*((gradient[0])[0,:nreal_])

#NL.buildPairsAndTriplesWithEleIndexPeriodic

	def EvalBPDirectEEUpdateSinglePeriodic(self, mol, Rr_cut, Ra_cut, Ree_cut, nreal, HasVdw = True):
		"""
		The energy, force and dipole routine for BPs_EE.
		"""
		mol_set=MSet()
		mol_set.mols.append(mol)
		nmols = len(mol_set.mols)
		dummy_energy = np.zeros((nmols))
		dummy_dipole = np.zeros((nmols, 3))
		self.TData.MaxNAtoms = mol.NAtoms()
		xyzs = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.TData.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.TData.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((nmols), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = nreal   # this is hacky.. K.Y.
		NL = NeighborListSetWithImages(xyzs, np.array([mol.NAtoms()]), np.array([nreal]),True, True, Zs, sort_=True)
		rad_p_ele, ang_t_elep, mil_j, mil_jk = NL.buildPairsAndTriplesWithEleIndexPeriodic(Rr_cut, Ra_cut, self.Instances.eles_np, self.Instances.eles_pairs_np)
		NLEE = NeighborListSetWithImages(xyzs, np.array([mol.NAtoms()]), np.array([nreal]), False, False,  Zs)
		rad_eep_e1e2 = NLEE.buildPairsWithBothEleIndex(Ree_cut, self.Instances.eles_np)
		Etotal, Ebp, Ebp_atom, Ecc, Evdw,  mol_dipole, atom_charge, gradient  = self.Instances.evaluate_periodic([xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, rad_eep_e1e2, mil_j, mil_jk, 1.0/natom], nreal)
		#return Etotal, Ebp, Ebp_atom ,Ecc, Evdw, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0][0][:nreal].reshape(1, nreal, 3)  # be consist with old code
		return Etotal, -JOULEPERHARTREE*gradient[0][0][:nreal].reshape(1, nreal, 3)  # be consist with old code


	def Prepare(self):
		self.Load()
		self.Instances= None # In order of the elements in TData
		if (self.NetType == "fc_classify"):
			self.Instances = MolInstance_fc_classify(None,  self.TrainedNetworks[0], None, Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = MolInstance_fc_sqdiff(None, self.TrainedNetworks[0], None, Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP"):
			self.Instances = MolInstance_fc_sqdiff_BP(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Update"):
			self.Instances = MolInstance_fc_sqdiff_BP_Update(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct"):
			self.Instances = MolInstance_DirectBP_NoGrad(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BPBond_Direct"):
			self.Instances = MolInstance_DirectBPBond_NoGrad(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad"):
			self.Instances = MolInstance_DirectBP_Grad(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad_noGradTrain"):
			self.Instances = MolInstance_DirectBP_Grad_noGradTrain(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad_NewIndex"):
			self.Instances = MolInstance_DirectBP_Grad_NewIndex(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad_Linear"):
			self.Instances = MolInstance_DirectBP_Grad_Linear(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_Grad_Linear_EmbOpt"):
			self.Instances = MolInstance_DirectBP_Grad_Linear_EmbOpt(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE"):
			self.Instances = MolInstance_DirectBP_EE(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_Update"):
			self.Instances = MolInstance_DirectBP_EE_Update(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_ChargeEncode"):
			self.Instances = MolInstance_DirectBP_EE_ChargeEncode(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_ChargeEncode_Update"):
			self.Instances = MolInstance_DirectBP_EE_ChargeEncode_Update(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw"):
			self.Instances = MolInstance_DirectBP_EE_ChargeEncode_Update_vdw(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu"):
			self.Instances = MolInstance_DirectBP_EE_ChargeEncode_Update_vdw_DSF_elu(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "Dipole_BP"):
			self.Instances = MolInstance_BP_Dipole(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "Dipole_BP_2"):
			self.Instances = MolInstance_BP_Dipole_2(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "Dipole_BP_2_Direct"):
			self.Instances = MolInstance_BP_Dipole_2_Direct(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		else:
			raise Exception("Unknown Network Type!")
		# Raise TF instances for each atom which have already been trained.
		return

# This has to be totally re-written to be more like the
# testing in TFInstance.

	def Test(self, save_file="mbe_test.dat"):
		ti, to = self.TData.LoadData( True)
		NTest = int(self.TData.TestRatio * ti.shape[0])
		ti= ti[ti.shape[0]-NTest:]
		to = to[to.shape[0]-NTest:]
		acc_nn = np.zeros((to.shape[0],2))
		nn, gradient=self.Eval(ti)
		acc_nn[:,0]=acc.reshape(acc.shape[0])
		acc_nn[:,1]=nn.reshape(nn.shape[0])
		mean, std = self.TData.Get_Mean_Std()
		acc_nn = acc_nn*std+mean
		np.savetxt(save_file,acc_nn)
		np.savetxt("dist_2b.dat", ti[:,1])
		return
