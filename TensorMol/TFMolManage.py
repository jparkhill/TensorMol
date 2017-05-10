#
# These work Moleculewise the versions without the mol prefix work atomwise.
# but otherwise the behavior of these is the same as TFManage etc.
#
from TFManage import *
from TensorMolData import *
from TFMolInstance import *
from TFMolInstanceEE import *
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
		if (self.TData.dig.eshape==None):
			raise Exception("Must Have Digester")
		# It's up the TensorData to provide the batches and input output shapes.
		if (self.NetType == "fc_classify"):
			self.Instances = MolInstance_fc_classify(self.TData, None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = MolInstance_fc_sqdiff(self.TData, None)
		elif (self.NetType == "fc_sqdiff_BP"):
			self.Instances = MolInstance_fc_sqdiff_BP(self.TData)
		elif (self.NetType == "Dipole_BP"):
			self.Instances = MolInstance_BP_Dipole(self.TData)
		else:
			raise Exception("Unknown Network Type!")
		self.Instances.train(self.n_train) # Just for the sake of debugging.
		nm = self.Instances.name
		# Here we should print some summary of the pupil's progress as well, maybe.
		if self.TrainedNetworks.count(nm)==0:
			self.TrainedNetworks.append(nm)
		self.Save()
		gc.collect()
		return

	def Continue_Training(self, maxsteps):   # test a pretrained network
		self.Instances.TData = self.TData
		self.Instances.TData.LoadDataToScratch(self.Instances.tformer)
		self.Instances.Prepare()
		self.Instances.continue_training(maxsteps)
		self.Save()
		return

	def Eval(self, inputs):
		if (self.Instances[mol_t.atoms[atom]].tformer.innorm != None):
			inputs = self.Instances[mol_t.atoms[atom]].tformer.NormalizeIns(inputs, train=False)
		outputs = self.Instances.evaluate(inputs)
		if (self.Instances[mol_t.atoms[atom]].tformer.outnorm != None):
			outputs = self.Instances[mol_t.atoms[atom]].tformer.UnNormalizeOuts(outputs)
		return outputs

	def EvalBPEnergy(self, mol_set, total_energy = False):
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
		for i in range (0, nmols):
			mol = mol_set.mols[i]
			if total_energy:
				total = mol_out[0][i]
				for j in range (0, mol.NAtoms()):
					total += ele_U[mol.atoms[j]]
			for j in range (0, mol.atoms.shape[0]):
				atom_type = mol.atoms[j]
				atom_index = self.TData.eles.index(atom_type)
				pointers[atom_index] += 1
		return  mol_out[0]

	def Eval_BPForce(self, mol, total_energy = False):
		mol_set = MSet()
		mol_set.mols = [mol]
		nmols = len(mol_set.mols)
		natoms = mol_set.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
		cases_grads = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)+list([3*natoms])))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((natoms, 4), dtype = np.int)
		casep = 0
		mols_done = 0
		t = time.time()
		for mol in mol_set.mols:
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
		mol_out, atom_out, nn_gradient = self.Instances.evaluate([inputs, matrices, dummy_outputs])
		pointers = [0 for ele in self.TData.eles]
		diff = 0
		for i in range (0, nmols):
			mol = mol_set.mols[i]
			if total_energy:
				total = mol_out[0][i]
				for j in range (0, mol.NAtoms()):
					total += ele_U[mol.atoms[j]]
			for j in range (0, mol.atoms.shape[0]):
				atom_type = mol.atoms[j]
				atom_index = self.TData.eles.index(atom_type)
				pointers[atom_index] += 1
		total_gradient = np.zeros((natoms*3))
		for i in range (0, len(nn_gradient)):
			for j in range (0, nn_gradient[i].shape[0]):
				total_gradient += np.sum(np.repeat(nn_gradient[i][j].reshape((nn_gradient[i][j].shape[0], 1)), natoms*3,  axis=1)*inputs_grads[i][j], axis=0)
		if (total_energy):
			return  total, (-627.509*total_gradient).reshape((-1,3))
		else:
			return  (-627.509*total_gradient).reshape((-1,3))

	def EvalBPDipole(self, mol_set,  ScaleCharge_ = False):
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
			print "for mol :", mol.name," energy:", mol.energy
			print "total atomization energy:", mol_out[0][i]
			#diff += abs(mol.energy - mol_out[0][i])
			if total_energy:
				total = mol_out[0][i]
				for j in range (0, mol.NAtoms()):
					total += ele_U[mol.atoms[j]]
				print "total electronic energy:", total
			for j in range (0, mol.bonds.shape[0]):
				bond_type = mol.bonds[j, 0]
				bond_index = self.TData.eles.index(bond_type)
				print "bond: ", mol.bonds[j], " energy:", atom_out[bond_index][0][pointers[bond_index]]
				pointers[bond_index] += 1
		#print "mol out:", mol_out, " atom_out", atom_out
		#return	diff / nmols
		return

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
		print "evaluating order:", self.TData.order
		nn, nn_deri=self.Eval(cases)
		#print "nn:",nn, "nn_deri:",nn_deri, "cm_deri:", cases_deri, "cases:",cases, "coord:", mol.coords
		mol.Set_Frag_Force_with_Order(cases_deri, nn_deri, self.TData.order)
		return nn.sum()

	def Prepare(self):
		self.Load()
		self.Instances= None # In order of the elements in TData
		if (self.NetType == "fc_classify"):
			self.Instances = MolInstance_fc_classify(None,  self.TrainedNetworks[0], None, Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = MolInstance_fc_sqdiff(None, self.TrainedNetworks[0], None, Trainable_ = self.Trainable)
		elif (self.NetType == "fc_sqdiff_BP"):
			self.Instances = MolInstance_fc_sqdiff_BP(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
		elif (self.NetType == "Dipole_BP"):
			self.Instances = MolInstance_BP_Dipole(None,self.TrainedNetworks[0], Trainable_ = self.Trainable)
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
