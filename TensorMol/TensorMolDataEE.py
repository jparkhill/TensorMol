#
# Contains Routines to generate training sets
# Combining a dataset, sampler and an embedding. (CM etc.)
#
# These work Moleculewise the versions without the mol prefix work atomwise.
# but otherwise the behavior of these is the same as Tensordata etc.
#
#
import os, gc
from Sets import *
from DigestMol import *
from TensorMolData import *


class TensorMolData_BP_Multipole(TensorMolData_BP):
	"""
			A tensordata for learning the multipole of molecules using Behler-Parinello scheme.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol"):
		TensorMolData_BP.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
		self.xyzMetadata=None # case X molecule index X element type (Strictly ascending)
		self.scratch_xyzmeta = None
		self.scratch_test_xyzmeta = None
		return

	def CleanScratch(self):
		TensorMolData_BP.CleanScratch(self)
		self.xyzMetadata=None # case X molecule index , element type , first atom in this mol, last atom in this mol (exclusive)
		self.scratch_xyzmeta = None
		self.scratch_test_xyzmeta = None
		return


	def LoadData(self):
		insname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_in.npy"
		outsname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_out.npy"
		metasname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_meta.npy" # Used aggregate
		xyzmetasname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_xyzmeta.npy"
		inf = open(insname,"rb")
		ouf = open(outsname,"rb")
		mef = open(metasname,"rb")
		xyzf = open(xyzmetasname,"rb")
		ti = np.load(inf)
		to = np.load(ouf)
		tm = np.load(mef)
		txyzm = np.load(xyzf)
		inf.close()
		ouf.close()
		mef.close()
		xyzf.close()
		to = to.reshape((to.shape[0],-1))  # flat labels to [mol, 1]
		return ti, to, tm, txyzm


	def LoadDataToScratch(self, tformer):
		"""
		Reads built training data off disk into scratch space.
		Divides training and test data.
		Normalizes inputs and outputs.
			note that modifies my MolDigester to incorporate the normalization
		Initializes pointers used to provide training batches.

		Args:
			random: Not yet implemented randomization of the read data.

		Note:
			Also determines mean stoichiometry
		"""
		if (self.ScratchState == 1):
			return
		ti, to, tm, txyzm = self.LoadData()
		if (tformer.innorm != None):
			ti = tformer.NormalizeIns(ti)
		if (tformer.outnorm != None):
			to = tformer.NormalizeOuts(to)
		self.NTestMols = int(self.TestRatio * to.shape[0])
		self.LastTrainMol = int(to.shape[0]-self.NTestMols)
		LOGGER.debug("LastTrainMol in TensorMolData: %i", self.LastTrainMol)
		LOGGER.debug("NTestMols in TensorMolData: %i", self.NTestMols)
		LOGGER.debug("Number of molecules in meta:: %i", tm[-1,0]+1)
		LastTrainCase=0
		#print tm
		# Figure out the number of atoms in training and test.
		for i in range(len(tm)):
			if (tm[i,0] == self.LastTrainMol):
				LastTrainCase = tm[i,2] # exclusive
				break
		LOGGER.debug("last train atom: %i",LastTrainCase)
		LOGGER.debug("Num Test atoms: %i",len(tm)-LastTrainCase)
		LOGGER.debug("Num atoms: %i",len(tm))
		self.NTrain = LastTrainCase
		self.NTest = len(tm)-LastTrainCase
		self.scratch_inputs = ti[:LastTrainCase]
		self.scratch_outputs = to[:self.LastTrainMol]
		self.scratch_meta = tm[:LastTrainCase]
		self.scratch_xyzmeta = txyzm[:LastTrainCase]
		self.scratch_test_inputs = ti[LastTrainCase:]
		self.scratch_test_outputs = to[self.LastTrainMol:]
		# metadata contains: molecule index, atom type, mol start, mol stop
		# these columns need to be shifted.
		self.scratch_test_meta = tm[LastTrainCase:]
		self.scratch_test_xyzmeta = txyzm[LastTrainCase:]
		self.test_begin_mol = self.scratch_test_meta[0,0]
#		print "before shift case  ", tm[LastTrainCase:LastTrainCase+30], "real", self.set.mols[tm[LastTrainCase, 0]].bonds, self.set.mols[self.test_begin_mol].bonds
		self.scratch_test_meta[:,0] -= self.scratch_test_meta[0,0]
		self.scratch_test_meta[:,3] -= self.scratch_test_meta[0,2]
		self.scratch_test_meta[:,2] -= self.scratch_test_meta[0,2]
		self.ScratchState = 1
		self.ScratchPointer = 0
		self.test_ScratchPointer=0
		# Compute mean Stoichiometry and number of atoms.
		self.eles = np.unique(tm[:,1]).tolist()
		atomcount = np.zeros(len(self.eles))
		self.MeanStoich = np.zeros(len(self.eles))
		for j in range(len(self.eles)):
			for i in range(len(ti)):
				if (tm[i,1]==self.eles[j]):
					atomcount[j]=atomcount[j]+1
		self.MeanStoich=atomcount/len(to)
		self.MeanNumAtoms = np.sum(self.MeanStoich)
		return



	def BuildTrain(self, name_="gdb9",  append=False, max_nmols_=1000000):
		self.CheckShapes()
		self.name=name_
		LOGGER.info("TensorMolData_BP_Multipole, self.type:"+self.type)
		if self.type=="frag":
			raise Exception("No BP frags now")
		nmols  = len(self.set.mols)
		natoms = self.set.NAtoms()
		LOGGER.info( "self.dig.eshape"+str(self.dig.eshape)+" self.dig.lshape"+str(self.dig.lshape))
		cases = np.zeros(tuple([natoms]+list(self.dig.eshape)))
		LOGGER.info( "cases:"+str(cases.shape))
		labels = np.zeros(tuple([nmols]+list(self.dig.lshape)))
		self.CaseMetadata = np.zeros((natoms, 4), dtype = np.int)
		self.xyzMetadata = np.zeros((natoms, 3))
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_in.npy"
		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_out.npy"
		metasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_meta.npy" # Used aggregate and properly sum network inputs and outputs.
		xyzmetasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_xyzmeta.npy" # Used aggregate and properly sum network inputs and outputs.
		casep=0
		# Generate the set in a random order.
		ord=np.random.permutation(len(self.set.mols))
		mols_done = 0
		for mi in ord:
			nat = self.set.mols[mi].NAtoms()
			#print "casep:", casep
			if (mols_done%1000==0):
				LOGGER.info("Mol:"+str(mols_done))
			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
			if not np.all(np.isfinite(ins)):
				print "find a bad case, writting down xyz.."
				self.set.mols[mi].WriteXYZfile(fpath=".", fname="bad_buildset_cases")
			#print mi, ins.shape, outs.shape
			cases[casep:casep+nat] = ins
			labels[mols_done] = outs
			center_xyz = self.set.mols[mi].coords - np.average(self.set.mols[mi].coords, axis=0)
			for j in range(casep,casep+nat):
				self.CaseMetadata[j,0] = mols_done
				self.CaseMetadata[j,1] = self.set.mols[mi].atoms[j-casep]
				self.CaseMetadata[j,2] = casep
				self.CaseMetadata[j,3] = casep+nat
				self.xyzMetadata[j] = center_xyz[j - casep]
			casep += nat
			mols_done = mols_done + 1
			if (mols_done>=max_nmols_):
				break
		inf = open(insname,"wb")
		ouf = open(outsname,"wb")
		mef = open(metasname,"wb")
		xyzf = open(xyzmetasname, "wb")
		np.save(inf,cases[:casep,:])
		np.save(ouf,labels[:mols_done,:])
		np.save(mef,self.CaseMetadata[:casep,:])
		np.save(xyzf, self.xyzMetadata[:casep,:])
		inf.close()
		ouf.close()
		mef.close()
		xyzf.close()
		self.AvailableDataFiles.append([insname,outsname,metasname, xyzmetasname])
		self.Save() #write a convenience pickle.
		return

	def GetTrainBatch(self,ncases,noutputs):
		"""
		Construct the data required for a training batch Returns inputs (sorted by element), and indexing matrices and outputs.
		Behler parinello batches need to have a typical overall stoichiometry.
		and a constant number of atoms, and must contain an integer number of molecules.

		Besides making sure all of that takes place this routine makes the summation matrices
		which map the cases => molecular energies in the Neural Network output.

		Args:
			ncases: the size of a training cases.
			noutputs: the maximum number of molecule energies which can be produced.
		Returns:
			A an **ordered** list of length self.eles containing
				a list of (num_of atom type X flattened input shape) matrix of input cases.
				a list of (num_of atom type X batchsize) matrices which linearly combines the elements
				a list of outputs.
		"""
		start_time = time.time()
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTrain):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.ScratchPointer+ncases >= self.NTrain):
			self.ScratchPointer = 0
		inputs = []#np.zeros((ncases, np.prod(self.dig.eshape)))
		matrices = []#np.zeros((len(self.eles), ncases, noutputs))
		offsets=[]
		coords = []
		# Get the number of molecules which would be contained in the desired batch size
		# and the number of element cases.
		# metadata contains: molecule index, atom type, mol start, mol stop
		bmols=np.unique(self.scratch_meta[self.ScratchPointer:self.ScratchPointer+ncases,0])
		nmols_out=len(bmols[1:-1])
		if (nmols_out > noutputs):
			raise Exception("Insufficent Padding. "+str(nmols_out)+" is greater than "+str(noutputs))
		inputpointer = 0
		outputpointer = 0
		#currentmol=self.scratch_meta[self.ScratchPointer,0]
		sto = np.zeros(len(self.eles),dtype = np.int32)
		offsets = np.zeros(len(self.eles),dtype = np.int32) # output pointers within each element block.
		destinations = np.zeros(ncases) # The index in the output of each case in the scratch.
		ignore_first_mol = 0
		for i in range(self.ScratchPointer,self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			elif (self.scratch_meta[i,0] == bmols[0]):
				ignore_first_mol += 1
			else:
				sto[self.eles.index(self.scratch_meta[i,1])]+=1
		currentmol=self.scratch_meta[self.ScratchPointer+ignore_first_mol,0]
		outputs = np.zeros((noutputs, 4))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			matrices.append(np.zeros((sto[e],noutputs)))
			coords.append(np.zeros((sto[e], 3)))
		for i in range(self.ScratchPointer+ignore_first_mol, self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_meta[i,0]
			# metadata contains: molecule index, atom type, mol start, mol stop
			e = (self.scratch_meta[i,1])
			ei = self.eles.index(e)
			# The offset for this element should be within the bounds or something is wrong...
			inputs[ei][offsets[ei],:] = self.scratch_inputs[i]
			matrices[ei][offsets[ei],outputpointer] = 1.0
			coords[ei][offsets[ei]] = self.scratch_xyzmeta[i]
			outputs[outputpointer] = self.scratch_outputs[self.scratch_meta[i,0]]
			offsets[ei] += 1
		#print "inputs",inputs
		#print "bounds",bounds
		#print "matrices",matrices
		#print "outputs",outputs
		self.ScratchPointer += ncases
		return [inputs, matrices, coords, outputs]

	def GetTestBatch(self,ncases,noutputs):
		"""
			Returns:
			A an **ordered** list of length self.eles containing
				a list of (num_of atom type X flattened input shape) matrix of input cases.
				a list of (num_of atom type X batchsize) matrices which linearly combines the elements
				a list of outputs.
				the number of output molecules.
		"""
		start_time = time.time()
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTest):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.test_ScratchPointer+ncases >= self.NTest):
			self.test_ScratchPointer = 0
			self.test_mols_done = True
		inputs = []#np.zeros((ncases, np.prod(self.dig.eshape)))
		matrices = []#np.zeros((len(self.eles), ncases, noutputs))
		coords = []
		offsets= []
		# Get the number of molecules which would be contained in the desired batch size
		# and the number of element cases.
		# metadata contains: molecule index, atom type, mol start, mol stop
		bmols=np.unique(self.scratch_test_meta[self.test_ScratchPointer:self.test_ScratchPointer+ncases,0])
		nmols_out=len(bmols[1:-1])
		#print "batch contains",nmols_out, "Molecules in ",ncases
		if (nmols_out > noutputs):
			raise Exception("Insufficent Padding. "+str(nmols_out)+" is greater than "+str(noutputs))
		inputpointer = 0
		outputpointer = 0
		#currentmol=self.scratch_meta[self.ScratchPointer,0]
		sto = np.zeros(len(self.eles),dtype = np.int32)
		offsets = np.zeros(len(self.eles),dtype = np.int32) # output pointers within each element block.
		destinations = np.zeros(ncases) # The index in the output of each case in the scratch.
		ignore_first_mol = 0
		for i in range(self.test_ScratchPointer,self.test_ScratchPointer+ncases):
			if (self.scratch_test_meta[i,0] == bmols[-1]):
				break
			elif (self.scratch_test_meta[i,0] == bmols[0]):
				ignore_first_mol += 1
			else:
				sto[self.eles.index(self.scratch_test_meta[i,1])]+=1
		currentmol=self.scratch_test_meta[self.test_ScratchPointer+ignore_first_mol,0]
		outputs = np.zeros((noutputs, 4))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			matrices.append(np.zeros((sto[e],noutputs)))
			coords.append(np.zeros((sto[e], 3)))
		for i in range(self.test_ScratchPointer+ignore_first_mol, self.test_ScratchPointer+ncases):
			if (self.scratch_test_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_test_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_test_meta[i,0]
			if not self.test_mols_done and self.test_begin_mol+currentmol not in self.test_mols:
					self.test_mols.append(self.test_begin_mol+currentmol)
#					if i < self.test_ScratchPointer+ignore_first_mol + 50:
#						print "i ",i, self.set.mols[self.test_mols[-1]].bonds
			# metadata contains: molecule index, atom type, mol start, mol stop
			e = (self.scratch_test_meta[i,1])
			ei = self.eles.index(e)
			# The offset for this element should be within the bounds or something is wrong...
			inputs[ei][offsets[ei],:] = self.scratch_test_inputs[i]
			matrices[ei][offsets[ei],outputpointer] = 1.0
			coords[ei][offsets[ei]] = self.scratch_test_xyzmeta[i]
			outputs[outputpointer] = self.scratch_test_outputs[self.scratch_test_meta[i,0]]
			offsets[ei] += 1
#			if i < self.test_ScratchPointer+ignore_first_mol + 50:
#				print "first 50 meta data :", i, self.test_ScratchPointer+ignore_first_mol, self.scratch_test_meta[i]
		#print "inputs",inputs
		#print "bounds",bounds
		#print "matrices",matrices
		#print "outputs",outputs
		self.test_ScratchPointer += ncases
#		print "length of test_mols:", len(self.test_mols)
#		print "outputpointer:", outputpointer
		return [inputs, matrices, coords, outputs]

class TensorMolData_BP_Multipole_2(TensorMolData_BP_Multipole):
	"""
    A tensordata for learning the multipole of molecules using Behler-Parinello scheme.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol"):
		TensorMolData_BP_Multipole.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
		return

	def GetTrainBatch(self,ncases,noutputs):
		"""
		Construct the data required for a training batch Returns inputs (sorted by element), and indexing matrices and outputs.
		Behler parinello batches need to have a typical overall stoichiometry.
		and a constant number of atoms, and must contain an integer number of molecules.

		Besides making sure all of that takes place this routine makes the summation matrices
		which map the cases => molecular energies in the Neural Network output.

		Args:
			ncases: the size of a training cases.
			noutputs: the maximum number of molecule energies which can be produced.
		Returns:
			A an **ordered** list of length self.eles containing
				a list of (num_of atom type X flattened input shape) matrix of input cases.
				a list of (num_of atom type X batchsize) matrices which linearly combines the elements
				a list of outputs.
		"""
		start_time = time.time()
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTrain):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.ScratchPointer+ncases >= self.NTrain):
			self.ScratchPointer = 0
		inputs = []#np.zeros((ncases, np.prod(self.dig.eshape)))
		matrices = []#np.zeros((len(self.eles), ncases, noutputs))
		offsets=[]
		coords = []
		# Get the number of molecules which would be contained in the desired batch size
		# and the number of element cases.
		# metadata contains: molecule index, atom type, mol start, mol stop
		bmols=np.unique(self.scratch_meta[self.ScratchPointer:self.ScratchPointer+ncases,0])
		nmols_out=len(bmols[1:-1])
		if (nmols_out > noutputs):
			raise Exception("Insufficent Padding. "+str(nmols_out)+" is greater than "+str(noutputs))
		inputpointer = 0
		outputpointer = 0
		#currentmol=self.scratch_meta[self.ScratchPointer,0]
		sto = np.zeros(len(self.eles),dtype = np.int32)
		offsets = np.zeros(len(self.eles),dtype = np.int32) # output pointers within each element block.
		destinations = np.zeros(ncases) # The index in the output of each case in the scratch.
		ignore_first_mol = 0
		for i in range(self.ScratchPointer,self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			elif (self.scratch_meta[i,0] == bmols[0]):
				ignore_first_mol += 1
			else:
				sto[self.eles.index(self.scratch_meta[i,1])]+=1
		currentmol=self.scratch_meta[self.ScratchPointer+ignore_first_mol,0]
		outputs = np.zeros((noutputs, 3))
		natom_in_mol  = np.zeros((noutputs, 1))
		natom_in_mol.fill(float('inf'))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			matrices.append(np.zeros((sto[e],noutputs)))
			coords.append(np.zeros((sto[e], 3)))
		for i in range(self.ScratchPointer+ignore_first_mol, self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_meta[i,0]
			# metadata contains: molecule index, atom type, mol start, mol stop
			e = (self.scratch_meta[i,1])
			ei = self.eles.index(e)
			# The offset for this element should be within the bounds or something is wrong...
			inputs[ei][offsets[ei],:] = self.scratch_inputs[i]
			matrices[ei][offsets[ei],outputpointer] = 1.0
			coords[ei][offsets[ei]] = self.scratch_xyzmeta[i]
			outputs[outputpointer] = self.scratch_outputs[self.scratch_meta[i,0]]
			natom_in_mol[outputpointer] = self.scratch_meta[i,3] - self.scratch_meta[i,2]
			offsets[ei] += 1
		#print "inputs",inputs
		#print "bounds",bounds
		#print "matrices",matrices
		#print "outputs",outputs
		self.ScratchPointer += ncases
		#for m in matrices:
		#	print "the matrix:", np.sum(m, axis=0)
		#print "natom:", natom_in_mol[0:2]
		return [inputs, matrices, coords, 1.0/natom_in_mol, outputs]

	def GetTestBatch(self,ncases,noutputs):
		"""
			Returns:
			A an **ordered** list of length self.eles containing
				a list of (num_of atom type X flattened input shape) matrix of input cases.
				a list of (num_of atom type X batchsize) matrices which linearly combines the elements
				a list of outputs.
				the number of output molecules.
		"""
		start_time = time.time()
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTest):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.test_ScratchPointer+ncases >= self.NTest):
			self.test_ScratchPointer = 0
			self.test_mols_done = True
		inputs = []#np.zeros((ncases, np.prod(self.dig.eshape)))
		matrices = []#np.zeros((len(self.eles), ncases, noutputs))
		coords = []
		offsets= []
		# Get the number of molecules which would be contained in the desired batch size
		# and the number of element cases.
		# metadata contains: molecule index, atom type, mol start, mol stop
		bmols=np.unique(self.scratch_test_meta[self.test_ScratchPointer:self.test_ScratchPointer+ncases,0])
		nmols_out=len(bmols[1:-1])
		#print "batch contains",nmols_out, "Molecules in ",ncases
		if (nmols_out > noutputs):
			raise Exception("Insufficent Padding. "+str(nmols_out)+" is greater than "+str(noutputs))
		inputpointer = 0
		outputpointer = 0
		#currentmol=self.scratch_meta[self.ScratchPointer,0]
		sto = np.zeros(len(self.eles),dtype = np.int32)
		offsets = np.zeros(len(self.eles),dtype = np.int32) # output pointers within each element block.
		destinations = np.zeros(ncases) # The index in the output of each case in the scratch.
		ignore_first_mol = 0
		for i in range(self.test_ScratchPointer,self.test_ScratchPointer+ncases):
			if (self.scratch_test_meta[i,0] == bmols[-1]):
				break
			elif (self.scratch_test_meta[i,0] == bmols[0]):
				ignore_first_mol += 1
			else:
				sto[self.eles.index(self.scratch_test_meta[i,1])]+=1
		currentmol=self.scratch_test_meta[self.test_ScratchPointer+ignore_first_mol,0]
		outputs = np.zeros((noutputs, 3))
		natom_in_mol = np.zeros((noutputs, 1))
		natom_in_mol.fill(float('inf'))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			matrices.append(np.zeros((sto[e],noutputs)))
			coords.append(np.zeros((sto[e], 3)))
		for i in range(self.test_ScratchPointer+ignore_first_mol, self.test_ScratchPointer+ncases):
			if (self.scratch_test_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_test_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_test_meta[i,0]
			if not self.test_mols_done and self.test_begin_mol+currentmol not in self.test_mols:
					self.test_mols.append(self.test_begin_mol+currentmol)
#					if i < self.test_ScratchPointer+ignore_first_mol + 50:
#						print "i ",i, self.set.mols[self.test_mols[-1]].bonds
			# metadata contains: molecule index, atom type, mol start, mol stop
			e = (self.scratch_test_meta[i,1])
			ei = self.eles.index(e)
			# The offset for this element should be within the bounds or something is wrong...
			inputs[ei][offsets[ei],:] = self.scratch_test_inputs[i]
			matrices[ei][offsets[ei],outputpointer] = 1.0
			coords[ei][offsets[ei]] = self.scratch_test_xyzmeta[i]
			natom_in_mol[outputpointer] = self.scratch_test_meta[i,3] - self.scratch_test_meta[i,2]
			outputs[outputpointer] = self.scratch_test_outputs[self.scratch_test_meta[i,0]]
			offsets[ei] += 1
#			if i < self.test_ScratchPointer+ignore_first_mol + 50:
#				print "first 50 meta data :", i, self.test_ScratchPointer+ignore_first_mol, self.scratch_test_meta[i]
		#print "inputs",inputs
		#print "bounds",bounds
		#print "matrices",matrices
		#print "outputs",outputs
#		self.test_ScratchPointer += ncases
#		print "length of test_mols:", len(self.test_mols)
#		print "outputpointer:", outputpointer
		return [inputs, matrices, coords, 1.0/natom_in_mol, outputs]


#	def BuildTrain(self, name_="gdb9",  append=False, max_nmols_=1000000):
#		self.CheckShapes()
#		self.name=name_
#		LOGGER.info("TensorMolData, self.type:"+self.type)
#		if self.type=="frag":
#			raise Exception("No BP frags now")
#		nmols  = len(self.set.mols)
#		natoms = self.set.NAtoms()
#		LOGGER.info( "self.dig.eshape"+str(self.dig.eshape)+" self.dig.lshape"+str(self.dig.lshape))
#		cases = np.zeros(tuple([natoms]+list(self.dig.eshape)))
#		LOGGER.info( "cases:"+str(cases.shape))
#		labels = np.zeros(tuple([nmols]+list(self.dig.lshape)))
#		self.CaseMetadata = np.zeros((natoms, 4), dtype = np.int)
#		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_in.npy"
#		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_out.npy"
#		metasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_meta.npy" # Used aggregate and properly sum network inputs and outputs.
#		casep=0
#		# Generate the set in a random order.
#		ord=np.random.permutation(len(self.set.mols))
#		mols_done = 0
#		for mi in ord:
#			nat = self.set.mols[mi].NAtoms()
#			#print "casep:", casep
#			if (mols_done%1000==0):
#				LOGGER.info("Mol:"+str(mols_done))
#			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
#			if not np.all(np.isfinite(ins)):
#				print "find a bad case, writting down xyz.."
#				self.set.mols[mi].WriteXYZfile(fpath=".", fname="bad_buildset_cases")
#			#print mi, ins.shape, outs.shape
#			cases[casep:casep+nat] = ins
#			labels[mols_done] = outs
#			for j in range(casep,casep+nat):
#				self.CaseMetadata[j,0] = mols_done
#				self.CaseMetadata[j,1] = self.set.mols[mi].atoms[j-casep]
#				self.CaseMetadata[j,2] = casep
#				self.CaseMetadata[j,3] = casep+nat
#			casep += nat
#			mols_done = mols_done + 1
#			if (mols_done>=max_nmols_):
#				break
#		inf = open(insname,"wb")
#		ouf = open(outsname,"wb")
#		mef = open(metasname,"wb")
#		np.save(inf,cases[:casep,:])
#		np.save(ouf,labels[:mols_done,:])
#		np.save(mef,self.CaseMetadata[:casep,:])
#		inf.close()
#		ouf.close()
#		mef.close()
#		self.AvailableDataFiles.append([insname,outsname,metasname])
#		self.Save() #write a convenience pickle.
#		return
#

	def BuildTrain(self, name_="gdb9",  append=False, max_nmols_=1000000):
		self.CheckShapes()
		self.name=name_
		LOGGER.info("TensorMolData_BP_Multipole, self.type:"+self.type)
		if self.type=="frag":
			raise Exception("No BP frags now")
		nmols  = len(self.set.mols)
		natoms = self.set.NAtoms()
		LOGGER.info( "self.dig.eshape"+str(self.dig.eshape)+" self.dig.lshape"+str(self.dig.lshape))
		cases = np.zeros(tuple([natoms]+list(self.dig.eshape)))
		LOGGER.info( "cases:"+str(cases.shape))
		labels = np.zeros(tuple([nmols]+list(self.dig.lshape)))
		self.CaseMetadata = np.zeros((natoms, 4), dtype = np.int)
		self.xyzMetadata = np.zeros((natoms, 3))
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_in.npy"
		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_out.npy"
		metasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_meta.npy" # Used aggregate and properly sum network inputs and outputs.
		xyzmetasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_xyzmeta.npy" # Used aggregate and properly sum network inputs and outputs.
		casep=0
		# Generate the set in a random order.
		ord=np.random.permutation(len(self.set.mols))
		mols_done = 0
		for mi in ord:
			nat = self.set.mols[mi].NAtoms()
			#print "casep:", casep
			if (mols_done%1000==0):
				LOGGER.info("Mol:"+str(mols_done))
			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
			if not np.all(np.isfinite(ins)):
				print "find a bad case, writting down xyz.."
				self.set.mols[mi].WriteXYZfile(fpath=".", fname="bad_buildset_cases")
			#print mi, ins.shape, outs.shape
			cases[casep:casep+nat] = ins
			labels[mols_done] = outs
			center_xyz = self.set.mols[mi].coords - np.average(self.set.mols[mi].coords, axis=0)
			for j in range(casep,casep+nat):
				self.CaseMetadata[j,0] = mols_done
				self.CaseMetadata[j,1] = self.set.mols[mi].atoms[j-casep]
				self.CaseMetadata[j,2] = casep
				self.CaseMetadata[j,3] = casep+nat
				self.xyzMetadata[j] = center_xyz[j - casep]
			casep += nat
			mols_done = mols_done + 1
			if (mols_done>=max_nmols_):
				break
		inf = open(insname,"wb")
		ouf = open(outsname,"wb")
		mef = open(metasname,"wb")
		xyzf = open(xyzmetasname, "wb")
		np.save(inf,cases[:casep,:])
		np.save(ouf,labels[:mols_done,:])
		np.save(mef,self.CaseMetadata[:casep,:])
		np.save(xyzf, self.xyzMetadata[:casep,:])
		inf.close()
		ouf.close()
		mef.close()
		xyzf.close()
		self.AvailableDataFiles.append([insname,outsname,metasname, xyzmetasname])
		self.Save() #write a convenience pickle.
		return

class TensorMolDataEE:
	"""
		Electrostatically embedded TensorMolData.
		This is also a re-write of our BP scheme with TensorFlow's
		New scatter gather capabilities to reduce padding.

		Inputs are xyz, embedding, (emb. derivative)
		Outputs are Energy, Dipole (force)
	"""
	def __init__(self,MSet_,Dig_,Name_, WithGrad_ = False):
		"""
		Args:
			The MSet_ should have energies, charges and optionally forces.
		"""
		TensorMolDataBP.__init__(self, MSet_, Dig_, Name_, order_=3, num_indis_=1, type_="mol", WithGrad_ = WithGrad_)

	def BuildTrain(self, name_="gdb9",  append=False, max_nmols_=1000000, WithGrad_=False):
		if (WithGrad_ and self.dig.OType != "AEAndForce"):
			raise Exception("Use to find forces.... ")
		self.CheckShapes()
		self.name=name_
		LOGGER.info("TensorMolData, self.type:"+self.type)
		if self.type=="frag":
			raise Exception("No BP frags now")
		nmols  = len(self.set.mols)
		natoms = self.set.NAtoms()
		LOGGER.info( "self.dig.eshape"+str(self.dig.eshape)+" self.dig.lshape"+str(self.dig.lshape))
		cases = np.zeros(tuple([natoms]+list(self.dig.eshape)))
		casesg = None
		if (WithGrad_):
			casesg = np.zeros(tuple([natoms]+list(self.dig.eshape)+[self.MaxN3]))
			self.HasGrad = True
		else:
			self.HasGrad = False
		LOGGER.info( "cases:"+str(cases.shape))
		labels = np.zeros(tuple([nmols]+list(self.dig.lshape)))
		if (WithGrad_):
			# Tediously if you have the gradient the lshape can't really be general....
			# We should figure out a more universal, differentiable way to do this.
			labels = np.zeros(tuple([nmols]+[self.MaxN3+1]))
		self.CaseMetadata = np.zeros((natoms, 4), dtype = np.int)
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_in.npy"
		ingname = self.path+"Mol_"+name_+"_"+self.dig.name+"_ing.npy"
		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_out.npy"
		metasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_meta.npy" # Used aggregate and properly sum network inputs and outputs.
		casep=0
		# Generate the set in a random order.
		ord=np.random.permutation(len(self.set.mols))
		mols_done = 0
		for mi in ord:
			nat = self.set.mols[mi].NAtoms()
			#print "casep:", casep
			if (mols_done%1000==0):
				LOGGER.info("Mol:"+str(mols_done))
			if (WithGrad_):
				ins, grads, outs = self.dig.TrainDigest(self.set.mols[mi])
			else:
				ins, outs = self.dig.TrainDigest(self.set.mols[mi])
			if not np.all(np.isfinite(ins)):
				print "find a bad case, writting down xyz.."
				self.set.mols[mi].WriteXYZfile(fpath=".", fname="bad_buildset_cases")
			#print mi, ins.shape, outs.shape
			cases[casep:casep+nat] = ins
			if (WithGrad_):
				if (grads.shape[2]%3 != 0):
					raise Exception("Bad Deriv.")
				casesg[casep:casep+nat,:,:grads.shape[2]] = grads # grads are atomXdesc dimX R
				labels[mols_done,:outs.shape[0]] = outs
			else:
				labels[mols_done] = outs
			for j in range(casep,casep+nat):
				self.CaseMetadata[j,0] = mols_done
				self.CaseMetadata[j,1] = self.set.mols[mi].atoms[j-casep]
				self.CaseMetadata[j,2] = casep
				self.CaseMetadata[j,3] = casep+nat
			casep += nat
			mols_done = mols_done + 1
			if (mols_done>=max_nmols_):
				break
		inf = open(insname,"wb")
		ouf = open(outsname,"wb")
		mef = open(metasname,"wb")
		if (WithGrad_):
			ingf = open(ingname,"wb")
			np.save(ingf,casesg[:casep,:])
			ingf.close()
			self.AvailableDataFiles.append([ingname])
		np.save(inf,cases[:casep,:])
		np.save(ouf,labels[:mols_done,:])
		np.save(mef,self.CaseMetadata[:casep,:])
		inf.close()
		ouf.close()
		mef.close()
		self.AvailableDataFiles.append([insname,outsname,metasname])
		self.Save() #write a convenience pickle.
		return

	def LoadData(self):
		insname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_in.npy"
		ingname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_ing.npy"
		outsname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_out.npy"
		metasname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_meta.npy" # Used aggregate
		inf = open(insname,"rb")
		ouf = open(outsname,"rb")
		mef = open(metasname,"rb")
		ti = np.load(inf)
		to = np.load(ouf)
		tm = np.load(mef)
		inf.close()
		ouf.close()
		mef.close()
		to = to.reshape((to.shape[0],-1))  # flat labels to [mol, 1]
		if (os.path.isfile(ingname) and self.HasGrad):
			ing = open(ingname,"rb")
			tig = np.load(ing)
			ing.close()
			return ti, tig, to, tm
		return ti, to, tm

	def LoadDataToScratch(self, tformer):
		"""
		Reads built training data off disk into scratch space.
		Divides training and test data.
		Normalizes inputs and outputs.
			note that modifies my MolDigester to incorporate the normalization
		Initializes pointers used to provide training batches.

		Args:
			random: Not yet implemented randomization of the read data.

		Note:
			Also determines mean stoichiometry
		"""
		if 'self.HasGrad' not in locals():
			print "do not find self.HasGrad, set to False"
			self.HasGrad = False
		if (self.ScratchState == 1):
			return
		if (self.HasGrad):
			ti, tig, to, tm = self.LoadData()
		else:
			ti, to, tm = self.LoadData()
		if (tformer.innorm != None):
			ti = tformer.NormalizeIns(ti)
		if (tformer.outnorm != None):
			to = tformer.NormalizeOuts(to)
		self.NTestMols = int(self.TestRatio * to.shape[0])
		self.LastTrainMol = int(to.shape[0]-self.NTestMols)
		LOGGER.debug("LastTrainMol in TensorMolData: %i", self.LastTrainMol)
		LOGGER.debug("NTestMols in TensorMolData: %i", self.NTestMols)
		LOGGER.debug("Number of molecules in meta:: %i", tm[-1,0]+1)
		LastTrainCase=0
		for i in range(len(tm)):
			if (tm[i,0] == self.LastTrainMol):
				LastTrainCase = tm[i,2] # exclusive
				break
		LOGGER.debug("last train atom: %i",LastTrainCase)
		LOGGER.debug("Num Test atoms: %i",len(tm)-LastTrainCase)
		LOGGER.debug("Num atoms: %i",len(tm))
		self.NTrain = LastTrainCase
		self.NTest = len(tm)-LastTrainCase
		self.scratch_inputs = ti[:LastTrainCase]
		self.scratch_outputs = to[:self.LastTrainMol]
		self.scratch_meta = tm[:LastTrainCase]
		self.scratch_test_inputs = ti[LastTrainCase:]
		self.scratch_test_outputs = to[self.LastTrainMol:]
		if (self.HasGrad):
			self.scratch_grads = tig[:LastTrainCase]
			self.scratch_test_grads = tig[LastTrainCase:]
		# metadata contains: molecule index, atom type, mol start, mol stop
		# these columns need to be shifted.
		self.scratch_test_meta = tm[LastTrainCase:]
		self.test_begin_mol = self.scratch_test_meta[0,0]
		#		print "before shift case  ", tm[LastTrainCase:LastTrainCase+30], "real", self.set.mols[tm[LastTrainCase, 0]].bonds, self.set.mols[self.test_begin_mol].bonds
		self.scratch_test_meta[:,0] -= self.scratch_test_meta[0,0]
		self.scratch_test_meta[:,3] -= self.scratch_test_meta[0,2]
		self.scratch_test_meta[:,2] -= self.scratch_test_meta[0,2]
		self.ScratchState = 1
		self.ScratchPointer = 0
		self.test_ScratchPointer=0
		# Compute mean Stoichiometry and number of atoms.
		self.eles = np.unique(tm[:,1]).tolist()
		atomcount = np.zeros(len(self.eles))
		self.MeanStoich = np.zeros(len(self.eles))
		for j in range(len(self.eles)):
			for i in range(len(ti)):
				if (tm[i,1]==self.eles[j]):
					atomcount[j]=atomcount[j]+1
		self.MeanStoich=atomcount/len(to)
		self.MeanNumAtoms = np.sum(self.MeanStoich)
		return

	def GetTrainBatch(self,ncases,noutputs):
		"""
		Construct the data required for a training batch Returns inputs (sorted by element), and indexing matrices and outputs.
		Behler parinello batches need to have a typical overall stoichiometry.
		and a constant number of atoms, and must contain an integer number of molecules.

		Besides making sure all of that takes place this routine makes the summation matrices
		which map the cases => molecular energies in the Neural Network output.

		Args:
			ncases: the size of a training cases.
			noutputs: the maximum number of molecule energies which can be produced.
		Returns:
			A an **ordered** list of length self.eles containing
				a list of (num_of atom type X flattened input shape) matrix of input cases.
				a list of (num_of atom type X batchsize) matrices which linearly combines the elements
				a list of outputs.

		"""
		start_time = time.time()
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTrain):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.ScratchPointer+ncases >= self.NTrain):
			self.ScratchPointer = 0
		inputs = []#np.zeros((eles,ncases, np.prod(self.dig.eshape)))
		inputgs = []#np.zeros((eles, ncases, (np.prod(self.dig.egshape),max(n3)))
		matrices = []#np.zeros((eles, ncases, (np.prod(self.dig.egshape),max(n3)))
		offsets=[]
		# Get the number of molecules which would be contained in the desired batch size
		# and the number of element cases.
		# metadata contains: molecule index, atom type, mol start, mol stop
		bmols=np.unique(self.scratch_meta[self.ScratchPointer:self.ScratchPointer+ncases,0])
		nmols_out=len(bmols[1:-1])
		if (nmols_out > noutputs):
			raise Exception("Insufficent Padding. "+str(nmols_out)+" is greater than "+str(noutputs))
		inputpointer = 0
		outputpointer = 0
		#currentmol=self.scratch_meta[self.ScratchPointer,0]
		sto = np.zeros(len(self.eles),dtype = np.int32)
		offsets = np.zeros(len(self.eles),dtype = np.int32) # output pointers within each element block.
		destinations = np.zeros(ncases) # The index in the output of each case in the scratch.
		ignore_first_mol = 0
		for i in range(self.ScratchPointer,self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			elif (self.scratch_meta[i,0] == bmols[0]):
				ignore_first_mol += 1
			else:
				sto[self.eles.index(self.scratch_meta[i,1])]+=1
		currentmol=self.scratch_meta[self.ScratchPointer+ignore_first_mol,0]
		outputs = None
		if (self.HasGrad):
			outputs = np.zeros((noutputs,self.dig.lshape[0]))
		else:
			outputs = np.zeros((noutputs))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			if (self.HasGrad):
				inputgs.append(np.zeros((sto[e],np.prod(self.dig.eshape),self.MaxN3)))
			matrices.append(np.zeros((sto[e],noutputs)))
		for i in range(self.ScratchPointer+ignore_first_mol, self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_meta[i,0]
			# metadata contains: molecule index, atom type, mol start, mol stop
			e = (self.scratch_meta[i,1])
			ei = self.eles.index(e)
			# The offset for this element should be within the bounds or something is wrong...
			inputs[ei][offsets[ei],:] = self.scratch_inputs[i]
			if (self.HasGrad):
				inputgs[ei][offsets[ei],:] = self.scratch_grads[i]
			matrices[ei][offsets[ei],outputpointer] = 1.0
			outputs[outputpointer] = self.scratch_outputs[self.scratch_meta[i,0]]
			offsets[ei] += 1
		#print "inputs",inputs
		#print "bounds",bounds
		#print "matrices",matrices
		#print "outputs",outputs
		self.ScratchPointer += ncases
		if (self.HasGrad):
			return [inputs, inputgs, matrices, outputs]
		else:
			return [inputs, matrices, outputs]

	def GetTestBatch(self,ncases,noutputs):
		"""
			Returns:
			A an **ordered** list of length self.eles containing
				a list of (num_of atom type X flattened input shape) matrix of input cases.
				a list of (num_of atom type X batchsize) matrices which linearly combines the elements
				a list of outputs.
				the number of output molecules.
		"""
		start_time = time.time()
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTest):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.test_ScratchPointer+ncases >= self.NTest):
			self.test_ScratchPointer = 0
			self.test_mols_done = True
		inputs = []#np.zeros((ncases, np.prod(self.dig.eshape)))
		inputgs = []#np.zeros((eles, ncases, (np.prod(self.dig.egshape),max(n3)))
		matrices = []#np.zeros((len(self.eles), ncases, noutputs))
		offsets= []
		# Get the number of molecules which would be contained in the desired batch size
		# and the number of element cases.
		# metadata contains: molecule index, atom type, mol start, mol stop
		bmols=np.unique(self.scratch_test_meta[self.test_ScratchPointer:self.test_ScratchPointer+ncases,0])
		nmols_out=len(bmols[1:-1])
		#print "batch contains",nmols_out, "Molecules in ",ncases
		if (nmols_out > noutputs):
			raise Exception("Insufficent Padding. "+str(nmols_out)+" is greater than "+str(noutputs))
		inputpointer = 0
		outputpointer = 0
		#currentmol=self.scratch_meta[self.ScratchPointer,0]
		sto = np.zeros(len(self.eles),dtype = np.int32)
		offsets = np.zeros(len(self.eles),dtype = np.int32) # output pointers within each element block.
		destinations = np.zeros(ncases) # The index in the output of each case in the scratch.
		ignore_first_mol = 0
		for i in range(self.test_ScratchPointer,self.test_ScratchPointer+ncases):
			if (self.scratch_test_meta[i,0] == bmols[-1]):
				break
			elif (self.scratch_test_meta[i,0] == bmols[0]):
				ignore_first_mol += 1
			else:
				sto[self.eles.index(self.scratch_test_meta[i,1])]+=1
		currentmol=self.scratch_test_meta[self.test_ScratchPointer+ignore_first_mol,0]
		outputs = None
		if (self.HasGrad):
			outputs = np.zeros((noutputs,self.dig.lshape[0]))
		else:
			outputs = np.zeros((noutputs))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			if (self.HasGrad):
				inputgs.append(np.zeros((sto[e],np.prod(self.dig.eshape),self.MaxN3)))
			matrices.append(np.zeros((sto[e],noutputs)))
		for i in range(self.test_ScratchPointer+ignore_first_mol, self.test_ScratchPointer+ncases):
			if (self.scratch_test_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_test_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_test_meta[i,0]
			if not self.test_mols_done and self.test_begin_mol+currentmol not in self.test_mols:
					self.test_mols.append(self.test_begin_mol+currentmol)
			# metadata contains: molecule index, atom type, mol start, mol stop
			e = (self.scratch_test_meta[i,1])
			ei = self.eles.index(e)
			# The offset for this element should be within the bounds or something is wrong...
			inputs[ei][offsets[ei],:] = self.scratch_test_inputs[i]
			if (self.HasGrad):
				inputgs[ei][offsets[ei],:] = self.scratch_test_grads[i]
			matrices[ei][offsets[ei],outputpointer] = 1.0
			outputs[outputpointer] = self.scratch_test_outputs[self.scratch_test_meta[i,0]]
			offsets[ei] += 1
		self.test_ScratchPointer += ncases
		if (self.HasGrad):
			return [inputs, inputgs, matrices, outputs]
		else:
			return [inputs, matrices, outputs]
