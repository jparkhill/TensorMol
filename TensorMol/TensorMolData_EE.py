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
from TensorData import *
from TensorMolData import *
#import tables should go to hdf5 soon...


class TensorMolData_BP_Multipole(TensorMolData_BP):
	"""
			A tensordata for learning the multipole of molecules using Behler-Parinello scheme.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol"):
		TensorMolData_BP.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
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
		self.CaseMetadata = np.zeros((natoms, 7), dtype = np.int)
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_in.npy"
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
			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
			if not np.all(np.isfinite(ins)):
				print "find a bad case, writting down xyz.."
				self.set.mols[mi].WriteXYZfile(fpath=".", fname="bad_buildset_cases")
			#print mi, ins.shape, outs.shape
			cases[casep:casep+nat] = ins
			labels[mols_done] = outs
			center_xyz	= self.set.mols[mi].coords - np.average(self.set.mols[mi].coords, axis=0)
			for j in range(casep,casep+nat):
				self.CaseMetadata[j,0] = mols_done
				self.CaseMetadata[j,1] = self.set.mols[mi].atoms[j-casep]
				self.CaseMetadata[j,2] = casep
				self.CaseMetadata[j,3] = casep+nat
				self.CaseMetadata[j,4:] = center_xyz[j - casep]
			casep += nat
			mols_done = mols_done + 1
			if (mols_done>=max_nmols_):
				break
		inf = open(insname,"wb")
		ouf = open(outsname,"wb")
		mef = open(metasname,"wb")
		np.save(inf,cases[:casep,:])
		np.save(ouf,labels[:mols_done,:])
		np.save(mef,self.CaseMetadata[:casep,:])
		inf.close()
		ouf.close()
		mef.close()
		self.AvailableDataFiles.append([insname,outsname,metasname])
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
			coords[ei][offsets[ei]] = self.scratch_meta[i, 4:]
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
			coords[ei][offsets[ei]] = self.scratch_meta[i, 4:]
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


