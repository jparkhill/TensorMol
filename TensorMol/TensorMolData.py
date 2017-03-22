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
#import tables should go to hdf5 soon...

class TensorMolData(TensorData):
	"""
		A Training Set is a Molecule set, with a sampler and an embedding
		The sampler chooses points in the molecular volume.
		The embedding turns that into inputs and labels for a network to regress.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="frag"):
		"""
			Args:
				MSet_: A molecule set from which to cull data.
				Dig_: A MolDigester object to create embeddings, and evaluate outputs.
				Name_: A name for this TensorMolData
				order_ : Order of many-body expansion to perform.
				num_indis_: Number of Indistinguishable Fragments.
				type_: Whether this TensorMolData is for "frag", "atom", or "mol"
		"""
		self.order = order_
		self.num_indis = num_indis_
		self.NTrain = 0
		TensorData.__init__(self, MSet_,Dig_,Name_, type_=type_)
		print "self.type:", self.type
		return

	def QueryAvailable(self):
		""" If Tensordata has already been made, this looks for it under a passed name."""
		# It should probably check the sanity of each input/outputfile as well...
		return

	def CheckShapes(self):
		# Establish case and label shapes.
		if self.type=="frag":
			tins,touts = self.dig.TrainDigest(self.set.mols[0].mbe_permute_frags[self.order][0])
		elif self.type=="mol":
			tins,touts = self.dig.TrainDigest(self.set.mols[0])
		else:
			raise Exception("Unkown Type")
		print "self.dig ", self.dig.name
		print "self.dig input shape: ", self.dig.eshape
		print "self.dig output shape: ", self.dig.lshape
		if (self.dig.eshape == None or self.dig.lshape ==None):
			raise Exception("Ain't got no fucking shape.")

	def BuildTrain(self, name_="gdb9",  append=False):
		self.CheckShapes()
		self.name=name_
		total_case = 0
		for mi in range(len(self.set.mols)):
			total_case += len(self.set.mols[mi].mbe_permute_frags[self.order])
		cases = np.zeros(tuple([total_case]+list(self.dig.eshape)))
		labels = np.zeros(tuple([total_case]+list(self.dig.lshape)))
		casep=0
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_"+str(self.order)+"_in.npy"
		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_"+str(self.order)+"_out.npy"
		if self.type=="frag":
			for mi in range(len(self.set.mols)):
				for frag in self.set.mols[mi].mbe_permute_frags[self.order]:
					#print  frag.dist[0], frag.frag_mbe_energy
					ins,outs = self.dig.TrainDigest(frag)
					cases[casep:casep+1] += ins
					labels[casep:casep+1] += outs
					casep += 1
		elif self.type=="mol":
			for mi in range(len(self.set.mols)):
				if (mi%10000==0):
					LOGGER.debug("Mol: "+str(mi))
				ins,outs = self.dig.TrainDigest(mi)
				cases[casep:casep+1] += ins
				labels[casep:casep+1] += outs
				casep += 1
		else:
			raise Exception("Unknown Type")
		alreadyexists = (os.path.isfile(insname) and os.path.isfile(outsname))
		if (append and alreadyexists):
			ti=None
			to=None
			inf = open(insname,"rb")
			ouf = open(outsname,"rb")
			ti = np.load(inf)
			to = np.load(ouf)
			inf.close()
			ouf.close()
			cases = np.concatenate((cases[:casep],ti))
			labels = np.concatenate((labels[:casep],to))
			inf = open(insname,"wb")
			ouf = open(outsname,"wb")
			np.save(inf,cases)
			np.save(ouf,labels)
			inf.close()
			ouf.close()
			self.AvailableDataFiles.append([insname,outsname])
			#self.SamplesPerElement.append(casep*self.dig.NTrainSamples)
		else:
			inf = open(insname,"wb")
			ouf = open(outsname,"wb")
			np.save(inf,cases[:casep])
			np.save(ouf,labels[:casep])
			inf.close()
			ouf.close()
			self.AvailableDataFiles.append([insname,outsname])
			#self.SamplesPerElement.append(casep*self.dig.NTrainSamples)
		self.Save() #write a convenience pickle.
		return

	def GetTrainBatch(self,ncases=1280,random=False):
		if (self.ScratchState != self.order):
			self.LoadDataToScratch()
		if (ncases>self.NTrain):
			raise Exception("Training Data is less than the batchsize... :( ")
		if ( self.ScratchPointer+ncases >= self.NTrain):
			self.ScratchPointer = 0 #Sloppy.
		tmp=(self.scratch_inputs[self.ScratchPointer:self.ScratchPointer+ncases], self.scratch_outputs[self.ScratchPointer:self.ScratchPointer+ncases])
		self.ScratchPointer += ncases
		return tmp

	def GetTestBatch(self,ncases=1280, ministep = 0):
		if (ncases>self.NTest):
			raise Exception("Test Data is less than the batchsize... :( ")
		return (self.scratch_test_inputs[ncases*(ministep):ncases*(ministep+1)], self.scratch_test_outputs[ncases*(ministep):ncases*(ministep+1)])

	def Randomize(self, ti, to, group):
		ti = ti.reshape((ti.shape[0]/group, group, -1))
		to = to.reshape((to.shape[0]/group, group, -1))
		random.seed(0)
		idx = np.random.permutation(ti.shape[0])
		ti = ti[idx]
		to = to[idx]
		ti = ti.reshape((ti.shape[0]*ti.shape[1],-1))
		to = to.reshape((to.shape[0]*to.shape[1],-1))
		return ti, to

	def LoadData(self, random=False):
		insname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_"+str(self.order)+"_in.npy"
		outsname = self.path+"Mol_"+self.name+"_"+self.dig.name+"_"+str(self.order)+"_out.npy"
		inf = open(insname,"rb")
		ouf = open(outsname,"rb")
		ti = np.load(inf)
		to = np.load(ouf)
		inf.close()
		ouf.close()
		if (ti.shape[0] != to.shape[0]):
			raise Exception("Bad Training Data.")
		ti = ti.reshape((ti.shape[0],-1))  # flat data to [ncase, num_per_case]
		to = to.reshape((to.shape[0],-1))  # flat labels to [ncase, 1]
		group = 1
		tmp = 1
		for i in range (1, self.order+1):
			group = group*i
		for i in range (1, self.num_indis+1):
			tmp = tmp*i
		group = group*(tmp**self.order)
		print "randomize group:", group
		if (random):
			ti, to = self.Randomize(ti, to, group)
		self.NTrain = to.shape[0]
		return ti, to

	def KRR(self):
		from sklearn.kernel_ridge import KernelRidge
		ti, to = self.LoadData(True)
		print "KRR: input shape", ti.shape, " output shape", to.shape
		#krr = KernelRidge()
		krr = KernelRidge(alpha=0.0001, kernel='rbf')
		trainsize = int(ti.shape[0]*0.5)
		krr.fit(ti[0:trainsize,:], to[0:trainsize])
		predict  = krr.predict(ti[trainsize:, : ])
		print predict.shape
		krr_acc_pred  = np.zeros((predict.shape[0],2))
		krr_acc_pred[:,0] = to[trainsize:].reshape(to[trainsize:].shape[0])
		krr_acc_pred[:,1] = predict.reshape(predict.shape[0])
		np.savetxt("krr_acc_pred.dat", krr_acc_pred)
		print "KRR train R^2:", krr.score(ti[0:trainsize, : ], to[0:trainsize])
		print "KRR test  R^2:", krr.score(ti[trainsize:, : ], to[trainsize:])
		return

	def LoadDataToScratch(self, tformer, random=True):
		ti, to = self.LoadData( random)
		if (tformer.innorm != None):
			ti = tformer.NormalizeIns(ti)
		if (tformer.outnorm != None):
			to = tformer.NormalizeOuts(to)
		self.NTest = int(self.TestRatio * ti.shape[0])
		self.NTrain = int(ti.shape[0]-self.NTest)
		self.scratch_inputs = ti[:ti.shape[0]-self.NTest]
		self.scratch_outputs = to[:ti.shape[0]-self.NTest]
		self.scratch_test_inputs = ti[ti.shape[0]-self.NTest:]
		self.scratch_test_outputs = to[ti.shape[0]-self.NTest:]
		if random==True:
			self.scratch_inputs, self.scratch_outputs = self.Randomize(self.scratch_inputs, self.scratch_outputs, 1)
			self.scratch_test_inputs, self.scratch_test_outputs = self.Randomize(self.scratch_test_inputs, self.scratch_test_outputs, 1)
		self.ScratchState = self.order
		self.ScratchPointer=0
		#
		# Also get the relevant Normalizations of input, output
		# and average stoichiometries, etc.
		#
		return

	def PrintSampleInformation(self):
		print "From files: ", self.AvailableDataFiles
		return

	def Save(self):
		self.CleanScratch()
		f=open(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+".tdt","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

	def EvaluateTestBatch(self, desired, predicted, tformer, nmols_=100):
		if (tformer.outnorm != None):
			desired = tformer.UnNormalizeOuts(desired)
			predicted = tformer.UnNormalizeOuts(predicted)
		LOGGER.info("desired.shape "+str(desired.shape)+" predicted.shape "+str(predicted.shape)+" nmols "+str(nmols_))
		LOGGER.info("Evaluating, "+str(len(desired))+" predictions... ")
		if (self.dig.OType=="GoEnergy" or self.dig.OType == "Energy" or self.dig.OType == "AtomizationEnergy"):
			predicted=predicted.flatten()[:nmols_]
			desired=desired.flatten()[:nmols_]
			LOGGER.info( "NCases: "+str(len(desired)))
			#LOGGER.info( "Mean Energy "+str(self.unscld(desired)))
			#LOGGER.info( "Mean Predicted Energy "+str(self.unscld(predicted)))
			for i in range(min(50,nmols_)):
				LOGGER.info( "Desired: "+str(i)+" "+str(desired[i])+" Predicted "+str(predicted[i]))
			LOGGER.info("MAE "+str(np.average(np.abs(desired-predicted))))
			LOGGER.info("STD "+str(np.std(desired-predicted)))
		else:
			raise Exception("Unknown Digester Output Type.")
		return


class TensorMolData_BP(TensorMolData):
	"""
			A tensordata for molecules and Behler-Parinello.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol"):
		# a Case is an input to the NN.
		self.CaseMetadata=None # case X molecule index X element type (Strictly ascending)
		self.LastTrainMol=0
		self.NTestMols=0
		self.scratch_meta = None
		self.scratch_test_meta = None
		TensorMolData.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
		self.eles = list(self.set.AtomTypes())
		self.eles.sort()
		self.MeanStoich=None
		self.MeanNAtoms=None
		print "TensorMolData_BP.eles", self.eles
		return

	def CleanScratch(self):
		TensorData.CleanScratch(self)
		self.CaseMetadata=None # case X molecule index , element type , first atom in this mol, last atom in this mol (exclusive)
		self.scratch_meta = None
		self.scratch_test_meta = None
		return

	def BuildTrain(self, name_="gdb9",  append=False, max_nmols_=1000000):
		self.CheckShapes()
		self.name=name_
		LOGGER.info("TensorMolData, self.type:"+self.type)
		if self.type=="frag":
			raise Exception("No BP frags now")
		nmols  = len(self.set.mols)
		natoms = self.set.NAtoms()
		LOGGER.info( "self.dig.eshape"+str(self.dig.eshape)+" self.dig.lshape"+str(self.dig.lshape))
		cases = np.zeros(tuple([natoms]+list(self.dig.eshape)))
		LOGGER.info( "cases:"+str(cases.shape))
		labels = np.zeros(tuple([nmols]+list(self.dig.lshape)))
		self.CaseMetadata = np.zeros((natoms, 4), dtype = np.int)
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
			if (mols_done%10000==0):
				LOGGER.info("Mol:"+str(mols_done))
			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
			#print mi, ins.shape, outs.shape
			cases[casep:casep+nat] = ins
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
		if (self.ScratchState == 1):
			return
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
		self.scratch_test_inputs = ti[LastTrainCase:]
		self.scratch_test_outputs = to[self.LastTrainMol:]
		# metadata contains: molecule index, atom type, mol start, mol stop
		# these columns need to be shifted.
		self.scratch_test_meta = tm[LastTrainCase:]
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
		inputs = []#np.zeros((ncases, np.prod(self.dig.eshape)))
		matrices = []#np.zeros((len(self.eles), ncases, noutputs))
		offsets=[]
		# Get the number of molecules which would be contained in the desired batch size
		# and the number of element cases.
		# metadata contains: molecule index, atom type, mol start, mol stop
		bmols=np.unique(self.scratch_meta[self.ScratchPointer:self.ScratchPointer+ncases,0])
		nmols_out=len(bmols[:-1])
		#print "batch contains",nmols_out, "Molecules in ",ncases
		if (nmols_out > noutputs):
			raise Exception("Insufficent Padding. "+str(nmols_out)+" is greater than "+str(noutputs))
		inputpointer = 0
		outputpointer = 0
		#print "ScratchPointer",self.ScratchPointer,self.NTrain
		currentmol=self.scratch_meta[self.ScratchPointer,0]
		sto = np.zeros(len(self.eles),dtype = np.int32)
		offsets = np.zeros(len(self.eles),dtype = np.int32) # output pointers within each element block.
		destinations = np.zeros(ncases) # The index in the output of each case in the scratch.
		for i in range(self.ScratchPointer,self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			sto[self.eles.index(self.scratch_meta[i,1])]+=1
		outputs = np.zeros((noutputs))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			matrices.append(np.zeros((sto[e],noutputs)))
		for i in range(self.ScratchPointer,self.ScratchPointer+ncases):
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
			outputs[outputpointer] = self.scratch_outputs[self.scratch_meta[i,0]]
			offsets[ei] += 1
		#print "inputs",inputs
		#print "bounds",bounds
		#print "matrices",matrices
		#print "outputs",outputs
		self.ScratchPointer += ncases
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
		if (ncases > len(self.scratch_test_inputs)):
			raise Exception("Insufficent test data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		inputs = []#np.zeros((ncases, np.prod(self.dig.eshape)))
		matrices = []#np.zeros((len(self.eles), ncases, noutputs))
		offsets=[]
		# outputs = np.zeros((noutputs, np.prod(self.dig.lshape)))
		# Get the number of molecules which would be contained in the desired batch size
		# and the number of element cases.
		# metadata contains: molecule index, atom type, mol start, mol stop
		bmols=np.unique(self.scratch_test_meta[:ncases,0])
		nmols_out=len(bmols[:-1])
		#print "batch contains",nmols_out, "Molecules in ",ncases
		if (nmols_out > noutputs):
			raise Exception("Insufficent Padding. "+str(nmols_out)+" is greater than "+str(noutputs))
		inputpointer = 0
		outputpointer = 0
		#print "ScratchPointer",self.ScratchPointer,self.NTrain
		currentmol=self.scratch_test_meta[0,0]
		sto = np.zeros(len(self.eles),dtype = np.int32)
		offsets = np.zeros(len(self.eles),dtype = np.int32) # output pointers within each element block.
		destinations = np.zeros(ncases) # The index in the output of each case in the scratch.
		for i in range(0,ncases):
			if (self.scratch_test_meta[i,0] == bmols[-1]):
				break
			sto[self.eles.index(self.scratch_test_meta[i,1])]+=1
		outputs = np.zeros((noutputs))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			matrices.append(np.zeros((sto[e],noutputs)))
		for i in range(ncases):
			if (self.scratch_test_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_test_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_test_meta[i,0]
			# metadata contains: molecule index, atom type, mol start, mol stop
			e = (self.scratch_test_meta[i,1])
			ei = self.eles.index(e)
			# The offset for this element should be within the bounds or something is wrong...
			inputs[ei][offsets[ei],:] = self.scratch_test_inputs[i]
			matrices[ei][offsets[ei],outputpointer] = 1.0
			outputs[outputpointer] = self.scratch_test_outputs[self.scratch_test_meta[i,0]-self.scratch_test_meta[0,0]]
			offsets[ei] += 1
		return [inputs, matrices, outputs, outputpointer]

	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchPointer",self.ScratchPointer
		print "self.test_ScratchPointer",self.test_ScratchPointer

	def Save(self):
	    self.CleanScratch()
	    f=open(self.path+self.name+"_"+self.dig.name+".tdt","wb")
	    pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
	    f.close()
	    return
