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
from Neighbors import *
#import tables should go to hdf5 soon...

class TensorMolData(TensorData):
	"""
	A Training Set is a Molecule set, with a sampler and an embedding
	The sampler chooses points in the molecular volume.
	The embedding turns that into inputs and labels for a network to regress.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol"):
		"""
		Args:
			MSet_: A molecule set from which to cull data.
			Dig_: A MolDigester object to create embeddings, and evaluate outputs.
			Name_: A name for this TensorMolData
			# These parameters should be removed ------------
			order_ : Order of many-body expansion to perform.
			num_indis_: Number of Indistinguishable Fragments.
			type_: Whether this TensorMolData is for "frag", "atom", or "mol"
		"""
		self.order = order_
		self.num_indis = num_indis_
		self.NTrain = 0
		#self.MaxNAtoms = MSet_.MaxNAtoms()
		TensorData.__init__(self, MSet_,Dig_,Name_, type_=type_)
		try:
			LOGGER.info("TensorMolData.type: %s",self.type)
			LOGGER.info("TensorMolData.dig.name: %s",self.dig.name)
			LOGGER.info("NMols in TensorMolData.set: %i", len(self.set.mols))
			self.raw_it = iter(self.set.mols)
		except:
			print " do not include MSet"
		self.MaxNAtoms = None
		try:
			if (MSet_ != None):
				self.MaxNAtoms = MSet_.MaxNAtoms()
		except:
			print "fail to load self.MaxNAtoms"
		return

	def QueryAvailable(self):
		""" If Tensordata has already been made, this looks for it under a passed name."""
		# It should probably check the sanity of each input/outputfile as well...
		return

	def CheckShapes(self):
		# Establish case and label shapes.
		if self.type=="frag":
			tins,touts = self.dig.Emb(test_mol.mbe_frags[self.order][0],False,False)
		elif self.type=="mol":
			if (self.set != None):
				test_mol = self.set.mols[0]
				tins,touts = self.dig.Emb(test_mol,True,False)
			else:
				return
		else:
			raise Exception("Unknown Type")
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
			total_case += len(self.set.mols[mi].mbe_frags[self.order])
		cases = np.zeros(tuple([total_case]+list(self.dig.eshape)))
		labels = np.zeros(tuple([total_case]+list(self.dig.lshape)))
		casep=0
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_"+str(self.order)+"_in.npy"
		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_"+str(self.order)+"_out.npy"
		if self.type=="frag":
			for mi in range(len(self.set.mols)):
				for frag in self.set.mols[mi].mbe_frags[self.order]:
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

	def RawBatch(self,nmol = 4096):
		"""
			Shimmy Shimmy Ya Shimmy Ya Shimmy Yay.
			This type of batch is not built beforehand
			because there's no real digestion involved.

			Args:
				nmol: number of molecules to put in the output.

			Returns:
				Ins: a #atomsX4 tensor (AtNum,x,y,z)
				Outs: output of the digester
				Keys: (nmol)X(MaxNAtoms) tensor listing each molecule's place in the input.
		"""
		ndone = 0
		natdone = 0
		self.MaxNAtoms = self.set.MaxNAtoms()
		Ins = np.zeros(tuple([nmol,self.MaxNAtoms,4]))
		Outs = np.zeros(tuple([nmol,self.MaxNAtoms,3]))
		while (ndone<nmol):
			try:
				m = self.raw_it.next()
#				print "m props", m.properties.keys()
#				print "m coords", m.coords
				ti, to = self.dig.Emb(m, True, False)
				n=ti.shape[0]

				Ins[ndone,:n,:] = ti.copy()
				Outs[ndone,:n,:] = to.copy()
				ndone += 1
				natdone += n
			except StopIteration:
				self.raw_it = iter(self.set.mols)
		return Ins,Outs

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
			a Case is an input to the NN.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol", WithGrad_ = False):
		self.CaseMetadata=None # case X molecule index X element type (Strictly ascending)
		self.LastTrainMol=0
		self.NTestMols=0
		self.scratch_meta = None
		self.scratch_test_meta = None
		self.scratch_grads = None
		self.scratch_test_grads = None
		self.HasGrad = WithGrad_ # whether to pass around the gradient.
		TensorMolData.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
		self.eles = []
		if (MSet_ != None):
			self.eles = list(MSet_.AtomTypes())
			self.eles.sort()
		self.MeanStoich=None
		self.MeanNAtoms=None
		self.test_mols_done = False
		self.test_begin_mol  = None
		self.test_mols = []
		self.MaxN3 = None # The most coordinates in the set.
		print "TensorMolData_BP.eles", self.eles
		print "self.HasGrad:", self.HasGrad
		if (self.HasGrad):
			self.MaxN3 = 3*np.max([m.NAtoms() for m in self.set.mols])
			print "TensorMolData_BP.MaxN3", self.MaxN3
		return

	def CleanScratch(self):
		TensorData.CleanScratch(self)
		self.raw_it=None
		self.CaseMetadata=None # case X molecule index , element type , first atom in this mol, last atom in this mol (exclusive)
		self.scratch_meta = None
		self.scratch_test_meta = None
		return

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
		t = time.time()
		for mi in ord:
			nat = self.set.mols[mi].NAtoms()
			#print "casep:", casep
			if (mols_done%1000==0):
				print "time cost:", time.time() -t, " second"
				LOGGER.info("Mol:"+str(mols_done))
				t = time.time()
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
		try:
			self.HasGrad
		except:
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
		inputs = []
		inputgs = []
		matrices = []
		offsets = []
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
			#print "inputs: ", inputs[0].shape, " inputgs:", inputgs[0], inputgs[0].shape, " outputs", outputs.shape, " matrices", matrices.shape
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
		inputs = []
		inputgs = []
		matrices = []
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

	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchPointer",self.ScratchPointer
		#print "self.test_ScratchPointer",self.test_ScratchPointer


	def Init_TraceBack(self):
		num_eles = [0 for ele in self.eles]
		for mol_index in self.test_mols:
			for ele in list(self.set.mols[mol_index].atoms):
				num_eles[self.eles.index(ele)] += 1
		self.test_atom_index = [np.zeros((num_eles[i],2), dtype = np.int) for i in range (0, len(self.eles))]

		pointer = [0 for ele in self.eles]
		for mol_index in self.test_mols:
			mol = self.set.mols[mol_index]
			for i in range (0, mol.atoms.shape[0]):
				atom_type = mol.atoms[i]
				self.test_atom_index[self.eles.index(atom_type)][pointer[self.eles.index(atom_type)]] = [int(mol_index), i]
				pointer[self.eles.index(atom_type)] += 1
		print self.test_atom_index
		f  = open("test_energy_real_atom_index_for_test.dat","wb")
		pickle.dump(self.test_atom_index, f)
		f.close()
		return

	def Save(self):
		self.CleanScratch()
		f=open(self.path+self.name+"_"+self.dig.name+".tdt","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return


class TensorMolData_Bond_BP(TensorMolData_BP):
	"""
	A tensordata for molecules and Bond-wise Behler-Parinello.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol"):
		TensorMolData_BP.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
		self.eles = list(self.set.BondTypes())
		self.eles.sort()
		#self. = self.set.
		#self.bonds = list(self.set.BondTypes())
		#self.bonds.sort()
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
		print "self.type:", self.type
		if self.type=="frag":
			raise Exception("No BP frags now")
		nmols  = len(self.set.mols)
		nbonds = self.set.NBonds()
		print "self.dig.eshape", self.dig.eshape, " self.dig.lshape", self.dig.lshape
		cases = np.zeros(tuple([nbonds]+list(self.dig.eshape)))
		print "cases:", cases.shape
		labels = np.zeros(tuple([nmols]+list(self.dig.lshape)))
		self.CaseMetadata = np.zeros((nbonds, 4), dtype = np.int)
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_in.npy"
		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_out.npy"
		metasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_meta.npy" # Used aggregate and properly sum network inputs and outputs.
		casep=0
		# Generate the set in a random order.
		#ord = range (0, len(self.set.mols))  # debug
		ord=np.random.permutation(len(self.set.mols))
		mols_done = 0
		for mi in ord:
			nbo = self.set.mols[mi].NBonds()
			if (mi == 0 or mi == 1):
				print "name of the first/second mol:", self.set.mols[mi].name
			#print "casep:", casep
			if (mols_done%1000==0):
				print "Mol:", mols_done
			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
			#print mi, ins.shape, outs.shape
			cases[casep:casep+nbo] = ins
			#if (self.set.mols[mi].name == "Comment: c60"):
			#	np.savetxt("c60_in.dat", ins)
			#print "ins:", ins, " cases:", cases[casep:casep+nat]
			labels[mols_done] = outs
			for j in range(casep,casep+nbo):
				self.CaseMetadata[j,0] = mols_done
				self.CaseMetadata[j,1] = self.set.mols[mi].bonds[j-casep, 0]
				self.CaseMetadata[j,2] = casep
				self.CaseMetadata[j,3] = casep+nbo
			casep += nbo
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

	def Init_TraceBack(self):
		num_eles = [0 for ele in self.eles]
		for mol_index in self.test_mols:
			for ele in list(self.set.mols[mol_index].bonds[:,0]):
				num_eles[self.eles.index(ele)] += 1
		self.test_atom_index = [np.zeros((num_eles[i],2), dtype = np.int) for i in range (0, len(self.eles))]
		pointer = [0 for ele in self.eles]
		for mol_index in self.test_mols:
			mol = self.set.mols[mol_index]
			for i in range (0, mol.bonds.shape[0]):
				bond_type = mol.bonds[i,0]
				self.test_atom_index[self.eles.index(bond_type)][pointer[self.eles.index(bond_type)]] = [int(mol_index), i]
				pointer[self.eles.index(bond_type)] += 1
		print self.test_atom_index
		f  = open("test_energy_bond_index_for_test.dat","wb")
		pickle.dump(self.test_atom_index, f)
		f.close()
		return


class TensorMolData_BP_Update(TensorMolData_BP):
	"""
			A update version of tensordata for molecules and Behler-Parinello.
			a Case is an input to the NN.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol", WithGrad_ = False):
		TensorMolData_BP.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
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
				a list of (num_of atom type X batchsize) array  which linearly combines the elements
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
		inputs = []
		inputgs = []
		atom_mol_index = [] # mol index of each atom
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
		sto = np.zeros(len(self.eles),dtype = np.int64)
		offsets = np.zeros(len(self.eles),dtype = np.int64) # output pointers within each element block.
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
			atom_mol_index.append(np.zeros((sto[e]), dtype=np.int64))
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
			atom_mol_index[ei][offsets[ei]] = outputpointer
			outputs[outputpointer] = self.scratch_outputs[self.scratch_meta[i,0]]
			offsets[ei] += 1
		#print "inputs",inputs
		#print "bounds",bounds
		#print "matrices",matrices
		#print "outputs",outputs
		self.ScratchPointer += ncases
		if (self.HasGrad):
			#print "inputs: ", inputs[0].shape, " inputgs:", inputgs[0], inputgs[0].shape, " outputs", outputs.shape, " matrices", matrices.shape
			return [inputs, inputgs, atom_mol_index, outputs]
		else:
			return [inputs, atom_mol_index, outputs]

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
		inputs = []
		inputgs = []
		atom_mol_index = []#np.zeros((len(self.eles), ncases, noutputs))
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
		sto = np.zeros(len(self.eles),dtype = np.int64)
		offsets = np.zeros(len(self.eles),dtype = np.int64) # output pointers within each element block.
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
			atom_mol_index.append(np.zeros((sto[e]), dtype=np.int64))
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
			atom_mol_index[ei][offsets[ei]] = outputpointer
			outputs[outputpointer] = self.scratch_test_outputs[self.scratch_test_meta[i,0]]
			offsets[ei] += 1
		self.test_ScratchPointer += ncases
		if (self.HasGrad):
			return [inputs, inputgs, atom_mol_index, outputs]
		else:
			return [inputs, atom_mol_index, outputs]

class TensorMolData_Bond_BP_Update(TensorMolData_BP_Update):
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol", WithGrad_ = False):
		TensorMolData_BP_Update.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
		self.eles = list(self.set.BondTypes())
		self.eles.sort()
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
		print "self.type:", self.type
		if self.type=="frag":
			raise Exception("No BP frags now")
		nmols  = len(self.set.mols)
		nbonds = self.set.NBonds()
		print "self.dig.eshape", self.dig.eshape, " self.dig.lshape", self.dig.lshape
		cases = np.zeros(tuple([nbonds]+list(self.dig.eshape)))
		print "cases:", cases.shape
		labels = np.zeros(tuple([nmols]+list(self.dig.lshape)))
		self.CaseMetadata = np.zeros((nbonds, 4), dtype = np.int)
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_in.npy"
		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_out.npy"
		metasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_meta.npy" # Used aggregate and properly sum network inputs and outputs.
		casep=0
		# Generate the set in a random order.
		#ord = range (0, len(self.set.mols))  # debug
		ord=np.random.permutation(len(self.set.mols))
		mols_done = 0
		for mi in ord:
			nbo = self.set.mols[mi].NBonds()
			if (mi == 0 or mi == 1):
				print "name of the first/second mol:", self.set.mols[mi].name
			#print "casep:", casep
			if (mols_done%1000==0):
				print "Mol:", mols_done
			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
			#print mi, ins.shape, outs.shape
			cases[casep:casep+nbo] = ins
			#if (self.set.mols[mi].name == "Comment: c60"):
			#	np.savetxt("c60_in.dat", ins)
			#print "ins:", ins, " cases:", cases[casep:casep+nat]
			labels[mols_done] = outs
			for j in range(casep,casep+nbo):
				self.CaseMetadata[j,0] = mols_done
				self.CaseMetadata[j,1] = self.set.mols[mi].bonds[j-casep, 0]
				self.CaseMetadata[j,2] = casep
				self.CaseMetadata[j,3] = casep+nbo
			casep += nbo
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

	def Init_TraceBack(self):
		num_eles = [0 for ele in self.eles]
		for mol_index in self.test_mols:
			for ele in list(self.set.mols[mol_index].bonds[:,0]):
				num_eles[self.eles.index(ele)] += 1
		self.test_atom_index = [np.zeros((num_eles[i],2), dtype = np.int) for i in range (0, len(self.eles))]
		pointer = [0 for ele in self.eles]
		for mol_index in self.test_mols:
			mol = self.set.mols[mol_index]
			for i in range (0, mol.bonds.shape[0]):
				bond_type = mol.bonds[i,0]
				self.test_atom_index[self.eles.index(bond_type)][pointer[self.eles.index(bond_type)]] = [int(mol_index), i]
				pointer[self.eles.index(bond_type)] += 1
		print self.test_atom_index
		f  = open("test_energy_bond_index_for_test.dat","wb")
		pickle.dump(self.test_atom_index, f)
		f.close()
		return

class TensorMolData_BP_Direct(TensorMolData):
	"""
	This tensordata serves up batches digested within TensorMol.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol", WithGrad_ = False):
		self.HasGrad = WithGrad_ # whether to pass around the gradient.
		TensorMolData.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
		self.eles = []
		if (MSet_ != None):
			self.eles = list(MSet_.AtomTypes())
			self.eles.sort()
			self.MaxNAtoms = np.max([m.NAtoms() for m in self.set.mols])
			print "self.MaxNAtoms:", self.MaxNAtoms
			self.Nmols = len(self.set.mols)
		self.MeanStoich=None
		self.MeanNAtoms=None
		self.test_mols_done = False
		self.test_begin_mol  = None
		self.test_mols = []
		self.MaxN3 = None # The most coordinates in the set.
		self.name = self.set.name
		print "TensorMolData_BP.eles", self.eles
		print "self.HasGrad:", self.HasGrad
		return

	def CleanScratch(self):
		TensorData.CleanScratch(self)
		self.raw_it = None
		self.xyzs = None
		self.Zs = None
		self.labels = None
		self.grads = None
		return

	def LoadData(self):
		self.ReloadSet()
		random.shuffle(self.set.mols)
		xyzs = np.zeros((self.Nmols, self.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((self.Nmols, self.MaxNAtoms), dtype = np.int32)
		if (self.dig.OType == "AtomizationEnergy"):
			labels = np.zeros((self.Nmols), dtype = np.float64)
		else:
			raise Exception("Output Type is not implemented yet")
		if (self.HasGrad):
			grads = np.zeros((self.Nmols, self.MaxNAtoms, 3), dtype=np.float64)
		for i, mol in enumerate(self.set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			if (self.dig.OType  == "AtomizationEnergy"):
				labels[i] = mol.properties["atomization"]
			else:
				raise Exception("Output Type is not implemented yet")
			if (self.HasGrad):
				grads[i][:mol.NAtoms()] = mol.properties["gradients"]
		if (self.HasGrad):
			return xyzs, Zs, labels, grads
		else:
			return xyzs, Zs, labels

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
		try:
			self.HasGrad
		except:
			self.HasGrad = False
		if (self.ScratchState == 1):
			return
		if (self.HasGrad):
			self.xyzs, self.Zs, self.labels, self.grads = self.LoadData()
		else:
			self.xyzs, self.Zs, self.labels  = self.LoadData()
		self.NTestMols = int(self.TestRatio * self.Zs.shape[0])
		self.LastTrainMol = int(self.Zs.shape[0]-self.NTestMols)
		self.NTrain = self.LastTrainMol
		self.NTest = self.NTestMols
		self.test_ScratchPointer = self.LastTrainMol
		self.ScratchPointer = 0
		self.ScratchState = 1
		LOGGER.debug("LastTrainMol in TensorMolData: %i", self.LastTrainMol)
		LOGGER.debug("NTestMols in TensorMolData: %i", self.NTestMols)
		return

	def GetTrainBatch(self,ncases):
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTrain):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.ScratchPointer+ncases >= self.NTrain):
			self.ScratchPointer = 0
		self.ScratchPointer += ncases
		xyzs = self.xyzs[self.ScratchPointer-ncases:self.ScratchPointer]
		Zs = self.Zs[self.ScratchPointer-ncases:self.ScratchPointer]
		labels = self.labels[self.ScratchPointer-ncases:self.ScratchPointer]
		if (self.HasGrad):
			return [xyzs, Zs, labels, self.grads[self.ScratchPointer-ncases:self.ScratchPointer]]
		else:
			return [xyzs, Zs, labels]

	def GetTestBatch(self,ncases):
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTest):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTest)+" vs "+str(ncases))
		if (self.test_ScratchPointer+ncases > self.Zs.shape[0]):
			self.test_ScratchPointer = self.LastTrainMol
		self.test_ScratchPointer += ncases
		xyzs = self.xyzs[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
		Zs = self.Zs[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
		labels = self.labels[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
		if (self.HasGrad):
			return [xyzs, Zs, labels, self.grads[self.test_ScratchPointer-ncases:self.test_ScratchPointer]]
		else:
			return [xyzs, Zs, labels]

	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchPointer",self.ScratchPointer
		#print "self.test_ScratchPointer",self.test_ScratchPointer

	def Save(self):
		self.CleanScratch()
		f=open(self.path+self.name+"_"+self.dig.name+".tdt","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

class TensorMolData_BPBond_Direct(TensorMolData):
	"""
	This tensordata serves up batches digested within TensorMol.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=1, num_indis_=1, type_="mol", WithGrad_ = False):
		self.HasGrad = WithGrad_ # whether to pass around the gradient.
		TensorMolData.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_)
		self.eles = []
		if (MSet_ != None):
			self.eles = list(MSet_.AtomTypes())
			self.eles.sort()
			self.MaxNAtoms = np.max([m.NAtoms() for m in self.set.mols])
			print "self.MaxNAtoms:", self.MaxNAtoms
			self.Nmols = len(self.set.mols)
		self.MeanStoich=None
		self.MeanNAtoms=None
		self.test_mols_done = False
		self.test_begin_mol  = None
		self.test_mols = []
		self.MaxN3 = None # The most coordinates in the set.
		self.name = self.set.name
		print "TensorMolData_BP.eles", self.eles
		print "self.HasGrad:", self.HasGrad
		return

	def CleanScratch(self):
		TensorData.CleanScratch(self)
		self.raw_it = None
		self.xyzs = None
		self.Zs = None
		self.labels = None
		self.grads = None
		return

	def RawBatch(self,nmol = 4096):
		"""
			Shimmy Shimmy Ya Shimmy Ya Shimmy Yay.
			This type of batch is not built beforehand
			because there's no real digestion involved.

			Args:
				nmol: number of molecules to put in the output.

			Returns:
				Ins: a #atomsX4 tensor (AtNum,x,y,z)
				Outs: output of the digester
				Keys: (nmol)X(MaxNAtoms) tensor listing each molecule's place in the input.
		"""
		ndone = 0
		natdone = 0
		self.MaxNAtoms = self.set.MaxNAtoms()
		Ins = np.zeros(tuple([nmol,self.MaxNAtoms,4]))
		NAtomsVec = np.zeros((nmol),dtype=np.int32)
		Outs = np.zeros(tuple([nmol]))
		while (ndone<nmol):
			try:
				m = self.raw_it.next()
				ti, to = self.dig.Emb(m, True, False)
				n=ti.shape[0]
				Ins[ndone,:n,:] = ti
				NAtomsVec[ndone] = m.NAtoms()
				Outs[ndone] = to
				ndone += 1
				natdone += n
			except StopIteration:
				self.raw_it = iter(self.set.mols)
		NL = NeighborListSet(Ins[:,:,1:],NAtomsVec)
		BondIdxMatrix = NL.buildPairs()
		return Ins,BondIdxMatrix,Outs

	def LoadData(self):
		if self.set == None:
			self.ReloadSet()
		random.shuffle(self.set.mols)
		xyzs = np.zeros((self.Nmols, self.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((self.Nmols, self.MaxNAtoms), dtype = np.int32)
		if (self.dig.OType == "AtomizationEnergy"):
			labels = np.zeros((self.Nmols), dtype = np.float64)
		else:
			raise Exception("Output Type is not implemented yet")
		if (self.HasGrad):
			grads = np.zeros((self.Nmols, self.MaxNAtoms, 3), dtype=np.float64)
		for i, mol in enumerate(self.set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			if (self.dig.OType  == "AtomizationEnergy"):
				labels[i] = mol.properties["atomization"]
			else:
				raise Exception("Output Type is not implemented yet")
			if (self.HasGrad):
				grads[i][:mol.NAtoms()] = mol.properties["gradients"]
		if (self.HasGrad):
			return xyzs, Zs, labels, grads
		else:
			return xyzs, Zs, labels

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
		try:
			self.HasGrad
		except:
			self.HasGrad = False
		if (self.ScratchState == 1):
			return
		if (self.HasGrad):
			self.xyzs, self.Zs, self.labels, self.grads = self.LoadData()
		else:
			self.xyzs, self.Zs, self.labels  = self.LoadData()
		self.NTestMols = int(self.TestRatio * self.Zs.shape[0])
		self.LastTrainMol = int(self.Zs.shape[0]-self.NTestMols)
		self.NTrain = self.LastTrainMol
		self.NTest = self.NTestMols
		self.test_ScratchPointer = self.LastTrainMol
		self.ScratchPointer = 0
		self.ScratchState = 1
		LOGGER.debug("LastTrainMol in TensorMolData: %i", self.LastTrainMol)
		LOGGER.debug("NTestMols in TensorMolData: %i", self.NTestMols)
		return


	# def GetTrainBatch(self,ncases):
	# 	if (self.ScratchState == 0):
	# 		self.LoadDataToScratch()
	# 	reset = False
	# 	if (ncases > self.NTrain):
	# 		raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
	# 	if (self.ScratchPointer+ncases >= self.NTrain):
	# 		self.ScratchPointer = 0
	# 	self.ScratchPointer += ncases
	# 	xyzs = self.xyzs[self.ScratchPointer-ncases:self.ScratchPointer]
	# 	Zs = self.Zs[self.ScratchPointer-ncases:self.ScratchPointer]
	# 	labels = self.labels[self.ScratchPointer-ncases:self.ScratchPointer]
	# 	if (self.HasGrad):
	# 		return [xyzs, Zs, labels, self.grads[self.ScratchPointer-ncases:self.ScratchPointer]]
	# 	else:
	# 		return [xyzs, Zs, labels]
	#
	# def GetTestBatch(self,ncases):
	# 	if (self.ScratchState == 0):
	# 		self.LoadDataToScratch()
	# 	reset = False
	# 	if (ncases > self.NTest):
	# 		raise Exception("Insufficent training data to fill a batch"+str(self.NTest)+" vs "+str(ncases))
	# 	if (self.test_ScratchPointer+ncases > self.Zs.shape[0]):
	# 		self.test_ScratchPointer = self.LastTrainMol
	# 	self.test_ScratchPointer += ncases
	# 	xyzs = self.xyzs[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
	# 	Zs = self.Zs[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
	# 	labels = self.labels[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
	# 	if (self.HasGrad):
	# 		return [xyzs, Zs, labels, self.grads[self.test_ScratchPointer-ncases:self.test_ScratchPointer]]
	# 	else:
	# 		return [xyzs, Zs, labels]

	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchPointer",self.ScratchPointer
		#print "self.test_ScratchPointer",self.test_ScratchPointer

	def Save(self):
		self.CleanScratch()
		f=open(self.path+self.name+"_"+self.dig.name+".tdt","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

class TensorMolData_BP_Direct_Linear(TensorMolData_BP_Direct):
	"""
	This tensordata serves up batches digested within TensorMol.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol", WithGrad_ = False):
		TensorMolData_BP_Direct.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_, WithGrad_)
		self.Rr_cut = PARAMS["AN1_r_Rc"] 
		self.Ra_cut = PARAMS["AN1_a_Rc"]
		return

	def LoadData(self):
		self.ReloadSet()
		random.shuffle(self.set.mols)
		xyzs = np.zeros((self.Nmols, self.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((self.Nmols, self.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((self.Nmols), dtype = np.int64)
		if (self.dig.OType == "AtomizationEnergy"):
			labels = np.zeros((self.Nmols), dtype = np.float64)
		else:
			raise Exception("Output Type is not implemented yet")
		if (self.HasGrad):
			grads = np.zeros((self.Nmols, self.MaxNAtoms, 3), dtype=np.float64)
		for i, mol in enumerate(self.set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = mol.NAtoms()
			if (self.dig.OType  == "AtomizationEnergy"):
				labels[i] = mol.properties["atomization"]
			else:
                        	raise Exception("Output Type is not implemented yet")
			if (self.HasGrad):
				grads[i][:mol.NAtoms()] = mol.properties["gradients"]
		if (self.HasGrad):
			return xyzs, Zs, labels, natom, grads
		else:
			return xyzs, Zs, labels, natom	

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
		try:
			self.HasGrad 
		except:
			self.HasGrad = False
		if (self.ScratchState == 1):
			return
		if (self.HasGrad):
			self.xyzs, self.Zs, self.labels, self.natom, self.grads = self.LoadData()
		else:
			self.xyzs, self.Zs, self.labels, self.natom  = self.LoadData()
		self.NTestMols = int(self.TestRatio * self.Zs.shape[0])
		self.LastTrainMol = int(self.Zs.shape[0]-self.NTestMols)
		self.NTrain = self.LastTrainMol
                self.NTest = self.NTestMols
		self.test_ScratchPointer = self.LastTrainMol
		self.ScratchPointer = 0
		self.ScratchState = 1 
		LOGGER.debug("LastTrainMol in TensorMolData: %i", self.LastTrainMol)
		LOGGER.debug("NTestMols in TensorMolData: %i", self.NTestMols)
		return

	def GetTrainBatch(self,ncases):
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTrain):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.ScratchPointer+ncases >= self.NTrain):
			self.ScratchPointer = 0
		self.ScratchPointer += ncases
		xyzs = self.xyzs[self.ScratchPointer-ncases:self.ScratchPointer]
		Zs = self.Zs[self.ScratchPointer-ncases:self.ScratchPointer]
		labels = self.labels[self.ScratchPointer-ncases:self.ScratchPointer]
		natom = self.natom[self.ScratchPointer-ncases:self.ScratchPointer]
		NL = NeighborListSet(xyzs, natom, True, True, Zs)
		rad_p, ang_t = NL.buildPairsAndTriples(self.Rr_cut, self.Ra_cut)
		if (self.HasGrad):
			return [xyzs, Zs, labels, self.grads[self.ScratchPointer-ncases:self.ScratchPointer], rad_p, ang_t]
		else:
			return [xyzs, Zs, labels, rad_p, ang_t]

	def GetTestBatch(self,ncases):
		if (self.ScratchState == 0):
			self.LoadDataToScratch()
		reset = False
		if (ncases > self.NTest):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTest)+" vs "+str(ncases))
		if (self.test_ScratchPointer+ncases > self.Zs.shape[0]):
			self.test_ScratchPointer = self.LastTrainMol
		self.test_ScratchPointer += ncases
		xyzs = self.xyzs[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
		Zs = self.Zs[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
		labels = self.labels[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
		natom = self.natom[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
		NL = NeighborListSet(xyzs, natom, True, True, Zs)
		rad_p, ang_t = NL.buildPairsAndTriples(self.Rr_cut, self.Ra_cut)
		if (self.HasGrad):
			return [xyzs, Zs, labels, self.grads[self.test_ScratchPointer-ncases:self.test_ScratchPointer], rad_p, ang_t]
		else:
			return [xyzs, Zs, labels, rad_p, ang_t]


	def GetBatch(self, ncases, Train_=True):
		if Train_:
			return self.GetTrainBatch(ncases)
		else:
			return self.GetTestBatch(ncases)
