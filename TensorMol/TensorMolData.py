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
from MolDigest import *
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
					print "Mol: ", mi
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

	def LoadDataToScratch(self, random=True):
		ti, to = self.LoadData( random)
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

	def NormalizeInputs(self):
		mean = (np.mean(self.scratch_inputs, axis=0)).reshape((1,-1))
		std = (np.std(self.scratch_inputs, axis=0)).reshape((1, -1))
		self.scratch_inputs = (self.scratch_inputs-mean)/std
		self.scratch_test_inputs = (self.scratch_test_inputs-mean)/std
		np.save(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_in_MEAN.npy", mean)
		np.save(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_in_STD.npy",std)
		return

#	We need to figure out a better way of incorporating this with the data.
	def NormalizeOutputs(self):
		print self.scratch_outputs
		mean = (np.mean(self.scratch_outputs, axis=0)).reshape((1,-1))
		std = (np.std(self.scratch_outputs, axis=0)).reshape((1, -1))
		self.scratch_outputs = (self.scratch_outputs-mean)/std
		self.scratch_test_outputs = (self.scratch_test_outputs-mean)/std
		np.save(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_out_MEAN.npy", mean)
		np.save(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_out_STD.npy",std)
		print mean, std, self.scratch_outputs
                return

	def Get_Mean_Std(self):
		mean = np.load(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_out_MEAN.npy")
		std  = np.load(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_out_STD.npy")
		return mean, std


	def ApplyNormalize(self, outputs):
		mean = np.load(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_out_MEAN.npy")
		std  = np.load(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_out_STD.npy")
		print mean,std, outputs, (outputs-mean)/std
		return (outputs-mean)/std

	def PrintSampleInformation(self):
		print "From files: ", self.AvailableDataFiles
		return

	def Save(self):
		self.CleanScratch()
		f=open(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+".tdt","wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
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
		self.NormalizeInputs = True
		self.NormalizeOutputs = True
		self.test_begin_mol  = None
		self.test_mols = []
		self.test_mols_done = False
		print "TensorMolData_BP.eles", self.eles
		print "TensorMolData_BP.MeanStoich", self.MeanStoich
		print "TensorMolData_BP.MeanNAtoms", self.MeanStoich
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
		natoms = self.set.NAtoms()
		print "self.dig.eshape", self.dig.eshape, " self.dig.lshape", self.dig.lshape
		cases = np.zeros(tuple([natoms]+list(self.dig.eshape)))
		print "cases:", cases.shape
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
				print "Mol:", mols_done
			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
			#print mi, ins.shape, outs.shape
			cases[casep:casep+nat] = ins
			#print "ins:", ins, " cases:", cases[casep:casep+nat]
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

	def LoadData(self, random=False):
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
		#if (random):
		#	print "Randomizing molecule case..."
		#	ti, to, tm = self.Randomize(ti, to, tm)
		return ti, to, tm

	def Randomize(self, ti, to, tm):
		new_mol_array = np.random.permutation(np.unique(tm[:,0]))
		new_ti = np.zeros((ti.shape[0], ti.shape[1]))
		new_to = np.zeros((to.shape[0], to.shape[1]))
		new_tm = np.zeros((tm.shape[0], 4), dtype = np.int)
		atom_pointer = 0
		for i in range (0, new_mol_array.shape[0]):
			#print i
			mol_index = np.where(tm[:,0]==new_mol_array[i])[0]
			new_tm[atom_pointer:atom_pointer+mol_index.shape[0], 0] = i
			new_tm[atom_pointer:atom_pointer+mol_index.shape[0], 1] = tm[mol_index, 1]
			new_tm[atom_pointer:atom_pointer+mol_index.shape[0], 2] = atom_pointer 
			new_tm[atom_pointer:atom_pointer+mol_index.shape[0], 3] = atom_pointer+mol_index.shape[0]
			new_to[i] = to[new_mol_array[i]]
			new_ti[atom_pointer:atom_pointer+mol_index.shape[0]] = ti[tm[mol_index[0], 2]:tm[mol_index[0], 3]]
			atom_pointer += mol_index.shape[0]
		print "randomizied data:",new_tm, new_ti, new_to
		return new_ti, new_to, new_tm

	def Normalize(self,ti,to):
		if (self.NormalizeInputs):
			for i in range(len(ti)):
				ti[i] = ti[i]/np.linalg.norm(ti[i])
		if (self.NormalizeOutputs):
			mo = np.average(to)
			to -= mo
			stdo = np.std(to)
			to /= stdo
			self.dig.AssignNormalization(mo,stdo)
		return ti, to

	def LoadDataToScratch(self, random=True):
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
		ti, to, tm = self.LoadData(random)
		#ti, to = self.Normalize(ti,to)
		#self.TestRatio = 0.99
		self.TestRatio = 0.2 # debug
		self.NTestMols = int(self.TestRatio * to.shape[0])
		#self.TrainRatio = 0.01
		#self.LastTrainMol = int(self.TrainRatio * to.shape[0])
		self.LastTrainMol = int(to.shape[0]-self.NTestMols)
		print "Using  BP"
		print "LastTrainMol in TensorMolData:", self.LastTrainMol
		print "NTestMols in TensorMolData:", self.NTestMols
		print "Number of molecules in meta:", tm[-1,0]+1
		LastTrainCase=0
		print tm
		# Figure out the number of atoms in training and test.
		for i in range(len(tm)):
			if (tm[i,0] == self.LastTrainMol):
				LastTrainCase = tm[i,2] # exclusive
				break
		print "last train atom: ", LastTrainCase
		print "Num Test atoms: ", len(tm)-LastTrainCase
		print "Num atoms: ", len(tm)

		self.NTrain = LastTrainCase
		self.NTest = len(tm)-LastTrainCase
		self.scratch_inputs = ti[:LastTrainCase]
		self.scratch_outputs = to[:self.LastTrainMol]
		self.scratch_meta = tm[:LastTrainCase]
		self.scratch_test_inputs = ti[LastTrainCase:]
		#print "self.scratch_test_inputs:", self.scratch_test_inputs
		self.scratch_test_outputs = to[self.LastTrainMol:]
		#print "self.LastTrainMol:",self.LastTrainMol, "self.LastTrainCase:", LastTrainCase
		# metadata contains: molecule index, atom type, mol start, mol stop
		# these columns need to be shifted.
		print ("before shift:", tm[LastTrainCase:])
		self.scratch_test_meta = tm[LastTrainCase:]
		self.test_begin_mol = self.scratch_test_meta[0,0]
#		print "before shift case  ", tm[LastTrainCase:LastTrainCase+30], "real", self.set.mols[tm[LastTrainCase, 0]].bonds, self.set.mols[self.test_begin_mol].bonds
		self.scratch_test_meta[:,0] -= self.scratch_test_meta[0,0]
		self.scratch_test_meta[:,3] -= self.scratch_test_meta[0,2]
		self.scratch_test_meta[:,2] -= self.scratch_test_meta[0,2]
		print ("after shift:", self.scratch_test_meta, " test begin mol:", self.test_begin_mol)

		self.ScratchState = 1
		self.ScratchPointer = 0
		self.test_ScratchPointer=0

		# Compute mean Stoichiometry and number of atoms.
		self.eles = np.unique(tm[:,1]).tolist()
		self.eles.sort()
		atomcount = np.zeros(len(self.eles))
		self.MeanStoich = np.zeros(len(self.eles))
		for j in range(len(self.eles)):
			for i in range(len(ti)):
				if (tm[i,1]==self.eles[j]):
					atomcount[j]=atomcount[j]+1
		self.MeanStoich=atomcount/len(to)
		self.MeanNumAtoms = np.sum(self.MeanStoich)
		#self.NormalizeIns()
                self.test_mols = []  # this needs to be removed some time, debug
                self.test_mols_done = False # this needs to be removed some time, debug
		return


        def NormalizeIns(self):
                mean = (np.mean(self.scratch_inputs, axis=0)).reshape((1,-1))
                std = (np.std(self.scratch_inputs, axis=0)).reshape((1, -1))
                self.scratch_inputs = (self.scratch_inputs-mean)/std
                self.scratch_test_inputs = (self.scratch_test_inputs-mean)/std
                np.save(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_in_MEAN.npy", mean)
                np.save(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_in_STD.npy",std)
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
			A an **ordered** list containing
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
				#print "this one is not included:", self.scratch_meta[i]
				break
			elif (self.scratch_meta[i,0] == bmols[0]):
				ignore_first_mol += 1
			else:
				sto[self.eles.index(self.scratch_meta[i,1])]+=1
		currentmol=self.scratch_meta[self.ScratchPointer+ignore_first_mol,0]
		outputs = np.zeros((noutputs))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			matrices.append(np.zeros((sto[e],noutputs)))
		for i in range(self.ScratchPointer+ignore_first_mol, self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_meta[i,0]
			#print "outputpointer:", outputpointer, " currentmol:", currentmol
			#print "currentmol:", currentmol
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
	
	def GetTestBatch(self, ncases, noutputs):
		reset = False
		if (ncases > self.NTest):
			raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
		if (self.test_ScratchPointer+ncases >= self.NTest):
			self.test_ScratchPointer = 0
			self.test_mols_done = True
		inputs = []#np.zeros((ncases, np.prod(self.dig.eshape)))
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
		outputs = np.zeros((noutputs))
		for e in range(len(self.eles)):
			inputs.append(np.zeros((sto[e],np.prod(self.dig.eshape))))
			matrices.append(np.zeros((sto[e],noutputs)))
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
		return [inputs, matrices, outputs]


	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchPointer",self.ScratchPointer
		#print "self.test_ScratchPointer",self.test_ScratchPointer


	def Save(self):
	    self.CleanScratch()
	    f=open(self.path+self.name+"_"+self.dig.name+".tdt","wb")
	    pickle.dump(self.__dict__, f, protocol=1)
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
                f  = open("test_energy_atom_index_for_test.dat","wb")
                pickle.dump(self.test_atom_index, f)
                f.close()
                return

