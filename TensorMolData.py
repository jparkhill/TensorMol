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
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="frag"):  # type can be mol or frag
		TensorData.__init__(self, MSet_,Dig_,Name_)
		self.order = order_
		self.type = type_
		self.num_indis = num_indis_
		self.NTrain = 0
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
		cases = np.zeros((total_case, self.dig.eshape))
		labels = np.zeros((total_case, self.dig.lshape))
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
			self.AvailableElements.append(element)
			self.SamplesPerElement.append(casep*self.dig.NTrainSamples)
		else:
			inf = open(insname,"wb")
			ouf = open(outsname,"wb")
			np.save(inf,cases[:casep])
			np.save(ouf,labels[:casep])
			inf.close()
			ouf.close()
			self.AvailableDataFiles.append([insname,outsname])
			self.AvailableElements.append(element)
			self.SamplesPerElement.append(casep*self.dig.NTrainSamples)
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
		print "TensorMolData_BP.MeanStoich", self.MeanStoich
		print "TensorMolData_BP.MeanNAtoms", self.MeanStoich
		return 

	def CleanScratch(self):
		TensorMolData.CleanScratch(self)
		self.CaseMetadata=None # case X molecule index , element type , first atom in this mol, last atom in this mol (exclusive)
		self.scratch_meta = None
		self.scratch_test_meta = None
		return

	def BuildTrain(self, name_="gdb9",  append=False):
		self.CheckShapes()
		self.name=name_
		print "self.type:", self.type
		if self.type=="frag":
			raise Exception("No BP frags now")
		nmols  = len(self.set.mols)
		natoms = self.set.NAtoms()
		cases = np.zeros(tuple([natoms]+list(self.dig.eshape)))
		labels = np.zeros(tuple([nmols]+list(self.dig.lshape)))
		self.CaseMetadata = np.zeros((natoms, 4), dtype = np.int)
		insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_in.npy"
		outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_out.npy"
		metasname = self.path+"Mol_"+name_+"_"+self.dig.name+"_meta.npy" # Used aggregate and properly sum network inputs and outputs.
		casep=0
		for mi in range(len(self.set.mols)):
			print "casep:", casep
			ins,outs = self.dig.TrainDigest(self.set.mols[mi])
			cases[casep:casep+outs.shape[0]] = ins
			labels[mi] = outs
			for j in range(casep,casep+self.set.mols[mi].NAtoms()):
				self.CaseMetadata[j,0] = mi
				self.CaseMetadata[j,1] = self.set.mols[mi].atoms[j-casep]
				self.CaseMetadata[j,2] = casep
				self.CaseMetadata[j,3] = casep+self.set.mols[mi].NAtoms()
			casep += self.set.mols[mi].NAtoms()
		inf = open(insname,"wb")
		ouf = open(outsname,"wb")
		mef = open(metasname,"wb")
		np.save(inf,cases)
		np.save(ouf,labels)
		np.save(mef,self.CaseMetadata)
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
		if (random):
			print "Cannot yet properly randomize molecule cases. Please implement soon."
			#ti, to, atom_index = self.Randomize(ti, to)
		return ti, to, tm

	def LoadDataToScratch(self, random=True):
		"""
		Reads built training data off disk into scratch space. 
		Divides training and test data. 
		Initializes pointers used to provide training batches.
		
		Args:
			random: Not yet implemented randomization of the read data.
			
		Note:
			Also determines mean stoichiometry
		"""
		ti, to, tm = self.LoadData(random)
		self.NTestMols = int(self.TestRatio * to.shape[0])
		self.LastTrainMol = int(to.shape[0]-self.NTestMols)
		print "LastTrainMol in TensorMolData:", self.LastTrainMol
		print "NTestMols in TensorMolData:", self.NTestMols
		print "Number of molecules in meta:", tm[-1,0]+1
		LastTrainCase=0
		print tm
		# Figure out the number of atoms in training and test.
		for i in range(len(tm)):
			if (tm[i,0] == self.LastTrainMol):
				LastTrainCase = tm[i,3] # exclusive
				break
		print "last train atom: ", LastTrainCase
		print "Num Test atoms: ", len(tm)-LastTrainCase
		print "Num atoms: ", len(tm)
	
		self.NTrain = LastTrainCase
		self.scratch_inputs = ti[:LastTrainCase]
		self.scratch_outputs = to[:self.LastTrainMol]
		self.scratch_meta = tm[:LastTrainCase]
		self.scratch_test_inputs = ti[LastTrainCase:]
		self.scratch_test_outputs = to[self.LastTrainMol:]
		self.scratch_test_meta = tm[LastTrainCase:]

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
			A an **ordered** list containing 
				a (batch_size X flattened input shape) matrix of input cases.
				a (num_ele X 2) int32 tensor of bounds of the input ** which must be passed in element order. **
				a (num_ele X Bnds_size[e] X batch_size_output) tensor which linearly combines the elements.
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

		inputs = np.zeros((ncases, np.prod(self.dig.eshape)))
		bounds = np.zeros((len(self.eles), 2), dtype=np.int32)
		matrices = np.zeros((len(self.eles), ncases, noutputs))
		outputs = np.zeros((noutputs, np.prod(self.dig.lshape)))
		
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
		for i in range(self.ScratchPointer,self.ScratchPointer+ncases):
			if (self.scratch_meta[i,0] == bmols[-1]):
				break
			if (currentmol != self.scratch_meta[i,0]):
				outputpointer = outputpointer+1
				currentmol = self.scratch_meta[i,0]
			inputs[inputpointer] = self.scratch_inputs[i]
			sto[self.eles.index(self.scratch_meta[i,1])]+=1
			outputs[outputpointer] = self.scratch_outputs[self.scratch_meta[i,0]]
			matrices[self.eles.index(self.scratch_meta[i,1]),inputpointer,outputpointer] = 1.0
			inputpointer = inputpointer+1
		
		for e in range(len(self.eles)):
			bounds[e,0]= np.sum(sto[:e]) #start
			bounds[e,1]= sto[e] #size
		
		#print "inputs",inputs
		#print "bounds",bounds
		#print "matrices",matrices
		#print "outputs",outputs
		
		self.ScratchPointer += ncases
		return [inputs, bounds, matrices, outputs]

	def GetTestBatch(self,ncases=1200, num_mol = 1200/6):
		start_time = time.time()
		if (num_mol> self.NTest):
				raise Exception("Test Data is less than the batchsize... :( ")
		reset = False
		if ( self.test_ScratchPointer+num_mol > self.NTest):
				reset = True
		for ele in self.eles:
				if (self.test_Ele_ScratchPointer[ele] >= self.num_test_atoms[ele]):
						reset = True
		if reset==True:
				self.test_ScratchPointer = 0
				for ele in self.eles:
						self.test_Ele_ScratchPointer[ele] = 0

		inputs = np.zeros((ncases, self.dig.eshape[1]))
		outputs = self.scratch_test_outputs[self.test_ScratchPointer:self.test_ScratchPointer+num_mol]
		number_atom_per_ele = dict()
		input_index=0
		for ele in self.eles:
				tmp = 0
				for i in range (self.test_ScratchPointer, self.test_ScratchPointer + num_mol):
						inputs[input_index:input_index+self.test_mol_len[ele][i]]=self.scratch_test_inputs[ele][self.test_Ele_ScratchPointer[ele]:self.test_Ele_ScratchPointer[ele]+self.test_mol_len[ele][i]]
						self.test_Ele_ScratchPointer[ele] += self.test_mol_len[ele][i]
						tmp += self.test_mol_len[ele][i]
						input_index += self.test_mol_len[ele][i]
				number_atom_per_ele[ele]=tmp
		# make the index matrix
		index_matrix = self.Make_Index_Matrix(number_atom_per_ele, num_mol, Train=False) # one needs to know the number of molcule that contained in the ncase atom
		self.test_ScratchPointer += num_mol
		return inputs, outputs, number_atom_per_ele, index_matrix

	def Sort_BPinput_By_Ele(self, inputs, atom_index):
		BPinput = dict()
		for ele in self.eles:
			BPinput[ele]=[]
		for i in range (0, inputs.shape[0]):
			for ele in self.eles:
				for j  in range (0, len(atom_index[ele][i])):
					BPinput[ele].append(inputs[i][atom_index[ele][i][j]])
		#print "sym_funcs: ", len(sym_funcs[1]), len(sym_funcs[8])
		for ele in self.eles:
			BPinput[ele]=np.asarray(BPinput[ele])
		return BPinput

	def Generate_Atom_Index(self):
		atom_index = dict() # for Mol: element X absolute atom X index in molecule
		total_case = 0
		if self.type=="frag":
			for mi in range(len(self.set.mols)):
					total_case += len(self.set.mols[mi].mbe_frags[self.order])
		elif self.type=="mol":
			total_case = len(self.set.mols)
		else:
			raise Exception ("Unknown Type")
		for ele in self.eles:
			atom_index[ele]=[[] for i in range(total_case)]
		loop_index = 0
		if self.type=="frag":
			for i in range (0, len(self.set.mols)):
				for  j, frag in enumerate(self.set.mols[i].mbe_frags[self.order]):
					for k in range (0, frag.NAtoms()):
						(atom_index[int(frag.atoms[k])][loop_index]).append(k)
					loop_index += 1
		elif self.type=="mol":
			for i in range (0, len(self.set.mols)):
				for k in range (0, self.set.mols[i].NAtoms()):
					(atom_index[int(self.set.mols[i].atoms[k])][loop_index]).append(k)
					loop_index += 1
		else:
			raise Exception ("Unknown Type")
		for ele in self.eles:
			atom_index[ele] = np.asarray(atom_index[ele])
		print "atom_index:", atom_index, atom_index[1].shape, atom_index[8].shape
		return atom_index

	def Make_Index_Matrix(self, number_atom_per_ele, num_mol, Train=True):
		index_matrix = dict() # element x
		if Train==True:
			for ele in self.eles:
				index_matrix[ele] = np.zeros((number_atom_per_ele[ele], num_mol),dtype=np.bool)
				atom_index = 0
				for i in range (self.ScratchPointer, self.ScratchPointer + num_mol):
					for j in range (atom_index, atom_index + self.train_mol_len[ele][i]):
						#print "index of mol:", i, "index of atom:", j
						index_matrix[ele][j][i-self.ScratchPointer]=True
					atom_index += self.train_mol_len[ele][i]
		else:
			for ele in self.eles:
				index_matrix[ele] = np.zeros((number_atom_per_ele[ele], num_mol),dtype=np.bool)
				atom_index = 0
				for i in range (self.test_ScratchPointer, self.test_ScratchPointer + num_mol):
					for j in range (atom_index, atom_index + self.test_mol_len[ele][i]):
						#print "index of mol:", i, "index of atom:", j
						index_matrix[ele][j][i-self.test_ScratchPointer]=True
					atom_index += self.test_mol_len[ele][i]	
		#print "index_matrix", index_matrix
		return index_matrix

	def KRR(self):
		from sklearn.kernel_ridge import KernelRidge
		ti, to, atom_index = self.LoadData(True)
		ti = ti.reshape(( to.shape[0], -1 ))
		print "KRR: input shape", ti.shape, " output shape", to.shape, " input", ti
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


	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchPointer",self.ScratchPointer
		print "self.test_ScratchPointer",self.test_ScratchPointer

