#
# Contains Routines to generate training sets 
# Combining a dataset, sampler and an embedding. (CM etc.)
#
import os, gc
from Sets import *
from Digest import *
#import tables should go to hdf5 soon...

class TensorData():
	"""
		A Training Set is a Molecule set, with a sampler and an embedding
		The sampler chooses points in the molecular volume.
		The embedding turns that into inputs and labels for a network to regress.
	"""
	def __init__(self, MSet_=None, Dig_=None, Name_=None, MxTimePerElement_=20400):
		self.path = "./trainsets/"
		self.suffix = ".pdb"
		self.set = MSet_
		self.dig = Dig_
		self.CurrentElement = None # This is a mode switch for when TensorData provides training data.
		self.SamplesPerElement = []
		self.AvailableElements = []
		self.AvailableDataFiles = []
		self.NTest = 0  # assgin this value when the data is loaded
		self.TestRatio = 0.2 # number of cases withheld for testing.
		self.MxTimePerElement=MxTimePerElement_
		self.MxMemPerElement=8000 # Max Array for an element in MB
		self.ScratchNCase = 0
		self.ScratchState=None
		self.ScratchPointer=0 # for non random batch iteration.
		self.scratch_inputs=None
		self.scratch_outputs=None
		self.scratch_test_inputs=None # These should be partitioned out by LoadElementToScratch
		self.scratch_test_outputs=None
		# Ordinarily during training batches will be requested repeatedly
		# for the same element. Introduce some scratch space for that.
		if (not os.path.isdir(self.path)):
			os.mkdir(self.path)
		if (Name_!= None):
			self.name = Name_
			self.Load()
			self.QueryAvailable() # Should be a sanity check on the data files.
			return
		elif (MSet_==None or Dig_==None):
			raise Exception("I need a set and Digester if you're not loading me.")
		self.name = ""
	
	def CleanScratch(self):
		self.ScratchState=None
		self.ScratchPointer=0 # for non random batch iteration.
		self.scratch_inputs=None
		self.scratch_outputs=None
		self.scratch_test_inputs=None # These should be partitioned out by LoadElementToScratch
		self.scratch_test_outputs=None
		return

	def NTrainCasesInScratch(self):
		return self.scratch_inputs.shape[0]
	def NTestCasesInScratch(self):
		return self.scratch_inputs.shape[0]

	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchNCase", self.ScratchNCase
		print "self.ScratchPointer",self.ScratchPointer
		if (self.scratch_outputs != None):
			print "self.scratch_inputs.shape",self.scratch_inputs.shape
			print "self.scratch_outputs.shape",self.scratch_outputs.shape
			print "scratch_test_inputs.shape",self.scratch_test_inputs.shape
			print "scratch_test_outputs.shape",self.scratch_test_outputs.shape

	def QueryAvailable(self):
		""" If Tensordata has already been made, this looks for it under a passed name."""
		for i in range(MAX_ATOMIC_NUMBER):
			if (os.path.isfile(self.path+self.name+"_"+self.dig.name+"_"+str(i)+"_in.npy") and os.path.isfile(self.path+self.name+"_"+self.dig.name+"_"+str(i)+"_out.npy")):
				if (self.AvailableElements.count(i)==0):
					self.AvailableElements.append(i)
		self.AvailableElements.sort()
		# It should probably check the sanity of each input/outputfile as well...
		return

	def CheckShapes(self):
		# Establish case and label shapes.
		tins,touts = self.dig.TrainDigest(self.set.mols[0],self.set.mols[0].atoms[0])
		print "self.dig input shape: ", self.dig.eshape
		print "self.dig output shape: ", self.dig.lshape
		print "TrainDigest input shape: ", tins.shape
		print "TrainDigest output shape: ", touts.shape
		if (self.dig.eshape == None or self.dig.lshape ==None):
			raise Exception("Ain't got no fucking shape.")

	def BuildTrain(self, name_="gdb9", atypes=[], append=False):
		""" 
			Generates probability inputs for all training data using the chosen digester.
			All the inputs for a given atom are built separately.
			Now requires some sort of PES information.
				If PESSamples = [] it will use a Go-model (CITE:http://dx.doi.org/10.1016/S0006-3495(02)75308-3)
				The code that uses ab-initio samples isn't written yet, but should be.
		"""
		self.CheckShapes()
		self.name=name_
		print "Generating Train set:", self.name, " from mol set ", self.set.name, " of size ", len(self.set.mols)," molecules"
		if (len(self.set.mols[0].PESSamples)==0):
			print "--- using a Go model of the PES ---"
		else:
			print "--- using ab-initio PES ---"
			raise Exception("Finish writing code to use an ab-initio PES")
		if (len(atypes)==0):
			atypes = self.set.AtomTypes()
		print "Will train atoms: ", atypes
		# Determine the size of the training set that will be made.
		nofe = [0 for i in range(MAX_ATOMIC_NUMBER)]
		for element in atypes:
			for m in self.set.mols:
				nofe[element] = nofe[element]+m.NumOfAtomsE(element)
		reqmem = [nofe[element]*self.dig.NTrainSamples*np.prod(self.dig.eshape)*4/1024.0/1024.0 for element in range(MAX_ATOMIC_NUMBER)]
		truncto = [nofe[i] for i in range(MAX_ATOMIC_NUMBER)]
		for element in atypes:
			print "AN: ", element, " contributes ", nofe[element]*self.dig.NTrainSamples , " samples, requiring ", reqmem[element], " MB of in-core memory. "
			if (reqmem[element]>self.MxMemPerElement):
				truncto[element]=int(self.MxMemPerElement/reqmem[element]*nofe[element])
				print "Truncating element ", element, " to ",truncto[element]," Samples"
		for element in atypes:
			print "Digesting atom: ", element
			cases = np.zeros(shape=tuple([truncto[element]*self.dig.NTrainSamples]+list(self.dig.eshape)), dtype=np.float32)
			labels = np.zeros(shape=tuple([truncto[element]*self.dig.NTrainSamples]+list(self.dig.lshape)), dtype=np.float32)
			casep = 0
			t0 = time.time()
			for mi in range(len(self.set.mols)):
				m = self.set.mols[mi]
				if (mi%100==0):
					print "Digested ", mi ," of ",len(self.set.mols)
				ins,outs = self.dig.TrainDigest(self.set.mols[mi],element)
				GotOut = outs.shape[0]
				if (GotOut!=ins.shape[0]):
					raise Exception("Insane Digest")
				if (truncto[element]<casep+GotOut):
					print "Truncating at ", casep, "Samples"
					break
				else:
					cases[casep:casep+outs.shape[0]] += ins
					labels[casep:casep+outs.shape[0]] += outs
					casep += outs.shape[0]
				if ((time.time()-t0)>self.MxTimePerElement):
					break
				if (mi==40 and casep>=40):
					print "Seconds to process 40 molecules: ", time.time()-t0
					print "Average label: ", np.average(labels[:casep])
					if  (not np.isfinite(np.average(labels[:casep]))):
						raise Exception("Bad Labels")
				#ins, outs = self.dig.TrainDigestWGoForce(self.set.mols[mi], element)
				if (mi%300):
					gc.collect()
				if (mi%1000==0):
					print mi
			# Write the numpy arrays for this element.
			insname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_in.npy"
			outsname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_out.npy"
			if (not append):
				inf = open(insname,"wb")
				ouf = open(outsname,"wb")
				np.save(inf,cases[:casep])
				np.save(ouf,labels[:casep])
				inf.close()
				ouf.close()
				self.AvailableDataFiles.append([insname,outsname])
				self.AvailableElements.append(element)
				self.SamplesPerElement.append(casep*self.dig.NTrainSamples)
			else:
				inf = open(insname,"a+b")
				ouf = open(outsname,"a+b")
				np.save(inf,cases)
				np.save(ouf,labels)
				inf.close()
				ouf.close()
		self.Save() #write a convenience pickle.
		return

	def BuildSamples(self,name_="gdb9", atypes=[],uniform=False):
		""" 
			Generates sampled data set without preparing the probabilities or embedding 
			if uniform is true, it generate a grid of uniform samples up to 4 angstrom away from 
			the central atom to generate known-good validation data. 
		"""
		self.name=name_
		print "Sampling set:", self.name, " from mol set ", self.set.name, " of size ", len(self.set.mols)," molecules"
		if (uniform):
			print "sampling uniformily"
		Cache=MSet(self.name)
		print "Will store completed PES samples in ./datasets/"+self.name
		if (len(atypes)==0):
			atypes = self.set.AtomTypes()
		print "Will sample atoms: ", atypes
		# Determine the size of the training set that will be made.
		nofe = [0 for i in range(MAX_ATOMIC_NUMBER)]
		for element in atypes:
			for m in self.set.mols:
				nofe[element] = nofe[element]+m.NumOfAtomsE(element)
		if (uniform): 
			for element in atypes:
				print "AN: ", element, " contributes ", nofe[element]*20*20*20 , " samples "
		else: 
			for element in atypes:
				print "AN: ", element, " contributes ", nofe[element]*self.dig.NTrainSamples , " samples "
		t0 = time.time()
		for element in atypes:
			print "Digesting atom: ", element
			casep = 0
			for mi in range(len(self.set.mols)):
				m = self.set.mols[mi]
				if (mi%1000==0):
					print "Digested ", mi ," of ",len(self.set.mols)
				self.dig.SampleDigestWPyscf(self.set.mols[mi],element,uniform)
				Cache.mols.append(self.set.mols[mi])
				if (mi%10==0 or uniform):
					Cache.Save()
				print mi
		self.Save() #write a convenience pickle.
		Cache.Save()
		return

	def GetTrainBatch(self,ele,ncases=2000,random=False):
		if (self.ScratchState != ele):
			self.LoadElementToScratch(ele)
		if (ncases>self.ScratchNCase):
			raise Exception("Training Data is less than the batchsize... :( ")
		if ( self.ScratchPointer+ncases >= self.scratch_inputs.shape[0]):
			self.ScratchPointer = 0 #Sloppy.
		tmp=(self.scratch_inputs[self.ScratchPointer:self.ScratchPointer+ncases], self.scratch_outputs[self.ScratchPointer:self.ScratchPointer+ncases])
		self.ScratchPointer += ncases
		return tmp

	def GetTestBatch(self,ele,ncases=200, ministep = 0):
		if (self.ScratchState != ele):
			self.LoadElementToScratch(ele)
		if (ncases>self.scratch_test_inputs.shape[0]):
			raise Exception("Test Data is less than the batchsize... :( ")
		return (self.scratch_test_inputs[ncases*(ministep):ncases*(ministep+1)], self.scratch_test_outputs[ncases*(ministep):ncases*(ministep+1)])

	def EvaluateTestBatch(self,desired,preds):
		self.dig.EvaluateTestOutputs(desired,preds)
		return

	def MergeWith(self,ASet_):
		''' Augments my training data with another set, which for example may have been generated on another computer.'''
		self.QueryAvailable()
		ASet_.QueryAvailable()
		print "Merging", self.name, " with ", ASet_.name
		for ele in ASet_.AvailableElements:
			if (self.AvailableElements.count(ele)==0):
				raise Exception("WriteME192837129874")
			else:
				mti,mto = self.LoadElement(ele)
				ati,ato = ASet_.LoadElement(ele)
				labelshapes = list(mti.shape)[1:]
				eshapes = list(mto.shape)[1:]
				ASet_labelshapes = list(ati.shape)[1:]
				ASet_eshapes = list(ato.shape)[1:]
				if (labelshapes != ASet_labelshapes or eshapes != ASet_eshapes):
					raise Exception("incompatible")
				if (self.dig.name != ASet_.dig.name):
					raise Exception("incompatible")
				print "Merging ", self.name, " element, ", ele ," with ", ASet_.name
				mti=np.concatenate((mti,ati),axis=0)
				mto=np.concatenate((mto,ato),axis=0)
				print "The new element train set will have", mti.shape[0], " cases in it"
				insname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in.npy"
				outsname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_out.npy"
				inf = open(insname,"wb")
				ouf = open(outsname,"wb")
				np.save(inf,mti)
				np.save(ouf,mto)
				inf.close()
				ouf.close()

	def LoadElement(self, ele, Random=False):
		insname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in.npy"
		outsname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_out.npy"
		try:
			inf = open(insname,"rb")
			ouf = open(outsname,"rb")
			ti = np.load(inf)
			to = np.load(ouf)
			inf.close()
			ouf.close()
		except Exception as Ex:
			print "Failed to read:",insname, " or ",outsname
			raise Ex
		if (ti.shape[0] != to.shape[0]):
			raise Exception("Bad Training Data.")
		#ti = ti.reshape((ti.shape[0],-1))  # flat data to [ncase, num_per_case]
		#to = to.reshape((to.shape[0],-1))  # flat labels to [ncase, 1]
		if (Random):
			idx = np.random.permutation(ti.shape[0])
			ti = ti[idx]
			to = to[idx]
		self.ScratchNCase = to.shape[0]
		return ti, to

	def LoadElementToScratch(self,ele, Random=True, ExpandIsometries=True):
		ti, to = self.LoadElement(ele, Random)
		if (ExpandIsometries and self.dig.name=="SensoryBasis" and self.dig.OType=="Disp"):
			print "Expanding the given set over isometries."
			ti,to = GRIDS.ExpandIsometries(ti,to)
		self.NTest = int(self.TestRatio * ti.shape[0])
		self.scratch_inputs = ti[:ti.shape[0]-self.NTest]
		self.scratch_outputs = to[:ti.shape[0]-self.NTest]
		self.scratch_test_inputs = ti[ti.shape[0]-self.NTest:]
		self.scratch_test_outputs = to[ti.shape[0]-self.NTest:]
		self.ScratchState = ele
		self.ScratchPointer=0
		return

	def NormalizeInputs(self, ele):
		mean = (np.mean(self.scratch_inputs, axis=0)).reshape((1,-1))
		std = (np.std(self.scratch_inputs, axis=0)).reshape((1, -1))
		self.scratch_inputs = (self.scratch_inputs-mean)/std
		self.scratch_test_inputs = (self.scratch_test_inputs-mean)/std
		np.save(self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in_MEAN.npy", mean)
		np.save(self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in_STD.npy",std)
		return

	def ApplyNormalize(self, inputs, ele):
		mean = np.load(self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in_MEAN.npy")
		std  = np.load(self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in_STD.npy")
		return (inputs-mean)/std

	def Save(self):
		self.CleanScratch()
		f=open(self.path+self.name+"_"+self.dig.name+".tdt","wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
		return

	def Load(self):
		print "Unpickling Tensordata"
		f = open(self.path+self.name+".tdt","rb")
		tmp=pickle.load(f)
		self.__dict__.update(tmp)
		f.close()
		self.CheckShapes()
		print "Training data manager loaded."
		print "Based on ", len(self.set.mols), " molecules "
		print "Based on files: ",self.AvailableDataFiles
		self.PrintSampleInformation()
		self.dig.Print()
		return

	def PrintSampleInformation(self):
		for i in range(len(self.AvailableElements)):
			print "AN: ", self.AvailableElements[i], " contributes ", self.SamplesPerElement[i] , " samples "
			print "From files: ", self.AvailableDataFiles[i]
		return




