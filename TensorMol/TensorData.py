"""
 Contains Routines to generate training sets
 Combining a dataset, sampler and an embedding. (CM etc.)
"""
import os, gc
from Sets import *
from Digest import *
from Transformer import *
#import tables should go to hdf5 soon...

class TensorData():
	"""
		A Training Set is a Molecule set, with a sampler and an embedding
		The sampler chooses points in the molecular volume.
		The embedding turns that into inputs and labels for a network to regress.
	"""
	def __init__(self, MSet_=None, Dig_=None, Name_=None, type_="atom"):
		"""
		make a tensordata object
		Several arguments of PARAMS affect this classes behavior
		Args:
			MSet_: A MoleculeSet
			Dig_: A Digester
			Name_: A Name
		"""
		self.path = "./trainsets/"
		self.suffix = ".pdb"
		self.set = MSet_
		self.dig = Dig_
		self.type = type_
		self.CurrentElement = None # This is a mode switch for when TensorData provides training data.
		self.SamplesPerElement = []
		self.AvailableElements = []
		self.AvailableDataFiles = []
		self.NTest = 0  # assgin this value when the data is loaded
		self.TestRatio = 0.2 # number of cases withheld for testing.
		self.Random=True # Whether to scramble training data (can be disabled for debugging purposes)
		self.ScratchNCase = 0
		self.ScratchState=None
		self.ScratchPointer=0 # for non random batch iteration.
		self.scratch_inputs=None
		self.scratch_outputs=None
		self.scratch_test_inputs=None # These should be partitioned out by LoadElementToScratch
		self.scratch_test_outputs=None
		self.Classify=PARAMS["Classify"]
		self.MxTimePerElement=PARAMS["MxTimePerElement"]
		self.MxMemPerElement=PARAMS["MxMemPerElement"]
		# self.NormalizeInputs = PARAMS["NormalizeInputs"]
		# self.NormalizeOutputs = PARAMS["NormalizeOutputs"]
		# self.NormalizeOutputsLog = PARAMS["NormalizeOutputsLog"]
		self.ChopTo = PARAMS["ChopTo"]
		self.ExpandIsometriesAltogether = False
		self.ExpandIsometriesBatchwise = False

		# Ordinarily during training batches will be requested repeatedly
		# for the same element. Introduce some scratch space for that.
		if (not os.path.isdir(self.path)):
			os.mkdir(self.path)
		if (Name_!= None):
			self.name = Name_
			self.Load()
			self.tform = Transformer(PARAMS["InNormRoutine"], PARAMS["OutNormRoutine"], self.dig.name, self.dig.OType)
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

	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchNCase", self.ScratchNCase
		print "self.NTrainCasesInScratch()", self.NTrainCasesInScratch()
		print "self.ScratchPointer",self.ScratchPointer
		if (self.scratch_outputs != None):
			print "self.scratch_inputs.shape",self.scratch_inputs.shape
			print "self.scratch_outputs.shape",self.scratch_outputs.shape
			print "scratch_test_inputs.shape",self.scratch_test_inputs.shape
			print "scratch_test_outputs.shape",self.scratch_test_outputs.shape

	def CheckShapes(self):
		# Establish case and label shapes.
		tins,touts = self.dig.TrainDigest(self.set.mols[0],self.set.mols[0].atoms[0])
		print "self.dig input shape: ", self.dig.eshape
		print "self.dig output shape: ", self.dig.lshape
		print "TrainDigest input shape: ", tins.shape
		print "TrainDigest output shape: ", touts.shape
		if (self.dig.eshape == None or self.dig.lshape ==None):
			raise Exception("Ain't got no fucking shape.")


	def BuildTrainMolwise(self, name_="gdb9", atypes=[], append=False, MakeDebug=False):
		"""
			Generates probability inputs for all training data using the chosen digester.
			This version builds all the elements at the same time.
			The other version builds each element separately
			If PESSamples = [] it will use a Go-model (CITE:http://dx.doi.org/10.1016/S0006-3495(02)75308-3)
		"""
		self.CheckShapes()
		self.name=name_
		LOGGER.info("Generating Train set: %s from mol set %s of size %i molecules", self.name, self.set.name, len(self.set.mols))
		if (len(self.set.mols[0].PESSamples)==0):
			LOGGER.info("--- using a Go model of the PES ---")
		else:
			LOGGER.info("--- using ab-initio PES ---")
			raise Exception("Finish writing code to use an ab-initio PES")
		if (len(atypes)==0):
			atypes = self.set.AtomTypes()
		LOGGER.debug("Will train atoms: "+str(atypes))
		# Determine the size of the training set that will be made.
		nofe = [0 for i in range(MAX_ATOMIC_NUMBER)]
		for element in atypes:
			for m in self.set.mols:
				nofe[element] = nofe[element]+m.NumOfAtomsE(element)
		truncto = [nofe[i] for i in range(MAX_ATOMIC_NUMBER)]
		cases_list = [np.zeros(shape=tuple([nofe[element]*self.dig.NTrainSamples]+list(self.dig.eshape)), dtype=np.float32) for element in atypes]
		labels_list = [np.zeros(shape=tuple([nofe[element]*self.dig.NTrainSamples]+list(self.dig.lshape)), dtype=np.float32) for element in atypes]
		casep_list = [0 for element in atypes]
		t0 = time.time()
		ord=np.random.permutation(len(self.set.mols))
		mols_done = 0
		try:
		 	for mi in ord:
				m = self.set.mols[mi]
				ins,outs = self.dig.TrainDigestMolwise(m)
				for i in range(m.NAtoms()):
					# Route all the inputs and outputs to the appropriate place...
					ai = atypes.tolist().index(m.atoms[i])
					cases_list[ai][casep_list[ai]] = ins[i]
					labels_list[ai][casep_list[ai]] = outs[i]
					casep_list[ai] = casep_list[ai]+1
				if (mols_done%10000==0 and mols_done>0):
					gc.collect()
				if (mols_done%10000==0 and mols_done>0):
					print mols_done
				if (mols_done==400):
					print "Seconds to process 400 molecules: ", time.time()-t0
				mols_done = mols_done + 1
		except Exception as Ex:
				print "Likely you need to re-install MolEmb.", Ex
		for element in atypes:
			# Write the numpy arrays for this element.
			ai = atypes.tolist().index(element)
			insname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_in.npy"
			outsname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_out.npy"
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
				try:
					cases = np.concatenate((cases_list[ai][:casep_list[ai]],ti))
					labels = np.concatenate((labels_list[ai][:casep_list[ai]],to))
				except Exception as Ex:
					print "Size mismatch with old training data, clear out trainsets"
				inf = open(insname,"wb")
				ouf = open(outsname,"wb")
				np.save(inf,cases)
				np.save(ouf,labels)
				inf.close()
				ouf.close()
				self.AvailableDataFiles.append([insname,outsname])
				self.AvailableElements.append(element)
				self.SamplesPerElement.append(casep_list[ai]*self.dig.NTrainSamples)
			else:
				inf = open(insname,"wb")
				ouf = open(outsname,"wb")
				np.save(inf,cases_list[ai][:casep_list[ai]])
				np.save(ouf,labels_list[ai][:casep_list[ai]])
				inf.close()
				ouf.close()
				self.AvailableDataFiles.append([insname,outsname])
				self.AvailableElements.append(element)
				self.SamplesPerElement.append(casep_list[ai]*self.dig.NTrainSamples)
		self.Save() #write a convenience pickle.
		return

	def BuildTrain(self, name_="gdb9", atypes=[], append=False, MakeDebug=False):
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
			DebugCases=[]
			print "Digesting atom: ", element
			cases = np.zeros(shape=tuple([truncto[element]*self.dig.NTrainSamples]+list(self.dig.eshape)), dtype=np.float32)
			labels = np.zeros(shape=tuple([truncto[element]*self.dig.NTrainSamples]+list(self.dig.lshape)), dtype=np.float32)
			casep = 0
			t0 = time.time()
			for mi in range(len(self.set.mols)):
				m = self.set.mols[mi]
				if (mi%100==0):
					print "Digested ", mi ," of ",len(self.set.mols)
				ins,outs=None,None
				if (MakeDebug):
					ins,outs,db = self.dig.TrainDigest(self.set.mols[mi],element,True)
					DebugCases = DebugCases + db
				else:
					ins,outs = self.dig.TrainDigest(self.set.mols[mi],element)
				GotOut = outs.shape[0]
				if (GotOut!=ins.shape[0]):
					raise Exception("Insane Digest")
				if (truncto[element]<casep+GotOut):
					print "Truncating at ", casep, "Samples because ",truncto[element], " is less than ",casep," plus ",GotOut
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
				if (mi%1000):
					gc.collect()
				if (mi%1000==0):
					print mi
			# Write the numpy arrays for this element.
			insname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_in.npy"
			outsname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_out.npy"
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
			if (MakeDebug):
				dbgname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_dbg.tdt"
				f=open(dbgname,"wb")
				pickle.dump(DebugCases, f, protocol=pickle.HIGHEST_PROTOCOL)
				f.close()
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

	def GetTrainBatch(self,ele,ncases=2000,random=True):
		if (self.ScratchState != ele):
			self.LoadElementToScratch(ele,random)
		if (ncases>self.NTrainCasesInScratch()):
			raise Exception("Training Data is less than the batchsize... :( ")
		if (self.ExpandIsometriesBatchwise):
			if ( self.ScratchPointer*GRIDS.NIso()+ncases >= self.NTrainCasesInScratch() ):
				self.ScratchPointer = 0 #Sloppy.
			neff = int(ncases/GRIDS.NIso())+1
			tmp=GRIDS.ExpandIsometries(self.scratch_inputs[self.ScratchPointer:self.ScratchPointer+neff], self.scratch_outputs[self.ScratchPointer:self.ScratchPointer+neff])
			#print tmp[0].shape, tmp[1].shape
			self.ScratchPointer += neff
			return tmp[0][:ncases], tmp[1][:ncases]
		else:
			if ( self.ScratchPointer+ncases >= self.NTrainCasesInScratch()):
				self.ScratchPointer = 0 #Sloppy.
			tmp=(self.scratch_inputs[self.ScratchPointer:self.ScratchPointer+ncases], self.scratch_outputs[self.ScratchPointer:self.ScratchPointer+ncases])
			self.ScratchPointer += ncases
			return tmp

	def GetTestBatch(self,ele,ncases=200, ministep = 0):
		if (self.ScratchState != ele):
			self.LoadElementToScratch(ele,False)
		if (ncases>self.scratch_test_inputs.shape[0]):
			print "Test Data is less than the batchsize... :( "
			tmpinputs=np.zeros(shape=tuple([ncases]+list(self.dig.eshape)), dtype=np.float32)
			tmpoutputs=np.zeros(shape=tuple([ncases]+list(self.dig.lshape)), dtype=np.float32)
			tmpinputs[0:self.scratch_test_inputs.shape[0]] += self.scratch_test_inputs
			tmpoutputs[0:self.scratch_test_outputs.shape[0]] += self.scratch_test_outputs
			return (tmpinputs[ncases*(ministep):ncases*(ministep+1)], tmpoutputs[ncases*(ministep):ncases*(ministep+1)])
		return (self.scratch_test_inputs[ncases*(ministep):ncases*(ministep+1)], self.scratch_test_outputs[ncases*(ministep):ncases*(ministep+1)])

	def EvaluateTestBatch(self, desired, predicted):
		#try:
		print "Evaluating, ", len(desired), " predictions... "
		print desired.shape, predicted.shape
		if (self.dig.OType=="Disp" or self.dig.OType=="Force" or self.dig.OType == "GoForce"):
			ders=np.zeros(len(desired))
			#comp=np.zeros(len(desired))
			if (self.tform.outnorm != None):
				for i in range(len(desired)):
					ders[i] = np.linalg.norm(self.tform.UnNormalizeOuts(predicted[i,-3:])-self.tform.UnNormalizeOuts(desired[i,-3:]))
				for i in range(100):
					print "Desired: ",i,self.tform.UnNormalizeOuts(desired[i,-3:])," Predicted: ",self.tform.UnNormalizeOuts(predicted[i,-3:])
			else:
				for i in range(len(desired)):
					ders[i] = np.linalg.norm(predicted[i,-3:]-desired[i,-3:])
				for i in range(100):
					print "Desired: ",i,desired[i,-3:]," Predicted: ",predicted[i,-3:]
			LOGGER.info("Test displacement errors direct (mean,std) %f,%f",np.average(ders),np.std(ders))
			LOGGER.info("Average learning target: %s, Average output (direct) %s", str(np.average(desired[:,-3:],axis=0)),str(np.average(predicted[:,-3:],axis=0)))
			print "Fraction of incorrect directions: ", np.sum(np.sign(desired[:,-3:])-np.sign(predicted[:,-3:]))/(6.*len(desired))
		elif (self.dig.OType == "GoForceSphere"):
			# Convert them back to cartesian
			desiredc = SphereToCartV(desired)
			predictedc = SphereToCartV(predicted)
			ders=np.zeros(len(desired))
			#comp=np.zeros(len(desired))
			if (self.tform.outnorm != None):
				for i in range(len(desiredc)):
					ders[i] = np.linalg.norm(self.tform.UnNormalizeOuts(predictedc[i,-3:])-self.tform.UnNormalizeOuts(desiredc[i,-3:]))
				for i in range(100):
					print "Desired: ",i,self.tform.UnNormalizeOuts(desiredc[i,-3:])," Predicted: ",self.tform.UnNormalizeOuts(predictedc[i,-3:])
			else:	
				for i in range(len(desiredc)):
					ders[i] = np.linalg.norm(predictedc[i,-3:]-desiredc[i,-3:])
				for i in range(100):
					print "Desired: ",i,desiredc[i,-3:]," Predicted: ",predictedc[i,-3:]
			LOGGER.info("Test displacement errors direct (mean,std) %f,%f",np.average(ders),np.std(ders))
			LOGGER.info("Average learning target: %s, Average output (direct) %s", str(np.average(desiredc[:,-3:],axis=0)),str(np.average(predictedc[:,-3:],axis=0)))
			print "Fraction of incorrect directions: ", np.sum(np.sign(desiredc[:,-3:])-np.sign(predictedc[:,-3:]))/(6.*len(desiredc))
		elif (self.dig.OType=="SmoothP"):
			ders=np.zeros(len(desired))
			iers=np.zeros(len(desired))
			comp=np.zeros(len(desired))
			for i in range(len(desired)):
				#print "Direct - desired disp", desired[i,-3:]," Pred disp", predicted[i,-3:]
				Pr = GRIDS.Rasterize(predicted[i,:GRIDS.NGau3])
				Pr /= np.sum(Pr)
				p=np.dot(GRIDS.MyGrid().T,Pr)
				#print "fit disp: ", p
				ders[i] = np.linalg.norm(predicted[i,-3:]-desired[i,-3:])
				iers[i] = np.linalg.norm(p-desired[i,-3:])
				comp[i] = np.linalg.norm(p-predicted[i,-3:])
			print "Test displacement errors direct (mean,std) ", np.average(ders),np.std(ders), " indirect ",np.average(iers),np.std(iers), " Comp ", np.average(comp), np.std(comp)
			print "Average learning target: ", np.average(desired[:,-3:],axis=0),"Average output (direct)",np.average(predicted[:,-3:],axis=0)
			print "Fraction of incorrect directions: ", np.sum(np.sign(desired[:,-3:])-np.sign(predicted[:,-3:]))/(6.*len(desired))
		elif (self.dig.OType=="StoP"):
			raise Exception("Unknown Digester Output Type.")
		elif (self.dig.OType=="Energy"):
			raise Exception("Unknown Digester Output Type.")
		elif (self.dig.OType=="GoForce_old_version"): # python version is fine for here
			raise Exception("Unknown Digester Output Type.")
		elif (self.dig.OType=="HardP"):
			raise Exception("Unknown Digester Output Type.")
		else:
			raise Exception("Unknown Digester Output Type.")
		#except Exception as Ex:
		#	print "Something went wrong"
		#	pass
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

	def Save(self):
		self.CleanScratch()
		f=open(self.path+self.name+"_"+self.dig.name+".tdt","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
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
		self.QueryAvailable()
		self.PrintSampleInformation()
		self.dig.Print()
		return

	def QueryAvailable(self):
		""" If Tensordata has already been made, this looks for it under a passed name."""
		self.AvailableElements=[]
		self.SamplesPerElement=[]
		for i in range(MAX_ATOMIC_NUMBER):
			if (os.path.isfile(self.path+self.name+"_"+self.dig.name+"_"+str(i)+"_in.npy") and os.path.isfile(self.path+self.name+"_"+self.dig.name+"_"+str(i)+"_out.npy")):
				self.AvailableElements.append(i)
				ifname = self.path+self.name+"_"+self.dig.name+"_"+str(i)+"_out.npy"
				ofname = self.path+self.name+"_"+self.dig.name+"_"+str(i)+"_out.npy"
				inf = open(ifname,"rb")
				ouf = open(ofname,"rb")
				ti = np.load(inf)
				to = np.load(ouf)
				inf.close()
				ouf.close()
				if (len(ti)!=len(to)):
					print "...Retrain element i"
				else:
					if (self.ChopTo!=None):
						self.SamplesPerElement.append(min(len(ti),self.ChopTo))
					else:
						self.SamplesPerElement.append(len(ti))
		self.AvailableElements.sort()
		# It should probably check the sanity of each input/outputfile as well...
		return

	def LoadElement(self, ele, Random=True, DebugData_=False):
		insname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in.npy"
		outsname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_out.npy"
		dbgname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_dbg.tdt"
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
		if (self.ChopTo!=None):
			ti = ti[:self.ChopTo]
			to = to[:self.ChopTo]
		if (DebugData_):
			print "DEBUGGING, ", len(ti), " cases.."
			f = open(dbgname,"rb")
			dbg=pickle.load(f)
			f.close()
			print "Found ", len(dbg), " pieces of debug information for this element... "
			for i in range(len(dbg)):
				print "CASE:", i, " was for ATOM", dbg[i][1], " At Point ", dbg[i][2]
				ds=GRIDS.Rasterize(ti[i])
				GridstoRaw(ds, GRIDS.NPts, "InpCASE"+str(i))
				print dbg[i][0].coords
				print dbg[i][0].atoms
		#ti = ti.reshape((ti.shape[0],-1))  # flat data to [ncase, num_per_case]
		#to = to.reshape((to.shape[0],-1))  # flat labels to [ncase, 1]
		if (Random):
			idx = np.random.permutation(ti.shape[0])
			ti = ti[idx]
			to = to[idx]
		self.ScratchNCase = to.shape[0]
		return ti, to

	def NormalizeInputs(self, ele):
		"""
		PLEASE MAKE THESE TWO WAYS OF NORMALIZING CONSISTENT.
		AND REMOVE THE OTHER ONE...
		JAP
		"""
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

	def Normalize(self,ti,to):
		"""
		PLEASE MAKE THESE TWO WAYS OF NORMALIZING CONSISTENT.
		AND REMOVE THE OTHER ONE...
		-JAP
		"""
		if (self.NormalizeInputs):
			for i in range(len(ti)):
				ti[i] = ti[i]/(np.linalg.norm(ti[i])+1.0E-8)
		if (self.NormalizeOutputs):
			mo = np.average(to)
			to -= mo
			stdo = np.std(to)
			to /= stdo
			self.tform.AssignNormalization(mo,stdo)
		if (self.NormalizeOutputsLog):
			for x in np.nditer(to, op_flags=["readwrite"]):
				if x > 0:
					x[...] = np.log10(x+1)
				if x < 0:
					x[...] = -np.log10(np.absolute(x-1))
		return ti, to

	def LoadElementToScratch(self,ele):
		"""
		Reads built training data off disk into scratch space.
		Divides training and test data.
		Normalizes inputs and outputs.
			note that modifies my MolDigester to incorporate the normalization
		Initializes pointers used to provide training batches.

		Args:
			random: Not yet implemented randomization of the read data.
		"""
		ti, to = self.LoadElement(ele, self.Random)
		if (self.dig.name=="SensoryBasis" and self.dig.OType=="Disp" and self.ExpandIsometriesAltogether):
			print "Expanding the given set over isometries."
			ti,to = GRIDS.ExpandIsometries(ti,to)
		# Here we should Check to see if we want to normalize inputs/outputs.
		if (self.tform.innorm != None):
			ti = self.tform.NormalizeIns(ti)
		if (self.tform.outnorm != None):
			to = self.tform.NormalizeOuts(to)
		#if (PARAMS["Normalize"]):
		#	ti, to = self.tform.Normalize(ti, to)
		self.NTest = int(self.TestRatio * ti.shape[0])
		self.scratch_inputs = ti[:ti.shape[0]-self.NTest]
		self.scratch_outputs = to[:ti.shape[0]-self.NTest]
		self.scratch_test_inputs = ti[ti.shape[0]-self.NTest:]
		self.scratch_test_outputs = to[ti.shape[0]-self.NTest:]
		self.ScratchState = ele
		self.ScratchPointer=0
		LOGGER.debug("Element "+str(ele)+" loaded...")
		return

	def NTrainCasesInScratch(self):
		if (self.ExpandIsometriesBatchwise):
			return self.scratch_inputs.shape[0]*GRIDS.NIso()
		else:
			return self.scratch_inputs.shape[0]

	def NTestCasesInScratch(self):
		return self.scratch_inputs.shape[0]

	def PrintSampleInformation(self):
		lim = min(len(self.AvailableElements),len(self.SamplesPerElement),len(self.AvailableDataFiles))
		for i in range(lim):
			print "AN: ", self.AvailableElements[i], " contributes ", self.SamplesPerElement[i] , " samples "
			print "From files: ", self.AvailableDataFiles[i]
		return
