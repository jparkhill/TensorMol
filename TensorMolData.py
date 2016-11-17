#
# Contains Routines to generate training sets 
# Combining a dataset, sampler and an embedding. (CM etc.)
#
import os, gc
from Sets import *
from MolDigest import *
#import tables should go to hdf5 soon...

class TensorMolData():
	"""
		A Training Set is a Molecule set, with a sampler and an embedding
		The sampler chooses points in the molecular volume.
		The embedding turns that into inputs and labels for a network to regress.
	"""
	def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1):
		self.path = "./trainsets/"
		self.suffix = ".pdb"
		self.set = MSet_
		self.dig = Dig_
		self.order = order_
		self.num_indis = num_indis_
		self.AvailableDataFiles = []
		self.NTest = 0  # assgin this value when the data is loaded
		self.TestRatio = 0.2 # number of cases withheld for testing.

		self.NTrain = 0
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

	def PrintStatus(self):
		print "self.ScratchState",self.ScratchState
		print "self.ScratchPointer",self.ScratchPointer
		if (self.scratch_outputs != None):
			print "self.scratch_inputs.shape",self.scratch_inputs.shape
			print "self.scratch_outputs.shape",self.scratch_outputs.shape
			print "scratch_test_inputs.shape",self.scratch_test_inputs.shape
			print "scratch_test_outputs.shape",self.scratch_test_outputs.shape

	def NTrain(self):
		return len(self.scratch_outputs)

	def QueryAvailable(self):
		""" If Tensordata has already been made, this looks for it under a passed name."""
		# It should probably check the sanity of each input/outputfile as well...
		return

	def CheckShapes(self):
		# Establish case and label shapes.
		tins,touts = self.dig.TrainDigest(self.set.mols[0].mbe_permute_frags[self.order][0])
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
		for mi in range(len(self.set.mols)):
			for frag in self.set.mols[mi].mbe_permute_frags[self.order]:
				#print  frag.dist[0], frag.frag_mbe_energy
				ins,outs = self.dig.TrainDigest(frag)
				cases[casep:casep+1] += ins
				labels[casep:casep+1] += outs
				casep += 1
		insname = self.path+name_+"_"+self.dig.name+"_"+str(self.order)+"_in.npy"
		outsname = self.path+name_+"_"+self.dig.name+"_"+str(self.order)+"_out.npy"
		if (not append):
			inf = open(insname,"wb")
			ouf = open(outsname,"wb")
			np.save(inf,cases[:casep])
			np.save(ouf,labels[:casep])
			inf.close()
			ouf.close()
			self.AvailableDataFiles.append([insname,outsname])
		else:
			inf = open(insname,"a+b")
			ouf = open(outsname,"a+b")
			np.save(inf,cases)
			np.save(ouf,labels)
			inf.close()
			ouf.close()
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
		ti = ti.reshape((ti.shape[0]/group,group, -1))
		to = to.reshape((to.shape[0]/group, group, -1))
		random.seed(0)
		idx = np.random.permutation(ti.shape[0])
		ti = ti[idx]
		to = to[idx]

		ti = ti.reshape((ti.shape[0]*ti.shape[1],-1))
		to = to.reshape((to.shape[0]*to.shape[1],-1))


		return ti, to

	def LoadData(self, random=False):
		insname = self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_in.npy"
		outsname = self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_out.npy"
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
		self.scratch_inputs = ti[:ti.shape[0]-self.NTest]
		self.scratch_outputs = to[:ti.shape[0]-self.NTest]
		self.scratch_test_inputs = ti[ti.shape[0]-self.NTest:]
		self.scratch_test_outputs = to[ti.shape[0]-self.NTest:]
		if random==True:
			self.scratch_inputs, self.scratch_outputs = self.Randomize(self.scratch_inputs, self.scratch_outputs, 1)
			self.scratch_test_inputs, self.scratch_test_outputs = self.Randomize(self.scratch_test_inputs, self.scratch_test_outputs, 1)
		self.ScratchState = self.order
		self.ScratchPointer=0
		
		return

	def NormalizeInputs(self):
		mean = (np.mean(self.scratch_inputs, axis=0)).reshape((1,-1))
		std = (np.std(self.scratch_inputs, axis=0)).reshape((1, -1))
		self.scratch_inputs = (self.scratch_inputs-mean)/std
		self.scratch_test_inputs = (self.scratch_test_inputs-mean)/std
		np.save(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_in_MEAN.npy", mean)
		np.save(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+"_in_STD.npy",std)
		return


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

	def Save(self):
		self.CleanScratch()
		f=open(self.path+self.name+"_"+self.dig.name+"_"+str(self.order)+".tdt","wb")
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
		print "order:", self.order
		self.PrintSampleInformation()
		self.dig.Print()
		return

	def PrintSampleInformation(self):
		print "From files: ", self.AvailableDataFiles
		return




