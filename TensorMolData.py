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


	def QueryAvailable(self):
		""" If Tensordata has already been made, this looks for it under a passed name."""
		# It should probably check the sanity of each input/outputfile as well...
		return

	def CheckShapes(self):
		# Establish case and label shapes.
		tins,touts = self.dig.TrainDigest(self.set.mols[0].mbe_frags[self.order][0])
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


class TensorMolData_BP(TensorMolData):
        """
                A Training Set is a Molecule set, with a sampler and an embedding
                The sampler chooses points in the molecular volume.
                The embedding turns that into inputs and labels for a network to regress.
        """
        def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1):
		self.atom_index = None
		self.test_atom_index = None
		self.train_atom_index = None
		self.Ele_ScratchPointer = None
		self.num_test_atoms = None
		self.test_mol_len = None
		self.num_train_atoms = None
		self.train_mol_len = None
		self.scratch_outputs = None
		self.scratch_test_outputs = None
		TensorMolData.__init__(self, MSet_, Dig_, Name_, order_, num_indis_)
		self.eles = list(self.set.AtomTypes()) 
		self.eles.sort()
		print "self.eles", self.eles
		return 

	def CleanScratch(self):
		TensorMolData.CleanScratch(self)
                self.test_atom_index = None
                self.train_atom_index = None
                self.Ele_ScratchPointer = None
                self.num_test_atoms = None
                self.test_mol_len = None
                self.num_train_atoms = None
                self.train_mol_len = None
                self.scratch_outputs = None
                self.scratch_test_outputs = None
                return


	def BuildTrain(self, name_="gdb9",  append=False):
                self.CheckShapes()
                self.name=name_
                total_case = 0
                for mi in range(len(self.set.mols)):
                        total_case += len(self.set.mols[mi].mbe_frags[self.order])
                cases = []
                labels = np.zeros((total_case, self.dig.lshape))
                casep=0
		self.atom_index = self.Generate_Atom_Index()
                for mi in range(len(self.set.mols)):
                        for frag in self.set.mols[mi].mbe_frags[self.order]:
                                #print  frag.dist[0], frag.frag_mbe_energy
                                ins,outs = self.dig.TrainDigest(frag)
                                cases.append(ins)
                                labels[casep:casep+1] += outs
                                casep += 1
				#print cases, labels, cases.shape, labels.shape
		cases = np.asarray(cases)
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

	def Sort_SymFunc_By_Ele(self, inputs, atom_index):
                sym_funcs = dict()
		for ele in self.eles:
			sym_funcs[ele]=[]
		
		for i in range (0, inputs.shape[0]):
			for ele in self.eles:
				for j  in range (0, len(atom_index[ele][i])):
					sym_funcs[ele].append(inputs[i][atom_index[ele][i][j]])
		#print "sym_funcs: ", len(sym_funcs[1]), len(sym_funcs[8])
		for ele in self.eles:
                        sym_funcs[ele]=np.asarray(sym_funcs[ele])
		return sym_funcs


	def Generate_Atom_Index(self):
		atom_index = dict()

		total_case = 0
                for mi in range(len(self.set.mols)):
                        total_case += len(self.set.mols[mi].mbe_frags[self.order])

		for ele in self.eles:
			atom_index[ele]=[[] for i in range(total_case)]

		loop_index = 0
		for i in range (0, len(self.set.mols)):
			for  j, frag in enumerate(self.set.mols[i].mbe_frags[self.order]):
				for k in range (0, frag.NAtoms()):
					(atom_index[int(frag.atoms[k])][loop_index]).append(k)
				loop_index += 1

		for ele in self.eles:
                        atom_index[ele] = np.asarray(atom_index[ele])
		print "atom_index:", atom_index, atom_index[1].shape, atom_index[8].shape
		return atom_index


	def Randomize(self, ti, to):
                random.seed(0)
                idx = np.random.permutation(ti.shape[0])
                ti = ti[idx]
                to = to[idx]
		atom_index =dict()
		for ele in self.eles:
			atom_index[ele] = self.atom_index[ele][idx]
                return ti, to, atom_index

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
                #ti = ti.reshape((ti.shape[0],-1))  # flat data to [ncase, num_per_case]
                to = to.reshape((to.shape[0],-1))  # flat labels to [ncase, 1]
                if (random):
                        ti, to, atom_index = self.Randomize(ti, to)
		print ti.shape, to.shape, atom_index[1].shape, atom_index[8].shape
                self.NTrain = to.shape[0]
                return ti, to, atom_index


	def LoadDataToScratch(self, random=True):
                ti, to, atom_index = self.LoadData( random)

                self.NTest = int(self.TestRatio * ti.shape[0])
		self.NTrain = int(ti.shape[0]-self.NTest) 

		self.scratch_outputs = to[:ti.shape[0]-self.NTest] 
		tmp_inputs = ti[:ti.shape[0]-self.NTest]
		train_atom_index=dict()
		self.train_mol_len = dict()
		for ele in self.eles:
			train_atom_index[ele]=atom_index[ele][:ti.shape[0]-self.NTest]
			self.train_mol_len[ele]=[]
			for i in range (0, train_atom_index[ele].shape[0]):
				self.train_mol_len[ele].append(len(train_atom_index[ele][i]))
		self.scratch_inputs = self.Sort_SymFunc_By_Ele(tmp_inputs, train_atom_index)

		self.num_train_atoms = dict()
		for ele in self.eles:
			self.num_train_atoms[ele] = sum(self.train_mol_len[ele]) 

                self.scratch_test_outputs = to[ti.shape[0]-self.NTest:]
		tmp_test_inputs = ti[ti.shape[0]-self.NTest:]
		test_atom_index=dict()
		self.test_mol_len = dict()
                for ele in self.eles:
                        test_atom_index[ele]=atom_index[ele][ti.shape[0]-self.NTest:]
			self.test_mol_len[ele]=[]
                        for i in range (0, test_atom_index[ele].shape[0]):
                                self.test_mol_len[ele].append(len(test_atom_index[ele][i]))
                self.scratch_test_inputs = self.Sort_SymFunc_By_Ele(tmp_test_inputs, test_atom_index)

		self.num_test_atoms = dict()
                for ele in self.eles:
                        self.num_test_atoms[ele] = sum(self.test_mol_len[ele])

                self.ScratchState = self.order
                self.ScratchPointer=0
		self.Ele_ScratchPointer=dict()
		for ele in self.eles:
			self.Ele_ScratchPointer[ele]=0
		#print self.test_mol_len, self.train_mol_len
		return

	def Make_Index_Matrix(self, number_atom_per_ele, num_mol):
		index_matrix = dict()
		for ele in self.eles:
			index_matrix[ele] = np.zeros((number_atom_per_ele[ele], num_mol),dtype=np.bool)
			atom_index = 0
			for i in range (self.ScratchPointer, self.ScratchPointer + num_mol):
				for j in range (atom_index, atom_index + self.train_mol_len[ele][i]):
					#print "index of mol:", i, "index of atom:", j
					index_matrix[ele][j][i-self.ScratchPointer]=True
				atom_index += self.train_mol_len[ele][i]	
		#print "index_matrix", index_matrix
		return index_matrix
		

        def GetTrainBatch(self,ncases=1200, num_mol = 1200/6):
		start_time = time.time()
                if (self.ScratchState != self.order):
                        self.LoadDataToScratch()
		#print "NTrain:", self.NTrain, "num_mol:", num_mol
                if (num_mol> self.NTrain):
                        raise Exception("Training Data is less than the batchsize... :( ")

		reset = False
                if ( self.ScratchPointer+num_mol >= self.NTrain):
			reset = True
		for ele in self.eles:
			if (self.Ele_ScratchPointer[ele] >= self.num_train_atoms[ele]):
				reset = True
		if reset==True:
			self.ScratchPointer = 0
			for ele in self.eles:
				self.Ele_ScratchPointer[ele] = 0
	
		inputs = np.zeros((ncases, self.dig.eshape[1]))
		outputs = self.scratch_outputs[self.ScratchPointer:self.ScratchPointer+num_mol]
		number_atom_per_ele = dict()
		input_index=0
		for ele in self.eles:
			tmp = 0
			for i in range (self.ScratchPointer, self.ScratchPointer + num_mol):
				inputs[input_index:input_index+self.train_mol_len[ele][i]]=self.scratch_inputs[ele][self.Ele_ScratchPointer[ele]:self.Ele_ScratchPointer[ele]+self.train_mol_len[ele][i]]
				self.Ele_ScratchPointer[ele] += self.train_mol_len[ele][i]
				tmp += self.train_mol_len[ele][i]
				input_index += self.train_mol_len[ele][i]
			number_atom_per_ele[ele]=tmp
			#print "ele:",ele, "number_atom_per_ele[ele]", number_atom_per_ele[ele],"self.Ele_ScratchPointer[ele]", self.Ele_ScratchPointer[ele], "self.ScratchPointer ", self.ScratchPointer 
		# make the index matrix
		index_matrix = self.Make_Index_Matrix(number_atom_per_ele, num_mol) # one needs to know the number of molcule that contained in the ncase atom
                #tmp=(self.scratch_inputs[self.ScratchPointer:self.ScratchPointer+ncases], self.scratch_outputs[self.ScratchPointer:self.ScratchPointer+num_mol])
                self.ScratchPointer += num_mol
		#print inputs[1], inputs[-1], outputs[1], outputs[-1],  number_atom_per_ele, index_matrix
		#print "time cost of get batch:", time.time() - start_time
                return inputs, outputs, number_atom_per_ele, index_matrix

        def GetTestBatch(self,ncases=1280, ministep = 0):
                if (ncases>self.NTest):
                        raise Exception("Test Data is less than the batchsize... :( ")
                return (self.scratch_test_inputs[ncases*(ministep):ncases*(ministep+1)], self.scratch_test_outputs[ncases*(ministep):ncases*(ministep+1)]) 


        def PrintStatus(self):
                print "self.ScratchState",self.ScratchState
                print "self.ScratchPointer",self.ScratchPointer
                if (self.scratch_outputs != None):
			print "number of training molecules:",self.NTrain, " number of training molecules:", self.NTest 
			for ele in self.eles:
                        	print "element: ",AtomicSymbol(ele),  " Input Shape:", self.scratch_inputs[ele].shape
