#
# Either trains, tests, evaluates or provides an interface for optimization.
#
from TensorMolData import *
from TFMolInstance import *
import numpy as np
import gc


class TFMolManage:
	def __init__(self, Name_="", TData_=None, Train_=True, NetType_="fc_sqdiff", Test_TData_=None):  #Test_TData_ is some other randon independent test data
	        self.path = "./networks/"	
		if (Name_ != ""):
			# This will unpickle and instantiate TData...
			self.name = Name_
			self.Prepare()
			return
		self.TData = TData_
		self.Test_TData = Test_TData_
		self.NetType = NetType_ 
		print self.TData.AvailableDataFiles
		self.name = self.TData.name+self.TData.dig.name+"_"+self.NetType+"_"+str(self.TData.order)
		print "--- TF will be fed by ---",self.TData.name

		self.TrainedAtoms=[] # In order of the elements in TData
		self.TrainedNetworks=[] # In order of the elements in TData
		self.Instances=None # In order of the elements in TData
		if (Train_):
			self.Train()
			return
		return

	def Print(self):
		print "-- TensorMol, Tensorflow Manager Status--"
		return
	

	def Save(self):
		print "Saving TFManager."
		self.TData.CleanScratch()
		f=open(self.path+self.name+".tfm","wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
		return

	def Load(self):
		print "Unpickling TFManager..."
		f = open(self.path+self.name+".tfm","rb")
		tmp=pickle.load(f)
		self.__dict__.update(tmp)
		f.close()
		print "TFManager Metadata Loaded, Reviving Networks."
		self.Print()
		return

	def Train(self, maxstep=10000):
		if (self.TData.dig.eshape==None):
			raise Exception("Must Have Digester")
		# It's up the TensorData to provide the batches and input output shapes.
		if (self.NetType == "fc_classify"):
			self.Instances = Instance_fc_classify(self.TData, None, self.Test_TData)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = Instance_fc_sqdiff(self.TData, None, self.Test_TData)
		elif (self.NetType == "fc_sqdiff_BP"):
			NAtom_in_mol = 29  # debug ..this needs to be a member in TensorMolData or Mol
			self.Instances = Instance_fc_sqdiff_BP(self.TData, NAtom_in_mol, None, self.Test_TData)

		else:
			raise Exception("Unknown Network Type!")

		self.Instances.train(maxstep) # Just for the sake of debugging.
		nm = self.Instances.name
		# Here we should print some summary of the pupil's progress as well, maybe.
		if self.TrainedNetworks.count(nm)==0:
			self.TrainedNetworks.append(nm)
		self.Save()
		gc.collect()
		return

	def Eval(self,  test_input):
		return self.Instances.evaluate(test_input)   

	def Prepare(self):
		self.Load()
		self.Instances= None # In order of the elements in TData
		if (self.NetType == "fc_classify"):
			self.Instances = Instance_fc_classify(None,  self.TrainedNetworks[0], None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = Instance_fc_sqdiff(None, self.TrainedNetworks[0], None)
		else:
			raise Exception("Unknown Network Type!")	
		# Raise TF instances for each atom which have already been trained.
		return

	def Eval_Mol(self, mol):
		total_case = len(mol.mbe_frags[self.TData.order])
		if total_case == 0:
			return 0.0
		natom = mol.mbe_frags[self.TData.order][0].NAtoms()
		cases = np.zeros((total_case, self.TData.dig.eshape))
		cases_deri = np.zeros((total_case, natom, natom, 6)) # x1,y1,z1,x2,y2,z2
		casep = 0 
		for frag in mol.mbe_frags[self.TData.order]:
			ins, embed_deri =  self.TData.dig.EvalDigest(frag)
			cases[casep:casep+1] += ins
			cases_deri[casep:casep+1]=embed_deri
                        casep += 1
		print "evaluating order:", self.TData.order
		nn, nn_deri=self.Eval(cases)
		mean, std = self.TData.Get_Mean_Std()
		nn = nn*std+mean
		nn_deri = nn_deri*std
		#print "nn:",nn, "nn_deri:",nn_deri, "cm_deri:", cases_deri, "cases:",cases, "coord:", mol.coords
		mol.Set_Frag_Force_with_Order(cases_deri, nn_deri, self.TData.order)
		return nn.sum()



	def Test(self, save_file="mbe_test.dat"):
		ti, to = self.TData.LoadData( True)
		NTest = int(self.TData.TestRatio * ti.shape[0])
                ti= ti[ti.shape[0]-NTest:]
                to = to[to.shape[0]-NTest:]
		acc_nn = np.zeros((to.shape[0],2))
		acc=self.TData.ApplyNormalize(to)
		nn, gradient=self.Eval(ti)
		acc_nn[:,0]=acc.reshape(acc.shape[0])
		acc_nn[:,1]=nn.reshape(nn.shape[0])
		mean, std = self.TData.Get_Mean_Std()	
		acc_nn = acc_nn*std+mean
		np.savetxt(save_file,acc_nn)
		np.savetxt("dist_2b.dat", ti[:,1])
		return

