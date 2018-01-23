"""
An asynchronous re-write of TFMolManage.
The manager will now direct the training set to generate batches
and manage a random queue to accept those batches.
 THIS WHOLE THING IS BASICALLY NOT WORKING AND UNAPPROACHED
"""
from __future__ import absolute_import
from __future__ import print_function
from .TFManage import *
from ..Containers.TensorMolData import *
from .TFMolInstance import *
import numpy as np
import gc

class TFMolQManage(TFMolManage):
	"""
	As of now only Behler-Parinello will be supported here.
	"""
	def __init__(self, Name_="", TData_=None, Train_=False, NetType_="fc_sqdiff", RandomTData_=True):
		"""
			Args:
				Name_: If not blank, will try to load a network with that name using Prepare()
				TData_: A TensorMolData instance to provide and process data.
				Train_: Whether to train the instances raised.
				NetType_: Choices of Various network architectures.
				RandomTData_: Modifes the preparation of training batches.
		"""
		TFManage.__init__(self, "", TData_, False, NetType_, RandomTData_)
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.TData.order)
		if (Name_!=""):
			self.name = Name_
			self.Prepare()
			return
		self.TrainedAtoms=[] # In order of the elements in TData
		self.TrainedNetworks=[] # In order of the elements in TData
		self.Instances=None # In order of the elements in TData
		#capacity, min_after_dequeue, dtypes, shapes=None, names=None, seed=None, shared_name=None, name='random_shuffle_queue'
		self.BatchCapacity = 200
		self.BatchQueue = None
		if (Train_):
			self.Train()
			return
		return

	def Train(self, maxstep=3000):
		"""
		Instantiates and trains a Molecular network.

		Args:
			maxstep: The number of training steps.
		"""
		if (self.TData.dig.eshape==None):
			raise Exception("Must Have Digester")
		# It's up the TensorData to provide the batches and input output shapes.
		if (self.NetType == "fc_classify"):
			self.Instances = MolInstance_fc_classify(self.TData, None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = MolInstance_fc_sqdiff(self.TData, None)
		elif (self.NetType == "fc_sqdiff_BP"):
			self.Instances = MolInstance_fc_sqdiff_BP(self.TData)
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

	def Eval(self, test_input):
		return self.Instances.evaluate(test_input)

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
		print("evaluating order:", self.TData.order)
		nn, nn_deri=self.Eval(cases)
		mean, std = self.TData.Get_Mean_Std()
		nn = nn*std+mean
		nn_deri = nn_deri*std
		#print "nn:",nn, "nn_deri:",nn_deri, "cm_deri:", cases_deri, "cases:",cases, "coord:", mol.coords
		mol.Set_Frag_Force_with_Order(cases_deri, nn_deri, self.TData.order)
		return nn.sum()

	def Prepare(self):
		self.Load()
		self.Instances= None # In order of the elements in TData
		if (self.NetType == "fc_classify"):
			self.Instances = MolInstance_fc_classify(None,  self.TrainedNetworks[0], None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = MolInstance_fc_sqdiff(None, self.TrainedNetworks[0], None)
		elif (self.NetType == "fc_sqdiff_BP"):
			self.Instances = MolInstance_fc_sqdiff_BP(None,self.TrainedNetworks[0],None)
		else:
			raise Exception("Unknown Network Type!")

		self.BatchQueue = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes, shapes=None, names=None, seed=None, shared_name=None, name='random_shuffle_queue')

		return

# This has to be totally re-written to be more like the
# testing in TFInstance.

	def Test(self, save_file="mbe_test.dat"):
		ti, to = self.TData.LoadData( True)
		NTest = int(self.TData.TestRatio * ti.shape[0])
		ti= ti[ti.shape[0]-NTest:]
		to = to[to.shape[0]-NTest:]
		acc_nn = np.zeros((to.shape[0],2))
		nn, gradient=self.Eval(ti)
		acc_nn[:,0]=acc.reshape(acc.shape[0])
		acc_nn[:,1]=nn.reshape(nn.shape[0])
		mean, std = self.TData.Get_Mean_Std()
		acc_nn = acc_nn*std+mean
		np.savetxt(save_file,acc_nn)
		np.savetxt("dist_2b.dat", ti[:,1])
		return
