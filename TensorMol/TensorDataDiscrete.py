"""
The two classes provide the same interfaces as
digester and Tensordata to TFInstances but discretize
what they are built on.
"""

import os, gc, copy
from Sets import *
from TensorData import *

class Discretziation():
    """
    An n-dimensional discretization of a digester.
    """
    __init__(self, dig_, ):

        self.classPerDim = PARAMS["classPerDim"]
        self.labels = None
        self.name = dig_.name

class TensorDataDiscete():
	"""
	Discretizes a digester into some bins for a classification output.
	This may be desireable, for example if your desired output is high-dimensional,
	or spans many orders of magnitude.

    Works with Discretizer to discretize a digester and Tensordata.
    The tensordata's training set should already be built.
	"""
	def __init__(self, TData_):
		"""
			make a tensordata object
            Several arguments of PARAMS affect this classes behavior
			Args:
				MSet_: A MoleculeSet
				Dig_: A Digester
				Name_: A Name
		"""
		self.TData = TData_
        self.dig = None
        self.disc_labels = None
        self.MakeDiscretization()

    def MakeDiscretization(self):
        for ele in self.TData.AvailableElements:
            self.TData.LoadElementToScratch(ele)

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

	def EvaluateTestBatch(self,desired,preds):
		self.dig.EvaluateTestOutputs(desired,preds)
		return

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
		ti, to = self.Normalize(ti, to)
		self.NTest = int(self.TestRatio * ti.shape[0])
		self.scratch_inputs = ti[:ti.shape[0]-self.NTest]
		self.scratch_outputs = to[:ti.shape[0]-self.NTest]
		self.scratch_test_inputs = ti[ti.shape[0]-self.NTest:]
		self.scratch_test_outputs = to[ti.shape[0]-self.NTest:]
		self.ScratchState = ele
		self.ScratchPointer=0
		print "Element ", ele, " loaded..."
		return

	def NTrainCasesInScratch(self):
		if (self.ExpandIsometriesBatchwise):
			return self.scratch_inputs.shape[0]*GRIDS.NIso()
		else:
			return self.scratch_inputs.shape[0]

	def NTestCasesInScratch(self):
		return self.scratch_inputs.shape[0]
