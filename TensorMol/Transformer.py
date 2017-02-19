from Mol import *
from Util import *
import numpy,os,sys,re
import cPickle as pickle
import LinearOperations

class Transformer:
	"""
	Data manipulation routines for normalizing and other transformations to the
	embedding and learning targets. TensorData initializes the transformer
	automatically if a .tdt file is loaded for training. The choice of transformation
	routines are set by PARAMS["InNormRoutine"] and PARAMS["OutNormRoutine"].
	"""
	def __init__(self, InNorm_ = None, OutNorm_ = None, Emb_ = None, OType_ = None):
		"""
		Args:
			InNorm_ : Embedding normalization type
			OutNorm_: Learning target normalization type
			Emb_: type of digester to reduce molecules to NN inputs.
			OType_: property of the molecule which will be learned (energy, force, etc)
		"""
		self.Emb = Emb_
		self.OType = OType_
		self.innorm = None
		self.outnorm = None
		if (InNorm_ != None):
			self.innorm = InNorm_
		if (OutNorm_ != None):
			self.outnorm = OutNorm_
		self.Print()
		#Should check that normalization routines match input/output types here

	def Print(self):
		LOGGER.info("-------------------- ")
		LOGGER.info("Transformer Information ")
		LOGGER.info("self.innorm: "+self.innorm)
		LOGGER.info("self.outnorm: "+self.outnorm)
		LOGGER.info("-------------------- ")

	def NormalizeIns(self, ins):
		if (self.innorm == "Frobenius"):
			return self.NormInFrobenius(ins)
		elif (self.innorm == "MeanStd"):
			return self.NormInMeanStd(ins)

	def NormalizeOuts(self, outs):
		if (self.outnorm == "MeanStd"):
			return self.NormOutMeanStd(outs)
		elif (self.outnorm == "Logarithmic"):
			return self.NormOutLogarithmic(outs)

	def NormInFrobenius(self, ins):
		for i in range(len(ins)):
			ins[i] = ins[i]/(np.linalg.norm(ins[i])+1.0E-8)
		return ins

	def AssignInMeanStd(self, ins):
		self.inmean = (np.mean(ins, axis=0)).reshape((1,-1))
		self.instd = (np.std(ins, axis=0)).reshape((1, -1))

	def NormInMeanStd(self, ins):
		self.AssignInMeanStd(ins)
		return (ins - self.inmean)/self.instd

	def AssignOutMeanStd(self, outs):
		self.outmean = np.mean(outs, axis=0)
		self.outstd = np.std(outs, axis=0)

	def NormOutMeanStd(self, outs):
		self.AssignMeanStd(outs)
		return (outs - self.outmean)/self.outstd

	def NormOutLogarithmic(self, outs):
		for x in np.nditer(outs, op_flags=["readwrite"]):
			if x > 0:
				x[...] = np.log10(x+1)
			if x < 0:
				x[...] = -np.log10(np.absolute(x-1))
		return outs

	def UnNormalizeOuts(self, outs):
		if (self.outnorm == "MeanStd"):
			return self.UnNormOutMeanStd(outs)
		elif (self.outnorm == "Logarithmic"):
			return self.UnNormOutLogarithmic(outs)

	def UnNormOutMeanStd(self, outs):
		return outs*self.outstd+self.outmean

	def UnNormOutLogarithmic(self, outs):
		for x in np.nditer(outs, op_flags=["readwrite"]):
			if x > 0:
				x[...] = (10**x)-1
			if x < 0:
				x[...] = (-1*(10**(-x)))+1
		return outs

	def MakeSamples_v2(self,point):    # with sampling function f(x)=M/(x+1)^2+N; f(0)=maxdisp,f(maxdisp)=0; when maxdisp =5.0, 38 % lie in (0, 0.1)
		disps = samplingfunc_v2(self.TrainSampDistance * np.random.random(self.NTrainSamples), self.TrainSampDistance)
		theta  = np.random.random(self.NTrainSamples)* math.pi
		phi = np.random.random(self.NTrainSamples)* math.pi * 2
		grids  = np.zeros((self.NTrainSamples,3),dtype=np.float32)
		grids[:,0] = disps*np.cos(theta)
		grids[:,1] = disps*np.sin(theta)*np.cos(phi)
		grids[:,2] = disps*np.sin(theta)*np.sin(phi)
		return grids + point

	def Blurs(self, diffs):
		dists=np.array(map(np.linalg.norm,diffs))
		return np.exp(dists*dists/(-1.0*self.BlurRadius*self.BlurRadius))/(np.power(2.0*Pi*self.BlurRadius*self.BlurRadius,3.0/2.0))

	def HardCut(self, diffs, cutoff=0.05):
		# 0, 1 output
		dists=np.array(map(np.linalg.norm,diffs))
		labels = np.clip(-(dists - cutoff), 0, (-(dists - cutoff)).max())
		labels[np.where(labels > 0)]=1
		return labels

#
#  Embedding functions, called by batch digests. Use outside of Digester() is discouraged.
#  Instead call a batch digest routine.
#

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
