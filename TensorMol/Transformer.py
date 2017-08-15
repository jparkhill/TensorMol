from __future__ import absolute_import
from .Mol import *
from .Util import *
import numpy,os,sys,re
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle
from . import LinearOperations

class Transformer:
	"""
	Data manipulation routines for normalizing and other transformations to the
	embedding and learning targets. TensorData initializes the transformer
	automatically if a .tdt file is loaded for training. The choice of transformation
	routines are set by PARAMS["InNormRoutine"] and PARAMS["OutNormRoutine"].
	"""
	def __init__(self, InNorm_ = None, OutNorm_ = None, ele_ = None, Emb_ = None, OType_ = None):
		"""
		Args:
			InNorm_ : Embedding normalization type
			OutNorm_: Learning target normalization type
			ele_: Element type for this transformer
			Emb_: Type of digester to reduce molecules to NN inputs.
			OType_: Property of the molecule which will be learned (energy, force, etc)
		"""
		self.Emb = Emb_
		self.OType = OType_
		self.innorm = InNorm_
		self.outnorm = OutNorm_


		#Should check that normalization routines match input/output types here

	def Print(self):
		LOGGER.info("-------------------- ")
		LOGGER.info("Transformer Information ")
		LOGGER.info("self.innorm: "+str(self.innorm))
		LOGGER.info("self.outnorm: "+str(self.outnorm))
		if (self.outnorm == "MeanStd"):
			LOGGER.info("self.outmean: "+str(self.outmean))
			LOGGER.info("self.outstd: "+str(self.outstd))
		LOGGER.info("-------------------- ")

	def NormalizeIns(self, ins, train=True):
		if (self.innorm == "Frobenius"):
			return self.NormInFrobenius(ins)
		elif (self.innorm == "MeanStd"):
			if (train):
				self.AssignInMeanStd(ins)
			return self.NormInMeanStd(ins)
		elif (self.innorm == "DeltaMeanStd"):
			return self.NormInDeltaMeanStd(ins)
		elif (self.innorm == "MinMax"):
			if (train):
				self.AssignInMinMax(ins)
			return self.NormInMinMax(ins)

	def NormalizeOuts(self, outs, train=True):
		if (self.outnorm == "MeanStd"):
			if (train):
				self.AssignOutMeanStd(outs)
			return self.NormOutMeanStd(outs)
		elif (self.outnorm == "Logarithmic"):
			return self.NormOutLogarithmic(outs)
		elif (self.outnorm == "Sign"):
			return self.NormOutSign(outs)

	def NormInFrobenius(self, ins):
		for i in range(len(ins)):
			ins[i] = ins[i]/(np.linalg.norm(ins[i])+1.0E-8)
		return ins

	def AssignInMeanStd(self, ins):
		self.inmean = (np.mean(ins, axis=0)).reshape((1,-1))
		self.instd = (np.std(ins, axis=0)).reshape((1, -1))

	def NormInMeanStd(self, ins):
		return (ins - self.inmean)/self.instd

	def NormInDeltaMeanStd(self, ins):
		ins[:,-3:] = (ins[:,-3:] - self.outmean)/self.outstd
		return ins

	def AssignInMinMax(self, ins):
		self.inmin = np.amin(ins)
		self.inmax = np.amax(ins)

	def NormInMinMax(self, ins):
		return (ins - self.inmin)/(self.inmax-self.inmin)

	def AssignOutMeanStd(self, outs):
		outs = outs[~np.all(np.equal(a, 0), axis=2)]
		self.outmean = np.mean(outs)
		self.outstd = np.std(outs)

	def NormOutMeanStd(self, outs):
		return (outs - self.outmean)/self.outstd

	def NormOutSign(self, outs):
		return np.sign(outs)

	def UnNormalizeOuts(self, outs):
		if (self.outnorm == "MeanStd"):
			return self.UnNormOutMeanStd(outs)
		elif (self.outnorm == "Logarithmic"):
			return self.UnNormOutLogarithmic(outs)

	def UnNormOutMeanStd(self, outs):
		return outs*self.outstd+self.outmean
