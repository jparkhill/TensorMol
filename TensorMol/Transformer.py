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
		self.innorm = None
		self.outnorm = None
		if (InNorm_ != None):
			self.innorm = InNorm_
		if (OutNorm_ != None):
			self.outnorm = OutNorm_
		#Should check that normalization routines match input/output types here

	def Print(self):
		LOGGER.info("-------------------- ")
		LOGGER.info("Transformer Information ")
		LOGGER.info("self.innorm: "+str(self.innorm))
		if (self.innorm == "MeanStd"):
			LOGGER.info("self.inmean: "+str(self.inmean))
			LOGGER.info("self.instd: "+str(self.instd))
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
				self.AssignInMeanStd(outs)
			return self.NormInMeanStd(ins)

	def NormalizeOuts(self, outs, train=True):
		if (self.outnorm == "MeanStd"):
			if (train):
				self.AssignOutMeanStd(outs)
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
		return (ins - self.inmean)/self.instd

	def AssignOutMeanStd(self, outs):
		self.outmean = np.mean(outs, axis=0)
		self.outstd = np.std(outs, axis=0)

	def NormOutMeanStd(self, outs):
		return (outs - self.outmean)/self.outstd

	def NormOutLogarithmic(self, outs):
		for x in np.nditer(outs, op_flags=["readwrite"]):
			if x > 0:
				x[...] = np.log10(x+1)
			if x < 0:
				x[...] = -np.log10(np.absolute(x-1))
		return outs

	def UnNormalizeOuts(self, outs):
		print "Unnormalizing outputs here!!!"
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

	#Don't remember where this junk comes from

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

	#This junk is from TMolData

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

	def Normalize(self,ti,to):
		if (self.NormalizeInputs):
			for i in range(len(ti)):
				ti[i] = ti[i]/np.linalg.norm(ti[i])
		if (self.NormalizeOutputs):
			mo = np.average(to)
			to -= mo
			stdo = np.std(to)
			to /= stdo
			self.dig.AssignNormalization(mo,stdo)
		return ti, to

	#From DigestMol
	def AssignNormalization(self,mn,sn):
		self.MeanNorm=mn
		self.StdNorm=sn
		return
