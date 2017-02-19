from Mol import *
from Util import *
import numpy,os,sys,re
import cPickle as pickle
import LinearOperations

class Transformer:
	"""
	Data manipulation routines for normalizing and other transformations to the
	embedding and learning targets. TensorData initializes the transformer if
	PARAMS["NormalizeInputs"] or PARAMS["NormalizeOutputs"] is set to True. The
	choice of transformation routines are set by PARAMS["InNormRoutine"] and
	PARAMS["OutNormRoutine"]
	"""
	def __init__(self, InNorm_ = None, OutNorm_ = None, Emb_ = None, OType_ = None):
		"""
		Args:
			InNorm_ : Input Normalization type
			OutNorm_ Output Normalization type
			Name_: type of digester to reduce molecules to NN inputs.
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
		#Should check that normalization routines match input/output types here
		return

	def NormalizeIns(self, ti):
		if (self.innorm == "Frobenius"):
			ti = self.NormInFrobenius
		return ti

	def NormalizeOuts(self, to):
		if (self.outnorm == "MeanStd"):
			to = self.NormOutMeanStd(to)
		if (self.outnorm == "Logarithmic"):
			to = self.NormOutLogarithmic(to)
		return to

	def NormInFrobenius(self, ins):
		for i in range(len(ti)):
			ins[i] = ins[i]/(np.linalg.norm(ins[i])+1.0E-8)
		return ti

	def AssignMeanStd(self, to):
		self.outmean = np.mean(to, axis=0)
		self.outstd = np.std(to, axis=0)
		return

	def NormOutMeanStd(self, to):
		self.AssignMeanStd(to)
		to = (to - self.outmean)/self.outstd
		return to

	def NormOutLogarithmic(self, to):
		for x in np.nditer(to, op_flags=["readwrite"]):
			if x > 0:
				x[...] = np.log10(x+1)
			if x < 0:
				x[...] = -np.log10(np.absolute(x-1))
		return to

	def UnNormalizeOuts(self, to):
		if (self.outnorm == "MeanStd"):
			to = self.UnNormOutMeanStd(to)
		if (self.outnorm == "Logarithmic"):
			to = self.UnNormOutLogarithmic(to)
		return to

	def UnNormOutMeanStd(self, to):
		return to*self.outstd+self.outmean

	def UnNormOutLogarithmic(self, to):
		tmp = a.copy()
		for x in np.nditer(tmp, op_flags=["readwrite"]):
			if x > 0:
				x[...] = (10**x)-1
			if x < 0:
				x[...] = (-1*(10**(-x)))+1
		return tmp

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

	def unscld(self,a):
		"""
		I really don't like this routine at all. I think we should give
		Digesters some sort of Translator object which allows the networks training
		target to be some simple transformations of the input/output
		without changing the digester, redoing training etc...
		Thinking about how to do this for all elements etc. is tricky.
		"""
		return (a*self.StdNorm+self.MeanNorm)

	def unlog(self, a):
		tmp = a.copy()
		for x in np.nditer(tmp, op_flags=["readwrite"]):
			if x > 0:
				x[...] = (10**x)-1
			if x < 0:
				x[...] = (-1*(10**(-x)))+1
		return tmp

	def EvaluateTestOutputs(self, desired, predicted):
		try:
			print "Evaluating, ", len(desired), " predictions... "
			#print desired.shape, predicted.shape
			if (self.OType=="HardP"):
				raise Exception("Unknown Digester Output Type.")
			elif (self.OType=="Disp" or self.OType=="Force" or self.OType == "GoForce"):
				ders=np.zeros(len(desired))
				#comp=np.zeros(len(desired))
				for i in range(len(desired)):
					ders[i] = np.linalg.norm(self.unscld(predicted[i,-3:])-self.unscld(desired[i,-3:]))
				LOGGER.info("Test displacement errors direct (mean,std) %f,%f",np.average(ders),np.std(ders))
				LOGGER.info("Average learning target: %s, Average output (direct) %s", str(np.average(desired[:,-3:],axis=0)),str(np.average(predicted[:,-3:],axis=0)))
				print "Fraction of incorrect directions: ", np.sum(np.sign(desired[:,-3:])-np.sign(predicted[:,-3:]))/(6.*len(desired))
				for i in range(100):
					print "Desired: ",i,self.unscld(desired[i,-3:])," Predicted: ",self.unscld(predicted[i,-3:])
			elif (self.OType == "GoForceSphere"):
				# Convert them back to cartesian
				desiredc = SphereToCartV(desired)
				predictedc = SphereToCartV(predicted)
				ders=np.zeros(len(desired))
				#comp=np.zeros(len(desired))
				for i in range(len(desiredc)):
					ders[i] = np.linalg.norm(self.unscld(predictedc[i,-3:])-self.unscld(desiredc[i,-3:]))
				LOGGER.info("Test displacement errors direct (mean,std) %f,%f",np.average(ders),np.std(ders))
				LOGGER.info("Average learning target: %s, Average output (direct) %s", str(np.average(desiredc[:,-3:],axis=0)),str(np.average(predictedc[:,-3:],axis=0)))
				print "Fraction of incorrect directions: ", np.sum(np.sign(desiredc[:,-3:])-np.sign(predictedc[:,-3:]))/(6.*len(desiredc))
				for i in range(100):
					print "Desired: ",i,self.unscld(desiredc[i,-3:])," Predicted: ",self.unscld(predictedc[i,-3:])
			elif (self.OType=="SmoothP"):
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
			elif (self.OType=="StoP"):
				raise Exception("Unknown Digester Output Type.")
			elif (self.OType=="Energy"):
				raise Exception("Unknown Digester Output Type.")
			elif (self.OType=="GoForce_old_version"): # python version is fine for here
				raise Exception("Unknown Digester Output Type.")
			else:
				raise Exception("Unknown Digester Output Type.")
		except Exception as Ex:
			print "Something went wrong"
			pass
		return
