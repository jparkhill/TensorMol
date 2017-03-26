"""
 Embedding Optimizer
 This file contains routines to optimize an embedding
"""

from Mol import *
from Util import *
from TensorData import *
from TFInstance import *
import os, sys, re, random, math, copy
import numpy as np
import cPickle as pickle
import LinearOperations, DigestMol, Digest, Opt, Ipecac

class EmbeddingOptimizer:
	"""
	Provides an objective function to optimize an embedding, maximizing the reversibility of the embedding, and the distance the embedding predicts between molecules which are not equivalent in their geometry or stoiciometry.
	"""
	def __init__(self, method_, set_, dig_, OType_ = None, Elements_ = []):
		print "Will produce an objective function to optimize basis parameters, and then optimize it"
		self.method = method_
		self.set = set_
		self.dig = dig_
		print "Optimizing ", self.dig.name, " based off of ", self.method
		if (self.method == "Ipecac"):
			# Distort each mol in set_
			self.mols=[]
			self.dmols=[]
			self.DesiredEmbs = []
			for mol in self.set.mols:
				self.mols.append(copy.deepcopy(mol))
				self.mols[-1].BuildDistanceMatrix()
				self.mols.append(copy.deepcopy(mol))
				self.mols[-1].Distort()
				self.mols[-1].BuildDistanceMatrix()
				# Have to figure out a way to make the Atomic numbers invertible too...
				#self.mols.append(copy.deepcopy(mol))
				#self.mols[-1].DistortAN()
				#self.mols[-1].BuildDistanceMatrix()
			# We further distort this set and keep that around so the test is the same throughout the optimization.
			for mol in self.mols:
				self.dmols.append(copy.deepcopy(mol))
				self.dmols[-1].Distort()
			print "Using ", len(self.mols), "unique molecules and ", len(self.dmols), " unique geometries"
		elif (self.method == "KRR"):
			self.OType = OType_
			if (self.OType == None):
				raise Exception("KRR optimization requires setting OType_ for the EmbeddingOptimizer.")
			self.elements = Elements_
			if (len(self.elements) == 0):
				raise Exception("KRR optimization requires setting Elements_ for the EmbeddingOptimizer.")
			self.TreatedAtoms = self.set.AtomTypes()
			print "Optimizing based off ", self.OType, " using elements"
		return

	def SetBasisParams(self,basisParams_):
		PARAMS["RBFS"] = basisParams_[:PARAMS["SH_NRAD"]*2].reshape(PARAMS["SH_NRAD"],2).copy()
		#PARAMS["ANES"][0] = basisParams_[PARAMS["SH_NRAD"]*2].copy()
		#PARAMS["ANES"][5] = basisParams_[PARAMS["SH_NRAD"]*2+1].copy()
		#PARAMS["ANES"][6] = basisParams_[PARAMS["SH_NRAD"]*2+2].copy()
		#PARAMS["ANES"][7] = basisParams_[PARAMS["SH_NRAD"]*2+3].copy()
		S_Rad = MolEmb.Overlap_RBF(PARAMS)
		try:
			if (np.amin(np.linalg.eigvals(S_Rad)) < 1.e-10):
				return 100.0
		except numpy.linalg.linalg.LinAlgError:
			return 100.0
		PARAMS["SRBF"] = MatrixPower(S_Rad,-1./2)
		return 0.0

	def Ipecac_Objective(self,basisParams_):
		"""
		Resets the parameters. Builds the overlap if neccesary. Resets the desired embeddings. Reverses the distorted set and computes an error.
		"""
		berror = self.SetBasisParams(basisParams_)
		self.SetEmbeddings()
		resultmols = []
		for i in range(len(self.dmols)):
			m = self.dmols[i]
			emb = self.DesiredEmbs[i]
			resultmols.append(Ipecac.ReverseAtomwiseEmbedding(self.dig, emb, m.atoms, guess_=m.coords))
		# Compute the various parts of the error.
		SelfDistances = 0.0
		OtherDistances = 0.0
		for i in range(len(self.dmols)):
			SelfDistances += self.mols[i].rms_inv(resultmols[i])
			print SelfDistances
			for j in range(len(self.dmols)):
				if (i != j and len(self.mols[i].atoms)==len(self.mols[j].atoms)):
					OtherDistances += np.exp(-1.0*self.mols[i].rms_inv(resultmols[j]))
		print "Using params_: ", basisParams_
		print "Got Error: ", berror+SelfDistances+OtherDistances
		return berror+SelfDistances+OtherDistances

	def KRR_Objective(self, basisParams_):
		"""
		Resets the parameters. Builds the overlap if neccesary. Resets the desired embeddings. Reverses the distorted set and computes an error.
		"""
		berror = self.SetBasisParams(basisParams_)
		mae = 0.0
		tset = TensorData(self.set,self.dig)
		tset.BuildTrainMolwise(self.set.name+"_BasisOpt")
		for ele in self.elements:
			ele_inst = Instance_KRR(tset, ele, None)
			mae += ele_inst.basis_opt_run()
		print "Using params_: ", basisParams_
		print "Got MAE: ", mae
		return mae+berror

	def PerformOptimization(self):
		prm0 = PARAMS["RBFS"][:PARAMS["SH_NRAD"]]
		print prm0
		import scipy.optimize
		print "Optimizing RBFS."
		if (self.method == "Ipecac"):
			obj = lambda x: self.Ipecac_Objective(x)
		elif (self.method == "KRR"):
			obj = lambda x: self.KRR_Objective(x)
		res=scipy.optimize.minimize(obj, prm0, method='L-BFGS-B', tol=0.0001, options={'disp':True, 'maxcor':30, 'eps':0.0001})
		print "Opt complete", res.message
		return

	def SetEmbeddings(self):
		self.DesiredEmbs = []
		for mol in self.mols:
			self.DesiredEmbs.append(self.dig.TrainDigestMolwise(mol,MakeOutputs_=False))
