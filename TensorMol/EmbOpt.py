"""
 Embedding Optimizer
 This file contains routines to optimize an embedding
"""

from Mol import *
from Util import *
import os, sys, re, random, math, copy
import numpy as np
import cPickle as pickle
import LinearOperations, DigestMol, Digest, Opt, Ipecac

class EmbeddingOptimizer:
	"""
	Provides an objective function to optimize an embedding, maximizing the reversibility of the embedding, and the distance the embedding predicts between molecules which are not equivalent in their geometry or stoiciometry.
	"""
	def __init__(self, set_, dig_):
		print "Will produce an objective function to optimize basis parameters, and then optimize it."
		# Distort each mol in set_
		self.mols=[]
		self.dmols=[]
		self.DesiredEmbs = []
		self.dig = dig_
		for mol in set_.mols:
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
		print "Optimizing off of ", len(self.dmols), " geometries"
		return

	def SetEmbeddings(self):
		self.DesiredEmbs = []
		for mol in self.mols:
			self.DesiredEmbs.append(self.dig.TrainDigestMolwise(mol,MakeOutputs_=False))

	def SetBasisParams(self,basisParams_):
		PARAMS["RBFS"] = basisParams_[:PARAMS["SH_NRAD"]*2].reshape(PARAMS["SH_NRAD"],2).copy()
		#PARAMS["ANES"][0] = basisParams_[PARAMS["SH_NRAD"]*2].copy()
		#PARAMS["ANES"][5] = basisParams_[PARAMS["SH_NRAD"]*2+1].copy()
		#PARAMS["ANES"][6] = basisParams_[PARAMS["SH_NRAD"]*2+2].copy()
		#PARAMS["ANES"][7] = basisParams_[PARAMS["SH_NRAD"]*2+3].copy()
		S_Rad = MolEmb.Overlap_RBF(PARAMS)
		mae=0.0
		try:
			if (np.amin(np.linalg.eigvals(S_Rad)) < 1.e-10):
				return 100.0
		except numpy.linalg.linalg.LinAlgError:
			return 100.0
		PARAMS["SRBF"] = MatrixPower(S_Rad,-1./2)
		return 0.0

	def MyObjective(self,basisParams_):
		"""
		Resets the parameters. Builds the overlap if neccesary. Resets the desired embeddings. Reverses the distorted set and computes an error.
		"""
		berror = self.SetBasisParams(basisParams_)
		self.SetEmbeddings()
		resultmols = []
		for i in range(len(self.dmols)):
			m = self.dmols[i]
			emb = self.DesiredEmbs[i]
			resultmols.append(Ipecac.ReverseAtomwiseEmbedding(m.atoms, self.dig, emb, guess_=m.coords))
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

	def PerformOptimization(self):
		prm0 = PARAMS["RBFS"][:PARAMS["SH_NRAD"]]
		print prm0
		import scipy.optimize
		print "Optimizing RBFS."
		obj = lambda x: self.MyObjective(x)
		res=scipy.optimize.minimize(obj, prm0, method='L-BFGS-B', tol=0.0001, options={'disp':True, 'maxcor':30, 'eps':0.0001})
		print "Opt complete", res.message
		return
