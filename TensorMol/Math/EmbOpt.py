"""
 Embedding Optimizer
 This file contains routines to optimize an embedding
"""

from __future__ import absolute_import
from __future__ import print_function
from ..Mol import *
from ..Util import *
from ..Containers.TensorData import *
from ..TFNetworks.TFInstance import *
from scipy import optimize
import os, sys, re, random, math, copy
import numpy as np
if sys.version_info[0] < 3:
	import pickle
else:
	import _pickle as pickle
from ..Math.LinearOperations import *
from ..Containers.DigestMol import *
from ..Containers.Digest import *
from ..Simulations.Opt import *
from ..Math.Ipecac import *

class EmbeddingOptimizer:
	"""
	Provides an objective function to optimize an embedding, maximizing the reversibility of the embedding, and the distance the embedding predicts between molecules which are not equivalent in their geometry or stoiciometry.
	"""
	def __init__(self, method_, set_, dig_, OptParam_, OType_ = None, Elements_ = None):
		print("Will produce an objective function to optimize basis parameters, and then optimize it")
		self.method = method_
		self.set = set_
		self.dig = dig_
		self.OptParam = OptParam_
		LOGGER.info("Optimizing %s part of %s based off of %s", self.OptParam, self.dig.name, self.method)
		if self.method == "Ipecac":
			self.ip = Ipecac.Ipecac(self.set, self.dig, eles_=[1,6,7,8])
			self.Mols = []
			self.DistortMols=[]
			self.DesiredEmbs=[]
			for mol in self.set.mols:
				self.Mols.append(copy.deepcopy(mol))
				self.Mols[-1].BuildDistanceMatrix()
				# for i in range(9):
				# 	self.Mols.append(copy.deepcopy(mol))
				# 	self.Mols[-1].Distort(0.15)
				# 	self.Mols[-1].BuildDistanceMatrix()
			for mol in self.Mols:
				self.DistortMols.append(copy.deepcopy(mol))
				self.DistortMols[-1].Distort(0.15)
				self.DistortMols[-1].BuildDistanceMatrix()
			LOGGER.info("Using %d unique geometries", len(self.Mols))
		elif self.method == "KRR":
			self.OType = OType_
			if self.OType == None:
				raise Exception("KRR optimization requires setting OType_ for the EmbeddingOptimizer.")
			self.elements = Elements_
			if self.elements == None:
				raise Exception("KRR optimization requires setting Elements_ for the EmbeddingOptimizer.")
			self.TreatedAtoms = self.set.AtomTypes()
			print("Optimizing based off ", self.OType, " using elements")
		return

	def SetBasisParams(self,basisParams_):
		if self.dig.name == "GauSH":
			if self.OptParam == "radial":
				PARAMS["RBFS"] = basisParams_[:PARAMS["SH_NRAD"]*2].reshape(PARAMS["SH_NRAD"],2).copy()
			elif self.OptParam == "atomic":
				PARAMS["ANES"][[0,5,6,7]] = basisParams_[[0,1,2,3]].copy()
		elif self.digname == "ANI1_Sym":
			raise Exception("Not yet implemented for ANI1")
		S_Rad = MolEmb.Overlap_RBF(PARAMS)
		PARAMS["SRBF"] = MatrixPower(S_Rad,-1./2)
		print("Eigenvalue Overlap Error: ", (1/np.amin(np.linalg.eigvals(S_Rad)))/1.e6)
		return np.abs((1/np.amin(np.linalg.eigvals(S_Rad)))/1.e6)
		#return 0.0

	def Ipecac_Objective(self,basisParams_):
		"""
		Resets the parameters. Builds the overlap if neccesary. Resets the desired embeddings. Reverses the distorted set and computes an error.
		"""
		berror = self.SetBasisParams(basisParams_)
		self.SetEmbeddings()
		resultmols = []
		for i in range(len(self.DistortMols)):
			m = copy.deepcopy(self.DistortMols[i])
			emb = self.DesiredEmbs[i]
			resultmols.append(self.ip.ReverseAtomwiseEmbedding(emb, m.atoms, guess_=m.coords, GdDistMatrix=self.Mols[i].DistMatrix))
		# Compute the various parts of the error.
		SelfDistances = 0.0
		OtherDistances = 0.0
		for i in range(len(self.DistortMols)):
			SelfDistances += self.Mols[i].rms_inv(resultmols[i])
			print(SelfDistances)
			for j in range(len(self.DistortMols)):
				if (i != j and len(self.Mols[i].atoms)==len(self.Mols[j].atoms)):
					OtherDistances += np.exp(-1.0*self.Mols[i].rms_inv(resultmols[j]))
		LOGGER.info("Using params_: %s", basisParams_)
		LOGGER.info("Got Error: %.6f", berror+SelfDistances+OtherDistances)
		return berror+SelfDistances+OtherDistances

	def KRR_Objective(self, basisParams_):
		"""
		Resets the parameters. Builds the overlap if neccesary. Resets the desired embeddings. Reverses the distorted set and computes an error.
		"""
		berror = self.SetBasisParams(basisParams_)
		sqerr = 0.0
		tset = TensorData(self.set,self.dig)
		tset.BuildTrainMolwise(self.set.name+"_BasisOpt")
		for ele in self.elements:
			ele_inst = Instance_KRR(tset, ele, None)
			sqerr += (ele_inst.basis_opt_run())**2
		LOGGER.info("Basis Params: %s", basisParams_)
		LOGGER.info("SqError: %f", sqerr+berror)
		return sqerr+berror

	def PerformOptimization(self):
		prm0 = PARAMS["RBFS"][:PARAMS["SH_NRAD"]].flatten()
		# prm0 = np.array((PARAMS["ANES"][0], PARAMS["ANES"][5], PARAMS["ANES"][6], PARAMS["ANES"][7]))
		print(prm0)

		print("Optimizing RBFS.")
		if (self.method == "Ipecac"):
			obj = lambda x: self.Ipecac_Objective(x)
		elif (self.method == "KRR"):
			obj = lambda x: self.KRR_Objective(x)
		res=optimize.minimize(obj, prm0, method='COBYLA', tol=1e-8, options={'disp':True, 'rhobeg':0.1})
		LOGGER.info("Opt complete: %s", res.message)
		LOGGER.info("Optimal Basis Parameters: %s", res.x)
		return

	def BasinHopping(self):
		prm0 = PARAMS["RBFS"][:PARAMS["SH_NRAD"]].flatten()
		# prm0 = np.array((PARAMS["ANES"][0], PARAMS["ANES"][5], PARAMS["ANES"][6], PARAMS["ANES"][7]))
		print(prm0)
		print("Optimizing RBFS.")
		if (self.method == "Ipecac"):
			obj = lambda x: self.Ipecac_Objective(x)
		elif (self.method == "KRR"):
			obj = lambda x: self.KRR_Objective(x)
		min_kwargs = {"method": "COBYLA", "options": "{'rhobeg':0.1}"}
		ret=optimize.basinhopping(obj, prm0, minimizer_kwargs=min_kwargs, niter=100)
		LOGGER.info("Optimal Basis Parameters: %s", res.x)
		return

	def SetEmbeddings(self):
		self.DesiredEmbs = []
		for mol in self.Mols:
			self.DesiredEmbs.append(self.dig.Emb(mol,MakeOutputs=False))
