"""
 Ipecac: a syrup which causes vomiting.
 This file contains routines which 'attempt' to reverse embeddings back to the geometry of a molecule.
 Ie: it's the inverse of Digest.py
"""

from Mol import *
from Util import *
import os, sys, re, random, math, copy
import numpy as np
import cPickle as pickle
import LinearOperations, DigestMol, Digest, Opt

def EmbAtomwiseErr(mol_,dig_,emb_):
	ins = dig_.TrainDigestMolwise(mol_,MakeOutputs_=False)
	err = np.sqrt(np.sum((ins-emb_)*(ins-emb_)))
	return err

def ReverseAtomwiseEmbedding(atoms_, dig_, emb_, guess_=None, GdDistMatrix=None):
	"""
	Args:
		atoms_: a list of element types for which this routine provides coords.
		dig_: a digester
		emb_: the embedding which we will try to construct a mol to match. Because this is atomwise this will actually be a (natom X embedding shape) tensor.
	Returns:
		A best-fit version of a molecule which produces an embedding as close to emb_ as possible.
	"""
	natom = len(atoms_)
	# Construct a random non-clashing guess.
	# this is the tricky step, a random guess probably won't work.
	coords = np.random.rand(natom,3)
	if (guess_==None):
	# This puts natom into a cube of length 1 so correct the density to be roughly 1atom/angstrom.
		coords *= natom
		mfit = Mol(atoms_,coords)
		mfit.WriteXYZfile("./results/", "RevLog")
		# Next optimize with an equilibrium distance matrix which is roughly correct for each type of species...
		mfit.DistMatrix = np.ones((natom,3))
		np.fill_diagonal(mfit.DistMatrix,0.0)
		opt = Optimizer(None)
		opt.OptGoForce(mfit)
		mfit.WriteXYZfile("./results/", "RevLog")
	else:
		coords = guess_
	# Now shit gets real. Create a function to minimize.
	objective = lambda crds: EmbAtomwiseErr(Mol(atoms_,crds.reshape(natom,3)),dig_,emb_)
	if (0):
		def callbk(x_):
			mn = Mol(atoms_, x_.reshape(natom,3))
			mn.BuildDistanceMatrix()
			print "Distance error : ", np.sqrt(np.sum((GdDistMatrix-mn.DistMatrix)*(GdDistMatrix-mn.DistMatrix)))
	import scipy.optimize
	res=scipy.optimize.minimize(objective,coords.reshape(natom*3),method='L-BFGS-B',tol=0.000001,options={"maxiter":5000000,"maxfun":10000000})#,callback=callbk)
	print "Reversal complete: ", res.message
	mfit = Mol(atoms_, res.x.reshape(natom,3))
	return mfit

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
		self.RBFS = np.array([[0.33177521, 0.50949676], [0.74890231, 0.99964731], [0.52021807, 0.42015268],[0.6151809, 0.39502989], [1.26607895, 1.24048779], [2.19569368, 2.39738431]])
		self.ANES = np.array([0.50068655, 1., 1., 1., 1., 1.12237954, 0.90361766, 1.06592739])
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
			resultmols.append(ReverseAtomwiseEmbedding(m.atoms, self.dig, emb, guess_=m.coords))
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
		prm0 = PARAMS["RBFS"]
		import scipy.optimize
		print "Optimizing RBFS."
		obj = lambda x: self.MyObjective(x)
		res=scipy.optimize.minimize(obj, prm0, method='L-BFGS-B', tol=0.0001)
		print "Opt complete", res.message
		return
