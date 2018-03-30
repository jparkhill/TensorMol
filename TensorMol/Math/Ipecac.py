"""
 Ipecac: a syrup which causes vomiting.
 This file contains routines which 'attempt' to reverse embeddings back to the geometry of a molecule.
 Ie: it's the inverse of Digest.py
"""

from __future__ import absolute_import
from __future__ import print_function
from .Mol import *
from ..Util import *
import os, sys, re, random, math, copy, itertools
import numpy as np
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle
from .LinearOperations import *
from ..Containers.DigestMol import *
from ..Containers.Digest import *
from ..Simulations.Opt import *
from scipy import optimize

class Ipecac:
	def __init__(self, set_, dig_, eles_):
		"""
		Args:
			set_ : an MSet of molecules
			dig_: embedding type for reversal
			eles_: list of possible elements for reversal
		"""
		self.set = set_
		self.dig = dig_
		self.eles = eles_

	def ReverseAtomwiseEmbedding(self, emb_,atoms_, guess_, GdDistMatrix):
		"""
		Args:
			atoms_: a list of element types for which this routine provides coords.
			dig_: a digester
			emb_: the embedding which we will try to construct a mol to match. Because this is atomwise this will actually be a (natoms X embedding shape) tensor.
		Returns:
			A best-fit version of a molecule which produces an embedding as close to emb_ as possible.
		"""
		natoms = emb_.shape[0]
		if atoms_ == None:
			atoms = np.full((natoms),6)
		else:
			atoms = atoms_
		# if (guess_==None):
		# 	coords = np.zeros((natoms, 3))
		# 	print self.EmbAtomwiseErr(Mol(atoms[:1], coords[:1,:]), emb_[:1,:])
		# 	return
		# 	func = lambda crds: self.EmbAtomwiseErr(Mol(atoms,crds.reshape(natoms,3)),emb_)
		# 	min_kwargs = {"method": "BFGS"}
		# 	# This puts natom into a cube of length 1 so correct the density to be roughly 1atom/angstrom.
		# 	coords = np.random.rand(natoms,3)
		# 	coords *= natoms
		# 	ret = optimize.basinhopping(func, coords, minimizer_kwargs=min_kwargs, niter=500)
		# 	mfit = Mol(atoms, coords)
		# 	atoms_ = self.BruteForceAtoms(mfit, emb_)
		# 	func = lambda crds: self.EmbAtomwiseErr(Mol(atoms_,crds.reshape(natoms,3)),emb_)
		# 	ret = optimize.basinhopping(func, coords, minimizer_kwargs=min_kwargs, niter=500)
		# 	print("global minimum: coords = %s, atoms = %s, f(x0) = %.4f" % (ret.x, atoms_, ret.fun))
		# 	# return
		# 	coords = ret.x.reshape(natoms, 3)
		# 	mfit = Mol(atoms_,coords)
		# 	mfit.WriteXYZfile("./results/", "RevLog")
		# 	# Next optimize with an equilibrium distance matrix which is roughly correct for each type of species...
		# 	# mfit.DistMatrix = np.ones((natoms,3))
		# 	# np.fill_diagonal(mfit.DistMatrix,0.0)
		# 	# opt = Optimizer(None)
		# 	# opt.OptGoForce(mfit)
		# 	# mfit.WriteXYZfile("./results/", "RevLog")
		# else:
		coords = guess_
		# atoms = np.ones(len(atoms_), dtype=np.uint8)
		# Now shit gets real. Create a function to minimize.
		objective = lambda crds: self.EmbAtomwiseErr(Mol(atoms,crds.reshape(natoms,3)),emb_)
		if (1):
			def callbk(x_):
				mn = Mol(atoms, x_.reshape(natoms,3))
				mn.BuildDistanceMatrix()
				print("Distance error : ", np.sqrt(np.sum((GdDistMatrix-mn.DistMatrix)*(GdDistMatrix-mn.DistMatrix))))
		import scipy.optimize
		step = 0
		res=optimize.minimize(objective,coords.reshape(natoms*3),method='L-BFGS-B',tol=1.e-12,options={"maxiter":5000000,"maxfun":10000000},callback=callbk)
		# res=scipy.optimize.minimize(objective,coords.reshape(natoms*3),method='SLSQP',tol=0.000001,options={"maxiter":5000000},callback=callbk)
		# coords = res.x.reshape(natoms,3)
		# res=scipy.optimize.minimize(objective,coords.reshape(natoms*3),method='Powell',tol=0.000001,options={"maxiter":5000000},callback=callbk)
		# while (self.EmbAtomwiseErr(Mol(atoms_,coords),emb_) > 1.e-5) and (step < 10):
		# 	step += 1
		# 	res=scipy.optimize.minimize(objective,coords.reshape(natoms*3),method='L-BFGS-B',tol=0.000001,options={"maxiter":5000000,"maxfun":10000000},callback=callbk)
		# 	print "Reversal complete: ", res.message
		# 	coords = res.x.reshape(natoms,3)
		# 	mfit = Mol(atoms_, coords)
		# 	atoms_ = self.BruteForceAtoms(mfit, emb_)
		mfit = Mol(atoms, res.x.reshape(natoms,3))
		self.DistanceErr(GdDistMatrix, mfit)
		return mfit

	def BruteForceAtoms(self, mol_, emb_):
		print("Searching for best atom fit")
		bestmol = copy.deepcopy(mol_)
		besterr = 100.0
		# posib_stoich = [x for x in itertools.product([1,6,7,8], repeat=len(mol_.atoms))]
		# for stoich in posib_stoich:
		for stoich in itertools.product([1,6,7,8], repeat=len(mol_.atoms)):
			tmpmol = Mol(np.array(stoich), mol_.coords)
			tmperr = self.EmbAtomwiseErr(tmpmol,emb_)
			if tmperr < besterr:
				bestmol = copy.deepcopy(tmpmol)
				besterr = tmperr
				print(besterr)
		print(bestmol.atoms)
		return bestmol.atoms

	def EmbAtomwiseErr(self, mol_,emb_):
		ins = self.dig.Emb(mol_,MakeOutputs=False)
		err = np.sqrt(np.sum((ins-emb_)*(ins-emb_)))
		# print "Emb err: ", err
		return err

	def DistanceErr(self, GdDistMatrix_, mol_):
		mol_.BuildDistanceMatrix()
		print("Final Distance error : ", np.sqrt(np.sum((GdDistMatrix_-mol_.DistMatrix)*(GdDistMatrix_-mol_.DistMatrix))))
