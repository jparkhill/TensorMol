"""
 Ipecac: a syrup which causes vomiting.
 This file contains routines which 'attempt' to reverse embeddings back to the geometry of a molecule.
 Ie: it's the inverse of Digest.py
"""

from Mol import *
from Util import *
import os, sys, re, random, math
import numpy as np
import cPickle as pickle
import LinearOperations, MolDigester, Digester
if (HAS_EMB):
	import MolEmb

def ReverseAtomwiseEmbedding(atoms_, dig_, emb_, guess_=None, metric_ = "SqDiffAligned"):
	"""
	Args:
		atoms_: a list of element types for which this routine provides coords.
		dig_: a digester
		emb_: the embedding which we will try to construct a mol to match.
	Returns:
		A best-fit version of a molecule which produces an embedding as close to emb_ as possible.
	"""
	natom = len(atoms_)
	# Construct a random non-clashing guess.
	# this is the tricky step, a random guess probably won't work.
	coords = np.random.rand(natom,3)
	# This puts natom into a cube of length 1 so correct the density to be roughly 1atom/angstrom.
	coords *= natom
	mfit = Mol(atoms_,coords)
	return mfit
