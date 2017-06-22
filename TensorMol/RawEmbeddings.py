"""
Raw => various descriptors.

The Raw format is a batch of rank three tensors.
mol X MaxNAtoms X 4
The final dim is atomic number, x,y,z (Angstrom)

https://www.youtube.com/watch?v=h2zgB93KANE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from TensorMol.TensorData import *
from TensorMol.ElectrostaticsTF import *
import numpy as np
import cPickle as pickle
import math, time, os, sys, os.path
if (HAS_TF):
	import tensorflow as tf

def TFAngles(r_, D_, zetas_, thetas_):
	"""
	Returns angle tensor of a batch.
	Args:
		r_: Nmol X MaxNAtom X 3 coordinate tensor
		D_: Input distance tensor for thresholding.
		zetas_: list of zetas for angular grid
		thetas: list of thetas for angular grid.
	Returns
		D: Nmol X MaxNAtom X MaxNAtom X MaxNAtom X nTheta angle tensor.
	"""
	rm = tf.einsum('ijk,ijk->ij',r_,r_) # Mols , Rsq.
	rshp = tf.shape(rm)
	rmt = tf.tile(rm, [1,rshp[1]])
	rmt = tf.reshape(rmt,[rshp[0],rshp[1],rshp[1]])
	rmtt = tf.transpose(rmt,perm=[0,2,1])
	# Tensorflow can only reverse mode grad of sqrt if all these elements
	# are nonzero
	D = rmt - 2*tf.einsum('ijk,ilk->ijl',r_,r_) + rmtt + 0.000000000000000000000000001
	return tf.sqrt(D)

def TFRadials(D_, Rs_, etas_):
	"""
	The second factor in the ANI-1 symmetry function.

	Args:
		r_: Nmol X MaxNAtom X 3 coordinate tensor
		D_: Input distance tensor for thresholding.
		etas_: list of etas for radial grid
	Returns
		D: Nmol X MaxNAtom X MaxNAtom X MaxNAtom X neta radial tensor.
	"""

	rm = tf.einsum('ijk,ijk->ij',r_,r_) # Mols , Rsq.
	rshp = tf.shape(rm)
	rmt = tf.tile(rm, [1,rshp[1]])
	rmt = tf.reshape(rmt,[rshp[0],rshp[1],rshp[1]])
	rmtt = tf.transpose(rmt,perm=[0,2,1])
	# Tensorflow can only reverse mode grad of sqrt if all these elements
	# are nonzero
	D = rmt - 2*tf.einsum('ijk,ilk->ijl',r_,r_) + rmtt + 0.000000000000000000000000001
	return tf.sqrt(D)

def TFSyms(RawBatch_, eles_, zetas_, etas_, thetas_, Rs_, R_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eles_ is a list which
	maps only the Z's present in the data.
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s

	Args:
		RawBatch_ : a raw batch of molecules.
		eles_: a list mapping Z->SF order.
		zetas_: List of zeta angular series parameters.
		etas_: list of eta radial parameters.
		thetas_: list of theta angular parameters.
		Rs_: list of Rs_ radial parameters.

	Returns:
		Digested batch. In the shape NMol X Natom X Eles X Eles X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(RawBatch_)
	nmol = inp_shp[0]
	maxnatom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nzeta = tf.shape(zetas_)[0]
	neta = tf.shape(etas_)[0]
	ntheta = tf.shape(thetas_)[0]
	nr = tf.shape(Rs_)[0]

	XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
	Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
	# Construct distance matrices.
	Ds = TFDistances(XYZs)
	Ds = tf.where(tf.is_nan(Ds), tf.zeros_like(Ds), Ds)
	# Construct Radial Triples nmol X maxNAtom X nele X nele X neta*nr
	Radials = TFRadials(XYZs, Ds, eles_, Rs_, etas_)
	# Construct Angle Triples nmol X maxNAtom X nele X nele X ntheta*nzeta
	Thetas = TFAngles(XYZs, Ds, eles_, zetas_, thetas_)
	# Assemble the finished symmetry function


	return DigestedBatch
