"""
This file contains routines for calculating electrostatic energies and forces
using tensorflow. No training etc. These functions are used as a utility in
other Instance models. Some of these functions mirror Electrostatics.py

Position units are Angstrom, and energy units are Hartree
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from TensorMol.TensorData import *
import numpy as np
import cPickle as pickle
import math time, os, sys, os.path
if (HAS_TF):
	import tensorflow as tf

def TFDistance(A):
	"""
	Compute a distance matrix of A, a coordinate matrix
	Using the factorization:
	Dij = <i|i> - 2<i|j> + <j,j>
	Args:
		A: a Nx3 matrix
	Returns:
		D: a NxN matrix
	"""
	r = tf.reduce_sum(A*A, 1)
	r = tf.reshape(r, [-1, 1]) # For the later broadcast.
	D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
	return tf.sqrt(D)

def TFDistances(r_):
	"""
	Returns distance matrices batched over mols
	Args:
		r_: Nmol X MaxNAtom X 3 coordinate tensor
	Returns
		D: Nmol X MaxNAtom X MaxNAtom Distance tensor.
	"""
	rm = tf.einsum('ijk,ijk->ij',r_,r_) # Mols , Rsq.
	rshp = tf.shape(rm)
	rmt = tf.tile(rm, [1,rshp[1]])
	rmt = tf.reshape(rmt,[rshp[0],rshp[1],rshp[1]])
	rmtt = tf.transpose(rmt,perm=[0,2,1])
	D = rmt - 2*tf.einsum('ijk,ilk->ijl',r_,r_) + rmtt
	return tf.sqrt(D)

def CoulombKernel(D):
	"""
	Args:
		D:  A square distance matrix (bohr)
		PARAMS["EESwitchFunc"]: The Kernel type
			None => 1/r, bare Coulomb
			'Cos' => 1/r -> (0.5*(cos(PI*r/EECutoff)+1))/r (if r<Cutoff else 0)
			'Tanh' => 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	"""
	Dp = tf.add(D,0.00000001)
	K = tf.div(1.0,Dp)
	K = tf.subtract(K,tf.diag(tf.diag_part(K)))
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle
	#K = tf.Print(K,[K],"K Kernel",-1,1000000)
	return K

def CosKernel(D):
	"""
	Args:
		D:  A square distance matrix (bohr)
		'Cos' => 1/r -> (1-0.5*(cos(PI*r/EECutoff)+1))/r (if r<Cutoff else 0)
	"""
	ones = tf.ones(tf.shape(D))
	CosScreen = tf.where(tf.greater(D, PARAMS["EECutoff"]),ones,0.0*D)
	CosScreen = tf.Print(CosScreen,[CosScreen],"CosScreen", 10000, 1000 )
	CosArg = tf.scalar_mul(Pi/PARAMS["EECutoff"],D)
	CosOut = tf.cos(CosArg)
	CosOut = tf.add(CosOut,1.0)
	CosCut = (1.0-tf.scalar_mul(0.5,CosOut))*CosScreen
	Dp = tf.add(D,0.00000001)
	K = tf.div(1.0,Dp)
	K = tf.subtract(K,tf.diag(tf.diag_part(K)))
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle
	#K = tf.Print(K,[K],"K Kernel",-1,1000000)
	return K*CosCut

def TanhKernel(D):
	"""
	Args:
		D:  A square distance matrix (bohr)
		'Tanh' => 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	"""
	ones = tf.ones(tf.shape(D))
	Screen = tf.where(tf.greater(D, PARAMS["EECutoff"]+3.0*PARAMS["EEdr"]),ones,0.0*D)
	TanhOut = 0.5*(tf.tanh((D - PARAMS["EECutoff"])/PARAMS["EEdr"]) + 1)
	Cut = TanhOut*Screen
	Dp = tf.add(D,0.00000001)
	K = tf.div(1.0,Dp)
	K = tf.subtract(K,tf.diag(tf.diag_part(K)))
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle
	return K*Cut

def XyzsToCoulomb(xyz_pl, q_pl):
	"""
	Args:
		This version is quadratic (inefficient) and should eventually
		only be used for training purposes.

		xyz_pl: a NMol, Natom X 3 tensor of coordinates.
		q_pl: an NMol X Natom X 1 tensor of atomic charges.

		PARAMS["EESwitchFunc"]: The Kernel type
			None => 1/r, bare Coulomb
			'Cos' => 1/r -> (1.-0.5*(cos(PI*r/EECutoff)+1))/r (if r<Cutoff else 0)
			'Tanh' => 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
		Returns:
			E mol = \sum_{atom1,atom2,cart} q_1*q_2*Kernel(sqrt(pow(atom1_cart - atom2_cart,2.0)))
	"""
	D = TFDistances(xyz_pl) # Make distance matrices for all mols.
	# Compute Kernel of the distances.
	K = None
	if (PARAMS["EESwitchFunc"] == None):
		K = tf.map_fn(CoulombKernel, D)
	if (PARAMS["EESwitchFunc"] == 'Cos'):
		K = tf.map_fn(CosKernel, D)
	if (PARAMS["EESwitchFunc"] == 'Tanh'):
		K = tf.map_fn(TanhKernel, D)
	#Ks = tf.shape(K)
	#Kpr = tf.Print(K,[tf.to_float(Ks)],"K Shape",-1,1000000)
	En1 = tf.einsum('aij,ai->aij', K, q_pl)
	En2 = tf.einsum('aij,aj->aij', En1, q_pl)
	Emols = tf.reduce_sum(En2,[1,2])
	return Emols

def TestCoulomb():
	xyz_ = tf.Variable([[0.,0.,0.],[10.0,0.,0.],[0.,0.,5.],[0.,0.,2.],[0.,1.,9.],[0.,1.,20.]])
	q_ = tf.Variable([1.,-1.,1.,-1.,0.5,0.5])
	molis = tf.Variable([[0,1,2],[3,4,5]])
	xyzs = tf.gather(xyz_,molis)
	charges = tf.gather(q_,molis)
	init = tf.global_variables_initializer()
	import sys
	sys.stderr = sys.stdout
	with tf.Session() as session:
		session.run(init)
		print session.run(TFDistances(xyzs))
		print session.run(charges)
		PARAMS["EESwitchFunc"] = None # options are Cosine, and Tanh.
		print session.run(XyzsToCoulomb(xyzs,charges))
		PARAMS["EESwitchFunc"] = "Cos" # options are Cosine, and Tanh.
		print session.run(XyzsToCoulomb(xyzs,charges))
		PARAMS["EESwitchFunc"] = "Tanh" # options are Cosine, and Tanh.
		print session.run(XyzsToCoulomb(xyzs,charges))
	return
