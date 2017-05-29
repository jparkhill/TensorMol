"""
This file contains routines for calculating electrostatic energies and forces
using tensorflow. No training etc. These functions are used as a utility in
other Instance models. Some of these functions mirror Electrostatics.py

Position units are Bohr, and energy units are Hartree
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from TensorMol.TensorData import *
import numpy as np
import cPickle as pickle
import math, time, os, sys, os.path
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
	# Tensorflow can only reverse mode grad the sqrt if all these elements
	# are nonzero
	D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r) + 0.000000000000000000000000001
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
	# Tensorflow can only reverse mode grad of sqrt if all these elements
	# are nonzero
	D = rmt - 2*tf.einsum('ijk,ilk->ijl',r_,r_) + rmtt + 0.000000000000000000000000001
	return tf.sqrt(D)

def MorseKernel(D,Z,Ae,De,Re):
	"""
	Args:
		D: A square distance matrix (bohr)
		Z: Atomic Numbers.
		Ae: a matrix of force constants.
		De: a matrix of Morse De parameters. (MaxAtomicNumber X MaxAtomicNumber)
		Re: a matrix of
	"""
	# Extract De_ij and Re_ij
	Zshp = tf.shape(Z)
	Zr = tf.reshape(Z,[Zshp[0],1])-1 # Indices start at 0 AN's start at 1.
	Zij1 = tf.tile(Zr,[1,Zshp[0]])
	Zij2 = tf.transpose(Zij1)
	Zij = tf.stack([Zij1,Zij2],axis=2) # atomXatomX2
	Zij = tf.reshape(Zij,[Zshp[0]*Zshp[0],2])
	Aeij = tf.reshape(tf.gather_nd(Ae,Zij),tf.shape(D))
	Deij = tf.reshape(tf.gather_nd(De,Zij),tf.shape(D))
	Reij = tf.reshape(tf.gather_nd(Re,Zij),tf.shape(D))
	Dt = D + tf.eye(Zshp[0])
	# actually compute the kernel.
	K = Deij*tf.pow(1.0 - tf.exp(-Aeij*(Dt-Reij)),2.0)
	K = tf.subtract(K,tf.diag(tf.diag_part(K)))
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle
	#K = tf.Print(K,[K],"K Kernel",-1,1000000)
	return K

def LJKernel(D,Z,Ee,Re):
	"""
	A Lennard-Jones Kernel
	Args:
		D: A square distance matrix (bohr)
		Z: Atomic Numbers.
		Ee: a matrix of LJ well depths.
		Re: a matrix of Bond minima.
	"""
	# Extract De_ij and Re_ij
	Zshp = tf.shape(Z)
	Zr = tf.reshape(Z,[Zshp[0],1])-1 # Indices start at 0 AN's start at 1.
	Zij1 = tf.tile(Zr,[1,Zshp[0]])
	Zij2 = tf.transpose(Zij1)
	Zij = tf.stack([Zij1,Zij2],axis=2) # atomXatomX2
	Zij = tf.reshape(Zij,[Zshp[0]*Zshp[0],2])
	Eeij = tf.reshape(tf.gather_nd(Ee,Zij),tf.shape(D))
	Reij = tf.reshape(tf.gather_nd(Re,Zij),tf.shape(D))
	Reij = tf.Print(Reij,[Reij],"Reij",10000,1000)
	Dt = D + tf.eye(Zshp[0])
	K = Eeij*(tf.pow(Reij/Dt,12.0)-2.0*tf.pow(Reij/Dt,6.0))
	K = tf.subtract(K,tf.diag(tf.diag_part(K)))
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle
	return K

def LJKernels(Ds,Zs,Ee,Re):
	"""
	Batched over molecules.
	Args:
		Ds: A batch of square distance matrix (bohr)
		Zs: A batch of Atomic Numbers.
		Ee: a matrix of LJ well depths.
		Re: a matrix of Bond minima.
	Returns
		A #Mols X MaxNAtoms X MaxNAtoms matrix of LJ kernel contributions.
	"""
	# Zero distances will be set to 100.0 then masked to zero energy contributions.
	ones = tf.ones(tf.shape(Ds))
	zeros = tf.zeros(tf.shape(Ds))
	ZeroTensor = tf.where(tf.less_equal(Ds,0.00001),ones,zeros)
	Ds += ZeroTensor
	# Zero atomic numbers will be set to 1
	Zones = tf.ones(tf.shape(Zs),dtype=tf.int32)
	ZZeroTensor = tf.cast(tf.where(tf.equal(Zs,0),Zones,0*Zs),tf.float32)
	Zs = tf.where(tf.equal(Zs,0),Zones,Zs)
	# Extract De_ij and Re_ij
	Zshp = tf.shape(Zs)
	Zr = tf.reshape(Zs,[Zshp[0],Zshp[1],1])-1 # Indices start at 0 AN's start at 1.
	Zij1 = tf.tile(Zr,[1,1,Zshp[1]]) # molXatomXatom
	Zij2 = tf.transpose(Zij1,perm=[0,2,1])
	Zij = tf.stack([Zij1,Zij2],axis=3) # molXatomXatomX2
	# Construct a atomic number masks.
	Zzij1 = tf.tile(ZZeroTensor,[1,1,Zshp[1]]) # mol X atom X atom.
	Zzij2 = tf.transpose(Zzij1,perm=[0,2,1]) # mol X atom X atom.
	# Gather desired LJ parameters.
	Zij = tf.reshape(Zij,[Zshp[0]*Zshp[1]*Zshp[1],2])
	Eeij = tf.reshape(tf.gather_nd(Ee,Zij),[Zshp[0],Zshp[1],Zshp[1]])
	Reij = tf.reshape(tf.gather_nd(Re,Zij),[Zshp[0],Zshp[1],Zshp[1]])
	R = Reij/Ds
	tf.assert_less(R,100000000.0)
	K = Eeij*(tf.pow(R,12.0)-2.0*tf.pow(R,6.0))
	# Use the ZeroTensors to mask the output for zero dist or AN.
	K = tf.where(tf.equal(ZeroTensor,1.0),tf.zeros_like(K),K)
	K = tf.where(tf.equal(Zzij1,1.0),tf.zeros_like(K),K)
	K = tf.where(tf.equal(Zzij2,1.0),tf.zeros_like(K),K)
	K = tf.where(tf.is_nan(K),tf.zeros_like(K),K)
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle of each.
	return K

def LJEnergies(XYZs_,Zs_,Ee_, Re_):
	"""
	Returns LJ Energies batched over molecules.
	Input can be padded with zeros. That will be
	removed by LJKernels.

	Args:
		XYZs_: nmols X maxatom X 3 coordinate tensor.
		Zs_: nmols X maxatom X 1 atomic number tensor.
		Ee_: MAX_ATOMIC_NUMBER X MAX_ATOMIC_NUMBER Epsilon parameter matrix.
		Re_: MAX_ATOMIC_NUMBER X MAX_ATOMIC_NUMBER Re parameter matrix.
	"""
	Ds = TFDistances(XYZs_)
	Ds = tf.where(tf.is_nan(Ds), tf.zeros_like(Ds), Ds)
	Ks = LJKernels(Ds,Zs_,Ee_,Re_)
	Ens = tf.reduce_sum(Ks,[1,2])
	return Ens

def CoulombKernel(D):
	"""
	Args:
		D:  A square distance matrix (bohr)
		PARAMS["EESwitchFunc"]: The Kernel type
			None => 1/r, bare Coulomb
			'Cos' => 1/r -> (0.5*(cos(PI*r/EECutoff)+1))/r (if r>Cutoff else 0)
			'Tanh' => 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	"""
	K = tf.div(1.0,D)
	K = tf.subtract(K,tf.diag(tf.diag_part(K)))
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle
	#K = tf.Print(K,[K],"K Kernel",-1,1000000)
	return K

def CosKernelLR(D):
	"""
	'Cos' => 1/r -> (1-0.5*(cos(PI*r/EECutoff)+1))/r (if r>Cutoff else 0)
	Args:
		D:  A square distance matrix (bohr)
		Long: Whether long range or short range

	"""
	ones = tf.ones(tf.shape(D))
	CosScreen = tf.where(tf.greater(D, PARAMS["EECutoff"]),ones,0.0*D)
	Cut = (1.0-0.5*(tf.cos(D*Pi/PARAMS["EECutoff"])+1))*CosScreen
	#Cut = tf.Print(Cut,[Cut],"CosCut", 10000, 1000 )
	return CoulombKernel(D)*Cut

def CosKernelSR(D):
	"""
	'Cos' => 1/r -> (1-0.5*(cos(PI*r/EECutoff)+1))/r (if r>Cutoff else 0)
	Args:
		D:  A square distance matrix (bohr)
		Long: Whether long range or short range

	"""
	ones = tf.ones(tf.shape(D))
	CosScreen = tf.where(tf.greater(D, PARAMS["EECutoff"]),ones,0.0*D)
	Cut = 1.0-(1.0-0.5*(tf.cos(D*Pi/PARAMS["EECutoff"])+1))*CosScreen
	#Cut = tf.Print(Cut,[Cut],"CosCut", 10000, 1000 )
	return CoulombKernel(D)*Cut

def TanhKernelLR(D):
	"""
	Args:
		D:  A square distance matrix (bohr)
		'Tanh' => 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	"""
	ones = tf.ones(tf.shape(D))
	Screen = tf.where(tf.greater(D, PARAMS["EECutoff"]+3.0*PARAMS["EEdr"]),ones,0.0*D)
	TanhOut = 0.5*(tf.tanh((D - PARAMS["EECutoff"])/PARAMS["EEdr"]) + 1)
	Cut = TanhOut*Screen
	K = CoulombKernel(D)
	#Cut = tf.Print(Cut,[Cut],"Cut", 10000, 1000 )
	return K*Cut

def TanhKernelSR(D):
	"""
	Args:
		D:  A square distance matrix (bohr)
		'Tanh' => 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	"""
	ones = tf.ones(tf.shape(D))
	Screen = tf.where(tf.greater(D, PARAMS["EECutoff"]+3.0*PARAMS["EEdr"]),ones,0.0*D)
	TanhOut = 0.5*(tf.tanh((D - PARAMS["EECutoff"])/PARAMS["EEdr"]) + 1)
	Cut = TanhOut*Screen
	K = CoulombKernel(D)
	Cut = 1.0-Cut
	#Cut = tf.Print(Cut,[Cut],"Cut", 10000, 1000 )
	return K*Cut

def XyzsToCoulomb(xyz_pl, q_pl, Long = True):
	"""
	Args:
		This version is quadratic (inefficient) and should eventually
		only be used for training purposes.

		xyz_pl: a NMol, Natom X 3 tensor of coordinates.
		q_pl: an NMol X Natom X 1 tensor of atomic charges.
		Long: Whether to use long-rage or short-range kernel.

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
	if (PARAMS["EESwitchFunc"] == 'CosLR'):
		K = tf.map_fn(CosKernelLR, D)
	if (PARAMS["EESwitchFunc"] == 'CosSR'):
		K = tf.map_fn(CosKernelSR, D)
	if (PARAMS["EESwitchFunc"] == 'TanhLR'):
		K = tf.map_fn(TanhKernelLR, D)
	if (PARAMS["EESwitchFunc"] == 'TanhSR'):
		K = tf.map_fn(TanhKernelSR, D)
	#Ks = tf.shape(K)
	#Kpr = tf.Print(K,[tf.to_float(Ks)],"K Shape",-1,1000000)
	En1 = tf.einsum('aij,ai->aij', K, q_pl)
	En2 = tf.einsum('aij,aj->aij', En1, q_pl)
	Emols = tf.reduce_sum(En2,[1,2])
	# dEmols = tf.gradients(Emols,xyz_pl) # This works just fine :)
	return Emols

def TestCoulomb():
	xyz_ = tf.Variable([[0.,0.,0.],[10.0,0.,0.],[0.,0.,5.],[0.,0.,2.],[0.,1.,9.],[0.,1.,20.]])
	q_ = tf.Variable([1.,-1.,1.,-1.,0.5,0.5])
	molis = tf.Variable([[0,1,2],[3,4,5]])
	xyzs = tf.gather(xyz_,molis)
	charges = tf.gather(q_,molis)
	Ds = TFDistances(xyzs)
	dDs = tf.gradients(Ds,xyz_)

	init = tf.global_variables_initializer()
	import sys
	sys.stderr = sys.stdout
	with tf.Session() as session:
		session.run(init)
		print(session.run(Ds))
		print(session.run(dDs))
		print(session.run(charges))
		PARAMS["EESwitchFunc"] = None # options are Cosine, and Tanh.
		print(session.run(XyzsToCoulomb(xyzs,charges)))
		PARAMS["EESwitchFunc"] = "CosSR" # options are Cosine, and Tanh.
		print(session.run(XyzsToCoulomb(xyzs,charges)))
		PARAMS["EESwitchFunc"] = "CosLR" # options are Cosine, and Tanh.
		print(session.run(XyzsToCoulomb(xyzs,charges)))
		PARAMS["EESwitchFunc"] = "TanhSR" # options are Cosine, and Tanh.
		print(session.run(XyzsToCoulomb(xyzs,charges)))
		PARAMS["EESwitchFunc"] = "TanhLR" # options are Cosine, and Tanh.
		print(session.run(XyzsToCoulomb(xyzs,charges)))
	return

def TestLJ():
	xyz_ = tf.Variable([[0.,0.,0.],[10.0,0.,0.],[0.,0.,5.],[0.,0.,2.],[0.,1.,9.],[0.,1.,20.]])
	Z_ = tf.Variable([1,2,3,4,5,6])
	Re_ = tf.ones([6,6])
	Ee_ = tf.ones([6,6])
	Ds = TFDistance(xyz_)

	init = tf.global_variables_initializer()
	import sys
	sys.stderr = sys.stdout
	with tf.Session() as session:
		session.run(init)
		print(session.run(Ds))
		print("LJ Kernel: ", session.run(LJKernel(Ds,Z_,Ee_,Re_)))
	return

def LJForce(xyz_,Z_,inds_,Ee_, Re_):
	XYZs = tf.gather(xyz_,inds_)
	Zs = tf.gather(Z_,inds_)
	Ens = LJEnergies(XYZs, Zs, Ee_, Re_)
	output = tf.gradients(Ens, XYZs)
	return output

def LearnLJ():
	xyz_ = tf.Variable([[0.,0.,0.],[10.0,0.,0.],[0.,0.,5.],[0.,0.,2.],[0.,1.,9.],[0.,1.,20.]],trainable=False)
	Z_ = tf.Variable([1,2,3,4,5,6],dtype = tf.int32,trainable=False)
	Re_ = tf.Variable(tf.ones([6,6]),trainable=True)
	Ee_ = tf.Variable(tf.ones([6,6]),trainable=True)
	inds_ = tf.Variable([[0,1,2],[3,4,5]],trainable=False)
	frcs = tf.Variable([[1.0,0.0,0.0],[-1.0,0.0,0.0],[0.,0.,1.],[0.,0.,1.],[0.,0.2,1.],[0.,0.,1.]],trainable=False)

	des_frces = tf.gather(frcs, inds_)
	loss = tf.nn.l2_loss(LJForce(xyz_, Z_, inds_, Ee_, Re_) - des_frces)
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	train = optimizer.minimize(loss)

	init = tf.global_variables_initializer()
	import sys
	sys.stderr = sys.stdout
	with tf.Session() as session:
		session.run(init)
		print()
		for step in range(1000):
			session.run(train)
			print("step", step, "Energies:", session.run(LJEnergies(tf.gather(xyz_,inds_), tf.gather(Z_,inds_), Ee_, Re_)), " Forces ", session.run(LJForce(xyz_, Z_, inds_, Ee_, Re_)), " loss ", session.run(loss))
	return
