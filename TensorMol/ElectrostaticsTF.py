"""
This file contains routines for calculating electrostatic energies and forces
using tensorflow. No training etc. These functions are used as a utility in
other Instance models. Some of these functions mirror Electrostatics.py

Position units are Bohr, and energy units are Hartree
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math, time, os, sys, os.path
from TensorMol.Util import *

if (HAS_TF):
	import tensorflow as tf

def TFMatrixPower(mat_,exp_):
	"""
	General Matrix Power in Tensorflow.
	"""
	s,u,v = tf.svd(mat_,full_matrices=True,compute_uv=True)
	return tf.transpose(tf.matmul(u,tf.matmul(tf.diag(tf.pow(s,exp_)),tf.transpose(v))))

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
	D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r) + tf.cast(1e-26,tf.float64)
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
	D = rmt - 2*tf.einsum('ijk,ilk->ijl',r_,r_) + rmtt + tf.cast(1e-26,tf.float64)
	return tf.sqrt(D)

def TFDistanceLinear(B,NZP):
	"""
	Compute a distance vector of B, a coordinate matrix
	Args:
		B: a Nx3 matrix
		NZP: a (nonzero pairs X 2) index matrix.
	Returns:
		D: a NZP X 1 tensor of distances.
	"""
	Ii = tf.slice(NZP,[0,0],[-1,1])
	Ij = tf.slice(NZP,[0,1],[-1,1])
	Ri = tf.gather_nd(B,Ii)
	Rj = tf.gather_nd(B,Ij)
	A = Ri - Rj + 1e-26
	return tf.sqrt(tf.reduce_sum(A*A, 1))

def TFDistancesLinear(B,NZP):
	"""
	Returns distance vector batched over mols
	With these sparse versions I think the mol dimension should be eliminated.

	Args:
		r_: Nmol X MaxNAtom X 3 coordinate tensor
		NZP: a ( nonzero pairs X 3) index matrix. (mol, i, j)
	Returns
		D: nonzero pairs  Distance vector. (Dij)
		The calling routine has to scatter back into mol dimension using NZP if desired.
	"""
	Ii = tf.slice(NZP,[0,0],[-1,2])
	Ij = tf.concat([tf.slice(NZP,[0,0],[-1,1]),tf.slice(NZP,[0,2],[-1,1])],1)
	Ri = tf.gather_nd(B,Ii)
	Rj = tf.gather_nd(B,Ij)
	A = Ri - Rj + 1e-26
	D = tf.sqrt(tf.reduce_sum(A*A, 1))
	return D

def BumpEnergy(h,w,xyz,x,nbump):
	"""
	A -1*potential energy which is just the sum of gaussians
	with height h and width w at positions xyz sampled at x.
	This uses distance matrices to maintain rotational invariance.
	The factor of negative 1 is because we only desire the force...

	Args:
		h: bump height
		w: bump width
		xyz: a nbump X N X 3 tensor of bump centers.
		x: (n X 3) tensor representing the point at which the energy is sampled.
		nbump: an integer determining the number of nonzero bumps.
	"""
	xshp = tf.shape(x)
	nx = xshp[0]
	Nzxyz = tf.slice(xyz,[0,0,0],[nbump,nx,3])
	Ds = TFDistances(Nzxyz) # nbump X MaxNAtom X MaxNAtom Distance tensor.
	Dx = TFDistance(x) # MaxNAtom X MaxNAtom Distance tensor.	#sqrt2pi = tf.constant(2.50662827463100,dtype = tf.float64)
	w2 = w*w
	rij = Ds - tf.tile(tf.reshape(Dx,[1,nx,nx]),[nbump,1,1])
	ToExp = tf.einsum('ijk,ijk->i',rij,rij)
	ToSum = -1.0*h*tf.exp(-0.5*ToExp/w2)
	return tf.reduce_sum(ToSum,axis=0)

def BowlEnergy(BowlK,x):
	"""
	A bowl which attracts everything to 0,0,0
	with the energy tf.sqrt(x.x)

	Args:
		BowlK: the bowl force constant.
		x: (n X 3) tensor representing the point at which the energy is sampled.
	"""
	xshp = tf.shape(x)
	nx = xshp[0]
	Ds = tf.einsum('ij,ij->i',x,x)
	return tf.reduce_sum(-1*BowlK*((Ds)+1e-26),axis=0)

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
	ones = tf.ones(tf.shape(Ds),dtype = tf.float64)
	zeros = tf.zeros(tf.shape(Ds),dtype = tf.float64)
	ZeroTensor = tf.where(tf.less_equal(Ds,0.000000001),ones,zeros)
	Ds += ZeroTensor
	# Zero atomic numbers will be set to 1 and masked elsewhere
	Zs = tf.where(tf.equal(Zs,0),tf.ones_like(Zs),Zs)
	# Extract De_ij and Re_ij
	Zshp = tf.shape(Zs)
	Zr = tf.reshape(Zs,[Zshp[0],Zshp[1],1])-1 # Indices start at 0 AN's start at 1.
	Zij1 = tf.tile(Zr,[1,1,Zshp[1]]) # molXatomXatom
	Zij2 = tf.transpose(Zij1,perm=[0,2,1])
	Zij = tf.stack([Zij1,Zij2],axis=3) # molXatomXatomX2
	# Gather desired LJ parameters.
	Zij = tf.reshape(Zij,[Zshp[0]*Zshp[1]*Zshp[1],2])
	Eeij = tf.reshape(tf.gather_nd(Ee,Zij),[Zshp[0],Zshp[1],Zshp[1]])
	Reij = tf.reshape(tf.gather_nd(Re,Zij),[Zshp[0],Zshp[1],Zshp[1]])
	R = Reij/Ds
	K = Eeij*(tf.pow(R,12.0)-2.0*tf.pow(R,6.0))
	# Use the ZeroTensors to mask the output for zero dist or AN.
	K = tf.where(tf.equal(ZeroTensor,1.0),tf.zeros_like(K),K)
	K = tf.where(tf.is_nan(K),tf.zeros_like(K),K)
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle of each.
	return K

def LJKernelLinear(Ds,Zs,Ee,Re,NZP):
	"""
	Linear Scaling Lennard-Jones Energy for a single Molecule.

	Args:
		Ds: Distances Enumerated by NZP (flat)
		Zs: A batch of Atomic Numbers. (maxatom X 1)
		Ee: a matrix of LJ well depths.
		Re: a matrix of Bond minima.
		NZP: a list of nonzero atom pairs NNZ X 2 = (i, j).
	Returns
		LJ energy.
	"""
	NZP_shape = tf.shape(NZP)
	Zs_shp = tf.shape(Zs)
	NZP_shape = tf.Print(NZP_shape,[NZP_shape])
	maxnpairs = tf.shape(NZP)[0]
	Ii = tf.slice(NZP,[0,0],[-1,1])
	Ij = tf.slice(NZP,[0,1],[-1,1])
	Zi = tf.reshape(tf.gather_nd(Zs,Ii),[tf.shape(Ii)[0],1])
	Zj = tf.reshape(tf.gather_nd(Zs,Ij),[tf.shape(Ii)[0],1])
	Zij = tf.concat([Zi,Zj],axis=1)
	Eeij = tf.gather_nd(Ee,Zij)
	Reij = tf.gather_nd(Re,Zij)
	R = Reij/Ds
	K = Eeij*(tf.pow(R,12.0)-2.0*tf.pow(R,6.0))
	K = tf.where(tf.is_nan(K),tf.zeros_like(K),K)
	K = tf.reduce_sum(K,axis=0)
	return K

def LJKernelsLinear(Ds,Zs,Ee,Re,NZP):
	"""
	Batched over molecules.
	Args:
		Ds: Distances Enumerated by NZP (flat)
		Zs: A batch of Atomic Numbers. (nmol X maxatom X 1)
		Ee: a matrix of LJ well depths.
		Re: a matrix of Bond minima.
		NZP: a list of nonzero atom pairs NNZ X (mol, i, j).
	Returns
		A #Mols vector of LJ energies.
	"""
	NZP_shape = tf.shape(NZP)
	Zs_shp = tf.shape(Zs)
	maxnpairs = NZP_shape[0]
	nmols = Zs_shp[0]
	Ii = tf.slice(NZP,[0,0],[-1,2])
	Ij = tf.concat([tf.slice(NZP,[0,0],[-1,1]),tf.slice(NZP,[0,2],[-1,1])],1)
	Zi = tf.reshape(tf.gather_nd(Zs,Ii),[maxnpairs])
	Zj = tf.reshape(tf.gather_nd(Zs,Ij),[maxnpairs])
	# Gather desired LJ parameters.
	Zij = tf.stack([Zi,Zj],axis=1)
	Eeij = tf.reshape(tf.gather_nd(Ee,Zij),[maxnpairs])
	Reij = tf.reshape(tf.gather_nd(Re,Zij),[maxnpairs])
	R = Reij/tf.reshape(Ds,[maxnpairs])
	K = Eeij*(tf.pow(R,12.0)-2.0*tf.pow(R,6.0))
	K = tf.where(tf.is_nan(K),tf.zeros_like(K),K)
	range_index = tf.reshape(tf.range(tf.cast(maxnpairs, tf.int64), dtype=tf.int64),[maxnpairs,1])
	mol_index = tf.reshape(tf.slice(NZP,[0,0],[-1,1]),[maxnpairs,1])
	inds = tf.reshape(tf.stack([mol_index,range_index],axis=1),[maxnpairs,2])
	# Now use the sparse reduce sum trick to scatter this into mols.
	sp_atomoutputs = tf.SparseTensor(inds, tf.reshape(K,[maxnpairs]), dense_shape=[tf.cast(nmols, tf.int64), tf.cast(maxnpairs, tf.int64)])
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def ElectrostaticDampedShiftedLinear(Ds,Qs,NZP,alpha,Rc):
	"""
	A tensorflow linear scaling implementation of the Damped Shifted Electrostatic Force
	http://aip.scitation.org.proxy.library.nd.edu/doi/pdf/10.1063/1.2206581
	Batched over molecules.

	Args:
		Ds: Distances Enumerated by NZP (flat)
		Qs: A batch of Atomic Charges. (nmol X maxatom)
		NZP: a list of nonzero atom pairs NNZ X (mol, i, j).
		alpha: DSF alpha parameter (~0.2)
		Rc: DSF Rc parameter. (15A)
	Returns
		A #Mols X MaxNAtoms X MaxNAtoms matrix of LJ kernel contributions.
	"""
	twooversqrtpi = tf.constant(1.1283791671,dtype=tf.float64)
	NZP_shape = tf.shape(NZP)
	Zs_shp = tf.shape(Zs)
	maxnpairs = NZP_shape[0]
	nmols = Zs_shp[0]
	Ii = tf.slice(NZP,[0,0],[-1,2])
	Ij = tf.concat([tf.slice(NZP,[0,0],[-1,1]),tf.slice(NZP,[0,2],[-1,1])],1)
	Qi = tf.reshape(tf.gather_nd(Qs,Ii),[maxnpairs])
	Qj = tf.reshape(tf.gather_nd(Qs,Ij),[maxnpairs])
	# Gather desired LJ parameters.
	Qij = Qi*Qj
	# This is Dan's Equation (18)
	XX = alpha*Rc
	ZZ = tf.erfc(XX)/Rc
	YY = twooversqrtpi*alpha*tf.exp(-XX*XX)/Rc
	K = Qij*(tf.erfc(alpha*Ds)/Ds - ZZ + (Ds-Rc)*(ZZ/Rc+YY))
	K = tf.where(tf.is_nan(K),tf.zeros_like(K),K)
	range_index = tf.reshape(tf.range(tf.cast(maxnpairs, tf.int64), dtype=tf.int64),[maxnpairs,1])
	mol_index = tf.reshape(tf.slice(NZP,[0,0],[-1,1]),[maxnpairs,1])
	inds = tf.reshape(tf.stack([mol_index,range_index],axis=1),[maxnpairs,2])
	# Now use the sparse reduce sum trick to scatter this into mols.
	sp_atomoutputs = tf.SparseTensor(inds, tf.reshape(K,[maxnpairs]), dense_shape=[tf.cast(nmols, tf.int64), tf.cast(maxnpairs, tf.int64)])
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def LJEnergy_Numpy(XYZ,Z,Ee,Re):
	"""
	The same as the routine below, but
	in numpy just to test.
	"""
	n = XYZ.shape[0]
	D = np.zeros((n,n))
	for i in range(n):
		D[i,i] = 1.0
		for j in range(n):
			if i == j:
				continue
			D[i,j] = np.linalg.norm(XYZ[i]-XYZ[j])
	R = 1.0/D
	K = 0.01*(np.power(R,12.0)-2.0*np.power(R,6.0))
	En = 0.0
	for i in range(n):
		for j in range(n):
			if j<=i:
				K[i,j] = 0.
			else:
				En += K[i,j]
	return En

def LJEnergy(XYZs_,Zs_,Ee_, Re_):
	"""
	Returns LJ Energy of single molecule.
	Input can be padded with zeros. That will be
	removed by LJKernels.

	Args:
		XYZs_: maxatom X 3 coordinate tensor.
		Zs_: maxatom X 1 atomic number tensor.
		Ee_: MAX_ATOMIC_NUMBER X MAX_ATOMIC_NUMBER Epsilon parameter matrix.
		Re_: MAX_ATOMIC_NUMBER X MAX_ATOMIC_NUMBER Re parameter matrix.
	"""
	Ds = TFDistance(XYZs_)
	Ds = tf.where(tf.is_nan(Ds), tf.zeros_like(Ds), Ds)
	Ks = LJKernel(Ds,Zs_,Ee_,Re_)
	Ens = tf.reduce_sum(Ks)
	return Ens

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
	LJe = Ee_*tf.ones([8,8],dtype = tf.float64)
	LJr = Re_*tf.ones([8,8],dtype = tf.float64)
	Ks = LJKernels(Ds,Zs_,LJe,LJr)
	Ens = tf.reduce_sum(Ks,[1,2])
	return Ens

def LJEnergyLinear(XYZs_,Zs_,Ee_, Re_, NZP_):
	"""
	Linear scaling Lennard-Jones energy.

	Args:
		XYZs_: maxatom X 3 coordinate tensor.
		Zs_: nmols X maxatom X 1 atomic number tensor.
		Ee_: MAX_ATOMIC_NUMBER X MAX_ATOMIC_NUMBER Epsilon parameter matrix.
		Re_: MAX_ATOMIC_NUMBER X MAX_ATOMIC_NUMBER Re parameter matrix.
		NZP_: Nonzero Pairs (nnzp X 3) matrix (mol, i, j)
	"""
	Ds = TFDistanceLinear(XYZs_[0,:,:],NZP_)
	LJe = Ee_*tf.ones([8,8],dtype = tf.float64)
	LJr = Re_*tf.ones([8,8],dtype = tf.float64)
	return LJKernelLinear(Ds, Zs_, LJe, LJr, NZP_)

def LJEnergiesLinear(XYZs_,Zs_,Ee_, Re_, NZP_):
	"""
	Returns LJ Energies batched over molecules.
	Input can be padded with zeros. That will be
	removed by LJKernels. This version is linear scaling with sparse indices NZP

	Args:
		XYZs_: nmols X maxatom X 3 coordinate tensor.
		Zs_: nmols X maxatom X 1 atomic number tensor.
		Ee_: MAX_ATOMIC_NUMBER X MAX_ATOMIC_NUMBER Epsilon parameter matrix.
		Re_: MAX_ATOMIC_NUMBER X MAX_ATOMIC_NUMBER Re parameter matrix.
		NZP_: Nonzero Pairs (nnzp X 3) matrix (mol, i, j)
	"""
	Ds = TFDistancesLinear(XYZs_,NZP_)
	LJe = Ee_*tf.ones([8,8],dtype = tf.float64)
	LJr = Re_*tf.ones([8,8],dtype = tf.float64)
	return LJKernelsLinear(Ds, Zs_, LJe, LJr, NZP_)

def HarmKernels(XYZs, Deqs, Keqs):
	"""
	Args:
		XYZs: a nmol X maxnatom X 3 tensor of coordinates.
		Deqs: a nmol X maxnatom X maxnatom tensor of Equilibrium distances
		Keqs: a nmol X maxnatom X maxnatom tensor of Force constants.
	"""
	Ds = TFDistances(XYZs)
	tmp = Ds - tf.cast(Deqs,tf.float64)
	tmp -= tf.matrix_diag(tf.matrix_diag_part(tmp))
	K = tf.cast(Keqs,tf.float64)*tmp*tmp
	#K = tf.Print(K,[K],"Kern",100)
	K = tf.matrix_band_part(K, 0, -1) # Extract upper triangle of each.
	return K

def CoulombKernel(D):
	"""
	Args:
		D:  A square distance matrix (bohr)
		PARAMS["EESwitchFunc"]: The Kernel type
			None => 1/r, bare Coulomb
			'Cos' => 1/r -> (0.5*(cos(PI*r/EECutoff)+1))/r (if r>Cutoff else 0)
			'Tanh' => 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	"""
	K = tf.div(tf.cast(1.0, dtype=tf.float64),D)
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
	ones = tf.ones_like(D)
	CosScreen = tf.where(tf.greater(D, PARAMS["EECutoff"]), ones, 0.0*D)
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
	ones = tf.ones_like(D)
	CosScreen = tf.where(tf.greater(D, PARAMS["EECutoff"]), ones, 0.0*D)
	Cut = 1.0-(1.0-0.5*(tf.cos(D*Pi/PARAMS["EECutoff"])+1))*CosScreen
	#Cut = tf.Print(Cut,[Cut],"CosCut", 10000, 1000 )
	return CoulombKernel(D)*Cut

def TanhKernelLR(D):
	"""
	Args:
		D:  A square distance matrix (bohr)
		'Tanh' => 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	"""
	ones = tf.ones_like(D)
	Screen = tf.where(tf.greater(D, PARAMS["EECutoff"]+3.0*PARAMS["EEdr"]), ones,0.0*D)
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
	ones = tf.ones_like(D)
	Screen = tf.where(tf.greater(D, PARAMS["EECutoff"]+3.0*PARAMS["EEdr"]), ones,0.0*D)
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
	D = TFDistances(xyz_pl)  # Make distance matrices for all mols.
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
	xyz_ = tf.Variable([[0.,0.,0.],[10.0,0.,0.],[0.,0.,5.],[0.,0.,2.],[0.,1.,9.],[0.,1.,20.]], dtype=tf.float64)
	q_ = tf.Variable([1.,-1.,1.,-1.,0.5,0.5], dtype=tf.float64)
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
