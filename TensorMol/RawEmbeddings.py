"""
Raw => various descriptors in Tensorflow code.

The Raw format is a batch of rank three tensors.
mol X MaxNAtoms X 4
The final dim is atomic number, x,y,z (Angstrom)

https://www.youtube.com/watch?v=h2zgB93KANE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from TensorMol.Neighbors import *
from TensorMol.TensorData import *
from TensorMol.ElectrostaticsTF import *
from tensorflow.python.client import timeline
import numpy as np
import math, time, os, sys, os.path
from tensorflow.python.framework import function
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle
if (HAS_TF):
	import tensorflow as tf

def AllTriples(rng):
	"""Returns all possible triples of an input list.

	Args:
		rng: a 1D integer tensor to be triply outer product'd
	Returns:
		A natom X natom X natom X 3 tensor of all triples of entries from rng.
	"""
	rshp = tf.shape(rng)
	natom = rshp[0]
	v1 = tf.tile(tf.reshape(rng,[natom,1]),[1,natom])
	v2 = tf.tile(tf.reshape(rng,[1,natom]),[natom,1])
	v3 = tf.transpose(tf.stack([v1,v2],0),perm=[1,2,0])
	# V3 is now all pairs (nat x nat x 2). now do the same with another to make nat X 3
	v4 = tf.tile(tf.reshape(v3,[natom,natom,1,2]),[1,1,natom,1])
	v5 = tf.tile(tf.reshape(rng,[1,1,natom,1]),[natom,natom,1,1])
	v6 = tf.concat([v4,v5], axis = 3) # All triples in the range.
	return v6

def AllTriplesSet(rng, prec=tf.int32):
	"""Returns all possible triples of integers between zero and natom.

	Args:
		rng: a 1D integer tensor to be triply outer product'd
	Returns:
		A Nmol X natom X natom X natom X 4 tensor of all triples.
	"""
	natom = tf.shape(rng)[1]
	nmol = tf.shape(rng)[0]
	v1 = tf.tile(tf.reshape(rng,[nmol,natom,1]),[1,1,natom])
	v2 = tf.tile(tf.reshape(rng,[nmol,1,natom]),[1,natom,1])
	v3 = tf.transpose(tf.stack([v1,v2],1),perm=[0,2,3,1])
	# V3 is now all pairs (nat x nat x 2). now do the same with another to make nat X 3
	v4 = tf.tile(tf.reshape(v3,[nmol,natom,natom,1,2]),[1,1,1,natom,1])
	v5 = tf.tile(tf.reshape(rng,[nmol,1,1,natom,1]),[1,natom,natom,1,1])
	v6 = tf.concat([v4,v5], axis = 4) # All triples in the range.
	v7 = tf.cast(tf.tile(tf.reshape(tf.range(nmol),[nmol,1,1,1,1]),[1,natom,natom,natom,1]), dtype=prec)
	v8 = tf.concat([v7,v6], axis = -1)
	return v8

def AllDoublesSet(rng, prec=tf.int32):
	"""Returns all possible doubles of integers between zero and natom.

	Args:
		natom: max integer
	Returns:
		A nmol X natom X natom X 3 tensor of all doubles.
	"""
	natom = tf.shape(rng)[1]
	nmol = tf.shape(rng)[0]
	v1 = tf.tile(tf.reshape(rng,[nmol,natom,1]),[1,1,natom])
	v2 = tf.tile(tf.reshape(rng,[nmol,1,natom]),[1,natom,1])
	v3 = tf.transpose(tf.stack([v1,v2],1),perm=[0,2,3,1])
	v4 = tf.cast(tf.tile(tf.reshape(tf.range(nmol),[nmol,1,1,1]),[1,natom,natom,1]),dtype=prec)
	v5 = tf.concat([v4,v3], axis = -1)
	return v5

def AllSinglesSet(rng, prec=tf.int32):
	"""Returns all possible triples of integers between zero and natom.

	Args:
		natom: max integer
	Returns:
		A nmol X natom X 2 tensor of all doubles.
	"""
	natom = tf.shape(rng)[1]
	nmol = tf.shape(rng)[0]
	v1 = tf.reshape(rng,[nmol,natom,1])
	v2 = tf.cast(tf.tile(tf.reshape(tf.range(nmol),[nmol,1,1]),[1,natom,1]), dtype=prec)
	v3 = tf.concat([v2,v1], axis = -1)
	return v3

def ZouterSet(Z):
	"""
	Returns the outer product of atomic numbers for all molecules.

	Args:
		Z: nMol X MaxNAtom X 1 Z tensor
	Returns
		Z1Z2: nMol X MaxNAtom X MaxNAtom X 2 Z1Z2 tensor.
	"""
	zshp = tf.shape(Z)
	Zs = tf.reshape(Z,[zshp[0],zshp[1],1])
	z1 = tf.tile(Zs, [1,1,zshp[1]])
	z2 = tf.transpose(z1,perm=[0,2,1])
	return tf.transpose(tf.stack([z1,z2],axis=1),perm=[0,2,3,1])

def DifferenceVectorsSet(r_,prec = tf.float64):
	"""
	Given a nmol X maxnatom X 3 tensor of coordinates this
	returns a nmol X maxnatom X maxnatom X 3 tensor of Rij
	"""
	natom = tf.shape(r_)[1]
	nmol = tf.shape(r_)[0]
	#ri = tf.tile(tf.reshape(r_,[nmol,1,natom,3]),[1,natom,1,1])
	ri = tf.tile(tf.reshape(tf.cast(r_,prec),[nmol,1,natom*3]),[1,natom,1])
	ri = tf.reshape(ri, [nmol, natom, natom, 3])
	rj = tf.transpose(ri,perm=(0,2,1,3))
	return (ri-rj)

def DifferenceVectorsLinear(B, NZP):
	"""
	B: Nmol X NmaxNAtom X 3 coordinate tensor
	NZP: a index matrix (nzp X 3)
	"""
	Ii = tf.slice(NZP,[0,0],[-1,2])
	Ij = tf.concat([tf.slice(NZP,[0,0],[-1,1]),tf.slice(NZP,[0,2],[-1,1])],1)
	Ri = tf.gather_nd(B,Ii)
	Rj = tf.gather_nd(B,Ij)
	A = Ri - Rj
	return A

def TFSymASet(R, Zs, eleps_, SFPs_, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	nzeta = pshape[1]
	neta = pshape[2]
	ntheta = pshape[3]
	nr = pshape[4]
	nsym = nzeta*neta*ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001

	# atom triples.
	ats = AllTriplesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0,0],[nmol,natom,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,0,1],[nmol,natom,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,0,2],[nmol,natom,natom,natom,1])
	Rk_inds = tf.slice(ats,[0,0,0,0,3],[nmol,natom,natom,natom,1])
	Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	Z1Z2 = ZouterSet(Zs)
	ZPairs = tf.gather_nd(Z1Z2,Rjk_inds) # should have shape nmol X natom3 X 2
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom3,1,2]),tf.reshape(eleps_,[1,1,nelep,2])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.logical_and(tf.not_equal(Ri_inds,Rj_inds),tf.not_equal(Ri_inds,Rk_inds)),[nmol,natom3,1]),[1,1,nelep])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.range(nelep)
	ats = tf.tile(tf.reshape(ats,[nmol,natom3,1,4]),[1,1,nelep,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nelep,1]),[nmol,natom3,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom3 * nelep X 5 (mol, i,j,k,l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	miks = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,3],[nnz,1])],axis=-1)
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	B = tf.gather_nd(Rij,miks)
	RijRik = tf.reduce_sum(A*B,axis=1)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)
	RikRik = tf.sqrt(tf.reduce_sum(B*B,axis=1)+infinitesimal)
	denom = RijRij*RikRik+infinitesimal
	# Mask any troublesome entries.
	ToACos = RijRik/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar,ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar,ToACos)
	Thetaijk = tf.acos(ToACos)
	zetatmp = tf.cast(tf.reshape(SFPs_[0],[1,nzeta,neta,ntheta,nr]),prec)
	thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[2],[1,nzeta,neta,ntheta,nr]),[nnz,1,1,1,1]),prec)
	# Broadcast the thetas and ToCos together
	tct = tf.tile(tf.reshape(Thetaijk,[nnz,1,1,1,1]),[1,nzeta,neta,ntheta,nr])
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zetatmp)*tf.pow((1.0+Tijk),zetatmp)
	etmp = tf.cast(tf.reshape(SFPs_[1],[1,nzeta,neta,ntheta,nr]),prec) # ijk X zeta X eta ....
	rtmp = tf.cast(tf.reshape(SFPs_[3],[1,nzeta,neta,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij+RikRik)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnz,1,1,1,1]),[1,nzeta,neta,ntheta,nr]) - rtmp
	ToExp2 = etmp*tet*tet
	ToExp3 = tf.where(tf.greater(ToExp2,30),-30.0*tf.ones_like(ToExp2),-1.0*ToExp2)
	fac2 = tf.exp(ToExp3)
	# And finally the last two factors
	fac3 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros_like(RijRij, dtype=prec),0.5*(tf.cos(3.14159265359*RijRij/R_cut)+1.0))
	fac4 = tf.where(tf.greater_equal(RikRik,R_cut),tf.zeros_like(RikRik, dtype=prec),0.5*(tf.cos(3.14159265359*RikRik/R_cut)+1.0))
	# assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnz,1,1,1,1]),[1,nzeta,neta,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnz*nzeta*neta*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	jk2 = tf.add(tf.multiply(tf.slice(GoodInds,[0,2],[nnz,1]), natom), tf.slice(GoodInds,[0,3],[nnz, 1]))
	mil_jk2 = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,4],[nnz,1]),tf.reshape(jk2,[nnz,1])],axis=-1)
	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnz,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.range(nzeta), neta*ntheta*nr),[nzeta,1]),[1,neta])
	p2_2 = tf.tile(tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.multiply(tf.range(neta),ntheta*nr),[1,neta]),[nzeta,1])],axis=-1),[nzeta,neta,1,2]),[1,1,ntheta,1])
	p3_2 = tf.tile(tf.reshape(tf.concat([p2_2,tf.tile(tf.reshape(tf.multiply(tf.range(ntheta),nr),[1,1,ntheta,1]),[nzeta,neta,1,1])],axis=-1),[nzeta,neta,ntheta,1,3]),[1,1,1,nr,1])
	p4_2 = tf.reshape(tf.concat([p3_2,tf.tile(tf.reshape(tf.range(nr),[1,1,1,nr,1]),[nzeta,neta,ntheta,1,1])],axis=-1),[1,nzeta,neta,ntheta,nr,4])
	p5_2 = tf.reshape(tf.reduce_sum(p4_2,axis=-1),[1,nsym,1]) # scatter_nd only supports upto rank 5... so gotta smush this...
	p6_2 = tf.tile(p5_2,[nnz,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,[nmol,natom,nelep,natom2,nsym])
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymASet_Update(R, Zs, eleps_, SFPs_, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	nzeta = pshape[1]
	neta = pshape[2]
	ntheta = pshape[3]
	nr = pshape[4]
	nsym = nzeta*neta*ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001

	# atom triples.
	ats = AllTriplesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0,0],[nmol,natom,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,0,1],[nmol,natom,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,0,2],[nmol,natom,natom,natom,1])
	Rk_inds = tf.slice(ats,[0,0,0,0,3],[nmol,natom,natom,natom,1])
	Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	Z1Z2 = ZouterSet(Zs)
	ZPairs = tf.gather_nd(Z1Z2,Rjk_inds) # should have shape nmol X natom3 X 2
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom3,1,2]),tf.reshape(eleps_,[1,1,nelep,2])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.logical_and(tf.not_equal(Ri_inds,Rj_inds),tf.not_equal(Ri_inds,Rk_inds)),[nmol,natom3,1]),[1,1,nelep])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.range(nelep)
	ats = tf.tile(tf.reshape(ats,[nmol,natom3,1,4]),[1,1,nelep,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nelep,1]),[nmol,natom3,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom3 * nelep X 5 (mol, i,j,k,l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	miks = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,3],[nnz,1])],axis=-1)
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	B = tf.gather_nd(Rij,miks)
	RijRik = tf.reduce_sum(A*B,axis=1)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)
	RikRik = tf.sqrt(tf.reduce_sum(B*B,axis=1)+infinitesimal)

	MaskDist1 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	MaskDist2 = tf.where(tf.greater_equal(RikRik,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	MaskDist12 = tf.logical_and(MaskDist1, MaskDist2) # nmol X natom3 X nelep
	GoodInds2 = tf.boolean_mask(GoodInds, MaskDist12)
	nnz2 = tf.shape(GoodInds2)[0]
	mijs2 = tf.slice(GoodInds2,[0,0],[nnz2,3])
	miks2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,3],[nnz2,1])],axis=-1)
	A2 = tf.gather_nd(Rij,mijs2)
	B2 = tf.gather_nd(Rij,miks2)
	RijRik2 = tf.reduce_sum(A2*B2,axis=1)
	RijRij2 = tf.sqrt(tf.reduce_sum(A2*A2,axis=1)+infinitesimal)
	RikRik2 = tf.sqrt(tf.reduce_sum(B2*B2,axis=1)+infinitesimal)

	denom = RijRij2*RikRik2
	# Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar,ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar,ToACos)
	Thetaijk = tf.acos(ToACos)
	zetatmp = tf.cast(tf.reshape(SFPs_[0],[1,nzeta,neta,ntheta,nr]),prec)
	thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[2],[1,nzeta,neta,ntheta,nr]),[nnz2,1,1,1,1]),prec)
	# Broadcast the thetas and ToCos together
	tct = tf.tile(tf.reshape(Thetaijk,[nnz2,1,1,1,1]),[1,nzeta,neta,ntheta,nr], name="tct")
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zetatmp)*tf.pow((1.0+Tijk),zetatmp)
	etmp = tf.cast(tf.reshape(SFPs_[1],[1,nzeta,neta,ntheta,nr]),prec) # ijk X zeta X eta ....
	rtmp = tf.cast(tf.reshape(SFPs_[3],[1,nzeta,neta,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnz2,1,1,1,1]),[1,nzeta,neta,ntheta,nr], name="tet") - rtmp
	fac2 = tf.exp(-etmp*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	# assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnz2,1,1,1,1]),[1,nzeta,neta,ntheta,nr], name="fac34t")
	Gm = tf.reshape(fac1*fac2*fac34t,[nnz2*nzeta*neta*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	jk2 = tf.add(tf.multiply(tf.slice(GoodInds2,[0,2],[nnz2,1]), natom), tf.slice(GoodInds2,[0,3],[nnz2, 1]))
	mil_jk2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,4],[nnz2,1]),tf.reshape(jk2,[nnz2,1])],axis=-1)
	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnz2,1,4]),[1,nsym,1], name="mil_jk_Outer2")
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.range(nzeta), neta*ntheta*nr),[nzeta,1]),[1,neta])
	p2_2 = tf.tile(tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.multiply(tf.range(neta),ntheta*nr),[1,neta]),[nzeta,1])],axis=-1),[nzeta,neta,1,2]),[1,1,ntheta,1])
	p3_2 = tf.tile(tf.reshape(tf.concat([p2_2,tf.tile(tf.reshape(tf.multiply(tf.range(ntheta),nr),[1,1,ntheta,1]),[nzeta,neta,1,1])],axis=-1),[nzeta,neta,ntheta,1,3]),[1,1,1,nr,1])
	p4_2 = tf.reshape(tf.concat([p3_2,tf.tile(tf.reshape(tf.range(nr),[1,1,1,nr,1]),[nzeta,neta,ntheta,1,1])],axis=-1),[1,nzeta,neta,ntheta,nr,4])
	p5_2 = tf.reshape(tf.reduce_sum(p4_2,axis=-1),[1,nsym,1]) # scatter_nd only supports upto rank 5... so gotta smush this...
	p6_2 = tf.tile(p5_2,[nnz2,1,1], name="p6_tile") # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnz2*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,[nmol,natom,nelep,natom2,nsym])
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymRSet(R, Zs, eles_, SFPs_, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	neta = pshape[1]
	nr = pshape[2]
	nsym = neta*nr
	infinitesimal = 0.000000000000000000000000001

	# atom triples.
	ats = AllDoublesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0],[nmol,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,1],[nmol,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,2],[nmol,natom,natom,1])
	#Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	ZAll = AllDoublesSet(Zs)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1]) # should have shape nmol X natom X natom X 1
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom2,1,1]),tf.reshape(eles_,[1,1,nele,1])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.not_equal(Ri_inds,Rj_inds),[nmol,natom2,1]),[1,1,nele])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.range(nele)
	ats = tf.tile(tf.reshape(ats,[nmol,natom2,1,3]),[1,1,nele,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nele,1]),[nmol,natom2,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom2 * nele X 4 (mol, i, j, l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)
	# Mask any troublesome entries.
	etmp = tf.cast(tf.reshape(SFPs_[0],[1,neta,nr]),prec) # ijk X zeta X eta ....
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,neta,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij,[nnz,1,1]),[1,neta,nr]) - rtmp
	fac1 = tf.exp(-etmp*tet*tet)
	# And finally the last two factors
	fac2 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros_like(RijRij, dtype=prec),0.5*(tf.cos(3.14159265359*RijRij/R_cut)+1.0))
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1,1]),[1,neta,nr])
	# assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*neta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,3],[nnz,1]),tf.slice(GoodInds,[0,2],[nnz,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.range(neta), nr),[neta,1,1]),[1,nr,1])
	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.range(nr),[1,nr,1]),[neta,1,1])],axis=-1),[1,neta,nr,2])
	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
	p4_2 = tf.tile(p3_2,[nnz,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,[nmol,natom,nele,natom,nsym])
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymRSet_Update(R, Zs, eles_, SFPs_, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	neta = pshape[1]
	nr = pshape[2]
	nsym = neta*nr
	infinitesimal = 0.000000000000000000000000001

	# atom triples.
	ats = AllDoublesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0],[nmol,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,1],[nmol,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,2],[nmol,natom,natom,1])
	#Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	ZAll = AllDoublesSet(Zs)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1]) # should have shape nmol X natom X natom X 1
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom2,1,1]),tf.reshape(eles_,[1,1,nele,1])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.not_equal(Ri_inds,Rj_inds),[nmol,natom2,1]),[1,1,nele])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.range(nele)
	ats = tf.tile(tf.reshape(ats,[nmol,natom2,1,3]),[1,1,nele,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nele,1]),[nmol,natom2,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom2 * nele X 4 (mol, i, j, l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)

	MaskDist = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	GoodInds2 = tf.boolean_mask(GoodInds, MaskDist)
	nnz2 = tf.shape(GoodInds2)[0]
	mijs2 = tf.slice(GoodInds2,[0,0],[nnz2,3])
	A2 = tf.gather_nd(Rij,mijs2)
	RijRij2 = tf.sqrt(tf.reduce_sum(A2*A2,axis=1)+infinitesimal)

	# Mask any troublesome entries.
	etmp = tf.cast(tf.reshape(SFPs_[0],[1,neta,nr]),prec) # ijk X zeta X eta ....
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,neta,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz2,1,1]),[1,neta,nr]) - rtmp
	fac1 = tf.exp(-etmp*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz2,1,1]),[1,neta,nr])
	# assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz2*neta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,3],[nnz2,1]),tf.slice(GoodInds2,[0,2],[nnz2,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz2,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.range(neta), nr),[neta,1,1]),[1,nr,1])
	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.range(nr),[1,nr,1]),[neta,1,1])],axis=-1),[1,neta,nr,2])
	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
	p4_2 = tf.tile(p3_2,[nnz2,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz2*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,[nmol,natom,nele,natom,nsym])
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymASet_Update2(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001

	# atom triples.
	ats = AllTriplesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]), dtype=tf.int64), prec=tf.int64)
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0,0],[nmol,natom,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,0,1],[nmol,natom,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,0,2],[nmol,natom,natom,natom,1])
	Rk_inds = tf.slice(ats,[0,0,0,0,3],[nmol,natom,natom,natom,1])
	Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	Z1Z2 = ZouterSet(Zs)
	ZPairs = tf.gather_nd(Z1Z2,Rjk_inds) # should have shape nmol X natom3 X 2
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom3,1,2]),tf.reshape(eleps_,[1,1,nelep,2])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.logical_and(tf.not_equal(Ri_inds,Rj_inds),tf.not_equal(Ri_inds,Rk_inds)),[nmol,natom3,1]),[1,1,nelep])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.cast(tf.range(nelep),dtype=tf.int64)
	ats = tf.tile(tf.reshape(ats,[nmol,natom3,1,4]),[1,1,nelep,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nelep,1]),[nmol,natom3,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom3 * nelep X 5 (mol, i,j,k,l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	miks = tf.concat([tf.slice(GoodInds,[0,0],[nnz,2]),tf.slice(GoodInds,[0,3],[nnz,1])],axis=-1)
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	B = tf.gather_nd(Rij,miks)
	RijRik = tf.reduce_sum(A*B,axis=1)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)
	RikRik = tf.sqrt(tf.reduce_sum(B*B,axis=1)+infinitesimal)

	MaskDist1 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	MaskDist2 = tf.where(tf.greater_equal(RikRik,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	MaskDist12 = tf.logical_and(MaskDist1, MaskDist2) # nmol X natom3 X nelep
	GoodInds2 = tf.boolean_mask(GoodInds, MaskDist12)
	nnz2 = tf.shape(GoodInds2)[0]
	mijs2 = tf.slice(GoodInds2,[0,0],[nnz2,3])
	miks2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,3],[nnz2,1])],axis=-1)
	A2 = tf.gather_nd(Rij,mijs2)
	B2 = tf.gather_nd(Rij,miks2)
	RijRik2 = tf.reduce_sum(A2*B2,axis=1)
	RijRij2 = tf.sqrt(tf.reduce_sum(A2*A2,axis=1)+infinitesimal)
	RikRik2 = tf.sqrt(tf.reduce_sum(B2*B2,axis=1)+infinitesimal)

	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnz2,1,1]),prec)
	# Broadcast the thetas and ToCos together
	tct = tf.tile(tf.reshape(Thetaijk,[nnz2,1,1]),[1,ntheta,nr])
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnz2,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	# assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnz2,1,1]),[1,ntheta,nr])
	#Gm = tf.reshape(fac2*fac34t,[nnz2*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm = tf.reshape(fac1*fac2*fac34t,[nnz2*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	jk2 = tf.add(tf.multiply(tf.slice(GoodInds2,[0,2],[nnz2,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(GoodInds2,[0,3],[nnz2, 1]))
	mil_jk2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,4],[nnz2,1]),tf.reshape(jk2,[nnz2,1])],axis=-1)
	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnz2,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.

	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.cast(tf.range(ntheta), dtype=tf.int64), tf.cast(nr, dtype=tf.int64)),[ntheta,1,1]),[1,nr,1])
	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[1,nr,1]),[ntheta,1,1])],axis=-1),[1,ntheta,nr,2])
	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
	p6_2 = tf.tile(p3_2,[nnz2,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnz2*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nelep,natom2,nsym], dtype=tf.int64))
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)



def TFSymRSet_Update2(R, Zs, eles_, SFPs_, eta, R_cut, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001

	# atom triples.
	ats = AllDoublesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]), dtype=tf.int64), prec=tf.int64)
	# before performing any computation reduce this to desired pairs.
	# Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
	# Get Rij, Rik...
	Rm_inds = tf.slice(ats,[0,0,0,0],[nmol,natom,natom,1])
	Ri_inds = tf.slice(ats,[0,0,0,1],[nmol,natom,natom,1])
	Rj_inds = tf.slice(ats,[0,0,0,2],[nmol,natom,natom,1])
	#Rjk_inds = tf.reshape(tf.concat([Rm_inds,Rj_inds,Rk_inds],axis=4),[nmol,natom3,3])
	ZAll = AllDoublesSet(Zs, prec=tf.int64)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1]) # should have shape nmol X natom X natom X 1
	ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nmol,natom2,1,1]),tf.reshape(eles_,[1,1,nele,1])),axis=-1) # nmol X natom3 X nelep
	# Zero out the diagonal contributions (i==j or i==k)
	IdentMask = tf.tile(tf.reshape(tf.not_equal(Ri_inds,Rj_inds),[nmol,natom2,1]),[1,1,nele])
	Mask = tf.logical_and(ElemReduceMask,IdentMask) # nmol X natom3 X nelep
	# Mask is true if atoms ijk => pair_l and many triples are unused.
	# So we create a final index tensor, which is only nonzero m,ijk,l
	pinds = tf.cast(tf.range(nele), dtype=tf.int64)
	ats = tf.tile(tf.reshape(ats,[nmol,natom2,1,3]),[1,1,nele,1])
	ps = tf.tile(tf.reshape(pinds,[1,1,nele,1]),[nmol,natom2,1,1])
	ToMask = tf.concat([ats,ps],axis=3)
	GoodInds = tf.boolean_mask(ToMask,Mask)
	nnz = tf.shape(GoodInds)[0]
	# Good Inds has shape << nmol * natom2 * nele X 4 (mol, i, j, l=element pair.)
	# and contains all the indices we actually want to compute, Now we just slice, gather and compute.
	mijs = tf.slice(GoodInds,[0,0],[nnz,3])
	Rij = DifferenceVectorsSet(R,prec) # nmol X atom X atom X 3
	A = tf.gather_nd(Rij,mijs)
	RijRij = tf.sqrt(tf.reduce_sum(A*A,axis=1)+infinitesimal)

	MaskDist = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros([nnz], dtype=tf.bool), tf.ones([nnz], dtype=tf.bool))
	GoodInds2 = tf.boolean_mask(GoodInds, MaskDist)
	nnz2 = tf.shape(GoodInds2)[0]
	mijs2 = tf.slice(GoodInds2,[0,0],[nnz2,3])
	A2 = tf.gather_nd(Rij,mijs2)
	RijRij2 = tf.sqrt(tf.reduce_sum(A2*A2,axis=1)+infinitesimal)

	# Mask any troublesome entries.
	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz2,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz2,1]),[1,nr])
	# assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz2*nr]) # nnz X nzeta X neta X ntheta X nr
	# Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds2,[0,0],[nnz2,2]),tf.slice(GoodInds2,[0,3],[nnz2,1]),tf.slice(GoodInds2,[0,2],[nnz2,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz2,1,4]),[1,nsym,1])
	# So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p2_2 = tf.reshape(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[nr,1]),[1,nr,1])
	p4_2 = tf.tile(p2_2,[nnz2,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz2*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymASet_Linear(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, Angtri, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		Angtri: angular triples within the cutoff.
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(Angtri)[0]

	Z1Z2 = ZouterSet(Zs)

	Rij_inds = tf.slice(Angtri,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(Angtri,[0,0],[nnzt,2]), tf.slice(Angtri,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(Angtri,[0,0],[nnzt,1]), tf.slice(Angtri,[0,2],[nnzt,2])],axis=-1)
	ZPairs = tf.gather_nd(Z1Z2, Rjk_inds)
	EleIndex = tf.slice(tf.where(tf.reduce_all(tf.equal(tf.reshape(ZPairs,[nnzt,1,2]), tf.reshape(eleps_,[1, nelep, 2])),axis=-1)),[0,1],[nnzt,1])
	GoodInds2 = tf.concat([Angtri,EleIndex],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.reshape(ToExp,[nnzt,1,1]) - rtmp
	#tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	jk2 = tf.add(tf.multiply(tf.slice(GoodInds2,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(GoodInds2,[0,3],[nnzt, 1]))
	mil_jk2 = tf.concat([tf.slice(GoodInds2,[0,0],[nnzt,2]),tf.slice(GoodInds2,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnzt,1,4]),[1,nsym,1])
	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.cast(tf.range(ntheta), dtype=tf.int64), tf.cast(nr, dtype=tf.int64)),[ntheta,1,1]),[1,nr,1])
	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[1,nr,1]),[ntheta,1,1])],axis=-1),[1,ntheta,nr,2])
	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
	p6_2 = tf.tile(p3_2,[nnzt,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnzt*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	#to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nelep,natom2,nsym], dtype=tf.int64))  # scatter_nd way to do it
	to_reduce2 = tf.SparseTensor(ind2, Gm, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(natom, tf.int64), tf.cast(nelep, tf.int64), tf.cast(natom2, tf.int64), tf.cast(nsym, tf.int64)])
	#to_reduce2_reorder = tf.sparse_reorder(to_reduce2)
	reduced2 = tf.sparse_reduce_sum_sparse(to_reduce2, axis=3)
	#to_reduce2_dense = tf.sparse_tensor_to_dense(to_reduce2, validate_indices=False)
	#return tf.sparse_reduce_sum(to_reduce2, axis=3)
	#return tf.reduce_sum(to_reduce2_dense, axis=3)
	return tf.sparse_tensor_to_dense(reduced2)

def TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	#thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	#tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
	tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	#jk2 = tf.add(tf.multiply(tf.slice(AngtriEle,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(AngtriEle,[0,3],[nnzt, 1]))
	#mil_jk2 = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]),tf.slice(AngtriEle,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk2,[0,3], [nnzt, 1])) + 1

	Gm2= tf.reshape(Gm, [nnzt, nsym])
	to_reduce2 = tf.scatter_nd(mil_jk2, Gm2, tf.cast([nmol,natom, nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
#	mil_jk_Outer2 = tf.tile(tf.reshape(mil_jk2,[nnzt,1,4]),[1,nsym,1])
#	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
#	p1_2 = tf.tile(tf.reshape(tf.multiply(tf.cast(tf.range(ntheta), dtype=tf.int64), tf.cast(nr, dtype=tf.int64)),[ntheta,1,1]),[1,nr,1])
#	p2_2 = tf.reshape(tf.concat([p1_2,tf.tile(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[1,nr,1]),[ntheta,1,1])],axis=-1),[1,ntheta,nr,2])
#	p3_2 = tf.reshape(tf.reduce_sum(p2_2,axis=-1),[1,nsym,1]) # scatter_nd only supports up to rank 5... so gotta smush this...
#	p6_2 = tf.tile(p3_2,[nnzt,1,1]) # should be nnz X nsym
#	ind2 = tf.reshape(tf.concat([mil_jk_Outer2,p6_2],axis=-1),[nnzt*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
#	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom, nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))  # scatter_nd way to do it
#	#to_reduce2 = tf.SparseTensor(ind2, Gm, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(natom, tf.int64), tf.cast(nelep, tf.int64), tf.cast(jk_max, tf.int64), tf.cast(nsym, tf.int64)])
#	#to_reduce2_reorder = tf.sparse_reorder(to_reduce2)
#	#reduced2 = tf.sparse_reduce_sum_sparse(to_reduce2, axis=3)
#	#to_reduce2_dense = tf.sparse_tensor_to_dense(to_reduce2, validate_indices=True)
#	#to_reduce2_dense = tf.sparse_to_dense(ind2, [tf.cast(nmol, tf.int64), tf.cast(natom, tf.int64), tf.cast(nelep, tf.int64), tf.cast(jk_max, tf.int64), tf.cast(nsym, tf.int64)], Gm)
#	#to_reduce2_dense = tf.sparse_to_dense(ind2, [tf.cast(nmol, tf.int64), tf.cast(natom, tf.int64), tf.cast(nelep, tf.int64), tf.cast(natom2, tf.int64), tf.cast(nsym, tf.int64)], Gm, validate_indices=True)
#	#return tf.sparse_reduce_sum(to_reduce2, axis=3)
	return tf.reduce_sum(to_reduce2, axis=3)
	#return tf.sparse_tensor_to_dense(reduced2), ind2


def TFSymASet_Linear_WithEle_UsingList(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]

	num_elep, num_dim = eleps_.get_shape().as_list()
	elep_range = tf.cast(tf.range(nelep),dtype=tf.int64)

	Asym_ByElep = []
	for e in range(num_elep):
		tomask = tf.equal(AngtriEle[:,4], tf.reshape(elep_range[e], [1,1]))
		AngtriEle_sub = tf.reshape(tf.boolean_mask(AngtriEle, tf.tile(tf.reshape(tomask,[-1,1]),[1,5])),[-1,5])

		tomask1 = tf.equal(mil_jk2[:,2], tf.reshape(elep_range[e], [1,1]))
		mil_jk2_sub = tf.reshape(tf.boolean_mask(mil_jk2, tf.tile(tf.reshape(tomask1,[-1,1]),[1,4])),[-1,4])
		mi_jk2_sub = tf.concat([mil_jk2_sub[:,0:2],  mil_jk2_sub[:,3:]], axis=-1)

		nnzt_sub = tf.shape(AngtriEle_sub)[0]
		Rij_inds = tf.slice(AngtriEle_sub,[0,0],[nnzt_sub,3])
		Rik_inds = tf.concat([tf.slice(AngtriEle_sub,[0,0],[nnzt_sub,2]), tf.slice(AngtriEle_sub,[0,3],[nnzt_sub,1])],axis=-1)
		Rjk_inds = tf.concat([tf.slice(AngtriEle_sub,[0,0],[nnzt_sub,1]), tf.slice(AngtriEle_sub,[0,2],[nnzt_sub,2])],axis=-1)

		Rij = DifferenceVectorsLinear(R, Rij_inds)
		RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
		Rik = DifferenceVectorsLinear(R, Rik_inds)
		RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
		RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
		denom = RijRij2*RikRik2
		#Mask any troublesome entries.
		ToACos = RijRik2/denom
		ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
		ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
		Thetaijk = tf.acos(ToACos)
		thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
		tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
		ToCos = tct-thetatmp
		Tijk = tf.cos(ToCos) # shape: natom3 X ...
		# complete factor 1
		fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
		rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
		ToExp = ((RijRij2+RikRik2)/2.0)
		tet = tf.tile(tf.reshape(ToExp,[nnzt_sub,1,1]),[1,ntheta,nr]) - rtmp
		fac2 = tf.exp(-eta*tet*tet)
		# And finally the last two factors
		fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
		fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
		## assemble the full symmetry function for all triples.
		fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt_sub,1,1]),[1,ntheta,nr])
		Gm = tf.reshape(fac1*fac2*fac34t,[nnzt_sub*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
		jk_max = tf.reduce_max(tf.slice(mil_jk2_sub,[0,3], [nnzt_sub, 1])) + 1
		Gm2= tf.reshape(Gm, [nnzt_sub, nsym])
		to_reduce2 = tf.scatter_nd(mi_jk2_sub, Gm2, tf.cast([nmol,natom, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
		Asym_ByElep.append(tf.reduce_sum(to_reduce2, axis=2))
	return tf.stack(Asym_ByElep, axis=2)


def TFSymASet_Linear_WithElePeriodic(R, Zs, eleps_, SFPs_, zeta, eta, R_cut, AngtriEle, mil_jk2, nreal, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves append ele pair index at the end of triples with
	sorted order: m, i, l, j, k

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eleps_: a nelepairs X 2 tensor of element pairs present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
		is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
		R_cut: Radial Cutoff
		AngtriEle: angular triples within the cutoff. m, i, j, k, l
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	natom3 = natom*natom2
	nelep = tf.shape(eleps_)[0]
	pshape = tf.shape(SFPs_)
	ntheta = pshape[1]
	nr = pshape[2]
	nsym = ntheta*nr
	infinitesimal = 0.000000000000000000000000001
	onescalar = 1.0 - 0.0000000000000001
	nnzt = tf.shape(AngtriEle)[0]


	Rij_inds = tf.slice(AngtriEle,[0,0],[nnzt,3])
	Rik_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]), tf.slice(AngtriEle,[0,3],[nnzt,1])],axis=-1)
	Rjk_inds = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,1]), tf.slice(AngtriEle,[0,2],[nnzt,2])],axis=-1)

	Rij = DifferenceVectorsLinear(R, Rij_inds)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Rik = DifferenceVectorsLinear(R, Rik_inds)
	RikRik2 = tf.sqrt(tf.reduce_sum(Rik*Rik,axis=1)+infinitesimal)
	RijRik2 = tf.reduce_sum(Rij*Rik, axis=1)
	denom = RijRij2*RikRik2
	#Mask any troublesome entries.
	ToACos = RijRik2/denom
	ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos, dtype=prec)*onescalar, ToACos)
	Thetaijk = tf.acos(ToACos)
	#thetatmp = tf.cast(tf.tile(tf.reshape(SFPs_[0],[1,ntheta,nr]),[nnzt,1,1]),prec)
	# Broadcast the thetas and ToCos together
	#tct = tf.tile(tf.reshape(Thetaijk,[nnzt,1,1]),[1,ntheta,nr])
	thetatmp = tf.cast(tf.expand_dims(SFPs_[0], axis=0),prec)
	tct = tf.expand_dims(tf.expand_dims(Thetaijk, axis=1), axis=-1)
	ToCos = tct-thetatmp
	Tijk = tf.cos(ToCos) # shape: natom3 X ...
	# complete factor 1
	fac1 = tf.pow(tf.cast(2.0, prec),1.0-zeta)*tf.pow((1.0+Tijk),zeta)
	rtmp = tf.cast(tf.reshape(SFPs_[1],[1,ntheta,nr]),prec) # ijk X zeta X eta ....
	ToExp = ((RijRij2+RikRik2)/2.0)
	tet = tf.reshape(ToExp,[nnzt,1,1]) - rtmp
	#tet = tf.tile(tf.reshape(ToExp,[nnzt,1,1]),[1,ntheta,nr]) - rtmp
	fac2 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac3 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac4 = 0.5*(tf.cos(3.14159265359*RikRik2/R_cut)+1.0)
	## assemble the full symmetry function for all triples.
	fac34t = tf.reshape(fac3*fac4,[nnzt,1,1])
	Gm = tf.reshape(fac1*fac2*fac34t,[nnzt, nsym]) # nnz X nzeta X neta X ntheta X nr
	#fac34t =  tf.tile(tf.reshape(fac3*fac4,[nnzt,1,1]),[1,ntheta,nr])
	#Gm = tf.reshape(fac1*fac2*fac34t,[nnzt*ntheta*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	#jk2 = tf.add(tf.multiply(tf.slice(AngtriEle,[0,2],[nnzt,1]), tf.cast(natom, dtype=tf.int64)), tf.slice(AngtriEle,[0,3],[nnzt, 1]))
	#mil_jk2 = tf.concat([tf.slice(AngtriEle,[0,0],[nnzt,2]),tf.slice(AngtriEle,[0,4],[nnzt,1]),tf.reshape(jk2,[nnzt,1])],axis=-1)
	jk_max = tf.reduce_max(tf.slice(mil_jk2,[0,3], [nnzt, 1])) + 1

	Gm2= tf.reshape(Gm, [nnzt, nsym])
	to_reduce2 = tf.scatter_nd(mil_jk2, Gm2, tf.cast([nmol,tf.cast(nreal, tf.int32), nelep, tf.cast(jk_max, tf.int32), nsym], dtype=tf.int64))
	return tf.reduce_sum(to_reduce2, axis=3)

def TFCoulomb(R, Qs, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of sparse-coulomb
	Madelung energy build.

	Args:
	    R: a nmol X maxnatom X 3 tensor of coordinates.
	    Qs : nmol X maxnatom X 1 tensor of atomic charges.
	    R_cut: Radial Cutoff
	    Radpair: None zero pairs X 3 tensor (mol, i, j)
	    prec: a precision.
	Returns:
	    Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=-1)+infinitesimal)
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee


def TFCoulombCosLR(R, Qs, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of long range cutoff sparse-coulomb
	Madelung energy build.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	# Generate LR cutoff Matrix
	Cut = (1.0-0.5*(tf.cos(RijRij2*Pi/R_cut)+1.0))
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2*Cut
	# Scatter Back
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee


def TFCoulombPolyLR(R, Qs, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of short range cutoff sparse-coulomb
	Madelung energy build. Using switch function 1+x^2(2x-3) in http://pubs.acs.org/doi/ipdf/10.1021/ct501131j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	R_width = PARAMS["Poly_Width"]*BOHRPERA
	R_begin = R_cut
	R_end =  R_cut+R_width
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	t = (RijRij2 - R_begin)/R_width
	Cut_step1  = tf.where(tf.greater(t, 0.0), -t*t*(2.0*t-3.0), tf.zeros_like(t))
	Cut = tf.where(tf.greater(t, 1.0), tf.ones_like(t), Cut_step1)
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2*Cut
	# Scatter Back
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee


def TFCoulombPolyLRSR(R, Qs, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of short range and long range cutoff sparse-coulomb
	Madelung energy build. Using switch function 1+x^2(2x-3) in http://pubs.acs.org/doi/ipdf/10.1021/ct501131j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	R_width = PARAMS["Poly_Width"]*BOHRPERA
	R_begin = R_cut
	R_end =  R_cut+R_width
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	t = (RijRij2 - R_begin)/R_width
	Cut_step1  = tf.where(tf.greater(t, 0.0), -t*t*(2.0*t-3.0), tf.zeros_like(t))
	Cut = tf.where(tf.greater(t, 1.0), tf.ones_like(t), Cut_step1)

	R_off = PARAMS["EECutoffOff"]*BOHRPERA
	t_off = (RijRij2 - (R_off-R_width))/R_width
	Cut_off_step1  = tf.where(tf.greater(t_off, 0.0), 1+t_off*t_off*(2.0*t_off-3.0), tf.ones_like(t_off))
	Cut_off  = tf.where(tf.greater(t_off, 1.0), tf.zeros_like(t), Cut_off_step1)
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2*Cut*Cut_off
	# Scatter Back
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee

def TFCoulombEluSRDSFLR(R, Qs, R_cut, Radpair, alpha, elu_a, elu_shift, prec=tf.float64):
	"""
	A tensorflow linear scaling implementation of the Damped Shifted Electrostatic Force with short range cutoff with elu function (const at short range).
	http://aip.scitation.org.proxy.library.nd.edu/doi/pdf/10.1063/1.2206581
	Batched over molecules.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_srcut: Short Range Erf Cutoff
		R_lrcut: Long Range DSF Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		alpha: DSF alpha parameter (~0.2)
	Returns
		Energy of  Mols
	"""
	alpha = alpha/BOHRPERA
	R_lrcut = PARAMS["EECutoffOff"]*BOHRPERA
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	SR_sub = tf.where(tf.greater(RijRij2, R_cut), elu_a*(RijRij2-R_cut)+elu_shift, elu_a*(tf.exp(RijRij2-R_cut)-1.0)+elu_shift)

	twooversqrtpi = tf.constant(1.1283791671,dtype=tf.float64)
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Gather desired LJ parameters.
	Qij = Qi*Qj
	# This is Dan's Equation (18)
	XX = alpha*R_lrcut
	ZZ = tf.erfc(XX)/R_lrcut
	YY = twooversqrtpi*alpha*tf.exp(-XX*XX)/R_lrcut
	LR = Qij*(tf.erfc(alpha*RijRij2)/RijRij2 - ZZ + (RijRij2-R_lrcut)*(ZZ/R_lrcut+YY))
	LR= tf.where(tf.is_nan(LR), tf.zeros_like(LR), LR)
	LR = tf.where(tf.greater(RijRij2,R_lrcut), tf.zeros_like(LR), LR)

	SR = Qij*SR_sub

	K = tf.where(tf.greater(RijRij2, R_cut), LR, SR)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	sparse_index = tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, K, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	# Now use the sparse reduce sum trick to scatter this into mols.
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def TFVdwPolyLR(R, Zs, eles, c6, R_vdw, R_cut, Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of short range cutoff sparse-coulomb
	Madelung energy build. Using switch function 1+x^2(2x-3) in http://pubs.acs.org/doi/ipdf/10.1021/ct501131j
	damping function in http://pubs.rsc.org/en/content/articlepdf/2008/cp/b810189b is used.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		c6 : nele. Grimmer C6 coff in a.u.
		R_vdw: nele. Grimmer vdw radius in a.u.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	R = tf.multiply(R, BOHRPERA)
	R_width = PARAMS["Poly_Width"]*BOHRPERA
	R_begin = R_cut
	R_end =  R_cut+R_width
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles)[0]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	t = (RijRij2 - R_begin)/R_width
	Cut_step1  = tf.where(tf.greater(t, 0.0), -t*t*(2.0*t-3.0), tf.zeros_like(t))
	Cut = tf.where(tf.greater(t, 1.0), tf.ones_like(t), Cut_step1)

	ZAll = AllDoublesSet(Zs, prec=tf.int64)
	ZPairs1 = tf.slice(ZAll,[0,0,0,1],[nmol,natom,natom,1])
	ZPairs2 = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1])
	Ri=tf.gather_nd(ZPairs1, Radpair)
	Rl=tf.gather_nd(ZPairs2, Radpair)
	ElemIndex_i = tf.slice(tf.where(tf.equal(Ri, tf.reshape(eles, [1,nele]))),[0,1],[nnz,1])
	ElemIndex_j = tf.slice(tf.where(tf.equal(Rl, tf.reshape(eles, [1,nele]))),[0,1],[nnz,1])

	c6_i=tf.gather_nd(c6, ElemIndex_i)
	c6_j=tf.gather_nd(c6, ElemIndex_j)
	Rvdw_i = tf.gather_nd(R_vdw, ElemIndex_i)
	Rvdw_j = tf.gather_nd(R_vdw, ElemIndex_j)
	Kern = -Cut*tf.sqrt(c6_i*c6_j)/tf.pow(RijRij2,6.0)*1.0/(1.0+6.0*tf.pow(RijRij2/(Rvdw_i+Rvdw_j),-12.0))

	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_vdw = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_vdw

def TFVdwPolyLRWithEle(R, Zs, eles, c6, R_vdw, R_cut, Radpair_E1E2, prec=tf.float64):
	"""
	Tensorflow implementation of short range cutoff sparse-coulomb
	Madelung energy build. Using switch function 1+x^2(2x-3) in http://pubs.acs.org/doi/ipdf/10.1021/ct501131j
	damping function in http://pubs.rsc.org/en/content/articlepdf/2008/cp/b810189b is used.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		c6 : nele. Grimmer C6 coff in a.u.
		R_vdw: nele. Grimmer vdw radius in a.u.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	Radpair = Radpair_E1E2[:,:3]
	R = tf.multiply(R, BOHRPERA)
	R_width = PARAMS["Poly_Width"]*BOHRPERA
	R_begin = R_cut
	R_end =  R_cut+R_width
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles)[0]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	t = (RijRij2 - R_begin)/R_width
	Cut_step1  = tf.where(tf.greater(t, 0.0), -t*t*(2.0*t-3.0), tf.zeros_like(t))
	Cut = tf.where(tf.greater(t, 1.0), tf.ones_like(t), Cut_step1)

	ElemIndex_i = tf.reshape(Radpair_E1E2[:,3],[nnz, 1])
	ElemIndex_j = tf.reshape(Radpair_E1E2[:,4],[nnz, 1])

	c6_i=tf.gather_nd(c6, ElemIndex_i)
	c6_j=tf.gather_nd(c6, ElemIndex_j)
	Rvdw_i = tf.gather_nd(R_vdw, ElemIndex_i)
	Rvdw_j = tf.gather_nd(R_vdw, ElemIndex_j)
	Kern = -Cut*tf.sqrt(c6_i*c6_j)/tf.pow(RijRij2,6.0)*1.0/(1.0+6.0*tf.pow(RijRij2/(Rvdw_i+Rvdw_j),-12.0))

	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_vdw = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_vdw

def PolynomialRangeSepCoulomb(R,Qs,Radpair,SRRc,LRRc,dx):
	"""
	A tensorflow linear scaling implementation of a short-range and long range cutoff
	coulomb kernel. The cutoff functions are polynomials subject to the constraint
	that 1/r is brought to 0 twice-differentiably at SR and LR+dx cutoffs.

	The SR cutoff polynomial is 4th order, and the LR is fifth.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		SRRc: Distance where SR polynomial ends.
		LRRc: Distance where LR polynomial begins.
		dx: Small interval after which the kernel is zero.
	Returns
		A #Mols X MaxNAtoms X MaxNAtoms matrix of LJ kernel contributions.
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	Ds = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	twooversqrtpi = tf.constant(1.1283791671,dtype=tf.float64)
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	Qij = Qi*Qj
	D2 = Ds*Ds
	D3 = D2*Ds
	D4 = D3*Ds
	D5 = D4*Ds

	asr = -5./(3.*tf.pow(SRRc,4.0))
	dsr = 5./(3.*SRRc)
	csr = 1./(tf.pow(SRRc,5.0))

	x0 = LRRc
	x02 = x0*x0
	x03 = x02*x0
	x04 = x03*x0
	x05 = x04*x0

	dx2 = dx*dx
	dx3 = dx2*dx
	dx4 = dx3*dx
	dx5 = dx4*dx

	alr = -((3.*(dx4+2.*dx3*x0-4.*dx2*x02+10.*dx*x03+20.*x04))/(dx5*x03))
	blr = -((-dx5-9*dx4*x0+8.*dx2*x03-60.0*dx*x04-60.0*x05)/(dx5*x03))
	clr = (3.*(dx3-dx2*x0+10.*x03))/(dx5*x03)
	dlr = -((3.*(dx5+3.*dx4*x0-2.*dx3*x02+dx2*x03+15.*dx*x04+10.*x05))/(dx5*x02))
	elr = (3.*(dx5+dx4*x0-dx3*x02+dx2*x03+4.*dx*x04+2.*x05))/(dx5*x0)
	flr = -((dx2-3.*dx*x0+6.*x02)/(dx5*x03))

	CK = (Qij/Ds)
	SRK = Qij*(asr*D3+csr*D4+dsr)
	LRK = Qij*(alr*D3 + blr*D2 + dlr*Ds + clr*D4 + elr + flr*D5)
	ZK = tf.zeros_like(Ds)

	K0 = tf.where(tf.less_equal(Ds,SRRc),SRK,CK)
	K1 = tf.where(tf.greater_equal(Ds,LRRc),LRK,K0)
	K = tf.where(tf.greater_equal(Ds,LRRc+dx),ZK,K1)

	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	sparse_index = tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, K, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	# Now use the sparse reduce sum trick to scatter this into mols.
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def TFCoulombErfLR(R, Qs, R_cut,  Radpair, prec=tf.float64):
	"""
	Tensorflow implementation of long range cutoff sparse-Erf
	Madelung energy build.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	R_width = PARAMS["Erf_Width"]*BOHRPERA
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	# Generate LR cutoff Matrix
	Cut = (1.0 + tf.erf((RijRij2 - R_cut)/R_width))*0.5
	# Grab the Q's.
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Finish the Kernel.
	Kern = Qi*Qj/RijRij2*Cut
	# Scatter Back
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	sparse_index =tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, Kern, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	E_ee = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
	return E_ee


def TFCoulombErfSRDSFLR(R, Qs, R_srcut, R_lrcut, Radpair, alpha, prec=tf.float64):
	"""
	A tensorflow linear scaling implementation of the Damped Shifted Electrostatic Force with short range cutoff
	http://aip.scitation.org.proxy.library.nd.edu/doi/pdf/10.1063/1.2206581
	Batched over molecules.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_srcut: Short Range Erf Cutoff
		R_lrcut: Long Range DSF Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		alpha: DSF alpha parameter (~0.2)
	Returns
		Energy of  Mols
	"""
	alpha = alpha/BOHRPERA
	R_width = PARAMS["Erf_Width"]*BOHRPERA
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	Cut = (1.0 + tf.erf((RijRij2 - R_srcut)/R_width))*0.5

	twooversqrtpi = tf.constant(1.1283791671,dtype=tf.float64)
	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	# Gather desired LJ parameters.
	Qij = Qi*Qj
	# This is Dan's Equation (18)
	XX = alpha*R_lrcut
	ZZ = tf.erfc(XX)/R_lrcut
	YY = twooversqrtpi*alpha*tf.exp(-XX*XX)/R_lrcut
	K = Qij*(tf.erfc(alpha*RijRij2)/RijRij2 - ZZ + (RijRij2-R_lrcut)*(ZZ/R_lrcut+YY))*Cut
	K = tf.where(tf.is_nan(K),tf.zeros_like(K),K)
	range_index = tf.range(tf.cast(nnz, tf.int64), dtype=tf.int64)
	mol_index = tf.cast(tf.reshape(tf.slice(Radpair,[0,0],[-1,1]),[nnz]), dtype=tf.int64)
	sparse_index = tf.stack([mol_index, range_index], axis=1)
	sp_atomoutputs = tf.SparseTensor(sparse_index, K, dense_shape=[tf.cast(nmol, tf.int64), tf.cast(nnz, tf.int64)])
	# Now use the sparse reduce sum trick to scatter this into mols.
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)


def TFSymRSet_Linear(R, Zs, eles_, SFPs_, eta, R_cut, Radpair, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	ZAll = AllDoublesSet(Zs, prec=tf.int64)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1])
	Rl=tf.gather_nd(ZPairs, Radpair)
	ElemIndex = tf.slice(tf.where(tf.equal(Rl, tf.reshape(eles_,[1,nele]))),[0,1],[nnz,1])
	GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds2,[0,0],[nnz,2]),tf.slice(GoodInds2,[0,3],[nnz,1]),tf.slice(GoodInds2,[0,2],[nnz,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])
	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p2_2 = tf.reshape(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[nr,1]),[1,nr,1])
	p4_2 = tf.tile(p2_2,[nnz,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(RadpairEle,[0,0],[nnz,2]),tf.slice(RadpairEle,[0,3],[nnz,1]),tf.slice(RadpairEle,[0,2],[nnz,1])],axis=-1)
	#mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])

	to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
#	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
#	p2_2 = tf.reshape(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[nr,1]),[1,nr,1])
#	p4_2 = tf.tile(p2_2,[nnz,1,1]) # should be nnz X nsym
#	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
#	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
#	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(to_reduce2, axis=3)

def TFSymRSet_Linear_WithEle_Release(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, mil_j, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1
	to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol, tf.cast(natom, tf.int32), nele, tf.cast(j_max, tf.int32), nsym], dtype=tf.int64))
	#mil_j = tf.concat([tf.slice(RadpairEle,[0,0],[nnz,2]),tf.slice(RadpairEle,[0,3],[nnz,1]),tf.slice(RadpairEle,[0,2],[nnz,1])],axis=-1)
	#to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	return tf.reduce_sum(to_reduce2, axis=3)


def TFSymRSet_Linear_WithElePeriodic(R, Zs, eles_, SFPs_, eta, R_cut, RadpairEle, mil_j, nreal, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version appends the element type (by its index in eles_) in RadpairEle, and it is sorted by m,i,l,j

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		RadpairEle: None zero pairs X 4 tensor (mol, i, j, l)
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(RadpairEle)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, tf.slice(RadpairEle,[0, 0],[nnz, 3]))
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	#GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr
	Gm2 = tf.reshape(Gm, [nnz, nr])
	## Finally scatter out the symmetry functions where they belong.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1
	#mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])
	to_reduce2 = tf.scatter_nd(mil_j, Gm2, tf.cast([nmol, tf.cast(nreal, tf.int32), nele, tf.cast(j_max, tf.int32), nsym], dtype=tf.int64))
	return tf.reduce_sum(to_reduce2, axis=3)

def TFSymRSet_Linear_Qs(R, Zs, eles_, SFPs_, eta, R_cut, Radpair, Qs, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		Qs: charge of each atom. nmol X maxnatom
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	natom2 = natom*natom
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)
	ZAll = AllDoublesSet(Zs, prec=tf.int64)
	ZPairs = tf.slice(ZAll,[0,0,0,2],[nmol,natom,natom,1])
	Rl=tf.gather_nd(ZPairs, Radpair)
	ElemIndex = tf.slice(tf.where(tf.equal(Rl, tf.reshape(eles_,[1,nele]))),[0,1],[nnz,1])
	GoodInds2 = tf.concat([Radpair, ElemIndex], axis=-1)

	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	Qit = tf.tile(tf.reshape(Qi,[nnz,1]),[1, nr])
	Qjt = tf.tile(tf.reshape(Qj,[nnz,1]),[1, nr])

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t*Qit*Qjt,[nnz*nr]) # nnz X nzeta X neta X ntheta X nr

	## Finally scatter out the symmetry functions where they belong.
	mil_j = tf.concat([tf.slice(GoodInds2,[0,0],[nnz,2]),tf.slice(GoodInds2,[0,3],[nnz,1]),tf.slice(GoodInds2,[0,2],[nnz,1])],axis=-1)
	mil_j_Outer = tf.tile(tf.reshape(mil_j,[nnz,1,4]),[1,nsym,1])
	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	p2_2 = tf.reshape(tf.reshape(tf.cast(tf.range(nr), dtype=tf.int64),[nr,1]),[1,nr,1])
	p4_2 = tf.tile(p2_2,[nnz,1,1]) # should be nnz X nsym
	ind2 = tf.reshape(tf.concat([mil_j_Outer,p4_2],axis=-1),[nnz*nsym,5]) # This is now nnz*nzeta*neta*ntheta*nr X 8 -  m,i,l,jk,zeta,eta,theta,r
	to_reduce2 = tf.scatter_nd(ind2,Gm,tf.cast([nmol,natom,nele,natom,nsym], dtype=tf.int64))
	#to_reduce2 = tf.sparse_to_dense(ind2, tf.convert_to_tensor([nmol, natom, nelep, natom2, nsym]), Gm)
	#to_reduce_sparse = tf.SparseTensor(ind2,[nmol, natom, nelep, natom2, nzeta, neta, ntheta, nr])
	return tf.reduce_sum(tf.reduce_sum(to_reduce2, axis=3), axis=2)



def TFSymRSet_Linear_Qs_Periodic(R, Zs, eles_, SFPs_, eta, R_cut, Radpair, Qs, mil_j, nreal, prec=tf.float64):
	"""
	A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
	Here j,k are all other atoms, but implicitly the output
	is separated across elements as well. eleps_ is a list of element pairs
	G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
	a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s
	This version improves on the previous by avoiding some
	heavy tiles.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a nelepairs X 1 tensor of elements present in the data.
		SFP: A symmetry function parameter tensor having the number of elements
		as the SF output. 2 X neta  X nRs.
		R_cut: Radial Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		Qs: charge of each atom. nmol X maxnatom
		prec: a precision.
	Returns:
		Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	nele = tf.shape(eles_)[0]
	pshape = tf.shape(SFPs_)
	nr = pshape[1]
	nsym = nr
	infinitesimal = 0.000000000000000000000000001
	nnz = tf.shape(Radpair)[0]
	#Rtmp = tf.concat([tf.slice(Radpair,[0,0],[nnz,1]), tf.slice(Radpair,[0,2],[nnz,1])], axis=-1)
	#Rreverse = tf.concat([Rtmp, tf.slice(Radpair,[0,1],[nnz,1])], axis=-1)
	#Rboth = tf.concat([Radpair, Rreverse], axis=0)
	Rij = DifferenceVectorsLinear(R, Radpair)
	RijRij2 = tf.sqrt(tf.reduce_sum(Rij*Rij,axis=1)+infinitesimal)

	Qii = tf.slice(Radpair,[0,0],[-1,2])
	Qji = tf.concat([tf.slice(Radpair,[0,0],[-1,1]),tf.slice(Radpair,[0,2],[-1,1])], axis=-1)
	Qi = tf.gather_nd(Qs,Qii)
	Qj = tf.gather_nd(Qs,Qji)
	Qit = tf.tile(tf.reshape(Qi,[nnz,1]),[1, nr])
	Qjt = tf.tile(tf.reshape(Qj,[nnz,1]),[1, nr])

	rtmp = tf.cast(tf.reshape(SFPs_[0],[1,nr]),prec) # ijk X zeta X eta ....
	tet = tf.tile(tf.reshape(RijRij2,[nnz,1]),[1,nr]) - rtmp
	fac1 = tf.exp(-eta*tet*tet)
	# And finally the last two factors
	fac2 = 0.5*(tf.cos(3.14159265359*RijRij2/R_cut)+1.0)
	fac2t = tf.tile(tf.reshape(fac2,[nnz,1]),[1,nr])
	## assemble the full symmetry function for all triples.
	Gm = tf.reshape(fac1*fac2t*Qit*Qjt,[nnz, nr]) # nnz X nzeta X neta X ntheta X nr
	## Finally scatter out the symmetry functions where they belong.
	## So the above is Mol, i, l... now must outer nzeta,neta,ntheta,nr to finish the indices.
	j_max = tf.reduce_max(tf.slice(mil_j, [0,3], [nnz, 1])) + 1
	to_reduce2 = tf.scatter_nd(mil_j, Gm, tf.cast([nmol, tf.cast(nreal, tf.int32), nele, tf.cast(j_max, tf.int32), nsym], dtype=tf.int64))
	return tf.reduce_sum(tf.reduce_sum(to_reduce2, axis=3), axis=2)


def TFSymSet(R, Zs, eles_, SFPsR_, Rr_cut, eleps_, SFPsA_, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet(R, Zs, eles_, SFPsR_, Rr_cut),[nmol, natom, -1])
	GMA = tf.reshape(TFSymASet(R, Zs, eleps_, SFPsA_, Ra_cut),[nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	return GM

def TFSymSet_Scattered(R, Zs, eles_, SFPsR_, Rr_cut, eleps_, SFPsA_, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet(R, Zs, eles_, SFPsR_, Rr_cut),[nmol, natom, -1])
	GMA = tf.reshape(TFSymASet(R, Zs, eleps_, SFPsA_, Ra_cut),[nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask = AllSinglesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	IndexList = []
	SymList=[]
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		SymList.append(tf.gather_nd(GM, GatherList[-1]))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		IndexList.append(tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle]))
	return SymList, IndexList

def TFSymSet_Scattered_Update(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Update(R, Zs, eles_, SFPsR_, Rr_cut), [nmol, natom, -1])
	GMA = tf.reshape(TFSymASet_Update(R, Zs, eleps_, SFPsA_, Ra_cut), [nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask = AllSinglesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	IndexList = []
	SymList=[]
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		SymList.append(tf.gather_nd(GM, GatherList[-1]))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		IndexList.append(tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle]))
	return SymList, IndexList


def TFSymSet_Scattered_Update2(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Update2(R, Zs, eles_, SFPsR_, eta, Rr_cut), [nmol, natom, -1])
	GMA = tf.reshape(TFSymASet_Update2(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut), [nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	IndexList = []
	SymList=[]
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		SymList.append(tf.gather_nd(GM, GatherList[-1]))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		IndexList.append(tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle]))
	return SymList, IndexList

def TFSymSet_Scattered_Update_Scatter(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	This also selects out which of the atoms will contribute to the BP energy on the
	basis of whether the atomic number is treated in the 'element types to do list.'
	according to kun? (Trusted source?)

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part

	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Update2(R, Zs, eles_, SFPsR_, eta, Rr_cut), [nmol, natom, -1])
	GMA = tf.reshape(TFSymASet_Update2(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut), [nmol, natom, -1])
	GM = tf.concat([GMR, GMA], axis=2)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64)
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, Radp, Angt):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear(R, Zs, eles_, SFPsR_, eta, Rr_cut, Radp),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  Angt), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList


def TFSymSet_Scattered_Linear_WithEle(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_jk):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk),[nmol, natom,-1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_Release(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear_WithEle_Release(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk),[nmol, natom,-1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_UsingList(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_jk):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithEle_UsingList(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk),[nmol, natom,-1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Scattered_Linear_WithEle_Periodic(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_, SFPsA_, zeta, eta, Ra_cut, RadpEle, AngtEle, mil_j, mil_jk, nreal):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GMR = tf.reshape(TFSymRSet_Linear_WithElePeriodic(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j, nreal),[nmol, nreal, -1], name="FinishGMR")
	GMA = tf.reshape(TFSymASet_Linear_WithElePeriodic(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk, nreal),[nmol, nreal, -1], name="FinishGMA")
	#return GMR, R_index, GMA, A_index
	#GMR = tf.reshape(TFSymRSet_Linear_WithEle(R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle),[nmol, natom,-1], name="FinishGMR")
	#GMA = tf.reshape(TFSymASet_Linear_WithEle(R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle), [nmol, natom,-1], name="FinishGMA")
	GM = tf.concat([GMR, GMA], axis=2, name="ConcatRadAng")
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	Zs_real = Zs[:,:nreal]
	MaskAll = tf.equal(tf.reshape(Zs_real,[nmol,nreal,1]),tf.reshape(eles_,[1,1,nele]), name="FormEleMask")
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(nreal),[1,nreal]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*nreal), [nmol, nreal, 1]), dtype=tf.int64, name="FormIndices")
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,nreal,1]),[nmol, nreal])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Radius_Scattered_Linear_Qs(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_,  eta,  Radp, Qs):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GM = tf.reshape(TFSymRSet_Linear_Qs(R, Zs, eles_, SFPsR_, eta, Rr_cut, Radp, Qs),[nmol, natom,-1])
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*natom), [nmol, natom, 1]), dtype=tf.int64)
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def TFSymSet_Radius_Scattered_Linear_Qs_Periodic(R, Zs, eles_, SFPsR_, Rr_cut,  eleps_,  eta,  Radp, Qs, mil_j, nreal):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom X 1 tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		SFPsR_: A symmetry function parameter of radius part
		Rr_cut: Radial Cutoff of radius part
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		SFPsA_: A symmetry function parameter of angular part
		RA_cut: Radial Cutoff of angular part
	Returns:
		Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
	"""
	inp_shp = tf.shape(R)
	nmol = inp_shp[0]
	natom = inp_shp[1]
	nele = tf.shape(eles_)[0]
	nelep = tf.shape(eleps_)[0]
	GM = tf.reshape(TFSymRSet_Linear_Qs_Periodic(R, Zs, eles_, SFPsR_, eta, Rr_cut, Radp, Qs, mil_j, nreal),[nmol, nreal,-1])
	#GM = tf.identity(GMA)
	num_ele, num_dim = eles_.get_shape().as_list()
	Zs_real = Zs[:,:nreal]
	MaskAll = tf.equal(tf.reshape(Zs_real,[nmol,nreal,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask1 = AllSinglesSet(tf.cast(tf.tile(tf.reshape(tf.range(nreal),[1,nreal]),[nmol,1]),dtype=tf.int64), prec=tf.int64)
	v = tf.cast(tf.reshape(tf.range(nmol*nreal), [nmol, nreal, 1]), dtype=tf.int64)
	ToMask = tf.concat([ToMask1, v], axis = -1)
	IndexList = []
	SymList= []
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,nreal,1]),[nmol, nreal])))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		SymList.append(tf.gather_nd(GM, tf.slice(GatherList[-1],[0,0],[NAtomOfEle,2])))
		mol_index = tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle, 1])
		atom_index = tf.reshape(tf.slice(GatherList[-1],[0,2],[NAtomOfEle,1]),[NAtomOfEle, 1])
		IndexList.append(tf.concat([mol_index, atom_index], axis = -1))
	return SymList, IndexList

def NNInterface(R, Zs, eles_, GM):
	"""
	A tensorflow implementation of the AN1 symmetry function for a set of molecule.
	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Zs : nmol X maxnatom  tensor of atomic numbers.
		eles_: a neles X 1 tensor of elements present in the data.
		eleps_: a nelepairs X 2 X 12tensor of elements pairs present in the data.
		GM: Unscattered ANI1 sym Func: nmol X natom X nele X Dim


	Returns:
		List of ANI SymFunc of each atom by element type.
		List of Mol index of each atom by element type.
	"""
	nele = tf.shape(eles_)[0]
	num_ele, num_dim = eles_.get_shape().as_list()
	R_shp = tf.shape(R)
	nmol = R_shp[0]
	natom = R_shp[1]
	MaskAll = tf.equal(tf.reshape(Zs,[nmol,natom,1]),tf.reshape(eles_,[1,1,nele]))
	ToMask = AllSinglesSet(tf.tile(tf.reshape(tf.range(natom),[1,natom]),[nmol,1]))
	IndexList = []
	SymList=[]
	GatherList = []
	for e in range(num_ele):
		GatherList.append(tf.boolean_mask(ToMask,tf.reshape(tf.slice(MaskAll,[0,0,e],[nmol,natom,1]),[nmol, natom])))
		SymList.append(tf.gather_nd(GM, GatherList[-1]))
		NAtomOfEle=tf.shape(GatherList[-1])[0]
		IndexList.append(tf.reshape(tf.slice(GatherList[-1],[0,0],[NAtomOfEle,1]),[NAtomOfEle]))
	return SymList, IndexList

def tf_pairs_list(xyzs, Zs, r_cutoff, element_pairs):
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	distance_tensor = tf.norm(delta_xyzs,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(distance_tensor, r_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
	permutation_identity_mask = tf.where(tf.less(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, permutation_identity_mask)), tf.int32)
	pair_distances = tf.expand_dims(tf.gather_nd(distance_tensor, pair_indices), axis=1)
	pair_elements = tf.stack([tf.gather_nd(Zs, pair_indices[:,0:2]), tf.gather_nd(Zs, pair_indices[:,0:3:2])], axis=-1)
	element_pair_mask = tf.cast(tf.where(tf.logical_or(tf.reduce_all(tf.equal(tf.expand_dims(pair_elements, axis=1), tf.expand_dims(element_pairs, axis=0)), axis=2),
						tf.reduce_all(tf.equal(tf.expand_dims(pair_elements, axis=1), tf.expand_dims(element_pairs[:,::-1], axis=0)), axis=2))), tf.int32)
	num_element_pairs = element_pairs.get_shape().as_list()[0]
	element_pair_distances = tf.dynamic_partition(pair_distances, element_pair_mask[:,1], num_element_pairs)
	mol_indices = tf.dynamic_partition(pair_indices[:,0], element_pair_mask[:,1], num_element_pairs)
	return element_pair_distances, mol_indices

def tf_triples_list(xyzs, Zs, r_cutoff, element_triples):
	num_mols = Zs.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	distance_tensor = tf.norm(delta_xyzs,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(distance_tensor, r_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
	permutation_identity_mask = tf.where(tf.less(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, permutation_identity_mask)), tf.int32)
	mol_pair_indices = tf.dynamic_partition(pair_indices, pair_indices[:,0], num_mols)
	mol_triples_indices = []
	tmp = []
	for i in xrange(num_mols):
		mol_common_atom_indices = tf.where(tf.reduce_all(tf.equal(tf.expand_dims(mol_pair_indices[i][:,0:2], axis=0), tf.expand_dims(mol_pair_indices[i][:,0:2], axis=1)), axis=2))
		permutation_pairs_mask = tf.where(tf.less(mol_common_atom_indices[:,0], mol_common_atom_indices[:,1]))
		mol_common_atom_indices = tf.squeeze(tf.gather(mol_common_atom_indices, permutation_pairs_mask), axis=1)
		tmp.append(mol_common_atom_indices)
		mol_triples_indices.append(tf.concat([tf.gather(mol_pair_indices[i], mol_common_atom_indices[:,0]), tf.expand_dims(tf.gather(mol_pair_indices[i], mol_common_atom_indices[:,1])[:,2], axis=1)], axis=1))
	triples_indices = tf.concat(mol_triples_indices, axis=0)
	triples_distances = tf.stack([tf.gather_nd(distance_tensor, triples_indices[:,:3]),
						tf.gather_nd(distance_tensor, tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1)),
						tf.gather_nd(distance_tensor, tf.concat([triples_indices[:,0:1], triples_indices[:,2:]], axis=1))], axis=-1)
	cos_thetas = tf.stack([(tf.square(triples_distances[:,0]) + tf.square(triples_distances[:,1]) - tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,0] * triples_distances[:,1]),
						(tf.square(triples_distances[:,0]) - tf.square(triples_distances[:,1]) + tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,0] * triples_distances[:,2]),
						(-tf.square(triples_distances[:,0]) + tf.square(triples_distances[:,1]) + tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,1] * triples_distances[:,2])], axis=-1)
	cos_thetas = tf.where(tf.greater_equal(cos_thetas, 1.0), tf.ones_like(cos_thetas) * (1.0 - 1.0e-24), cos_thetas)
	cos_thetas = tf.where(tf.less_equal(cos_thetas, -1.0), -1.0 * tf.ones_like(cos_thetas) * (1.0 - 1.0e-24), cos_thetas)
	triples_angles = tf.acos(cos_thetas)
	triples_distances_angles = tf.concat([triples_distances, triples_angles], axis=1)
	triples_elements = tf.stack([tf.gather_nd(Zs, triples_indices[:,0:2]), tf.gather_nd(Zs, triples_indices[:,0:3:2]), tf.gather_nd(Zs, triples_indices[:,0:4:3])], axis=-1)
	sorted_triples_elements, _ = tf.nn.top_k(triples_elements, k=3)
	element_triples_mask = tf.cast(tf.where(tf.reduce_all(tf.equal(tf.expand_dims(sorted_triples_elements, axis=1), tf.expand_dims(element_triples, axis=0)), axis=2)), tf.int32)
	num_element_triples = element_triples.get_shape().as_list()[0]
	element_triples_distances_angles = tf.dynamic_partition(triples_distances_angles, element_triples_mask[:,1], num_element_triples)
	mol_indices = tf.dynamic_partition(triples_indices[:,0], element_triples_mask[:,1], num_element_triples)
	return element_triples_distances_angles, mol_indices


def matrix_power(matrix, power):
	"""
	Raise a Hermitian Matrix to a possibly fractional power.

	Args:
		matrix (tf.float): Diagonalizable matrix
		power (tf.float): power to raise the matrix to

	Returns:
		matrix_to_power (tf.float): matrix raised to the power

	Note:
		As of tensorflow v1.3, tf.svd() does not have gradients implimented
	"""
	s, U, V = tf.svd(matrix)
	s = tf.maximum(s, tf.pow(10.0, -14.0))
	return tf.matmul(U, tf.matmul(tf.diag(tf.pow(s, power)), tf.transpose(V)))

def matrix_power2(matrix, power):
	"""
	Raises a matrix to a possibly fractional power

	Args:
		matrix (tf.float): Diagonalizable matrix
		power (tf.float): power to raise the matrix to

	Returns:
		matrix_to_power (tf.float): matrix raised to the power
	"""
	matrix_eigenvals, matrix_eigenvecs = tf.self_adjoint_eig(matrix)
	matrix_to_power = tf.matmul(matrix_eigenvecs, tf.matmul(tf.matrix_diag(tf.pow(matrix_eigenvals, power)), tf.transpose(matrix_eigenvecs)))
	return matrix_to_power

def tf_gaussian_overlap(gaussian_params):
	r_nought = gaussian_params[:,0]
	sigma = gaussian_params[:,1]
	scaling_factor = tf.cast(tf.sqrt(np.pi / 2), eval(PARAMS["tf_prec"]))
	exponential_factor = tf.exp(-tf.square(tf.expand_dims(r_nought, axis=0) - tf.expand_dims(r_nought, axis=1))
	/ (2.0 * (tf.square(tf.expand_dims(sigma, axis=0)) + tf.square(tf.expand_dims(sigma, axis=1)))))
	root_inverse_sigma_sum = tf.sqrt((1.0 / tf.expand_dims(tf.square(sigma), axis=0)) + (1.0 / tf.expand_dims(tf.square(sigma), axis=1)))
	erf_numerator = (tf.expand_dims(r_nought, axis=0) * tf.expand_dims(tf.square(sigma), axis=1)
				+ tf.expand_dims(r_nought, axis=1) * tf.expand_dims(tf.square(sigma), axis=0))
	erf_denominator = (tf.sqrt(tf.cast(2.0, eval(PARAMS["tf_prec"]))) * tf.expand_dims(tf.square(sigma), axis=0) * tf.expand_dims(tf.square(sigma), axis=1)
				* root_inverse_sigma_sum)
	erf_factor = 1 + tf.erf(erf_numerator / erf_denominator)
	overlap_matrix = scaling_factor * exponential_factor * erf_factor / root_inverse_sigma_sum
	return overlap_matrix

def tf_gaussians(distance_tensor, Zs, gaussian_params):
	exponent = (tf.square(tf.expand_dims(distance_tensor, axis=-1) - tf.expand_dims(tf.expand_dims(gaussian_params[:,0], axis=0), axis=1))) \
				/ (-2.0 * (gaussian_params[:,1] ** 2))
	gaussian_embed = tf.where(tf.greater(exponent, -25.0), tf.exp(exponent), tf.zeros_like(exponent))
	gaussian_embed *= tf.expand_dims(tf.where(tf.not_equal(distance_tensor, 0), tf.ones_like(distance_tensor),
						tf.zeros_like(distance_tensor)), axis=-1)
	return gaussian_embed

def tf_gaussians_cutoff(distance_tensor, Zs, gaussian_params):
	exponent = (tf.square(tf.expand_dims(distance_tensor, axis=-1) - tf.expand_dims(tf.expand_dims(gaussian_params[:,0], axis=0), axis=1))) \
				/ (-2.0 * (gaussian_params[:,1] ** 2))
	gaussian_embed = tf.where(tf.greater(exponent, -25.0), tf.exp(exponent), tf.zeros_like(exponent))
	gaussian_embed *= tf.expand_dims(tf.where(tf.not_equal(distance_tensor, 0), tf.ones_like(distance_tensor),
						tf.zeros_like(distance_tensor)), axis=-1)
	xi = (distance_tensor - 5.5) / (6.5 - 5.5)
	cutoff_factor = 1 - 3 * tf.square(xi) + 2 * tf.pow(xi, 3.0)
	cutoff_factor = tf.where(tf.greater(distance_tensor, 6.5), tf.zeros_like(cutoff_factor), cutoff_factor)
	cutoff_factor = tf.where(tf.less(distance_tensor, 5.5), tf.ones_like(cutoff_factor), cutoff_factor)
	return gaussian_embed * tf.expand_dims(cutoff_factor, axis=-1)

def tf_spherical_harmonics_0(inverse_distance_tensor):
	return tf.fill(tf.shape(inverse_distance_tensor), tf.constant(0.28209479177387814, dtype=eval(PARAMS["tf_prec"])))

def tf_spherical_harmonics_1(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_0(tf.expand_dims(inverse_distance_tensor, axis=-1))
	l1_harmonics = 0.4886025119029199 * tf.stack([delta_xyzs[...,1], delta_xyzs[...,2], delta_xyzs[...,0]],
										axis=-1) * tf.expand_dims(inverse_distance_tensor, axis=-1)
	return tf.concat([lower_order_harmonics, l1_harmonics], axis=-1)

def tf_spherical_harmonics_2(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_1(delta_xyzs, inverse_distance_tensor)
	l2_harmonics = tf.stack([(-1.0925484305920792 * delta_xyzs[...,0] * delta_xyzs[...,1]),
			(1.0925484305920792 * delta_xyzs[...,1] * delta_xyzs[...,2]),
			(-0.31539156525252005 * (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2]))),
			(1.0925484305920792 * delta_xyzs[...,0] * delta_xyzs[...,2]),
			(0.5462742152960396 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])))], axis=-1) \
			* tf.expand_dims(tf.square(inverse_distance_tensor),axis=-1)
	return tf.concat([lower_order_harmonics, l2_harmonics], axis=-1)

def tf_spherical_harmonics_3(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_2(delta_xyzs, inverse_distance_tensor)
	l3_harmonics = tf.stack([(-0.5900435899266435 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]))),
			(-2.890611442640554 * delta_xyzs[...,0] * delta_xyzs[...,1] * delta_xyzs[...,2]),
			(-0.4570457994644658 * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 4. \
				* tf.square(delta_xyzs[...,2]))),
			(0.3731763325901154 * delta_xyzs[...,2] * (-3. * tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1]) \
				+ 2. * tf.square(delta_xyzs[...,2]))),
			(-0.4570457994644658 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 4. \
				* tf.square(delta_xyzs[...,2]))),
			(1.445305721320277 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2]),
			(0.5900435899266435 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])))], axis=-1) \
				* tf.expand_dims(tf.pow(inverse_distance_tensor,3),axis=-1)
	return tf.concat([lower_order_harmonics, l3_harmonics], axis=-1)

def tf_spherical_harmonics_4(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_3(delta_xyzs, inverse_distance_tensor)
	l4_harmonics = tf.stack([(2.5033429417967046 * delta_xyzs[...,0] * delta_xyzs[...,1] * (-1. * tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]))),
			(-1.7701307697799304 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2]),
			(0.9461746957575601 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2]))),
			(-0.6690465435572892 * delta_xyzs[...,1] * delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) + 3. \
				* tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2]))),
			(0.10578554691520431 * (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 24. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) + 6. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2])))),
			(-0.6690465435572892 * delta_xyzs[...,0] * delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) + 3.
				* tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2]))),
			(-0.47308734787878004 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2]))),
			(1.7701307697799304 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2]),
			(0.6258357354491761 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) \
				+ tf.pow(delta_xyzs[...,1], 4)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,4),axis=-1)
	return tf.concat([lower_order_harmonics, l4_harmonics], axis=-1)

def tf_spherical_harmonics_5(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_4(delta_xyzs, inverse_distance_tensor)
	l5_harmonics = tf.stack([(0.6563820568401701 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4))),
			(8.302649259524166 * delta_xyzs[...,0] * delta_xyzs[...,1] * (-1. * tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2]),
			(0.4892382994352504 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2]))),
			(4.793536784973324 * delta_xyzs[...,0] * delta_xyzs[...,1] * delta_xyzs[...,2] \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2]))),
			(0.45294665119569694 * delta_xyzs[...,1] * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) - 12. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) + 2. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2])))),
			(0.1169503224534236 * delta_xyzs[...,2] * (15. * tf.pow(delta_xyzs[...,0], 4) + 15. * tf.pow(delta_xyzs[...,1], 4) \
				- 40. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) + 10. \
				* tf.square(delta_xyzs[...,0]) * (3. * tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2])))),
			(0.45294665119569694 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) - 12. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) + 2. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2])))),
			(-2.396768392486662 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2] \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2]))),
			(-0.4892382994352504 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2]))),
			(2.0756623148810416 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2]),
			(0.6563820568401701 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + 5. * tf.pow(delta_xyzs[...,1], 4)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,5),axis=-1)
	return tf.concat([lower_order_harmonics, l5_harmonics], axis=-1)

def tf_spherical_harmonics_6(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_5(delta_xyzs, inverse_distance_tensor)
	l6_harmonics = tf.stack([(-1.3663682103838286 * delta_xyzs[...,0] * delta_xyzs[...,1] * (3. * tf.pow(delta_xyzs[...,0], 4) \
				- 10. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + 3. * tf.pow(delta_xyzs[...,1], 4))),
			(2.366619162231752 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2]),
			(2.0182596029148967 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2]))),
			(0.9212052595149236 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) \
				* delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) + 3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2]))),
			(-0.9212052595149236 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) \
				- 16. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. * tf.pow(delta_xyzs[...,2], 4) \
				+ 2. * tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])))),
			(0.5826213625187314 * delta_xyzs[...,1] * delta_xyzs[...,2] * (5. * tf.pow(delta_xyzs[...,0], 4) + 5. * tf.pow(delta_xyzs[...,1], 4) \
				- 20. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) \
				+ 10. * tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2])))),
			(-0.06356920226762842 * (5. * tf.pow(delta_xyzs[...,0], 6) + 5. * tf.pow(delta_xyzs[...,1], 6) - 90. \
				* tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 120. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) - 16. * tf.pow(delta_xyzs[...,2], 6) + 15. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2])) + 15. * tf.square(delta_xyzs[...,0]) \
				* (tf.pow(delta_xyzs[...,1], 4) - 12. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4)))),
			(0.5826213625187314 * delta_xyzs[...,0] * delta_xyzs[...,2] * (5. * tf.pow(delta_xyzs[...,0], 4) + 5. \
				* tf.pow(delta_xyzs[...,1], 4) - 20. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. \
				* tf.pow(delta_xyzs[...,2], 4) + 10. * tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 2. \
				* tf.square(delta_xyzs[...,2])))),
			(0.4606026297574618 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * (tf.pow(delta_xyzs[...,0], 4) \
				+ tf.pow(delta_xyzs[...,1], 4) - 16. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. \
				* tf.pow(delta_xyzs[...,2], 4) + 2. * tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 8. \
				* tf.square(delta_xyzs[...,2])))),
			(-0.9212052595149236 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2] \
				* (3. * tf.square(delta_xyzs[...,0]) + 3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2]))),
			(-0.5045649007287242 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) \
				+ tf.pow(delta_xyzs[...,1], 4)) * (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2]))),
			(2.366619162231752 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + 5. * tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2]),
			(0.6831841051919143 * (tf.pow(delta_xyzs[...,0], 6) - 15. * tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) \
				+ 15. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 1. * tf.pow(delta_xyzs[...,1], 6)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,6),axis=-1)
	return tf.concat([lower_order_harmonics, l6_harmonics], axis=-1)

def tf_spherical_harmonics_7(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_6(delta_xyzs, inverse_distance_tensor)
	l7_harmonics = tf.stack([(-0.7071627325245962 * delta_xyzs[...,1] * (-7. * tf.pow(delta_xyzs[...,0], 6) + 35. \
				* tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) - 21. * tf.square(delta_xyzs[...,0]) \
				* tf.pow(delta_xyzs[...,1], 4) + tf.pow(delta_xyzs[...,1], 6))),
			(-5.291921323603801 * delta_xyzs[...,0] * delta_xyzs[...,1] * (3. * tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + 3. * tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2]),
			(-0.5189155787202604 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4)) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 12. * tf.square(delta_xyzs[...,2]))),
			(4.151324629762083 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) - 1. \
				* tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) + 3. \
				* tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2]))),
			(-0.15645893386229404 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) \
				* (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 60. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 80. * tf.pow(delta_xyzs[...,2], 4) + 6. * tf.square(delta_xyzs[...,0]) \
				* (tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2])))),
			(-0.4425326924449826 * delta_xyzs[...,0] * delta_xyzs[...,1] * delta_xyzs[...,2] * (15. * tf.pow(delta_xyzs[...,0], 4) \
				+ 15. * tf.pow(delta_xyzs[...,1], 4) - 80. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 48. \
				* tf.pow(delta_xyzs[...,2], 4) + 10. * tf.square(delta_xyzs[...,0]) * (3. * tf.square(delta_xyzs[...,1]) - 8. \
				* tf.square(delta_xyzs[...,2])))),
			(-0.0903316075825173 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 6) + 5. * tf.pow(delta_xyzs[...,1], 6) - 120. \
				* tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 240. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) - 64. * tf.pow(delta_xyzs[...,2], 6) + 15. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])) + 15. * tf.square(delta_xyzs[...,0]) \
				* (tf.pow(delta_xyzs[...,1], 4) - 16. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. \
				* tf.pow(delta_xyzs[...,2], 4)))),
			(0.06828427691200495 * delta_xyzs[...,2] * (-35. * tf.pow(delta_xyzs[...,0], 6) - 35. * tf.pow(delta_xyzs[...,1], 6) \
				+ 210. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) - 168. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) + 16. * tf.pow(delta_xyzs[...,2], 6) - 105. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2])) - 21. * tf.square(delta_xyzs[...,0]) \
				* (5. * tf.pow(delta_xyzs[...,1], 4) - 20. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) \
				+ 8. * tf.pow(delta_xyzs[...,2], 4)))),
			(-0.0903316075825173 * delta_xyzs[...,0] * (5. * tf.pow(delta_xyzs[...,0], 6) + 5. * tf.pow(delta_xyzs[...,1], 6) \
				- 120. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 240. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) - 64. * tf.pow(delta_xyzs[...,2], 6) + 15. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])) + 15. * tf.square(delta_xyzs[...,0]) \
				* (tf.pow(delta_xyzs[...,1], 4) - 16. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. \
				* tf.pow(delta_xyzs[...,2], 4)))),
			(0.2212663462224913 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2] \
				* (15. * tf.pow(delta_xyzs[...,0], 4) + 15. * tf.pow(delta_xyzs[...,1], 4) - 80. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 48. * tf.pow(delta_xyzs[...,2], 4) + 10. * tf.square(delta_xyzs[...,0]) \
				* (3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])))),
			(0.15645893386229404 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) \
				* (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 60. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 80. * tf.pow(delta_xyzs[...,2], 4) + 6. * tf.square(delta_xyzs[...,0]) \
				* (tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2])))),
			(-1.0378311574405208 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) \
				+ tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) \
				+ 3. * tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2]))),
			(-0.5189155787202604 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + 5. * tf.pow(delta_xyzs[...,1], 4)) * (tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]) - 12. * tf.square(delta_xyzs[...,2]))),
			(2.6459606618019005 * (tf.pow(delta_xyzs[...,0], 6) - 15. * tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) \
				+ 15. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 1. * tf.pow(delta_xyzs[...,1], 6)) * delta_xyzs[...,2]),
			(0.7071627325245962 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 6) - 21. * tf.pow(delta_xyzs[...,0], 4) \
				* tf.square(delta_xyzs[...,1]) + 35. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 7. \
				* tf.pow(delta_xyzs[...,1], 6)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,7),axis=-1)
	return tf.concat([lower_order_harmonics, l7_harmonics], axis=-1)

def tf_spherical_harmonics_8(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_7(delta_xyzs, inverse_distance_tensor)
	l8_harmonics = tf.stack([(-5.831413281398639 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.pow(delta_xyzs[...,0], 6) \
				- 7. * tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) + 7. * tf.square(delta_xyzs[...,0]) \
				* tf.pow(delta_xyzs[...,1], 4) - 1. * tf.pow(delta_xyzs[...,1], 6))),
			(-2.9157066406993195 * delta_xyzs[...,1] * (-7. * tf.pow(delta_xyzs[...,0], 6) + 35. * tf.pow(delta_xyzs[...,0], 4) \
				* tf.square(delta_xyzs[...,1]) - 21. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) \
				+ tf.pow(delta_xyzs[...,1], 6)) * delta_xyzs[...,2]),
			(1.0646655321190852 * delta_xyzs[...,0] * delta_xyzs[...,1] * (3. * tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + 3. * tf.pow(delta_xyzs[...,1], 4)) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 14. * tf.square(delta_xyzs[...,2]))),
			(-3.449910622098108 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2] * (tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2]))),
			(-1.9136660990373227 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) - 1. \
				* tf.square(delta_xyzs[...,1])) * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) - 24. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 40. * tf.pow(delta_xyzs[...,2], 4) + 2. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 12. * tf.square(delta_xyzs[...,2])))),
			(-1.2352661552955442 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) \
				* delta_xyzs[...,2] * (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 20. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. * tf.pow(delta_xyzs[...,2], 4) \
				+ tf.square(delta_xyzs[...,0]) * (6. * tf.square(delta_xyzs[...,1]) - 20. * tf.square(delta_xyzs[...,2])))),
			(0.912304516869819 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.pow(delta_xyzs[...,0], 6) + tf.pow(delta_xyzs[...,1], 6) \
				- 30. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 80. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) - 32. * tf.pow(delta_xyzs[...,2], 6) + 3. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2])) + tf.square(delta_xyzs[...,0]) \
				* (3. * tf.pow(delta_xyzs[...,1], 4) - 60. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 80. \
				* tf.pow(delta_xyzs[...,2], 4)))),
			(-0.10904124589877995 * delta_xyzs[...,1] * delta_xyzs[...,2] * (35. * tf.pow(delta_xyzs[...,0], 6) + 35. \
				* tf.pow(delta_xyzs[...,1], 6) - 280. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 336. \
				* tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 4) - 64. * tf.pow(delta_xyzs[...,2], 6) + 35. \
				* tf.pow(delta_xyzs[...,0], 4) * (3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])) + 7. \
				* tf.square(delta_xyzs[...,0]) * (15. * tf.pow(delta_xyzs[...,1], 4) - 80. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 48. * tf.pow(delta_xyzs[...,2], 4)))),
			(0.009086770491564996 * (35. * tf.pow(delta_xyzs[...,0], 8) + 35. * tf.pow(delta_xyzs[...,1], 8) - 1120. \
				* tf.pow(delta_xyzs[...,1], 6) * tf.square(delta_xyzs[...,2]) + 3360. * tf.pow(delta_xyzs[...,1], 4) \
				* tf.pow(delta_xyzs[...,2], 4) - 1792. * tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 6) + 128. \
				* tf.pow(delta_xyzs[...,2], 8) + 140. * tf.pow(delta_xyzs[...,0], 6) * (tf.square(delta_xyzs[...,1]) - 8. \
				* tf.square(delta_xyzs[...,2])) + 210. * tf.pow(delta_xyzs[...,0], 4) * (tf.pow(delta_xyzs[...,1], 4) - 16. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. * tf.pow(delta_xyzs[...,2], 4)) + 28. \
				* tf.square(delta_xyzs[...,0]) * (5. * tf.pow(delta_xyzs[...,1], 6) - 120. * tf.pow(delta_xyzs[...,1], 4) \
				* tf.square(delta_xyzs[...,2]) + 240. * tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 4) - 64. \
				* tf.pow(delta_xyzs[...,2], 6)))),
			(-0.10904124589877995 * delta_xyzs[...,0] * delta_xyzs[...,2] * (35. * tf.pow(delta_xyzs[...,0], 6) + 35. \
				* tf.pow(delta_xyzs[...,1], 6) - 280. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 336. \
				* tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 4) - 64. * tf.pow(delta_xyzs[...,2], 6) + 35. \
				* tf.pow(delta_xyzs[...,0], 4) * (3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])) + 7. \
				* tf.square(delta_xyzs[...,0]) * (15. * tf.pow(delta_xyzs[...,1], 4) - 80. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 48. * tf.pow(delta_xyzs[...,2], 4)))),
			(-0.4561522584349095 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * (tf.pow(delta_xyzs[...,0], 6) \
				+ tf.pow(delta_xyzs[...,1], 6) - 30. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 80. \
				* tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 4) - 32. * tf.pow(delta_xyzs[...,2], 6) + 3. \
				* tf.pow(delta_xyzs[...,0], 4) * (tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2])) \
				+ tf.square(delta_xyzs[...,0]) * (3. * tf.pow(delta_xyzs[...,1], 4) - 60. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 80. * tf.pow(delta_xyzs[...,2], 4)))),
			(1.2352661552955442 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) \
				* delta_xyzs[...,2] * (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 20. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. * tf.pow(delta_xyzs[...,2], 4) \
				+ tf.square(delta_xyzs[...,0]) * (6. * tf.square(delta_xyzs[...,1]) - 20. * tf.square(delta_xyzs[...,2])))),
			(0.47841652475933066 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) \
				+ tf.pow(delta_xyzs[...,1], 4)) * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) - 24. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 40. * tf.pow(delta_xyzs[...,2], 4) + 2. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 12. * tf.square(delta_xyzs[...,2])))),
			(-3.449910622098108 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + 5. * tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2] * (tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2]))),
			(-0.5323327660595426 * (tf.pow(delta_xyzs[...,0], 6) - 15. * tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) \
				+ 15. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 1. * tf.pow(delta_xyzs[...,1], 6)) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 14. * tf.square(delta_xyzs[...,2]))),
			(2.9157066406993195 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 6) - 21. * tf.pow(delta_xyzs[...,0], 4) \
				* tf.square(delta_xyzs[...,1]) + 35. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 7. \
				* tf.pow(delta_xyzs[...,1], 6)) * delta_xyzs[...,2]),
			(0.7289266601748299 * (tf.pow(delta_xyzs[...,0], 8) - 28. * tf.pow(delta_xyzs[...,0], 6) * tf.square(delta_xyzs[...,1]) \
				+ 70. * tf.pow(delta_xyzs[...,0], 4) * tf.pow(delta_xyzs[...,1], 4) - 28. * tf.square(delta_xyzs[...,0]) \
				* tf.pow(delta_xyzs[...,1], 6) + tf.pow(delta_xyzs[...,1], 8)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,8),axis=-1)
	return tf.concat([lower_order_harmonics, l8_harmonics], axis=-1)

def tf_spherical_harmonics(delta_xyzs, distance_tensor, max_l):
	inverse_distance_tensor = tf.where(tf.greater(distance_tensor, 1.e-9), tf.reciprocal(distance_tensor), tf.zeros_like(distance_tensor))
	if max_l == 8:
		harmonics = tf_spherical_harmonics_8(delta_xyzs, inverse_distance_tensor)
	elif max_l == 7:
		harmonics = tf_spherical_harmonics_7(delta_xyzs, inverse_distance_tensor)
	elif max_l == 6:
		harmonics = tf_spherical_harmonics_6(delta_xyzs, inverse_distance_tensor)
	elif max_l == 5:
		harmonics = tf_spherical_harmonics_5(delta_xyzs, inverse_distance_tensor)
	elif max_l == 4:
		harmonics = tf_spherical_harmonics_4(delta_xyzs, inverse_distance_tensor)
	elif max_l == 3:
		harmonics = tf_spherical_harmonics_3(delta_xyzs, inverse_distance_tensor)
	elif max_l == 2:
		harmonics = tf_spherical_harmonics_2(delta_xyzs, inverse_distance_tensor)
	elif max_l == 1:
		harmonics = tf_spherical_harmonics_1(delta_xyzs, inverse_distance_tensor)
	elif max_l == 0:
		harmonics = tf_spherical_harmonics_0(inverse_distance_tensor)
	else:
		raise Exception("Spherical Harmonics only implemented up to l=8. Choose a lower order")
	return harmonics

def tf_gaussian_spherical_harmonics_element(xyzs, Zs, element, gaussian_params, atomic_embed_factors, l_max, labels=None):
	"""
	Encodes atoms into a gaussians and spherical harmonics embedding

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		element (int): element to return embedding/labels for
		gaussian_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		atomic_embed_factors (tf.float): MaxElementNumber tensor of scaling factors for elements
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use (needs implemented)
		labels (tf.Tensor): NMol x MaxNAtoms x label shape tensor of learning targets

	Returns:
		embedding (tf.float): atom embeddings for element
		labels (tf.float): atom labels for element
	"""
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	element_indices = tf.cast(tf.where(tf.equal(Zs, element)), tf.int32)
	num_batch_elements = tf.shape(element_indices)[0]
	element_delta_xyzs = tf.gather_nd(delta_xyzs, element_indices)
	element_Zs = tf.gather(Zs, element_indices[:,0])
	distance_tensor = tf.norm(element_delta_xyzs+1.e-16,axis=2)
	atom_scaled_gaussians, min_eigenval = tf_gaussians(tf.expand_dims(distance_tensor, axis=-1), element_Zs, gaussian_params, atomic_embed_factors)
	spherical_harmonics = tf_spherical_harmonics(element_delta_xyzs, distance_tensor, l_max)
	element_embedding = tf.reshape(tf.einsum('jkg,jkl->jgl', atom_scaled_gaussians, spherical_harmonics),
							[num_batch_elements, tf.shape(gaussian_params)[0] * (l_max + 1) ** 2])
	if labels != None:
		element_labels = tf.gather_nd(labels, element_indices)
		return element_embedding, element_labels, element_indices, min_eigenval
	else:
		return element_embedding, element_indices, min_eigenval

def tf_gaussian_spherical_harmonics(xyzs, Zs, elements, gaussian_params, atomic_embed_factors, l_max):
	"""
	Encodes atoms into a gaussians and spherical harmonics embedding

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		element (int): element to return embedding/labels for
		gaussian_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		atomic_embed_factors (tf.float): MaxElementNumber tensor of scaling factors for elements
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use (needs implemented)
		labels (tf.Tensor): NMol x MaxNAtoms x label shape tensor of learning targets

	Returns:
		embedding (tf.float): atom embeddings for element
		labels (tf.float): atom labels for element
	"""
	num_elements = elements.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	element_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(Zs, axis=-1), tf.reshape(elements, [1, 1, tf.shape(elements)[0]]))), tf.int32)
	distance_tensor = tf.norm(delta_xyzs+1.e-16,axis=3)
	atom_scaled_gaussians = tf_gaussians(distance_tensor, Zs, gaussian_params, atomic_embed_factors, orthogonalize)
	spherical_harmonics = tf_spherical_harmonics(delta_xyzs, distance_tensor, l_max)
	embeddings = tf.reshape(tf.einsum('ijkg,ijkl->ijgl', atom_scaled_gaussians, spherical_harmonics),
							[tf.shape(Zs)[0], tf.shape(Zs)[1], tf.shape(gaussian_params)[0] * (l_max + 1) ** 2])
	embeddings = tf.gather_nd(embeddings, element_indices[:,0:2])
	element_embeddings = tf.dynamic_partition(embeddings, element_indices[:,2], num_elements)
	molecule_indices = tf.dynamic_partition(element_indices[:,0:2], element_indices[:,2], num_elements)
	return element_embeddings, molecule_indices

#
# John, we still need a sparse version of this.
#
def tf_gaussian_spherical_harmonics_channel(xyzs, Zs, elements, gaussian_params, l_max):
	"""
	Encodes atoms into a gaussians * spherical harmonics embedding
	Works on a batch of molecules. This is the embedding routine used
	in BehlerParinelloDirectGauSH.

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		element (int): element to return embedding/labels for
		gaussian_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use (needs implemented)
		labels (tf.Tensor): NMol x MaxNAtoms x label shape tensor of learning targets

	Returns:
		embedding (tf.float): atom embeddings for element
		molecule_indices (tf.float): mapping between atoms and molecules.
	"""
	num_elements = elements.get_shape().as_list()[0]
	num_molecules = Zs.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	mol_atom_indices = tf.where(tf.not_equal(Zs, 0))
	delta_xyzs = tf.gather_nd(delta_xyzs, mol_atom_indices)
	distance_tensor = tf.norm(delta_xyzs+1.e-16,axis=2)
	gaussians = tf_gaussians_cutoff(distance_tensor, Zs, gaussian_params)
	spherical_harmonics = tf_spherical_harmonics(delta_xyzs, distance_tensor, l_max)
	channel_scatter_bool = tf.gather(tf.equal(tf.expand_dims(Zs, axis=1), tf.reshape(elements, [1, num_elements, 1])), mol_atom_indices[:,0])

	channel_scatter = tf.where(channel_scatter_bool, tf.ones_like(channel_scatter_bool, dtype=eval(PARAMS["tf_prec"])),tf.zeros_like(channel_scatter_bool, dtype=eval(PARAMS["tf_prec"])))

	element_channel_gaussians = tf.expand_dims(gaussians, axis=1) * tf.expand_dims(channel_scatter, axis=-1)
	element_channel_harmonics = tf.expand_dims(spherical_harmonics, axis=1) * tf.expand_dims(channel_scatter, axis=-1)

	embeddings = tf.reshape(tf.einsum('ijkg,ijkl->ijgl', element_channel_gaussians, element_channel_harmonics),
	[tf.shape(mol_atom_indices)[0], -1])

	partition_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(tf.gather_nd(Zs, mol_atom_indices), axis=-1), tf.expand_dims(elements, axis=0)))[:,1], tf.int32)

	element_embeddings = tf.dynamic_partition(embeddings, partition_indices, num_elements)
	molecule_indices = tf.dynamic_partition(mol_atom_indices, partition_indices, num_elements)
	return element_embeddings, molecule_indices

def tf_random_rotate(xyzs, rotation_params, labels = None, return_matrix = False):
	"""
	Rotates molecules and optionally labels in a uniformly random fashion

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		labels (tf.float, optional): NMol x MaxNAtoms x label shape tensor of learning targets
		return_matrix (bool): Returns rotation tensor if True

	Returns:
		new_xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor of randomly rotated molecules
		new_labels (tf.float): NMol x MaxNAtoms x label shape tensor of randomly rotated learning targets
	"""
	r = tf.sqrt(rotation_params[...,2])
	v = tf.stack([tf.sin(rotation_params[...,1]) * r, tf.cos(rotation_params[...,1]) * r, tf.sqrt(2.0 - rotation_params[...,2])], axis=-1)
	zero_tensor = tf.zeros_like(rotation_params[...,1])

	R1 = tf.stack([tf.cos(rotation_params[...,0]), tf.sin(rotation_params[...,0]), zero_tensor], axis=-1)
	R2 = tf.stack([-tf.sin(rotation_params[...,0]), tf.cos(rotation_params[...,0]), zero_tensor], axis=-1)
	R3 = tf.stack([zero_tensor, zero_tensor, tf.ones_like(rotation_params[...,1])], axis=-1)
	R = tf.stack([R1, R2, R3], axis=-2)
	M = tf.matmul((tf.expand_dims(v, axis=-2) * tf.expand_dims(v, axis=-1)) - tf.eye(3, dtype=eval(PARAMS["tf_prec"])), R)
	new_xyzs = tf.einsum("lij,lkj->lki", M, xyzs)
	if labels != None:
		new_labels = tf.einsum("lij,lkj->lki",M, (xyzs + labels)) - new_xyzs
		if return_matrix:
			return new_xyzs, new_labels, M
		else:
			return new_xyzs, new_labels
	elif return_matrix:
		return new_xyzs, M
	else:
		return new_xyzs

def tf_symmetry_functions(xyzs, Zs, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta):
	"""
	Encodes atoms into the symmetry function embedding as implemented in the ANI-1 Neural Network (doi: 10.1039/C6SC05720A)

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		num_atoms (tf.int32): NMol number of atoms numpy array
		elements (tf.int32): NElements tensor containing sorted unique atomic numbers present
		element_pairs (tf.int32): NElementPairs x 2 tensor containing sorted unique pairs of atomic numbers present
		radial_cutoff (tf.float): scalar tensor with the cutoff for radial pairs
		angular_cutoff (tf.float): scalar tensor with the cutoff for the angular triples
		radial_rs (tf.float): NRadialGridPoints tensor with R_s values for the radial grid
		angular_rs (tf.float): NAngularGridPoints tensor with the R_s values for the radial part of the angular grid
		theta_s (tf.float): NAngularGridPoints tensor with the theta_s values for the angular grid
		zeta (tf.float): scalar tensor with the zeta parameter for the symmetry functions
		eta (tf.float): scalar tensor with the eta parameter for the symmetry functions

	Returns:
		element_embeddings (list of tf.floats): List of NAtoms x (NRadial_rs x NElements + NAngular_rs x NTheta_s x NElementPairs)
				tensors of the same element type
		mol_indices (list of tf.int32s): List of NAtoms of the same element types with the molecule index of each atom
	"""
	num_molecules = Zs.get_shape().as_list()[0]
	num_elements = elements.get_shape().as_list()[0]
	num_element_pairs = element_pairs.get_shape().as_list()[0]

	radial_embedding, pair_indices, pair_elements = tf_symmetry_functions_radial_grid(xyzs, Zs, radial_cutoff, radial_rs, eta)
	angular_embedding, triples_indices, triples_element, triples_element_pairs = tf_symmetry_function_angular_grid(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta)

	pair_element_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(pair_elements[:,1], axis=-1),
							tf.expand_dims(elements, axis=0))), tf.int32)[:,1]
	triples_elements_indices = tf.cast(tf.where(tf.reduce_all(tf.equal(tf.expand_dims(triples_element_pairs, axis=-2),
									element_pairs), axis=-1)), tf.int32)[:,1]
	radial_scatter_indices = tf.concat([pair_indices, tf.expand_dims(pair_element_indices, axis=1)], axis=1)
	angular_scatter_indices = tf.concat([triples_indices, tf.expand_dims(triples_elements_indices, axis=1)], axis=1)

	radial_molecule_embeddings = tf.dynamic_partition(radial_embedding, pair_indices[:,0], num_molecules)
	radial_atom_indices = tf.dynamic_partition(radial_scatter_indices[:,1:], pair_indices[:,0], num_molecules)
	angular_molecule_embeddings = tf.dynamic_partition(angular_embedding, triples_indices[:,0], num_molecules)
	angular_atom_indices = tf.dynamic_partition(angular_scatter_indices[:,1:], triples_indices[:,0], num_molecules)

	embeddings = []
	mol_atom_indices = []
	for molecule in range(num_molecules):
		atom_indices = tf.cast(tf.where(tf.not_equal(Zs[molecule], 0)), tf.int32)
		molecule_atom_elements = tf.gather_nd(Zs[molecule], atom_indices)
		num_atoms = tf.shape(molecule_atom_elements)[0]
		radial_atom_embeddings = tf.reshape(tf.reduce_sum(tf.scatter_nd(radial_atom_indices[molecule], radial_molecule_embeddings[molecule],
								[num_atoms, num_atoms, num_elements, tf.shape(radial_rs)[0]]), axis=1), [num_atoms, -1])
		angular_atom_embeddings = tf.reshape(tf.reduce_sum(tf.scatter_nd(angular_atom_indices[molecule], angular_molecule_embeddings[molecule],
									[num_atoms, num_atoms, num_atoms, num_element_pairs, tf.shape(angular_rs)[0] * tf.shape(theta_s)[0]]),
									axis=[1,2]), [num_atoms, -1])
		embeddings.append(tf.concat([radial_atom_embeddings, angular_atom_embeddings], axis=1))
		mol_atom_indices.append(tf.concat([tf.fill([num_atoms, 1], molecule), atom_indices], axis=1))

	embeddings = tf.concat(embeddings, axis=0)
	mol_atom_indices = tf.concat(mol_atom_indices, axis=0)
	atom_Zs = tf.gather_nd(Zs, tf.where(tf.not_equal(Zs, 0)))
	atom_Z_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(atom_Zs, axis=1), tf.expand_dims(elements, axis=0)))[:,1], tf.int32)

	element_embeddings = tf.dynamic_partition(embeddings, atom_Z_indices, num_elements)
	mol_indices = tf.dynamic_partition(mol_atom_indices, atom_Z_indices, num_elements)
	return element_embeddings, mol_indices

def tf_symmetry_functions_radial_grid(xyzs, Zs, radial_cutoff, radial_rs, eta, prec=tf.float64):
	"""
	Encodes the radial grid portion of the symmetry functions. Should be called by tf_symmetry_functions_2()

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		num_atoms (np.int32): NMol number of atoms numpy array
		radial_cutoff (tf.float): scalar tensor with the cutoff for radial pairs
		radial_rs (tf.float): NRadialGridPoints tensor with R_s values for the radial grid
		eta (tf.float): scalar tensor with the eta parameter for the symmetry functions

	Returns:
		radial_embedding (tf.float): tensor of radial embeddings for all atom pairs within the radial_cutoff
		pair_indices (tf.int32): tensor of the molecule, atom, and pair atom indices
		pair_elements (tf.int32): tensor of the atomic numbers for the atom and its pair atom
	"""
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	distance_tensor = tf.norm(delta_xyzs + 1.e-16,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(distance_tensor, radial_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
	identity_mask = tf.where(tf.not_equal(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, identity_mask)), tf.int32)
	pair_distances = tf.gather_nd(distance_tensor, pair_indices)
	pair_elements = tf.stack([tf.gather_nd(Zs, pair_indices[:,0:2]), tf.gather_nd(Zs, pair_indices[:,0:3:2])], axis=-1)
	gaussian_factor = tf.exp(-eta * tf.square(tf.expand_dims(pair_distances, axis=-1) - tf.expand_dims(radial_rs, axis=0)))
	cutoff_factor = tf.expand_dims(0.5 * (tf.cos(3.14159265359 * pair_distances / radial_cutoff) + 1.0), axis=-1)
	radial_embedding = gaussian_factor * cutoff_factor
	return radial_embedding, pair_indices, pair_elements

def tf_symmetry_function_angular_grid(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta):
	"""
	Encodes the radial grid portion of the symmetry functions. Should be called by tf_symmetry_functions_2()

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		angular_cutoff (tf.float): scalar tensor with the cutoff for the angular triples
		angular_rs (tf.float): NAngularGridPoints tensor with the R_s values for the radial part of the angular grid
		theta_s (tf.float): NAngularGridPoints tensor with the theta_s values for the angular grid
		zeta (tf.float): scalar tensor with the zeta parameter for the symmetry functions
		eta (tf.float): scalar tensor with the eta parameter for the symmetry functions

	Returns:
		angular_embedding (tf.float): tensor of radial embeddings for all atom pairs within the radial_cutoff
		triples_indices (tf.int32): tensor of the molecule, atom, and triples atom indices
		triples_elements (tf.int32): tensor of the atomic numbers for the atom
		sorted_triples_element_pairs (tf.int32): sorted tensor of the atomic numbers of triples atoms
	"""
	num_mols = Zs.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	distance_tensor = tf.norm(delta_xyzs + 1.e-16,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.cast(tf.where(tf.logical_and(tf.logical_and(tf.less(distance_tensor, angular_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1))), tf.int32)
	identity_mask = tf.where(tf.not_equal(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, identity_mask)), tf.int32)
	mol_pair_indices = tf.dynamic_partition(pair_indices, pair_indices[:,0], num_mols)
	triples_indices = []
	tmp = []
	for i in xrange(num_mols):
		mol_common_pair_indices = tf.where(tf.equal(tf.expand_dims(mol_pair_indices[i][:,1], axis=1),
									tf.expand_dims(mol_pair_indices[i][:,1], axis=0)))
		mol_triples_indices = tf.concat([tf.gather(mol_pair_indices[i], mol_common_pair_indices[:,0]),
								tf.gather(mol_pair_indices[i], mol_common_pair_indices[:,1])[:,-1:]], axis=1)
		permutation_identity_pairs_mask = tf.where(tf.less(mol_triples_indices[:,2], mol_triples_indices[:,3]))
		mol_triples_indices = tf.squeeze(tf.gather(mol_triples_indices, permutation_identity_pairs_mask))
		triples_indices.append(mol_triples_indices)
	triples_indices = tf.concat(triples_indices, axis=0)

	triples_elements = tf.gather_nd(Zs, triples_indices[:,0:2])
	triples_element_pairs, _ = tf.nn.top_k(tf.stack([tf.gather_nd(Zs, triples_indices[:,0:3:2]),
							tf.gather_nd(Zs, triples_indices[:,0:4:3])], axis=-1), k=2)
	sorted_triples_element_pairs = tf.reverse(triples_element_pairs, axis=[-1])

	triples_distances = tf.stack([tf.gather_nd(distance_tensor, triples_indices[:,:3]), tf.gather_nd(distance_tensor,
						tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1))], axis=1)
	r_ijk_s = tf.square(tf.expand_dims(tf.reduce_sum(triples_distances, axis=1) / 2.0, axis=-1) - tf.expand_dims(angular_rs, axis=0))
	exponential_factor = tf.exp(-eta * r_ijk_s)

	xyz_ij_ik = tf.reduce_sum(tf.gather_nd(delta_xyzs, triples_indices[:,:3]) * tf.gather_nd(delta_xyzs,
						tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1)), axis=1)
	cos_theta = xyz_ij_ik / (triples_distances[:,0] * triples_distances[:,1])
	cos_theta = tf.where(tf.greater_equal(cos_theta, 1.0), tf.ones_like(cos_theta) - 1.0e-16, cos_theta)
	cos_theta = tf.where(tf.less_equal(cos_theta, -1.0), -1.0 * tf.ones_like(cos_theta) - 1.0e-16, cos_theta)
	triples_angle = tf.acos(cos_theta)
	theta_ijk_s = tf.expand_dims(triples_angle, axis=-1) - tf.expand_dims(theta_s, axis=0)
	cos_factor = tf.pow((1 + tf.cos(theta_ijk_s)), zeta)

	cutoff_factor = 0.5 * (tf.cos(3.14159265359 * triples_distances / angular_cutoff) + 1.0)
	scalar_factor = tf.pow(tf.cast(2.0, eval(PARAMS["tf_prec"])), 1.0-zeta)

	angular_embedding = tf.reshape(scalar_factor * tf.expand_dims(cos_factor * tf.expand_dims(cutoff_factor[:,0] * cutoff_factor[:,1], axis=-1), axis=-1) \
						* tf.expand_dims(exponential_factor, axis=-2), [tf.shape(triples_indices)[0], tf.shape(theta_s)[0] * tf.shape(angular_rs)[0]])
	return angular_embedding, triples_indices, triples_elements, sorted_triples_element_pairs

def tf_coulomb_dsf_elu(xyzs, charges, Radpair, elu_width, dsf_alpha, cutoff_dist):
	"""
	A tensorflow linear scaling implementation of the Damped Shifted Electrostatic Force with short range cutoff with elu function (const at short range).
	http://aip.scitation.org.proxy.library.nd.edu/doi/pdf/10.1063/1.2206581
	Batched over molecules.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_srcut: Short Range Erf Cutoff
		R_lrcut: Long Range DSF Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		alpha: DSF alpha parameter (~0.2)
	Returns
		Energy of  Mols
	"""
	xyzs *= BOHRPERA
	elu_width *= BOHRPERA
	dsf_alpha /= BOHRPERA
	cutoff_dist *= BOHRPERA
	inp_shp = tf.shape(xyzs)
	num_mol = tf.cast(tf.shape(xyzs)[0], dtype=tf.int64)
	num_pairs = tf.cast(tf.shape(Radpair)[0], tf.int64)
	elu_shift, elu_alpha = tf_dsf_potential(elu_width, cutoff_dist, dsf_alpha, return_grad=True)

	Rij = DifferenceVectorsLinear(xyzs + 1e-16, Radpair)
	RijRij2 = tf.norm(Rij, axis=1)
	qij = tf.gather_nd(charges, Radpair[:,:2]) * tf.gather_nd(charges, Radpair[:,::2])

	coulomb_potential = qij * tf.where(tf.greater(RijRij2, elu_width), tf_dsf_potential(RijRij2, cutoff_dist, dsf_alpha),
						elu_alpha * (tf.exp(RijRij2 - elu_width) - 1.0) + elu_shift)

	range_index = tf.range(num_pairs, dtype=tf.int64)
	mol_index = tf.cast(Radpair[:,0], dtype=tf.int64)
	sparse_index = tf.cast(tf.stack([mol_index, range_index], axis=1), tf.int64)
	sp_atomoutputs = tf.SparseTensor(sparse_index, coulomb_potential, [num_mol, num_pairs])
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def tf_dsf_potential(dists, cutoff_dist, dsf_alpha, return_grad=False):
	dsf_potential = tf.erfc(dsf_alpha * dists) / dists - tf.erfc(dsf_alpha * cutoff_dist) / cutoff_dist \
					+ (dists - cutoff_dist) * (tf.erfc(dsf_alpha * cutoff_dist) / tf.square(cutoff_dist) \
					+ 2.0 * dsf_alpha * tf.exp(-tf.square(dsf_alpha * cutoff_dist)) / (tf.sqrt(np.pi) * cutoff_dist))
	dsf_potential = tf.where(tf.greater(dists, cutoff_dist), tf.zeros_like(dsf_potential), dsf_potential)
	if return_grad:
		dsf_gradient = -(tf.erfc(dsf_alpha * dists) / tf.square(dists) - tf.erfc(dsf_alpha * cutoff_dist) / tf.square(cutoff_dist) \
	 					+ 2.0 * dsf_alpha / tf.sqrt(np.pi) * (tf.exp(-tf.square(dsf_alpha * dists)) / dists \
						- tf.exp(-tf.square(dsf_alpha * cutoff_dist)) / tf.sqrt(np.pi)))
		return dsf_potential, dsf_gradient
	else:
		return dsf_potential


class ANISym:
	def __init__(self, mset_):
		self.set = mset_
		self.MaxAtoms = self.set.MaxNAtoms()
		self.nmol = len(self.set.mols)
		self.MolPerBatch = 10000
		self.SymOutput = None
		self.xyz_pl= None
		self.Z_pl = None
		self.SFPa = None
		self.SFPr = None
		self.SymGrads = None
		self.TDSSet = None
		self.vdw_R = np.zeros(2)
		self.C6 = np.zeros(2)
		for i, ele in enumerate([1,8]):
			self.C6[i] = C6_coff[ele]* (BOHRPERA*10.0)**6.0 / JOULEPERHARTREE # convert into a.u.
			self.vdw_R[i] = atomic_vdw_radius[ele]*BOHRPERA

	def SetANI1Param(self):
		zetas = np.array([[8.0]], dtype = np.float64)
		etas = np.array([[4.0]], dtype = np.float64)
		self.zeta = 8.0
		self.eta = 4.0
		AN1_num_a_As = 8
		thetas = np.array([ 2.0*Pi*i/AN1_num_a_As for i in range (0, AN1_num_a_As)], dtype = np.float64)
		AN1_num_a_Rs = 8
		AN1_a_Rc = 3.1
		self.Ra_cut = 3.1
		self.Rr_cut = 4.6
		rs =  np.array([ AN1_a_Rc*i/AN1_num_a_Rs for i in range (0, AN1_num_a_Rs)], dtype = np.float64)
		Ra_cut = AN1_a_Rc
		# Create a parameter tensor. 4 x nzeta X neta X ntheta X nr
		p1 = np.tile(np.reshape(zetas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(etas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
		p3 = np.tile(np.reshape(thetas,[1,1,AN1_num_a_As,1,1]),[1,1,1,AN1_num_a_Rs,1])
		p4 = np.tile(np.reshape(rs,[1,1,1,AN1_num_a_Rs,1]),[1,1,AN1_num_a_As,1,1])
		SFPa = np.concatenate([p1,p2,p3,p4],axis=4)
		self.SFPa = np.transpose(SFPa, [4,0,1,2,3])
		#self.P5 = Tile_P5(1, 1, AN1_num_a_As, AN1_num_a_Rs)

		# Create a parameter tensor. 2 x ntheta X nr
		p1 = np.tile(np.reshape(thetas,[AN1_num_a_As,1,1]),[1,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(rs,[1,AN1_num_a_Rs,1]),[AN1_num_a_As,1,1])
		SFPa2 = np.concatenate([p1,p2],axis=2)
		self.SFPa2 = np.transpose(SFPa2, [2,0,1])

		etas_R = np.array([[4.0]], dtype = np.float64)
		AN1_num_r_Rs = 32
		AN1_r_Rc = 4.6
		rs_R =  np.array([ AN1_r_Rc*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)], dtype = np.float64)
		Rr_cut = AN1_r_Rc
		# Create a parameter tensor. 2 x  neta X nr
		p1_R = np.tile(np.reshape(etas_R,[1,1,1]),[1,AN1_num_r_Rs,1])
		p2_R = np.tile(np.reshape(rs_R,[1,AN1_num_r_Rs,1]),[1,1,1])
		SFPr = np.concatenate([p1_R,p2_R],axis=2)
		self.SFPr = np.transpose(SFPr, [2,0,1])
		# Create a parameter tensor. 1  X nr
		p1_new = np.reshape(rs_R,[AN1_num_r_Rs,1])
		self.SFPr2 = np.transpose(p1_new, [1,0])
		#self.P3 = Tile_P3(1,  AN1_num_r_Rs)
		#self.TDSSet = [AllTriplesSet_Np(self.MolPerBatch, self.MaxAtoms), AllDoublesSet_Np(self.MolPerBatch, self.MaxAtoms), AllSinglesSet_Np(self.MolPerBatch, self.MaxAtoms)]


	def Prepare(self):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
		        continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyz_pl=tf.placeholder(tf.float64, shape=tuple([self.MolPerBatch, self.MaxAtoms,3]))
			self.Z_pl=tf.placeholder(tf.int64, shape=tuple([self.MolPerBatch, self.MaxAtoms]))
			self.Radp_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.Angt_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.RadpEle_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.AngtEle_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_j_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_jk_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			#self.nreal = tf.Variable(self.Num_Real, dtype = tf.int32)
			self.Qs_pl=tf.placeholder(tf.float64, shape=tuple([self.MolPerBatch, self.MaxAtoms]))
			Ele = tf.Variable([[1],[8]], dtype = tf.int64)
			Elep = tf.Variable([[1,1],[1,8],[8,8]], dtype = tf.int64)
			C6 = tf.Variable(self.C6, tf.float64)
			R_vdw = tf.Variable(self.vdw_R, tf.float64)
			#zetas = tf.Variable([[8.0]], dtype = tf.float64)
			#etas = tf.Variable([[4.0]], dtype = tf.float64)
			SFPa = tf.Variable(self.SFPa, tf.float64)
			SFPr = tf.Variable(self.SFPr, tf.float64)
			SFPa2 = tf.Variable(self.SFPa2, tf.float64)
			SFPr2 = tf.Variable(self.SFPr2, tf.float64)
			#P3 = tf.Variable(self.P3, tf.int32)
			#P5 = tf.Variable(self.P5, tf.int32)
			Ra_cut = 3.1
			Rr_cut = 4.6
			Ree_on = 0.0
			self.Radp_pl = self.RadpEle_pl[:,:3]
			#self.A, self.B, self.C, self.D, self.E, self.F, self.G, self.H, self.I = TFVdwPolyLR(self.xyz_pl,  self.Z_pl, Ele, C6, R_vdw, Ree_on, self.Radp_pl)
			#self.Scatter_Sym, self.Sym_Index = TFSymSet_Scattered(self.xyz_pl, self.Z_pl, Ele, SFPr, Rr_cut, Elep, SFPa, Ra_cut)
			#self.Scatter_Sym_Update, self.Sym_Index_Update = TFSymSet_Scattered_Update(self.xyz_pl, self.Z_pl, Ele, SFPr, Rr_cut, Elep, SFPa, Ra_cut)
			#self.Scatter_Sym_Update2, self.Sym_Index_Update2 = TFSymSet_Scattered_Update2(self.xyz_pl, self.Z_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, self.zeta, self.eta, Ra_cut)
			#self.Scatter_Sym_Update, self.Sym_Index_Update = TFSymSet_Scattered_Update_Scatter(self.xyz_pl, self.Z_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, self.zeta, self.eta, Ra_cut)
			self.Scatter_Sym_Linear_Qs, self.Sym_Index_Linear_Qs = TFSymSet_Radius_Scattered_Linear_Qs(self.xyz_pl, self.Z_pl, Ele, SFPr2, Rr_cut, Elep,  self.eta,  self.Radp_pl, self.Qs_pl)

			self.Scatter_Sym_Linear_Qs_Periodic, self.Sym_Index_Linear_Qs_Periodic = TFSymSet_Radius_Scattered_Linear_Qs_Periodic(self.xyz_pl, self.Z_pl, Ele, SFPr2, Rr_cut, Elep,  self.eta,  self.Radp_pl, self.Qs_pl, self.mil_j_pl, self.nreal)
			#self.Scatter_Sym_Linear, self.Sym_Index_Linear = TFSymSet_Scattered_Linear(self.xyz_pl, self.Z_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, self.zeta, self.eta, Ra_cut, self.Radp_pl, self.Angt_pl)

			self.Scatter_Sym_Linear_Ele, self.Sym_Index_Linear_Ele = TFSymSet_Scattered_Linear_WithEle(self.xyz_pl, self.Z_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, self.zeta, self.eta, Ra_cut, self.RadpEle_pl, self.AngtEle_pl, self.mil_jk_pl)
			self.Scatter_Sym_Linear_Ele_Periodic, self.Sym_Index_Linear_Ele_Periodic = TFSymSet_Scattered_Linear_WithEle_Periodic(self.xyz_pl, self.Z_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, self.zeta, self.eta, Ra_cut, self.RadpEle_pl, self.AngtEle_pl, self.mil_j_pl, self.mil_jk_pl, self.nreal)
			#self.Scatter_Sym_Linear_tmp, self.Sym_Index_Linear_tmp = TFSymSet_Scattered_Linear_tmp(self.xyz_pl, self.Z_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, self.zeta, self.eta, Ra_cut, self.RadpEle_pl, self.AngtEle_pl, self.mil_jk_pl)
			#self.Eee, self.Kern, self.index = TFCoulombCosLR(self.xyz_pl, tf.cast(self.Z_pl, dtype=tf.float64), Rr_cut, self.Radp_pl)
			#self.gradient = tf.gradients(self.Scatter_Sym, self.xyz_pl)
			#self.gradient_update2 = tf.gradients(self.Scatter_Sym_Update2, self.xyz_pl)
			#self.gradient = tf.gradients(self.Scatter_Sym_Update, self.xyz_pl)
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.sess.run(init)
			self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			self.run_metadata = tf.RunMetadata()
		return

	#def fill_feed_dict(self, batch_data, coord_pl, atom_pl, radp_pl,  angt_pl, Qs_pl, radpEle_pl,  angtEle_pl, mil_j_pl, mil_jk_pl):
	#	return {coord_pl: batch_data[0], atom_pl: batch_data[1], radp_pl:batch_data[2], angt_pl:batch_data[3], Qs_pl:batch_data[4], radpEle_pl:batch_data[5], angtEle_pl:batch_data[6], mil_j_pl:batch_data[7], mil_jk_pl:batch_data[8]}


	def fill_feed_dict(self, batch_data, coord_pl, atom_pl, radpEle_pl,  angtEle_pl, mil_j_pl, mil_jk_pl, Qs_pl):
		return {coord_pl: batch_data[0], atom_pl: batch_data[1], radpEle_pl:batch_data[2], angtEle_pl:batch_data[3], mil_j_pl:batch_data[4], mil_jk_pl:batch_data[5], Qs_pl:batch_data[6]}

	def TestPeriodic(self):
		"""
		Tests a Simple Periodic optimization.
		Trying to find the HCP minimum for the LJ crystal.
		"""
		a=MSet("water_tiny", center_=False)
		a.ReadXYZ("water_tiny")
		m = a.mols[-1]
		m.coords = m.coords - np.min(m.coords)
		print("Original coords:", m.coords)
		# Generate a Periodic Force field.
		cellsize = 9.3215
		lat = cellsize*np.eye(3)
		self.MolPerBatch = 1
		PF = TensorMol.TFPeriodicForces.TFPeriodicVoxelForce(15.0,lat)
		#zp, xp = PF(m.atoms,m.coords,lat)  # tessilation in TFPeriodic seems broken

		zp = np.zeros(m.NAtoms()*PF.tess.shape[0], dtype=np.int32)
		xp = np.zeros((m.NAtoms()*PF.tess.shape[0], 3))
		for i in range(0, PF.tess.shape[0]):
			zp[i*m.NAtoms():(i+1)*m.NAtoms()] = m.atoms
			xp[i*m.NAtoms():(i+1)*m.NAtoms()] = m.coords + cellsize*PF.tess[i]

		self.MaxAtoms = xp.shape[0]
		self.nreal = m.NAtoms()

		qs = np.zeros((1, self.MaxAtoms))
		for j in range (0, m.NAtoms()):
			if m.atoms[j] == 1:
				qs[0][j] = 0.5
			else:
				qs[0][j] = -1.0



		#self.Num_Real = m.NAtoms()
		self.SetANI1Param()
		self.Prepare()
		t_total = time.time()
		Ele = np.asarray([1,8])
		Elep = np.asarray([[1,1],[1,8],[8,8]])

		t0 = time.time()
		NL = NeighborListSetWithImages(xp.reshape((1,-1,3)), np.array([zp.shape[0]]), np.array([m.NAtoms()]), True, True, zp.reshape((1,-1)), sort_=True)
		rad_p_ele, ang_t_elep, mil_j, mil_jk = NL.buildPairsAndTriplesWithEleIndexPeriodic(4.6, 3.1, np.array([1,8]), np.array([[1,1],[1,8],[8,8]]))
		print ("python time cost:", time.time() - t0)
		batch_data = [xp.reshape((1,-1,3)), zp.reshape((1,-1)), rad_p_ele, ang_t_elep, mil_j, mil_jk, qs]
		feed_dict = self.fill_feed_dict(batch_data, self.xyz_pl, self.Z_pl, self.RadpEle_pl, self.AngtEle_pl, self.mil_j_pl, self.mil_jk_pl, self.Qs_pl)

		A, B, C, D = self.sess.run([self.Scatter_Sym_Linear_Qs, self.Sym_Index_Linear_Qs, self.Scatter_Sym_Linear_Qs_Periodic, self.Sym_Index_Linear_Qs_Periodic], feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
		print ("A:\n",A[0].shape, len(A))
		print ("B:\n",B[0].shape)
		print ("C:\n",C[0].shape, len(C))
		print ("D:\n",D[0].shape)
		#print ("A==C:", A[0]==C[0])
		#print ("B==D", B[0]==D[0])
		print ("total time cost:", time.time() - t0)
		np.set_printoptions(threshold=np.nan)
		print (A[0][54],A[0][53])
		for i in range(0, 54):
			print (np.array_equal(A[0][i], C[0][i]))

		for i in range(0, 27):
			print (np.array_equal(A[1][i], C[1][i]))
		#print ("B:",B[0][:200])

	def Generate_ANISYM(self):
		xyzs = np.zeros((self.nmol, self.MaxAtoms, 3),dtype=np.float64)
		qs = np.zeros((self.nmol, self.MaxAtoms),dtype=np.float64)
		Zs = np.zeros((self.nmol, self.MaxAtoms), dtype=np.int64)
		nnz_atom = np.zeros((self.nmol), dtype=np.int64)
		#random.shuffle(self.set.mols)
		for i, mol in enumerate(self.set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			nnz_atom[i] = mol.NAtoms()
			for j in range (0, mol.NAtoms()):
				if Zs[i][j] == 1:
					qs[i][j] = 0.5
				else:
					qs[i][j] = -1.0
		self.SetANI1Param()
		self.Prepare()
		t_total = time.time()
		Ele = np.asarray([1,8])
		Elep = np.asarray([[1,1],[1,8],[8,8]])
		m = self.set.mols[0]
		#vdw = 0.0
		#for i in range(0, m.NAtoms()):
		#	for j in range(i+1, m.NAtoms()):
		#		dist = np.sum(np.square(m.coords[i]-m.coords[j]))**0.5
		#		c6_i = C6_coff[m.atoms[i]]*(10.0)**6.0 / JOULEPERHARTREE
		#		c6_j = C6_coff[m.atoms[j]]*(10.0)**6.0 / JOULEPERHARTREE
		#		Ri = atomic_vdw_radius[m.atoms[i]]
		#		Rj = atomic_vdw_radius[m.atoms[j]]
		#		if dist > 4.6:
		#			cut = 1.0
		#		else:
		#			t = dist/4.6
		#			cut = -t*t*(2.0*t-3.0)
		#		print (dist*BOHRPERA, -(c6_i*c6_j)**0.5/dist**6 /(1.0+6*(dist/(Ri+Rj))**-12)*cut, cut, 1.0/(1.0+6*(dist/(Ri+Rj))**-12))
		#		vdw += -(c6_i*c6_j)**0.5/dist**6 /(1.0+6*(dist/(Ri+Rj))**-12)*cut
		#print ("vdw:", vdw,"\n\n\n")


		for i in range (0, int(self.nmol/self.MolPerBatch-1)):
			t = time.time()
			NL = NeighborListSet(xyzs[i*self.MolPerBatch: (i+1)*self.MolPerBatch], nnz_atom[i*self.MolPerBatch: (i+1)*self.MolPerBatch], True, True, Zs[i*self.MolPerBatch: (i+1)*self.MolPerBatch], sort_=True)
			rad_p, ang_t = NL.buildPairsAndTriples(self.Rr_cut, self.Ra_cut)
			rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(self.Rr_cut, self.Ra_cut, Ele, Elep)

			NLEE = NeighborListSet(xyzs[i*self.MolPerBatch: (i+1)*self.MolPerBatch], nnz_atom[i*self.MolPerBatch: (i+1)*self.MolPerBatch], False, False,  None)
			rad_p = NLEE.buildPairs(self.Rr_cut)
			print ("rad_p:", rad_p[:20])
			#print ("time to build pairs:", time.time() - t)
			#print ("rad_p_ele:\n", rad_p_ele, "\nang_t_elep:\n", ang_t_elep)
			#raise Exception("Debug..")
			t = time.time()
			batch_data = [xyzs[i*self.MolPerBatch: (i+1)*self.MolPerBatch], Zs[i*self.MolPerBatch: (i+1)*self.MolPerBatch], rad_p,  ang_t, qs[i*self.MolPerBatch: (i+1)*self.MolPerBatch], rad_p_ele, ang_t_elep, mil_jk]
			feed_dict = self.fill_feed_dict(batch_data, self.xyz_pl, self.Z_pl, self.Radp_pl, self.Angt_pl, self.Qs_pl, self.RadpEle_pl, self.AngtEle_pl, self.mil_jk_pl)
			t = time.time()
			#sym_output, grad = self.sess.run([self.SymOutput, self.SymGrads], feed_dict = feed_dict)
			#sym_output_update2, sym_index_update2, sym_output, sym_index, gradient_update2 = self.sess.run([self.Scatter_Sym_Update2, self.Sym_Index_Update2, self.Scatter_Sym_Update, self.Sym_Index_Update, self.gradient_update2], feed_dict = feed_dict)
			#sym_output_update, sym_index_update, sym_output, sym_index, gradient, gradient_update = self.sess.run([self.Scatter_Sym_Update, self.Sym_Index_Update, self.Scatter_Sym, self.Sym_Index, self.gradient, self.gradient_update], feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
			#sym_output, sym_index  = self.sess.run([self.Scatter_Sym_Update2, self.Sym_Index_Update2], feed_dict = feed_dict)
			#sym_output, sym_index  = self.sess.run([self.Scatter_Sym, self.Sym_Index], feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
			#A, B  = self.sess.run([self.Scatter_Sym_Update, self.Sym_Index_Update], feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
			#A, B, C, D  = self.sess.run([self.Scatter_Sym_Linear, self.Sym_Index_Linear, self.Scatter_Sym_Linear_Qs, self.Sym_Index_Linear_Qs], feed_dict = feed_dict)

			#A, B  = self.sess.run([self.Scatter_Sym_Linear, self.Sym_Index_Linear], feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
			#C, D  = self.sess.run([self.Scatter_Sym_Linear_Ele, self.Sym_Index_Linear_Ele], feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
			A, B, C, D, E, F, H, G, I = self.sess.run([self.A, self.B, self.C, self.D, self.E, self.F, self.H, self.G, self.I], feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
			#A, B, C, D  = self.sess.run([self.Scatter_Sym_Linear, self.Sym_Index_Linear, self.Scatter_Sym_Linear_Ele, self.Sym_Index_Linear_Ele], feed_dict = feed_dict)
			#print ("A:\n", A[0].shape, "\nC:\n", C[0].shape)
			#np.set_printoptions(threshold=np.nan)
			print ("A:",A, "B:",B)
			print ("C:",C, "D:",D)
			print ("E:",E, " F:",F)
			print ("H:", H, "G:", G)
			print ("I:", I)
			return
			#np.savetxt("rad_sym.dat", A[0][0][:64])
			#np.savetxt("Zs_sym.dat", C[0][0])
			#np.savetxt("rad_ang_sym.dat", A[0][0])
			#raise Exception("End Here")
			#print ("i: ", i,  "sym_ouotput: ", len(sym_output)," time:", time.time() - t, " second", "gpu time:", time.time()-t1, sym_index)
			#print ("sym_output_update:", np.array_equal(sym_output_update2[0], sym_output[0]))
			#print ("sym_output_update:", np.sum(np.abs(sym_output_update2[0]-sym_output[0])))
			#print ("gradient_update:", np.sum(np.abs(gradient[0]-gradient_update[0])))
			#print ("sym_index_update:", np.array_equal(sym_index_update[0], sym_index[0]))
			#print ("gradient:", gradient[0].shape)
			fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			chrome_trace = fetched_timeline.generate_chrome_trace_format()
			with open('timeline_step_%d_new.json' % i, 'w') as f:
				f.write(chrome_trace)
			print ("inference time:", time.time() - t)
