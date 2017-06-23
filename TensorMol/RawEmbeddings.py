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

def DifferenceVectors(r_):
    """
    Given a maxnatom X 3 tensor of coordinates this
    returns a maxnatom X maxnatom X 3 tensor of Rij
    """
    natom = tf.shape(r_)[0]
    ri = tf.tile(tf.reshape(r_,[1,natom,3]),[natom,1,1])
    rj = tf.transpose(ri,perm=(1,0,2))
    return (ri-rj)

def DifferenceVectorsSet(r_):
    """
    Given a nmol X maxnatom X 3 tensor of coordinates this
    returns a nmol X maxnatom X maxnatom X 3 tensor of Rij
    """
    natom = tf.shape(r_)[1]
    nmol = tf.shape(r_)[0]
    ri = tf.tile(tf.reshape(r_,[nmol,1,natom,3]),[1,natom,1,1])
    rj = tf.transpose(ri,perm=(0,2,1,3))
    return (ri-rj)

def AllTriples(rng):
    """Returns all possible triples of an input list.

    Args:
        rng: a 1-D tensor.
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


def AllTriplesSet(rng):
    """Returns all possible triples of integers between zero and natom.

    Args:
        natom: max integer
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
    v7 = tf.tile(tf.reshape(tf.range(nmol),[nmol,1,1,1,1]),[1,natom,natom,natom,1])
    v8 = tf.concat([v7,v6], axis = -1)
    return v8


def Zouter(Z):
    """
    Returns the outer product of atomic numbers for one molecule.

    Args:
        Z: MaxNAtom X 1 Z tensor
    Returns
        Z1Z2: MaxNAtom X MaxNAtom X 2 Z1Z2 tensor.
    """
    zshp = tf.shape(Z)
    Zs = tf.reshape(Z,[zshp[0],1])
    z1 = tf.tile(Zs, [1,zshp[0]])
    z2 = tf.transpose(z1,perm=[1,0])
    return tf.transpose(tf.stack([z1,z2],axis=0),perm=[1,2,0])

def ZouterSet(Z):
    """
    Returns the outer product of atomic numbers for one molecule.

    Args:
        Z: MaxNAtom X 1 Z tensor
    Returns
        Z1Z2: MaxNAtom X MaxNAtom X 2 Z1Z2 tensor.
    """
    zshp = tf.shape(Z)
    Zs = tf.reshape(Z,[zshp[0],zshp[1],1])
    z1 = tf.tile(Zs, [1,1,zshp[1]])
    z2 = tf.transpose(z1,perm=[0,2,1])
    return tf.transpose(tf.stack([z1,z2],axis=1),perm=[0,2,3,1])

def jacobian(y, x):
    y_flat = tf.reshape(y, (-1,))
    jacobian_flat = tf.stack([tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
    return tf.reshape(jacobian_flat, y.shape.concatenate(x.shape))

def TFSymR(R, Zs, elems, RSFPs, R_cut):
    """
    A tensorflow implementation of the radial AN1 symmetry function for a single input molecule.

    Args:
        R: a maxnatom X 3 tensor of coordinates.
        Zs : maxnatom X 1 tensor of atomic numbers.
        elems_: a tensor of elements present in the data.
        RSFPs: A symmetry function parameter tensor 2 X neta X nRs.
        For example, RSFPs[0,0,0] is the first eta parameter. RSFPs[1,1] is the second R parameter.
        R_cut: Radial Cutoff

    Returns:
        Digested Mol. In the shape maxnatom X nelem X nEta X nRs
    """
    inp_shp = tf.shape(R)
    natom = inp_shp[0]
    natom2 = natom*natom
    natom3 = natom*natom2
    nele = tf.shape(elems)[0]
    pshape = tf.shape(RSFPs)
    neta = pshape[0]
    nr = pshape[1]

    # atom triples.
    ats = AllTriples(tf.range(natom))
    # Z triples. Used to scatter element contribs.
    ZTrips = AllTriples(Zs)

    # Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
    # Get Rij, Rik...
    Ri_inds = tf.slice(ats,[0,0,0,0],[natom,natom,natom,1])
    Rj_inds = tf.slice(ats,[0,0,0,1],[natom,natom,natom,1])
    Rk_inds = tf.slice(ats,[0,0,0,2],[natom,natom,natom,1])
    Rij_inds = tf.reshape(tf.concat([Ri_inds,Rj_inds],axis=3),[natom3,2])
    Rik_inds = tf.reshape(tf.concat([Ri_inds,Rk_inds],axis=3),[natom3,2])
    Rij = DifferenceVectors(R) # atom X atom X 3
    # Pull out the appropriate pairs of distances from Rij.
    A = tf.gather_nd(Rij,Rij_inds)
    B = tf.gather_nd(Rij,Rik_inds)
    RijRik = tf.einsum("ij,ij->i",A,B)
    RijRij = tf.sqrt(tf.einsum("ij,ij->i",A,A)+infinitesimal)
    RikRik = tf.sqrt(tf.einsum("ij,ij->i",B,B)+infinitesimal)
    denom = RijRij*RikRik
    # Mask any troublesome entries.
    ToACos = RijRik/denom
    ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos),ToACos)
    ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos),ToACos)
    Thetaijk = tf.acos(ToACos)
    # Finally construct the thetas for all the triples.
    zetatmp = tf.reshape(SFPs_[0],[1,nzeta,neta,ntheta,nr])
    thetatmp = tf.reshape(SFPs_[2],[1,nzeta,neta,ntheta,nr])
    # Broadcast the thetas and ToCos together
    tct = tf.reshape(Thetaijk,[natom3,1,1,1,1])
    Tijk = tf.cos(tct-thetatmp) # shape: natom3 X ...
    # complete factor 1 for all j,k
    fac1 = tf.pow(2.0,1.0-zetatmp)*tf.pow((1.0+Tijk),zetatmp)
    # Construct Rij + Rik/2  for all jk!=i
    # make the etas,R's broadcastable onto this and vice versa.
    etmp = tf.reshape(SFPs_[1],[1,nzeta,neta,ntheta,nr]) # ijk X zeta X eta ....
    rtmp = tf.reshape(SFPs_[3],[1,nzeta,neta,ntheta,nr]) # ijk X zeta X eta ....
    ToExp = ((RijRij+RikRik)/2.0)
    tet = tf.reshape(ToExp,[natom3,1,1,1,1]) - rtmp
    fac2 = tf.exp(-etmp*tet*tet)
    # And finally the last two factors
    fac3 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros_like(RijRij),0.5*(tf.cos(3.14159265359*RijRij/R_cut)+1.0))
    fac4 = tf.where(tf.greater_equal(RikRik,R_cut),tf.zeros_like(RikRik),0.5*(tf.cos(3.14159265359*RikRik/R_cut)+1.0))
    # Zero out the diagonal contributions (i==j or i==k)
    mask1 = tf.reshape(tf.where(tf.equal(Ri_inds,Rj_inds),tf.zeros_like(Ri_inds,dtype=tf.float32),tf.ones_like(Ri_inds,dtype=tf.float32)),[natom3])
    mask2 = tf.reshape(tf.where(tf.equal(Ri_inds,Rk_inds),tf.zeros_like(Ri_inds,dtype=tf.float32),tf.ones_like(Ri_inds,dtype=tf.float32)),[natom3])
    fac34t =  tf.reshape(fac3*fac4*mask1*mask2,[natom3,1,1,1,1])
    # assemble the full symmetry function for all triples.
    # Use broadcasting to mask these out...
    Gm = fac1*fac2*fac34t # Gm so far has shape atom3 X nzeta X neta X ntheta X nr
    # Now, finally Scatter the element contributions and sum over jk.
    Rjk_inds = tf.reshape(tf.concat([Rj_inds,Rk_inds],axis=3),[natom3,2])
    Z1Z2 = Zouter(Z)
    ZPairs = tf.gather_nd(Z1Z2,Rjk_inds) # should have shape natom3 X 2
    # Create a tensor which selects out components where jk = elep[i]
    # This is done by broadcasting our natom X natom X natom X zeta... tensor.
    # onto a tensor which has an added dimension for the element pairs, and reducing over jk.
    ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[natom3,1,2]),tf.reshape(eleps_,[1,nelep,2])),axis=-1)
    # So this tensor has dim natom3 X nelep we broadcast over it's shape and reduce_sum.
    GmToMask = tf.tile(tf.reshape(Gm,[natom,natom2,1,nzeta,neta,ntheta,nr]),[1,1,nelep,1,1,1,1])
    ERMask = tf.tile(tf.reshape(ElemReduceMask,[natom,natom2,nelep,1,1,1,1]),[1,1,1,nzeta,neta,ntheta,nr])
    ToRS = tf.where(ERMask,GmToMask,tf.zeros_like(GmToMask))
    GMA = tf.reduce_sum(ToRS,axis=[1])
    return GMA

def TFSymA(R, Zs, eleps_, SFPs_, R_cut):
    """
    A tensorflow implementation of the angular AN1 symmetry function for a single input molecule.
    Here j,k are all other atoms, but implicitly the output
    is separated across elements as well. eleps_ is a list of element pairs
    G = 2**(1-zeta) \sum_{j,k \neq i} (Angular triple) (radial triple) f_c(R_{ij}) f_c(R_{ik})
    a-la MolEmb.cpp. Also depends on PARAMS for zeta, eta, theta_s r_s

    Args:
        R: a maxnatom X 3 tensor of coordinates.
        Zs : maxnatom X 1 tensor of atomic numbers.
        eleps_: a nelepairs X 2 tensor of element pairs present in the data.
        SFP: A symmetry function parameter tensor having the number of elements
        as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
        is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
        R_cut: Radial Cutoff

    Returns:
        Digested Mol. In the shape maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
    """
    inp_shp = tf.shape(R)
    natom = inp_shp[0]
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

    # atom triples.
    ats = AllTriples(tf.range(natom))
    # Z triples. Used to scatter element contribs.
    #ZTrips = AllTriples(Zs)

    # Construct the angle triples acos(<Rij,Rik>/|Rij||Rik|) and mask them onto the correct output
    # Get Rij, Rik...
    Ri_inds = tf.slice(ats,[0,0,0,0],[natom,natom,natom,1])
    Rj_inds = tf.slice(ats,[0,0,0,1],[natom,natom,natom,1])
    Rk_inds = tf.slice(ats,[0,0,0,2],[natom,natom,natom,1])
    Rij_inds = tf.reshape(tf.concat([Ri_inds,Rj_inds],axis=3),[natom3,2])
    Rik_inds = tf.reshape(tf.concat([Ri_inds,Rk_inds],axis=3),[natom3,2])
    Rij = DifferenceVectors(R) # atom X atom X 3
    # Pull out the appropriate pairs of distances from Rij.
    A = tf.gather_nd(Rij,Rij_inds)
    B = tf.gather_nd(Rij,Rik_inds)
    RijRik = tf.einsum("ij,ij->i",A,B)
    RijRij = tf.sqrt(tf.einsum("ij,ij->i",A,A)+infinitesimal)
    RikRik = tf.sqrt(tf.einsum("ij,ij->i",B,B)+infinitesimal)
    denom = RijRij*RikRik
    # Mask any troublesome entries.
    ToACos = RijRik/denom
    ToACos = tf.where(tf.greater_equal(ToACos,1.0),tf.ones_like(ToACos),ToACos)
    ToACos = tf.where(tf.less_equal(ToACos,-1.0),-1.0*tf.ones_like(ToACos),ToACos)
    Thetaijk = tf.acos(ToACos)
    # Finally construct the thetas for all the triples.
    zetatmp = tf.reshape(SFPs_[0],[1,nzeta,neta,ntheta,nr])
    thetatmp = tf.reshape(SFPs_[2],[1,nzeta,neta,ntheta,nr])
    # Broadcast the thetas and ToCos together
    tct = tf.reshape(Thetaijk,[natom3,1,1,1,1])
    Tijk = tf.cos(tct-thetatmp) # shape: natom3 X ...
    # complete factor 1 for all j,k
    fac1 = tf.pow(2.0,1.0-zetatmp)*tf.pow((1.0+Tijk),zetatmp)
    # Construct Rij + Rik/2  for all jk!=i
    # make the etas,R's broadcastable onto this and vice versa.
    etmp = tf.reshape(SFPs_[1],[1,nzeta,neta,ntheta,nr]) # ijk X zeta X eta ....
    rtmp = tf.reshape(SFPs_[3],[1,nzeta,neta,ntheta,nr]) # ijk X zeta X eta ....
    ToExp = ((RijRij+RikRik)/2.0)
    tet = tf.reshape(ToExp,[natom3,1,1,1,1]) - rtmp
    fac2 = tf.exp(-etmp*tet*tet)
    # And finally the last two factors
    fac3 = tf.where(tf.greater_equal(RijRij,R_cut),tf.zeros_like(RijRij),0.5*(tf.cos(3.14159265359*RijRij/R_cut)+1.0))
    fac4 = tf.where(tf.greater_equal(RikRik,R_cut),tf.zeros_like(RikRik),0.5*(tf.cos(3.14159265359*RikRik/R_cut)+1.0))
    # Zero out the diagonal contributions (i==j or i==k)
    mask1 = tf.reshape(tf.where(tf.equal(Ri_inds,Rj_inds),tf.zeros_like(Ri_inds,dtype=tf.float32),tf.ones_like(Ri_inds,dtype=tf.float32)),[natom3])
    mask2 = tf.reshape(tf.where(tf.equal(Ri_inds,Rk_inds),tf.zeros_like(Ri_inds,dtype=tf.float32),tf.ones_like(Ri_inds,dtype=tf.float32)),[natom3])
    # Also mask out the lower triangle. (j>k)
    # mask3 = tf.reshape(tf.where(tf.greater(Rj_inds,Rk_inds),tf.zeros_like(Ri_inds,dtype=tf.float32),tf.ones_like(Ri_inds,dtype=tf.float32)),[natom3])
    # assemble the full symmetry function for all triples.
    fac34t =  tf.reshape(fac3*fac4*mask1*mask2,[natom3,1,1,1,1])
    # Use broadcasting to mask these out...
    Gm = fac1*fac2*fac34t # Gm so far has shape atom3 X nzeta X neta X ntheta X nr
    # Now, finally Scatter the element contributions and sum over jk.
    Rjk_inds = tf.reshape(tf.concat([Rj_inds,Rk_inds],axis=3),[natom3,2])
    Z1Z2 = Zouter(Z)
    ZPairs = tf.gather_nd(Z1Z2,Rjk_inds) # should have shape natom3 X 2
    # Create a tensor which selects out components where jk = elep[i]
    # This is done by broadcasting our natom X natom X natom X zeta... tensor.
    # onto a tensor which has an added dimension for the element pairs, and reducing over jk.
    ElemReduceMask = tf.reduce_all(tf.equal(tf.reshape(ZPairs,[natom3,1,2]),tf.reshape(eleps_,[1,nelep,2])),axis=-1)
    # So this tensor has dim natom3 X nelep we broadcast over it's shape and reduce_sum.
    GmToMask = tf.tile(tf.reshape(Gm,[natom,natom2,1,nzeta,neta,ntheta,nr]),[1,1,nelep,1,1,1,1])
    ERMask = tf.tile(tf.reshape(ElemReduceMask,[natom,natom2,nelep,1,1,1,1]),[1,1,1,nzeta,neta,ntheta,nr])
    ToRS = tf.where(ERMask,GmToMask,tf.zeros_like(GmToMask))
    GMA = tf.reduce_sum(ToRS,axis=[1])
    return GMA
