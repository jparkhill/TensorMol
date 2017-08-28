"""
This adds a layer of periodicity to a previously aperiodic local force in tensorflow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math, time, os, sys, os.path
from TensorMol.Util import *
if (HAS_TF):
	import tensorflow as tf

def PeriodicWrapping(z_, xyz_, lat_, tess_):
	"""
	This tesselates xyz_ using the lattice vectors lat_
	according to tess_, so that lat_ can be differentiable tensorflow varaiables.
	ie: the stress tensor can be calculated. The first block of atoms are
	the 'real' atoms (ie: atoms whose forces will be calculated)
	the remainder are images.

	Args:
		z_: vector of atomic numbers
		xyz_: nprimitive atoms X 3 coordinates.
		lat_: 3x3 lattice matrix.
		tess_: an ntess X 3 matrix of tesselations to perform. The identity is the assumed initial state.
	"""
	ntess = tf.shape(tess_)[0]
	i0 = tf.constant(0,tf.int32)
	natold = tf.shape(z_)
	natnew = tf.shape(z_)*ntess
	cond = lambda i,z,x: tf.less(i,ntess)
	body = lambda i,z,x: [i+1, tf.concat([z,z],axis=0),tf.concat([x,xyz_+tess_[i,0]*lat_[0]+tess_[i,1]*lat_[1]+tess_[i,2]*lat_[2]],axis=0)]
	i,z,x = tf.while_loop(cond,body,loop_vars=[i0,z_,xyz_],shape_invariants=[i0.get_shape(),tf.TensorShape([None,1]),tf.TensorShape([None,3])])
	return z,x

def PeriodicWrappingOfCoords(xyz_, lat_, tess_):
	"""
	This tesselates xyz_ using the lattice vectors lat_
	according to tess_, so that lat_ can be differentiable tensorflow varaiables.
	ie: the stress tensor can be calculated. The first block of atoms are
	the 'real' atoms (ie: atoms whose forces will be calculated)
	the remainder are images.

	Args:
		xyz_: nprimitive atoms X 3 coordinates.
		lat_: 3x3 lattice matrix.
		tess_: an ntess X 3 matrix of tesselations to perform. The identity is the assumed initial state.
	"""
	ntess = tf.shape(tess_)[0]
	i0 = tf.constant(0,tf.int32)
	natold = tf.shape(z_)
	natnew = tf.shape(z_)*ntess
	cond = lambda i,x: tf.less(i,ntess)
	body = lambda i,x: [i+1,tf.concat([x,xyz_+tess_[i,0]*lat_[0]+tess_[i,1]*lat_[1]+tess_[i,2]*lat_[2]],axis=0)]
	i,z,x = tf.while_loop(cond,body,loop_vars=[i0,xyz_],shape_invariants=[i0.get_shape(),tf.TensorShape([None,3])])
	return x
