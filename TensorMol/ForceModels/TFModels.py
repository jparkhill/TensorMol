"""
	Tensorflow implementations of simple chemical models to test out learning approaches.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..TFNetworks.TFInstance import *
from ..Containers.TensorMolData import *
from ..TFNetworks.TFMolInstance import *
from .ElectrostaticsTF import *
from .TFForces import *
from ..ForceModifiers.Neighbors import *
from . import *
from tensorflow.python.client import timeline
import threading


class MorseModel(ForceHolder):
	def __init__(self,natom_=3):
		"""
		simple morse model for three atoms for a training example.
		"""
		ForceHolder.__init__(self,natom_)
		self.lat_pl = None # boundary vectors of the cell.
		#self.a = sqrt(self.k/(2.0*self.de))
		self.Prepare()
	def PorterKarplus(self,x_pl):
		x1 = x_pl[0] - x_pl[1]
		x2 = x_pl[2] - x_pl[1]
		x12 = x_pl[0] - x_pl[2]
		r1 = tf.norm(x1)
		r2 = tf.norm(x2)
		r12 = tf.norm(x12)
		v1 = 0.7*tf.pow(1.-tf.exp(-(r1-0.7)),2.0)
		v2 = 0.7*tf.pow(1.-tf.exp(-(r2-0.7)),2.0)
		v3 = 0.7*tf.pow(1.-tf.exp(-((r12)-0.7)),2.0)
		return v1+v2+v3
	def Prepare(self):
		self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.natom,3]))
		self.Energy = self.PorterKarplus(self.x_pl)
		self.Force =tf.gradients(-1.0*self.PorterKarplus(self.x_pl),self.x_pl)
		init = tf.global_variables_initializer()
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		self.sess.run(init)
	def __call__(self,x_):
		"""
		Args:
			x_: the coordinates on which to evaluate the force.
			lat_: the lattice boundary vectors.
		Returns:
			the Energy and Force (Eh and Eh/ang.) associated with the quadratic walls.
		"""
#		print("lat",lat_)
#		print("Xinlat",self.sess.run([XInLat(self.x_pl,self.lat_pl)], feed_dict = {self.x_pl:x_, self.lat_pl:lat_}))
		e,f = self.sess.run([self.Energy,self.Force], feed_dict = {self.x_pl:x_})
		#print("Min max and lat", np.min(x_), np.max(x_), lat_, e ,f)
		return e, f[0]

class QuantumElectrostatic(ForceHolder):
	def __init__(self,natom_=3):
		"""
		This is a huckle-like model, something like BeH2
		four valence charges are exchanged between the atoms
		which experience a screened coulomb interaction
		"""
		ForceHolder.__init__(self,natom_)
		self.Prepare()
	def HuckelBeH2(self,x_pl):
		r = tf.reduce_sum(x_pl*x_pl, 1)
		r = tf.reshape(r, [-1, 1]) # For the later broadcast.
		# Tensorflow can only reverse mode grad the sqrt if all these elements
		# are nonzero
		D = tf.sqrt(r - 2*tf.matmul(x_pl, tf.transpose(x_pl)) + tf.transpose(r) + tf.cast(1e-26,tf.float64))
		emat = tf.diag(self.en0s)
		J = tf.matrix_band_part(-1.0/tf.pow((D + 0.5*0.5*0.5),1.0/3.0), 0, -1)
		emat += J + tf.transpose(J)
		e,v = tf.self_adjoint_eig(emat)
		popd = tf.nn.top_k(-1.*e, 2, sorted=True).indices
		# The lowest two orbitals are populated.
		Energy = e[popd[0]]+e[popd[1]]
		q1=-1.0+v[popd[0],0]*v[popd[0],0]+v[popd[1],0]*v[popd[1],0]
		q2=-0.5+v[popd[0],1]*v[popd[0],1]+v[popd[1],1]*v[popd[1],1]
		q3=-0.5+v[popd[0],2]*v[popd[0],2]+v[popd[1],2]*v[popd[1],2]
		# compute the dipole moment.
		Dipole = (q1*x_pl[0]+q2*x_pl[1]+q3*x_pl[2])/3.0
		return Energy, Dipole, [q1,q2,q3]
	def Prepare(self):
		self.en0s = tf.constant([-1.1,-0.5,-0.5],dtype=tf.float64)
		self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.natom,3]))
		self.Energy,self.Dipole,self.Charges = self.HuckelBeH2(self.x_pl)
		self.Force =tf.gradients(-1.0*self.Energy,self.x_pl)
		init = tf.global_variables_initializer()
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		self.sess.run(init)
	def __call__(self,x_):
		"""
		Args:
			x_: the coordinates on which to evaluate the force.
		Returns:
			the Energy and Force (Eh and Eh/ang.) associated with the quadratic walls.
		"""
		e,f,d,q = self.sess.run([self.Energy,self.Force,self.Dipole,self.Charges], feed_dict = {self.x_pl:x_})
		#print("Min max and lat", np.min(x_), np.max(x_), lat_, e ,f)
		return e, f[0], d, q

class ExtendedHuckel(ForceHolder):
	def __init__(self,):
		"""

		"""
		ForceHolder.__init__(self,natom_)
		self.Prepare()
	def HuckelBeH2(self,x_pl):
		r = tf.reduce_sum(x_pl*x_pl, 1)
		r = tf.reshape(r, [-1, 1]) # For the later broadcast.
		# Tensorflow can only reverse mode grad the sqrt if all these elements
		# are nonzero
		D = tf.sqrt(r - 2*tf.matmul(x_pl, tf.transpose(x_pl)) + tf.transpose(r) + tf.cast(1e-26,tf.float64))
		emat = tf.diag(self.en0s)
		J = tf.matrix_band_part(-1.0/tf.pow((D + 0.5*0.5*0.5),1.0/3.0), 0, -1)
		emat += J + tf.transpose(J)
		e,v = tf.self_adjoint_eig(emat)
		popd = tf.nn.top_k(-1.*e, 2, sorted=True).indices
		# The lowest two orbitals are populated.
		Energy = e[popd[0]]+e[popd[1]]
		q1=-1.0+v[popd[0],0]*v[popd[0],0]+v[popd[1],0]*v[popd[1],0]
		q2=-0.5+v[popd[0],1]*v[popd[0],1]+v[popd[1],1]*v[popd[1],1]
		q3=-0.5+v[popd[0],2]*v[popd[0],2]+v[popd[1],2]*v[popd[1],2]
		# compute the dipole moment.
		Dipole = (q1*x_pl[0]+q2*x_pl[1]+q3*x_pl[2])/3.0
		return Energy, Dipole, [q1,q2,q3]
	def Prepare(self):
		self.en0s = tf.constant([-1.1,-0.5,-0.5],dtype=tf.float64)
		self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.natom,3]))
		self.Energy,self.Dipole,self.Charges = self.HuckelBeH2(self.x_pl)
		self.Force =tf.gradients(-1.0*self.Energy,self.x_pl)
		init = tf.global_variables_initializer()
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		self.sess.run(init)
	def __call__(self,x_):
		"""
		Args:
			x_: the coordinates on which to evaluate the force.
		Returns:
			the Energy and Force (Eh and Eh/ang.) associated with the quadratic walls.
		"""
		e,f,d,q = self.sess.run([self.Energy,self.Force,self.Dipole,self.Charges], feed_dict = {self.x_pl:x_})
		#print("Min max and lat", np.min(x_), np.max(x_), lat_, e ,f)
		return e, f[0], d, q
