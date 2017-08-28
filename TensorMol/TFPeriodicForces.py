"""
	This provides raw inputs to sparse, linear scaling periodic forces.
	The lattice vectors are a part of this object and are thus differentiable.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from TensorMol.Periodic import *
from TensorMol.PeriodicTF import *
from TensorMol.TFMolInstanceDirect import *

class LinearVoxelBase:
	"""
	Base class which provides a linear scaling pair list using two voxel grids.
	First it divides space into cubes of size 2*rng then uses the quadratic alg.
	within each. There are two such grids offset by a half lattice, such
	that every point has a full range within a single voxel
	"""
	def __init__(self,rng_):
		"""
		Args:
			rng_: the range of the pairs (only unique)
		"""
		self.rng = rng_
		self.rngv = None
		self.xyz_pl = None
		self.x0 = None # The origin of the first voxel
		self.x0p = None # The origin of the first voxel in grid 2
		self.nvox = None # number in each direction
		self.Prepare()
		return

	def Prepare(self):
		with tf.Graph().as_default():
			self.rngv = tf.Variable(self.rng*1.414213,dtype = tf.float64)
			self.rngvec = tf.Variable([self.rngv,self.rngv,self.rngv],dtype = tf.float64)
			self.x0 = tf.Variable([0.,0.,0.],dtype = tf.float64)
			self.nvox = tf.Variable(0,dtype = tf.int32)
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def MakeVoxelList(self, x_, rng_):
		"""
		Given a monolithic set of coordinates x_
		This routine:
		- Chooses origin of voxel and # of voxels.
		- The number of voxels is 2x what is required to cover x_
		- Assigns the voxel indices owned by each atom (real and image)
		- Determines MaxNAtom within these voxels.
		- Nvox = NMol for a new NMol X MaxNAtom X 3 coordinate array.
		"""
		xmn = tf.min(x_)
		xmx = tf.max(x_)
		self.x0 = tf.ones(3)*xmn-2*self.rngvec
		self.x0p = tf.ones(3)*xmn-self.rngvec
		xp = tf.ones(3)*xmx+2*self.rngvec
		diam = tf.sqrt(tf.reduce_sum(xp-self.x0))
		self.nvox = tf.cast(diam/(2.0*self.rngv)+1
		#Determine the voxel each atom belongs to in each lattice.
		v1is = tf.floordiv(x_ - self.x0,2.0*self.rngv)
		v2is = tf.floordiv(x_ - self.x0p,2.0*self.rngv)
		#Determine which lattice is the 'real' lattice for each point
		reali = tf.where()
		return

class PeriodicLJ(LinearVoxelBase):
	def __init__(self, rcut_ = 18.0):
		"""
		Holds a periodic lennard-jones tensorflow Evaluator.
		And serves as a prototype for other simple differentiable
		periodic and linear-scaling pairwise forces. properly softens the
		force at it's cutoff.

		Args:
			rcut_: a cutoff radius (A) used for softness.
			lat_: in
		"""
		self.sess = None
		self.rcut = rcut_
		self.lat_pl = None
		self.tess_pl = None
		self.nzp_pl = None
		self.Prepare()
		return

	def Prepare(self):
		with tf.Graph().as_default():
			self.z_pl=tf.placeholder(tf.float64, shape=tuple([None,3]))
			self.xyz_pl=tf.placeholder(tf.float64, shape=tuple([None,3]))
			self.lat_pl=tf.placeholder(tf.float64, shape=tuple([3,3]))
			self.tess_pl=tf.placeholder(tf.float64, shape=tuple([None,3]))
			self.nzp_pl=tf.placeholder(tf.float64, shape=tuple([None,2]))
			self.Rc = tf.Variable(rcut,dtype = tf.float64)

			self.energy,self.force,self.stress = self.LJLinear(xyz_pl, lat_pl, tess_pl, nzp_pl)

			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def __call__(self, x_, lat_, tess_, nz_):
		"""
		Gives all the stuff needed to do variable cell dynamics.

		Args:
			x_: Coordinates within the unit cell.
			lat_: 3x3 matrix of lattice coords.
			tess_: List of non-identity tesselations to perform.
		Returns:
			returns the energy, force within the cell, and stress tensor.
		"""
		feeddict = {self.x_pl:x_, self.nzp_pl:nz_, self.lat_pl:lat_, self.tess_pl:tess_}
		En,Frc,Strs = self.sess.run([self.energy, self.force, self.stress],feed_dict=feeddict)
		return En, JOULEPERHARTREE*Frc[0], JOULEPERHARTREE*Strs[0] # Returns energies and forces and stresses

	def LJLinear(self, x_pl, lat_pl, tess_pl, nzp_pl):
		"""

		Args:
			x_pl: placeholder for the Natom X 3 tensor of x,y,z
			lat_pl: placeholder for the 3 lattice vectors
			tess_pl: placeholder for the ntess X 3 tesselations.
			nzp_pl: placeholder for the NMol X 3 tensor of nonzero pairs.

		Returns:
			The energy force and stress for the periodic system with the appropriate cutoff.
		"""
		tessx = PeriodicWrappingOfCoords(x_pl, lat_pl, tess_pl)
		x_shp = tf.shape(tessx)
		xpl = tf.reshape(x,[1,x_plshp[0],x_plshp[1]])
		Ens = LJEnergyLinear(xpl, nzp_pl)
		frcs = -1.0*(tf.gradients(Ens, xpl)[0])
		stress = -1.0*(tf.gradients(Ens, lat_pl)[0])
		return Ens, frcs, stress

	def LJEnergyLinear(self,XYZs,NZP):
		"""
		Linear Scaling Lennard-Jones Energy for a single Molecule.

		Args:
			Ds: Distances Enumerated by NZP (flat)
			NZP: a list of nonzero atom pairs NNZ X 2 = (i, j).
		Returns
			LJ energy.
		"""
		Ds = TFDistanceLinear(XYZs_[0,:,:], NZP)
		Cut = 0.5*(1.0-tf.erf(Ds - self.rcut))
		R = 1.0/Ds
		K = 0.2*Cut*(tf.pow(R,12.0)-2.0*tf.pow(R,6.0))
		K = tf.where(tf.is_nan(K),tf.zeros_like(K),K)
		K = tf.reduce_sum(K,axis=0)
		return K
