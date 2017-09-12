"""
	This file provides raw sparse, linear scaling periodic forces.

	This is made in two parts, the first part is LinearVoxelBase
	which is a linear scaling pair list using cubic voxels.

	the second part is a periodic force evaluator which handles tesselation and
	force evaluation. It can use LinearVoxelBase to generate pair lists, and
	voxel-decomposed coordinate tensors (nvox X maxnatom X 4)

	Derived classes can inherit both these functionalities.
	The lattice vectors are a part of the periodic object and are thus differentiable.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from TensorMol.Periodic import *
from TensorMol.PeriodicTF import *
from TensorMol.TFMolInstanceDirect import *

class LinearVoxelBase:
	"""
	Base class which provides a linear scaling voxel decompoosition and pair list. There is a single set of overlapping
	voxels such that every point lies within the sub-cube with a skin of depth rng. This routine can also create
	pair lists for sparse algorithms. First it divides space into cubes then uses the quadratic alg. within each.
	"""
	def __init__(self,rng_):
		"""
		Args:
		    rng_: the range of the pairs (only unique)
		"""
		self.rng = rng_
		self.xv_pl = None
		self.x0 = None # The origin of the first voxel
		self.dx = rng_ # the offset between voxels (stride) which is equal to the core length
		self.dl = rng_ # length to core from surface. The voxel has side-length 2dl+dx
		self.dxv = None # the offset between voxels (stride) in one cartesian direction.
		self.nvox = None # number in each direction determined by dx
		self.sess = None
		self.VL = None # Computes List of voxels.
		self.g = tf.Graph()
		return
	def Prepare(self):
		with self.g.as_default():
			self.dx = tf.constant(self.rng,dtype = tf.float64) # Stride
			self.dl = tf.constant(self.rng,dtype = tf.float64)
			self.dxv = tf.Variable([self.rng,self.rng,self.rng],dtype = tf.float64)
			self.x0 = tf.Variable([0.,0.,0.],dtype = tf.float64)
			self.nvox = tf.Variable(0, dtype = tf.int64)
			self.xv_pl = tf.placeholder(tf.float64, shape=(None,3), name="LVBx_pl")
			self.VL = self.VoxelList(self.xv_pl)
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return
	def TFRaster3(self):
		with self.g.as_default():
			rngs1 = tf.tile(tf.range(3,dtype=tf.int64)[:,tf.newaxis,tf.newaxis],[1,3,3])
			rngs2 = tf.tile(tf.range(3,dtype=tf.int64)[tf.newaxis,:,tf.newaxis],[3,1,3])
			rngs3 = tf.tile(tf.range(3,dtype=tf.int64)[tf.newaxis,tf.newaxis,:],[3,3,1])
			rngs = tf.reshape(tf.transpose(tf.stack([rngs1,rngs2,rngs3]),perm=[1,2,3,0]),[3*3*3,3])
			return rngs
	def TFRasterN(self,n_):
		with self.g.as_default():
			n=tf.reshape(tf.cast(n_,dtype=tf.int32),())
			rngs1 = tf.tile(tf.range(n)[:,tf.newaxis,tf.newaxis],[1,n,n])
			rngs2 = tf.tile(tf.range(n)[tf.newaxis,:,tf.newaxis],[n,1,n])
			rngs3 = tf.tile(tf.range(n)[tf.newaxis,tf.newaxis,:],[n,n,1])
			rngs = tf.reshape(tf.transpose(tf.stack([rngs1,rngs2,rngs3]),perm=[1,2,3,0]),[n*n*n,3])
			return rngs
	def VoxelIndices(self,pts_,x0_,dx_,nvox_):
		"""
		Returns the xyz core indices of a voxel containing pts_ within their core
		"""
		with self.g.as_default():
			return tf.floordiv(pts_ - x0_ - dx_, dx_)
	def VoxelCenter(self,ind):
		"""
		Returns the center of a Voxel.
		"""
		with self.g.as_default():
			return self.x0 + self.dl + (ind+0.5)*(self.dxv)
	def VoxelCenters(self,ind,x0,dxv):
		"""
		Returns the center of a Voxel.
		"""
		with self.g.as_default():
			tmp1 = x0[tf.newaxis,:]
			tmp2 = dxv[tf.newaxis,:]
			tmp4 = 0.5*tf.ones((1,3),dtype=tf.float64)
			tmp3 = tf.cast(ind,tf.float64)
			return tmp1+tmp2+(tmp3+tmp4)*(self.dxv[tf.newaxis,:])
	def PointsWithin(self, pts_, VoxInd_, rng_):
		"""
		Returns indices of pts_ which are within the core rng of VoxInd_
		"""
		with self.g.as_default():
			npts = tf.shape(pts_)[0]
			return vis
	def IndicesToRasters(self, inds_, x0_, nvox_):
		"""
		Convert a list of voxel index triples into
		raster indices.
		"""
		with self.g.as_default():
			nvox = tf.cast(nvox_,dtype=tf.int64)
			inds = tf.cast(inds_,tf.int64)
			tmp=inds[:,:,0]*nvox*nvox+inds[:,:,1]*nvox+inds[:,:,2]
			return tmp
	def RastersToIndices(self, r_):
		with self.g.as_default():
			ind0 = tf.floordiv(r_,self.nvox*self.nvox)
			tmp0 = r_ - ind0*self.nvox*self.nvox
			ind1 = tf.floordiv(tmp0,self.nvox)
			ind2 = tmp0 - ind1*self.nvox
			return tf.concat([ind0[:,tf.newaxis],ind1[:,tf.newaxis],ind2[:,tf.newaxis]],axis = 1)
	def MemberOfVoxels(self, vis_, x0_, dx_, nvox_):
		"""
		Returns the raster indices of the 27 voxels each point belongs to.
		The 13th is always the core of the voxel.

		Args:
		 vis_: index triples of the core voxel.
		Returns:
		 the raster indices of the surrounding 27 voxels this point can belong to.
		"""
		with self.g.as_default():
			coreis = vis_[:,tf.newaxis,:]
			toadd = self.TFRaster3()[tf.newaxis,:,:]-tf.ones(shape=[27,3],dtype=tf.int64)[tf.newaxis,:,:]
			carts = tf.cast(coreis,tf.int64)+toadd
			return self.IndicesToRasters(carts, x0_, nvox_)
	def VoxelList(self, x_):
		"""
		Given a monolithic set of coordinates x_
		This routine:
		- Chooses origin of voxel and # of voxels.
		- The number of voxels is 2x what is required to cover x_
		- Assigns the voxel indices owned by each atom (real)

		Args:
		 x_: a nptsX3 coordinate array.

		Returns:
		 voxel origin, nvox, voxel centers, and voxel membership. npts X 27
		"""
		with self.g.as_default():
			xmn = tf.cast(tf.reduce_min(x_),tf.float64)
			xmx = tf.cast(tf.reduce_max(x_),tf.float64)
			x0 = tf.ones(3,dtype=tf.float64)*xmn-2.0*self.dxv
			xp = tf.ones(3,dtype=tf.float64)*xmx+2.0*self.dxv
			diam = tf.reduce_max(xp-self.x0)
			nvox = tf.round(diam/(self.dx))+1
			# Determine the voxel each atom belongs to (core and tesselated)
			vis = self.VoxelIndices(x_,x0,self.dx,nvox)
			return x0, nvox, self.MemberOfVoxels(vis,x0,self.dx,nvox)
			# Below can be used to test this is working, which it is.
			#return xmn,xmx,x0,nvox,self.VoxelCenters(vis,x0,self.dxv),self.MemberOfVoxels(vis,x0,self.dx,nvox), self.VoxelCenters(self.TFRasterN(nvox),x0,self.dxv), vis
	def MakeVoxelList(self,x_):
		print(np.min(x_),np.max(x_))
		tmp = self.sess.run(self.VL, feed_dict={self.xv_pl:x_} )
		return tmp
	def MakeVoxelMols(self,z_,x_):
		"""
		This routine makes a batch of molecules which can be used to
		form energies in a linear scaling way.
		It also assigns NNZ for each molecule and maxnatom.
		Ie to form an energy with linear scaling you form the atomwise energies
		these mols, and only sum up the energy up-to atom nreal.

		Args:
		  x_, z_ : a coordinate and atomic number tensor.

		Returns:
		  mol batches of atoms, coords, NNZ and CoreReal (number of energetic atoms in this mol)
		"""
		x0,nvox_,voxs = self.MakeVoxelList(x_)
		print("nvox, Voxel assignments: ", nvox_, voxs)
		nvox = int(nvox_)
		nvox3 = nvox*nvox*nvox
		fvox = voxs.flatten()
		mxvx = np.max(fvox)
		mnvx = np.min(fvox)
		membercounts = np.bincount(fvox)
		core = voxs[:,13]
		corec = np.bincount(core)
		#        print corec, np.sum(corec)
		vxkey = np.zeros(nvox3,dtype=np.int)
		vxkey -= 1
		nnzvx = 0
		# Determine the number of nonzero voxels (which contain real atoms)
		for i in range(corec.shape[0]):
			if corec[i] > 0:
				vxkey[i] = nnzvx
				nnzvx += 1
		maxnatom = np.max(membercounts)
		minnatom = np.min(membercounts)
		filling = np.zeros(nnzvx,dtype=np.int)
		coords = np.zeros((nnzvx,maxnatom,3))
		atoms = np.zeros((nnzvx,maxnatom,1),dtype=np.uint8)
		absindex = np.zeros((nnzvx,maxnatom,1),dtype=np.uint8)
		RealOrImage = np.zeros((nnzvx,maxnatom,1),dtype=np.uint8)
		for i in range(x_.shape[0]):
			for k,vx in enumerate(voxs[i]):
				vk = vxkey[vx]
				if (k==13):
				    RealOrImage[vk,filling[vk],0] += 1
				if (vk >= 0):
					coords[vk,filling[vk],:] = x_[i]
					atoms[vk,filling[vk],:] = z_[i]
					absindex[vk,filling[vk],:] = i
					filling[vk] += 1
		return atoms, coords, filling, RealOrImage, absindex

class TFPeriodicLocalForce(LinearVoxelBase):
	def __init__(self , rcut_ = 16.0, lat_ = np.array(np.eye(3))):
		"""
		Base periodic local tensorflow evaluator.
		And serves as a prototype for other simple differentiable
		periodic forces. properly softens the
		force at it's cutoff. Check out PeriodicTF.py

		One key option is self.StressDifferentiable.
		If that's true, then the stress tensor can be calculated
		at the cost of a much more complex computational graph.
		Because the formation of the voxel molecules must be done
		within tensorflow.

		Args:
			rcut_: a cutoff radius (A) used for softness.
			lat_: in
		"""
		# Determine the required tesselations to perform.
		self.StressDifferentiable = False
		self.lat = lat_.copy()
		self.ntess = 2
		tessrng = range(-self.ntess,self.ntess+1)
		self.tess = np.array([[[[i,j,k] for k in tessrng] for j in tessrng] for i in tessrng]).reshape((pow(len(tessrng),3),3))
		LinearVoxelBase.__init__(self, rcut_)
		self.rcut = rcut_
		# Placeholders....
		self.x_pl = None
		self.z_pl = None
		self.lat_pl = None
		self.tess_pl = None
		# Molwise-voxel batch placeholders.
		xs_pl = None
		zs_pl = None
		# Graphs of desired outputs produced by __call__
		self.energy = None
		self.force = None
		self.stress = None
		# Graphs of desired outputs produced by __call__
		self.PW = None
		self.energies = None
		self.forces = None
		self.stresses = None
		self.Prepare()
		return

	def Prepare(self):
		LinearVoxelBase.Prepare(self)
		with self.g.as_default():
			self.z_pl = tf.placeholder(tf.float64, shape=tuple([None]))
			self.x_pl = tf.placeholder(tf.float64, shape=tuple([None,3]))
			self.lat_pl = tf.placeholder(tf.float64, shape=tuple([3,3]))
			self.tess_pl = tf.placeholder(tf.float64, shape=tuple([None,3]))
			self.xs_pl = tf.placeholder(tf.float64, shape=tuple([None, None, 3]))
			self.zs_pl = tf.placeholder(tf.float64, shape=tuple([None, None, 1]))
			self.Rc = tf.Variable(self.rcut,dtype = tf.float64)
			self.PW = self.PeriodicWrapping(self.z_pl, self.x_pl, self.lat_pl, self.tess_pl)
			self.energies = self.LJEnergies(self.xs_pl)
			self.forces = tf.gradients(self.energies,self.xs_pl)[0]
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def PeriodicWrapping(self, z_, xyz_, lat_, tess_):
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
		with self.g.as_default():
			ntess = tf.shape(tess_)[0]
			i0 = tf.zeros(1,tf.int32)[0]
			natold = tf.shape(z_)[0]
			natnew = tf.shape(z_)*ntess
			outz = tf.TensorArray(tf.float64, size=ntess)
			outx = tf.TensorArray(tf.float64, size=ntess)
			cond = lambda i,z,x: tf.less(i,ntess)
			body = lambda i,z,x: [i+1, outz.write(i,z_) ,outx.write(i,xyz_+tess_[i,0]*lat_[0]+tess_[i,1]*lat_[1]+tess_[i,2]*lat_[2])]
			inital = (0,outz,outx)
			i,zr,xr = tf.while_loop(cond,body,inital)
			return zr.concat(),xr.concat()

	def __call__(self, z_, x_, lat_):
		"""
		This doesn't support direct calculation of
		stress tensor derivatives yet... It's really
		annoying to put the batch creation into tensorflow

		Args:
			z_: atoms within the unit cell. (numpy array)
			x_: Coordinates within the unit cell.
			lat_: 3x3 matrix of lattice coords.
			tess_: List of non-identity tesselations to perform.

		Returns:
			returns the energy, force within the cell, and stress tensor.
		"""
		# Step 1: Tesselate the cell.
		print ("Tess?",self.tess)
		feeddict = {self.z_pl:z_, self.x_pl:x_, self.lat_pl: lat_, self.tess_pl:self.tess}
		zp, xp = self.sess.run(self.PW,feed_dict=feeddict)
		print("z,xp.shape after tess: ", zp.shape, xp.shape)
		# Step 2: Make a voxel batch out of that.
		ats, crds, fill, roi, inds = self.MakeVoxelMols(zp,xp)
		print("ats.shape after voxelization: ", crds.shape)
		# Step 3: evaluate the force using quadratic algs. over the voxels.
		feeddict = {self.zs_pl:ats, self.xs_pl:crds}
		# This routine returns partial forces partitioned over atoms.
		Ens,Frc,Strs = self.sess.run([self.energies, self.forces, self.stresses],feed_dict=feeddict)
		# Grab back out the desired paritial energy and force due to the real atoms in the unit cell.
		nrealat = x_.shape[0]
		FrcToRe = np.zeros(x_.shape)
		EnToRe = 0.0
		StrsToRe = np.zeros(3,3)
		for i in range(crds.shape):
			if (inds[i]<nrealat and roi[i] ==1):
				FrcToRe[inds[i]] = Frc[i]
				EnToRe += Ens[i]
		return En, JOULEPERHARTREE*Frc[0], JOULEPERHARTREE*Strs[0] # Returns energies and forces and stresses

	def LJEnergies(self,XYZs_):
		"""
		Returns LJ Energies batched over molecules.
		Input can be padded with zeros. That will be
		removed by LJKernels.

		Args:
			XYZs_: nmols X maxatom X 3 coordinate tensor.
		Returns:
			A vector of partial energies for each atom given as an argument.
		"""
		Ds = TFDistances(XYZs_)
		Ds = tf.where(tf.is_nan(Ds), tf.zeros_like(Ds), Ds)
		ones = tf.ones(tf.shape(Ds),dtype = tf.float64)
		zeros = tf.zeros(tf.shape(Ds),dtype = tf.float64)
		ZeroTensor = tf.where(tf.less_equal(Ds,0.000000001),ones,zeros)
		Ds += ZeroTensor
		R = 0.8/Ds
		Ks = 0.5*(tf.pow(R,12.0)-2.0*tf.pow(R,6.0))
		# Use the ZeroTensors to mask the output for zero dist or AN.
		Ks = tf.where(tf.equal(ZeroTensor,1.0),tf.zeros_like(Ks),Ks)
		Ks = tf.where(tf.is_nan(Ks),tf.zeros_like(Ks),Ks)
		Ks = tf.matrix_band_part(Ks, 0, -1)
		Ens = tf.reduce_sum(Ks,[2])
		return Ens

class PeriodicLJ(TFPeriodicLocalForce):
	def __init__(self, rcut_ = 18.0):
		"""
		Holds a periodic lennard-jones tensorflow Evaluator.
		This version tries to make the stress tensor analytically differentiable.
		And serves as a prototype for other simple differentiable
		periodic forces. properly softens the force at it's cutoff.

		There are two ingredients:

		This relies on LinearVoxelBase to divide a system into padded voxels.
		In linear time. Then dense routines are used on each 'molecule' = 'voxel'

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
		tessx = self.PeriodicWrapping(tf.zeros((tf.shape(x_pl)[0],1)),x_pl, lat_pl, tess_pl)
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
