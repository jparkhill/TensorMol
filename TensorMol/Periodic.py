"""
No symmetry but general unit cells supported.
Maintenance of the unit cell, etc. are handled by PeriodicForce.
Only linear scaling forces with energy are supported.
"""
from __future__ import absolute_import
from __future__ import print_function
from .Neighbors import *
from .Electrostatics import *
from .SimpleMD import *

class Lattice:
	def __init__(self, latvec_):
		"""
		Build a periodic lattice

		Args:
			latvec_: A 3x3 tensor of lattice vectors.
		"""
		self.lattice = latvec_.copy()
		self.latticeMetric = MatrixPower(np.array([[np.dot(self.lattice[i],self.lattice[j]) for j in range(3)] for i in range(3)]),-1/2.)
		self.latticeCenter = (self.lattice[0]+self.lattice[1]+self.lattice[2])/2.0
		self.latticeMinDiameter = 2.0*min([np.linalg.norm(self.lattice[0]-self.latticeCenter),np.linalg.norm(self.lattice[1]-self.latticeCenter),np.linalg.norm(self.lattice[2]-self.latticeCenter)])
		self.ntess = 1 # number of shells over which to tesselate.
		return
	def CenteredInLattice(self, mol):
		return Mol(mol.atoms,self.ModuloLattice(mol.coords - mol.Center() + self.latticeCenter))
	def InLat(self,crds):
		"""
		Express coordinates (atom X 3 cart)
		In units of the lattice vectors.
		"""
		latmet = MatrixPower(np.dot(self.lattice, self.lattice.T),-1)
		return 	np.dot(crds,np.dot(self.lattice.T,latmet))
	def FromLat(self,crds):
		"""
		Express coordinates (atom X 3 lat)
		In cartesian units.
		"""
		return np.dot(crds,self.lattice)
	def ModuloLattice(self, crds):
		"""
		Transports all coordinates into the primitive cell.

		Args:
			crds: a natom X 3 ndarray of atomic coordinates.
		Returns:
			crds modulo the primitive lattice.
		"""
		tmp = self.InLat(crds)
		fpart = np.fmod(tmp,1.0)
		revs=np.where(fpart < 0.0)
		fpart[revs] = 1.0 + fpart[revs]
		return self.FromLat(fpart)
	def TessNTimes(self, atoms_, coords_, ntess_):
		"""
		Enlarges a molecule to allow for accurate calculation of a short-ranged force

		Args:
			mol_: a molecule.
			rng_: minimum distance from center covered by the tesselation. (Angstrom)

		Returns:
			An enlarged molecule where the real coordinates preceed 'fake' periodic images.
		"""
		ntess = ntess_
		natom = atoms_.shape[0]
		nimages = pow(ntess_,3)
		#print("Doing",nimages,"images... of ",natom)
		newAtoms = np.zeros(nimages*natom,dtype=np.uint8)
		newCoords = np.zeros((nimages*natom,3))
		newAtoms[:natom] = atoms_
		newCoords[:natom,:3] = coords_
		ind = 1
		for i in range(ntess):
			for j in range(ntess):
				for k in range(ntess):
					if (i==0 and j==0 and k ==0):
						continue
					newAtoms[ind*natom:(ind+1)*natom] = atoms_
					newCoords[ind*natom:(ind+1)*natom,:] = coords_ + i*self.lattice[0] + j*self.lattice[1] + k*self.lattice[2]
					print(i,j,k,ind,nimages)
					ind = ind + 1
		print(newAtoms, newCoords.shape)
		return newAtoms, newCoords
	def TessLattice(self, atoms_, coords_, rng_):
		"""
		Enlarges a molecule to allow for accurate calculation of a short-ranged force

		Args:
			mol_: a molecule.
			rng_: minimum distance from center covered by the tesselation. (Angstrom)

		Returns:
			An enlarged molecule where the real coordinates preceed 'fake' periodic images.
		"""
		if (rng_ > self.latticeMinDiameter):
			self.ntess = int(rng_/self.latticeMinDiameter)+1
			#print("Doing",self.ntess,"tesselations...")
		natom = atoms_.shape[0]
		nimages = pow(2*self.ntess+1,3)
		#print("Doing",nimages,"images... of ",natom)
		newAtoms = np.zeros(nimages*natom,dtype=np.uint8)
		newCoords = np.zeros((nimages*natom,3))
		newAtoms[:natom] = atoms_
		newCoords[:natom,:3] = coords_
		ind = 1
		for i in range(-self.ntess,self.ntess+1):
			for j in range(-self.ntess,self.ntess+1):
				for k in range(-self.ntess,self.ntess+1):
					if (i==0 and j==0 and k ==0):
						continue
					newAtoms[ind*natom:(ind+1)*natom] = atoms_
					newCoords[ind*natom:(ind+1)*natom,:] = coords_ + i*self.lattice[0] + j*self.lattice[1] + k*self.lattice[2]
					ind = ind + 1
		return newAtoms, newCoords

class LocalForce:
	def __init__(self, f_, rng_=5.0, NeedsTriples_=False):
		self.range = rng_
		self.func=f_
		self.NeedsTriples = NeedsTriples_
		return
	def __call__(self, z, x, NZ):
		"""
		Generic call to a linear scaling local force.

		Args:
			z: atomic number vector
			x: atoms X 3 coordinate vector.
			NZ: pair or triples matrix. (NZP X 2)
		returns:
			energy number, and force vector with same shape as x.
		"""
		tmp = self.func(z, x, NZ)
		return tmp

class PeriodicForce:
	def __init__(self, pm_, lat_):
		"""
		A periodic force evaluator. The force consists of two components
		Short-Ranged forces, and long-ranged forces. Short ranged forces are
		evaluated by tesselation. Long-range forces are not supported yet.

		Args:
			pm_: a molecule.
			lat_: lattice vectors.
		"""
		self.lattice = Lattice(lat_)
		self.NL = None
		self.mol0 = self.lattice.CenteredInLattice(pm_)
		self.atoms = self.mol0.atoms.copy()
		self.natoms = self.mol0.NAtoms()
		self.natomsReal = pm_.NAtoms()
		self.maxrng = 0.0
		self.LocalForces = []
		self.lastx = np.zeros(pm_.coords.shape)
		self.nlthresh = 0.05 #per-atom Threshold for NL rebuild. (A)
		#self.LongForces = [] Everything is real-space courtesy of DSF.
		return
	def AdjustLattice(m, lat_):
		"""
		Adjusts the lattice and rescales the coordinates of m relative to previous lattice.
		"""
		il = self.lattice.InLat(m.coords)
		self.lattice = Lattice(lat_)
		m.coords = self.lattice.FromLat(il)
		return m
	def BindForce(self, lf_, rng_):
		"""
		Adds a local force to be computed when the PeriodicForce is called.

		Args:
			lf_: a function which takes z,x and returns atom energies, atom forces.
		"""
		self.LocalForces.append(LocalForce(lf_,rng_))
	def __call__(self,x_):
		"""
		Returns the Energy per unit cell and force on all primitive atoms

		Args:
			x_: a primitive geometry of atoms matching self.atoms.
		"""
		# Compute local energy.
		etore = 0.0
		ftore = np.zeros((self.natomsReal,3))
		if (self.maxrng == 0.0):
			self.maxrng = max([f.range for f in self.LocalForces])
		# Tesselate atoms.
		z,x = self.lattice.TessLattice(self.atoms,x_, self.maxrng)
		# Construct NeighborList determine if rebuild is necessary.
		if (self.NL==None):
			NeedsTriples = any([f.NeedsTriples for f in self.LocalForces])
			self.NL = NeighborList(x, DoTriples_=NeedsTriples, DoPerms_=False, ele_=None, alg_=None, sort_=False)
			self.NL.Update(x, self.maxrng, 0.0, None, self.natomsReal)
		rms = np.mean(np.linalg.norm(x_-self.lastx,axis=1))
		if (rms > self.nlthresh):
			self.NL.Update(x, self.maxrng, 0.0, None, self.natomsReal)
			self.lastx = x_.copy()
			#print("NL update",rms)
		else:
			print("No NL update",rms)
		#print(self.NL.pairs)
		# Compute forces and energies.
		for f in self.LocalForces:
			einc, finc = f(z,x,self.NL.pairs)
			etore += np.sum(einc)
			ftore += finc[:self.natomsReal]
		return etore, ftore
