"""
This is a dirty dirty hack which makes forces evaluable even when there is
no trained network for an atom type, etc. does QM/MM Type embeddings etc.
"""
from __future__ import absolute_import
from __future__ import print_function
from .Periodic import *

class Force:
	def __init__(self, rng_=1.0, NeedsTriples_=False):
		"""
		This is a dummy force, which basically nullify's an atom's contribution to the
		total energy.
		"""
		self.range = rng_
		self.NeedsTriples = NeedsTriples_
		return
	def __call__(self, z, x, NZ, DoForce = True):
		"""
		Generic call to a linear scaling local force.

		Args:
			z: atomic number vector
			x: atoms X 3 coordinate vector.
			NZ: pair or triples matrix. (NZP X 2)
		returns:
			energy number, and force vector with same shape as x.
		"""
		return np.zeros(x.shape)

class ForceAdaptor:
	def __init__(self, m):
		"""
		To use the ForceAdaptor you bind local forces to the accomodator with lists that
		specify what atom-types they can embed, evaluate forces on, and which atoms you'd like them
		to apply to. The force accomodator will break up the overall energy into these contributions.
		"""
		self.NL = None
		self.mol0 = self.lattice.CenteredInLattice(pm_)
		self.atoms = self.mol0.atoms.copy()
		self.natoms = self.mol0.NAtoms()
		self.natomsReal = pm_.NAtoms()
		self.maxrng = 0.0
		self.Forces = []
		self.ForcedAtoms=[] # Numforces list of atoms which experience the force.
		self.lastx = np.zeros(pm_.coords.shape)
		return
	def BindForce(self, lf_, fa_, rng_=15.0):
		"""
		Adds a local force to be computed when the PeriodicForce is called.

		Args:
			lf_: a function which takes z,x and returns atom energies, atom forces.
			rng_: the visibility radius of this force (A)
			fa_: Integer list of atoms in m to which this force applies.
		"""
		self.Forces.append(Force(lf_,rng_))
		self.ForcedAtoms.append(np.array(fa,dtype=np.uint32))
	def __call__(self,x_):
		"""
		Returns the Energy per unit cell and force on all atoms

		Args:
			x_: a primitive geometry of atoms matching self.atoms.
		"""
		etore = 0.0
		ftore = np.zeros((self.natomsReal,3))
		if (self.maxrng == 0.0):
			self.maxrng = max([f.range for f in self.LocalForces])
		for i in range(len(self.Forces)):
			# Compute all the atoms within the sensory radius
			# and all forced atoms, and increment the forces. 
