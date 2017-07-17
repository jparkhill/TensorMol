"""
A periodic version of SimpleMD
No symmetry but general unit cells supported.

Maintenance of the unit cell, etc. are handled by PeriodicForce.
Only linear scaling forces with energy are supported.
"""

from Sets import *
from TFManage import *
from Neighbors import *
from Electrostatics import *
from QuasiNewtonTools import *
from SimpleMD import *

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
		return
	def CenteredInLattice(self, mol):
		return Mol(mol.atoms,self.ModuloLattice(mol.coords - mol.Center() + self.latticeCenter))
	def ModuloLattice(self, crds):
		"""
		Transports all coordinates into the primitive cell.

		Args:
			crds: a natom X 3 ndarray of atomic coordinates.
		Returns:
			crds modulo the primitive lattice.
		"""
		#print "CoordsL", crds
		tmp = np.einsum("ij,jk->ij", crds, self.latticeMetric ) # atom X lattice.
		fpart = np.fmod(tmp,1.0)
		revs=np.where(fpart < 0.0)
		fpart[revs] = 1.0 + fpart[revs]
		tore = np.einsum("ij,jk->ik", fpart, self.lattice) # atom X coord
		return tore

	def TessLattice(self, atoms_, coords_, rng_):
		"""
		Enlarges a molecule to allow for accurate calculation of a short-ranged force

		Args:
			mol_: a molecule.
			rng_: how much tesselation is required (Angstrom)
		returns
			An enlarged molecule where the real coordinates preceed 'fake' periodic images.
		"""
		if (rng_ > self.latticeMinDiameter):
			raise Exception("Enlarge Cell")
		natom = atoms_.shape[0]
		newAtoms = np.zeros(27*natom,dtype=np.uint8)
		newCoords = np.zeros((27*natom,3))
		newAtoms[:natom] = atoms_
		newCoords[:natom,:3] = coords_
		ind = 1
		for i in range(-1,2):
			for j in range(-1,2):
				for k in range(-1,2):
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
		self.LocalForces = []
		#self.LongForces = [] Everything is real-space courtesy of DSF.
		return
	def AddLocal(self, lf_,rng_):
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
		mxrng = max([f.range for f in self.LocalForces])
		NeedsTriples = any([f.NeedsTriples for f in self.LocalForces])
		# Tesselate atoms.
		z,x = self.lattice.TessLattice(self.atoms,x_, mxrng)
		# Construct NeighborList
		if (self.NL==None):
			self.NL = NeighborList(x,mxrng,NeedsTriples)
		self.NL.Update(x,mxrng,None,self.natomsReal)
		# Compute forces and energies.
		for f in self.LocalForces:
			einc, finc = f(z,x,self.NL.pairs)
			etore += np.sum(einc)
			ftore += finc[:self.natomsReal]
		return etore, ftore
	def Ewald(self):
		"""
		http://thynnine.github.io/pysic/coulombsummation%20class.html
		"""
		return

def PeriodicVelocityVerletStep(pf_, a_, x_, v_, m_, dt_):
	"""
	A Periodic Velocity Verlet Step (just modulo's the vectors.)

	Args:
		pf_: The PERIODIC force class (returns Joules/Angstrom)
		a_: The acceleration at current step. (A^2/fs^2)
		x_: Current coordinates (A)
		v_: Velocities (A/fs)
		m_: the mass vector. (kg)
	Returns:
		x: updated positions
		v: updated Velocities
		a: updated accelerations
		e: Energy at midpoint per unit cell.
	"""
	x = x_ + v_*dt_ + (1./2.)*a_*dt_*dt_
	x = pf_.lattice.ModuloLattice(x)
	e, f_x_ = pf_(x)
	a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/m_) # m^2/s^2 => A^2/Fs^2
	v = v_ + (1./2.)*(a_+a)*dt_
	return x,v,a,e

class PeriodicNoseThermostat(NoseThermostat):
	def __init__(self,m_,v_):
		NoseThermostat.__init__(self,m_,v_)
		return
	def step(self, pf_, a_, x_, v_, m_, dt_ ):
		"""
		A periodic Nose thermostat velocity verlet step
		http://www2.ph.ed.ac.uk/~dmarendu/MVP/MVP03.pdf

		Args:
			pf_: a Periodic force class.
			a_: acceleration
			x_: coordinates
			v_: velocities
			m_: masses
			dt_: timestep.
		"""
		# Recompute these stepwise in case of variable T.
		self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.tau = 20.0*PARAMS["MDdt"]*self.N
		self.Q = self.kT*self.tau*self.tau
		x = x_ + v_*dt_ + (1./2.)*(a_ - self.eta*v_)*dt_*dt_
		x = pf_.lattice.ModuloLattice(x)
		vdto2 = v_ + (1./2.)*(a_ - self.eta*v_)*dt_
		e, f_x_ = pf_(x)
		a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/m_) # m^2/s^2 => A^2/Fs^2
		ke = (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_),m_)
		etadto2 = self.eta + (dt_/(2.*self.Q))*(ke - (((3.*self.N+1)/2.))*self.kT)
		kedto2 = (1./2.)*np.dot(np.einsum("ia,ia->i",vdto2,vdto2),m_)
		self.eta = etadto2 + (dt_/(2.*self.Q))*(kedto2 - (((3.*self.N+1)/2.))*self.kT)
		v = (vdto2 + (dt_/2.)*a)/(1 + (dt_/2.)*self.eta)
		return x,v,a,e

class PeriodicVelocityVerlet(VelocityVerlet):
	def __init__(self, Force_, name_ ="PdicMD"):
		"""
		Molecular dynamics

		Args:
			Force_: A PERIODIC energy, force CLASS.
			PMol_: initial molecule.
			PARAMS["MDMaxStep"]: Number of steps to take.
			PARAMS["MDTemp"]: Temperature to initialize or Thermostat to.
			PARAMS["MDdt"]: Timestep.
			PARAMS["MDV0"]: Sort of velocity initialization (None, or "Random")
			PARAMS["MDLogTrajectory"]: Write MD Trajectory.
		Returns:
			Nothing.
		"""
		self.PForce = Force_
		VelocityVerlet.__init__(self, None, self.PForce.mol0, name_, self.PForce.__call__)
		if (PARAMS["MDThermostat"]=="Nose"):
			self.Tstat = PeriodicNoseThermostat(self.m,self.v)
		else:
			print "Unthermostated Periodic Velocity Verlet."
		return

	def Prop(self):
		"""
		Propagate VelocityVerlet
		"""
		step = 0
		self.md_log = np.zeros((self.maxstep, 7)) # time Dipoles Energy
		while(step < self.maxstep):
			t = time.time()
			self.t = step*self.dt
			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/IDEALGASR
			if (PARAMS["MDThermostat"]==None):
				self.x , self.v, self.a, self.EPot = PeriodicVelocityVerletStep(self.PForce, self.a, self.x, self.v, self.m, self.dt)
			else:
				self.x , self.v, self.a, self.EPot = self.Tstat.step(self.PForce, self.a, self.x, self.v, self.m, self.dt)
			self.md_log[step,0] = self.t
			self.md_log[step,4] = self.KE
			self.md_log[step,5] = self.EPot
			self.md_log[step,6] = self.KE+(self.EPot-self.EPot0)*JOULEPERHARTREE

			if (step%3==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			if (step%500==0):
				np.savetxt("./results/"+"MDLog"+self.name+".txt",self.md_log)

			step+=1
			LOGGER.info("Step: %i time: %.1f(fs) <KE>(kJ/mol): %.5f <|a|>(m/s2): %.5f <EPot>(Eh): %.5f <Etot>(kJ/mol): %.5f Teff(K): %.5f", step, self.t, self.KE/1000.0,  np.linalg.norm(self.a) , self.EPot, self.KE/1000.0+self.EPot*KJPERHARTREE, Teff)
			print ("per step cost:", time.time() -t )
		return
