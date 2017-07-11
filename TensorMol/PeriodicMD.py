"""
A periodic version of SimpleMD
No symmetry but general unit cells supported.
"""

from Sets import *
from TFManage import *
from Electrostatics import *
from QuasiNewtonTools import *
from SimpleMD import *

class PeriodicMol(Mol):
		def __init__(self, atoms_ =  np.zeros(1,dtype=np.uint8), coords_ = np.zeros(shape=(1,1),dtype=np.float),latvec_ = [[10.0,0.0,0.0],[0.0,10.0,0.0],[0.,0.,10.0]]):
			"""
			The convention here is that the molecule is initially moved to the center
			of the unit cell. Asteroids boundary conditions thereafter.
			"""
			Mol.__init__(self,atoms_,coords_)
			self.lattice = np.array(latvec_)
			self.latticeMetric = MatrixPower(self.lattice,-1/2.)
			self.latticeCenter = np.average(self.lattice,axis=0)
			self.coords = self.coords - self.Center() + self.latticeCenter
			self.tcoords = None

		def ModuloLattice(self,crds):
			"""
			Transports all coordinates into the primitive cell.

			Args:
				crds: a natom X 3 ndarray of atomic coordinates.
			Returns:
				crds modulo the primitive lattice.
			"""
			tmp = np.einsum("ij,ik->ik",self.latticeMetric,crds) # Lat X atom.
			fpart, ipart = np.fmod(tmp)
			return np.einsum("ij,ik->kj",self.lattice,fpart) # atom X coord

		def TessSelf(self,range = 5.0):
			"""
			Returns a copy of this mol, tesselated such that finite range forces
			can be directly evaluated for first N atoms, the second group of
			atoms (in tcoords) should not be included in the energy, they only affect the first.
			"""
			return

class PeriodicForce:
		def __init__(self,pm_,lat_):
			"""
			A periodic force evaluator base class.
			"""
			return
		def EnergyForce(self):
			"""
			Returns the Energy per unit cell and force on all primitive atoms
			"""
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
		pm_: the periodic moleule.
	Returns:
		x: updated positions
		v: updated Velocities
		a: updated accelerations
		e: Energy at midpoint per unit cell.
	"""
	x = x_ + v_*dt_ + (1./2.)*a_*dt_*dt_
	x = pf_.ModuloLattice(x)
	e, f_x_ = pf_.EnergyForce(x)
	a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/m_) # m^2/s^2 => A^2/Fs^2
	v = v_ + (1./2.)*(a_+a)*dt_
	return x,v,a,e

class PeriodicNoseThermostat(NoseThermostat):
	def __init__(self,m_,v_):
		NoseThermostat.__init__(m_,v_)
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
		x = pf_.ModuloLattice(x)
		vdto2 = v_ + (1./2.)*(a_ - self.eta*v_)*dt_
		e, f_x_ = pf_.EnergyForce(x)
		a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/m_) # m^2/s^2 => A^2/Fs^2
		ke = (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_),m_)
		etadto2 = self.eta + (dt_/(2.*self.Q))*(ke - (((3.*self.N+1)/2.))*self.kT)
		kedto2 = (1./2.)*np.dot(np.einsum("ia,ia->i",vdto2,vdto2),m_)
		self.eta = etadto2 + (dt_/(2.*self.Q))*(kedto2 - (((3.*self.N+1)/2.))*self.kT)
		v = (vdto2 + (dt_/2.)*a)/(1 + (dt_/2.)*self.eta)
		if frc_:
			return x,v,a,e,f_x_
		else:
			return x,v,a,e

class PeriodicVelocityVerlet(VelocityVerlet):
	def __init__(self, Force_, PMol_, name_ =""):
		"""
		Molecular dynamics

		Args:
			Force_: A PERIODIC energy, force CLASS.
			PMol_: initial PERIODIC molecule.
			PARAMS["MDMaxStep"]: Number of steps to take.
			PARAMS["MDTemp"]: Temperature to initialize or Thermostat to.
			PARAMS["MDdt"]: Timestep.
			PARAMS["MDV0"]: Sort of velocity initialization (None, or "Random")
			PARAMS["MDLogTrajectory"]: Write MD Trajectory.
		Returns:
			Nothing.
		"""
		self.Force = Force_
		VelocityVerlet.__init__(self, Force_.JustForce, g0_, name_, Force_.EnergyForce)
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
				self.x , self.v, self.a, self.EPot, self.force = self.Tstat.step(self.PForce, self.a, self.x, self.v, self.m, self.dt)

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
