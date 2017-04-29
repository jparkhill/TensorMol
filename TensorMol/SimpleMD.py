"""
The Units chosen are Angstrom * Fs.
I convert the force outside from kcal/(mol angstrom) to Joules/(mol angstrom)
kcal/angstrom. = 4.184e+13 joules/meter  = 4184 joules/angstrom
Kb = 8.314 J/Mol K
"""

from Sets import *
from TFManage import *

def VelocityVerletstep(f_, a_, x_, v_, m_, dt_ ):
	""" A Velocity Verlet Step
	Args:
	f_: The force function (returns Joules/Angstrom)
	a_: The acceleration at current step. (A^2/fs^2)
	x_: Current coordinates (A)
	v_: Velocities (A/fs)
	m_: the mass vector. (kg)
	"""
	x = x_ + v_*dt_ + (1./2.)*a_*dt_*dt_
	a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_(x), 1.0/m_) # m^2/s^2 => A^2/Fs^2
	#print "dt", dt_ # fs
	#print "a", a
	#print "v", v_
	v = v_ + (1./2.)*(a_+a)*dt_
	return x,v,a

def KineticEnergy(v_, m_):
	""" The KineticEnergy
	Args:
		The masses are in kg.
		v_: Velocities (A/fs)
		m_: the mass vector. (kg/mol)
	Returns:
		The kinetic energy per atom (J/mol)
	"""
	return (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_)*pow(10.0,10.0),m_)/len(m_)

class Thermostat:
	def __init__(self,m_):
		"""
		Velocity Verlet step with a Rescaling Thermostat
		"""
		self.m = m_.copy()
		self.T = PARAMS["MDTemp"]  # Length of NH chain.
		self.kT = 8.314*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.tau = 30*PARAMS["MDdt"]
		return
	def VVstep(self,f_, a_, x_, v_, m_, dt_ ):
		x = x_ + v_*dt_ + (1./2.)*a_*dt_*dt_
		a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_(x), 1.0/m_) # m^2/s^2 => A^2/Fs^2
		v = v_ + (1./2.)*(a_+a)*dt_
		Teff = (2./3.)*KineticEnergy(v,self.m)/8.314
		v *= np.sqrt(self.T/Teff)
		return x,v,a

class NoseThermostat(Thermostat):
	def __init__(self,m_):
		"""
		Velocity Verlet step with a Nose-Hoover Thermostat.
		"""
		self.M = 12 # Depth of NH chain.
		self.m = m_.copy()
		self.N = len(m_)
		self.T = PARAMS["MDTemp"]  # Length of NH chain.
		self.kT = 8.314*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.tau = 30*PARAMS["MDdt"]
		self.Q = self.kT*self.tau*self.tau
		self.eta = 0.0
		self.Peta = 0.0
		self.Aeta = 0.0
	def VVstep(self,f_, a_, x_, v_, m_, dt_ ):
		"""
		http://www2.ph.ed.ac.uk/~dmarendu/MVP/MVP03.pdf
		"""
		x = x_ + v_*dt_ + (1./2.)*(a_ - self.eta*v_)*dt_*dt_
		vdto2 = v_ + (1./2.)*(a_ - self.eta*v_)*dt_
		a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_(x), 1.0/m_) # m^2/s^2 => A^2/Fs^2
		ke = (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_),m_)
		etadto2 = self.eta + (dt_/(2.*self.Q))*(ke - (((3.*self.N+1)/2.))*self.kT)
		kedto2 = (1./2.)*np.dot(np.einsum("ia,ia->i",vdto2,vdto2),m_)
		self.eta = etadto2 + (dt_/(2.*self.Q))*(ke - (((3.*self.N+1)/2.))*self.kT)
		v = (vdto2 + (dt_/2.)*a)/(1 + (dt_/2.)*self.eta)
		return x,v,a

class NoseChainThermostat(Thermostat):
	def __init__(self,x_,m_):
		"""
		Velocity Verlet step with a Nose-Hoover Thermostat.
		"""
		self.M = 12 # Depth of NH chain.
		self.T = PARAMS["MDTemp"]  # Length of NH chain.
		self.kT = 8.314*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.tau = 30*PARAMS["MDdt"]
		self.Qs = np.zeros(self.M) # Masses.
		self.eta = np.zeros(self.M) # Masses.
		self.Peta = np.zeros(self.M) # Masses.
		self.Aeta = np.zeros(self.M) # Masses.
	def MartynaQs(self):
		self.Qs = self.kT*self.tau*self.tau
		self.Qs[0] = 3.*len(x_)*self.kT*self.tau*self.tau
		return

class VelocityVerlet:
	def __init__(self,f_,g0_):
		"""
		Molecular dynamics
		Args:
			f_: a force routine
			m0_: initial molecule.
		Returns:
			A reaction path.
		"""
		self.maxstep = PARAMS["MDMaxStep"]
		self.T = PARAMS["MDTemp"]
		self.dt = PARAMS["MDdt"]
		self.ForceFunction = f_
		self.t = 0.0
		self.KE = 0.0
		self.atoms = g0_.atoms.copy()
		self.m = np.array(map(lambda x: ATOMICMASSES[x],self.atoms))
		self.natoms = len(self.atoms)
		self.x = g0_.coords.copy()
		self.v = np.zeros(self.x.shape)
		self.a = np.zeros(self.x.shape)
		return

	def WriteTrajectory(self):
		m=Mol(self.atoms,self.x)
		m.properties["Time"]=self.t
		m.properties["KineticEnergy"]=self.KE
		m.WriteXYZfile("./results/", "MDTrajectory")
		return

	def Prop(self):
		"""
		Propagate VelocityVerlet
		"""
		step = 0
		Tstat = None
		if (PARAMS["MDThermostat"]=="Rescaling"):
			Tstat = Thermostat(self.m)
		if (PARAMS["MDThermostat"]=="Nose"):
			Tstat = NoseThermostat(self.m)
		while(step < self.maxstep):
			self.t = step*self.dt
			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/8.314
			if (PARAMS["MDThermostat"]==None):
				self.x , self.v, self.a = VelocityVerletstep(self.ForceFunction, self.a, self.x, self.v, self.m, self.dt)
			else:
				self.x , self.v, self.a = Tstat.VVstep(self.ForceFunction, self.a, self.x, self.v, self.m, self.dt)
			self.WriteTrajectory()
			step+=1
			LOGGER.info("Step: %i time: %.1f(fs) <KE>(J): %.5f Teff(K): %.5f", step, self.t, self.KE,Teff)
		return
