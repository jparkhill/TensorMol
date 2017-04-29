"""
The Units chosen are Angstrom * Fs.
But I convert the force outside from kcal/(mol angstrom) to Joules/(mol angstrom)
The f_ are in kcal/angstrom. = 4.184e+13 joules/meter  = 4184 joules/angstrom
Kb = 8.314 J/Mol K
"""

from Sets import *
from TFManage import *

def VelocityVerletstep(f_, a_, x_, v_, m_, dt_ ):
	""" A Velocity Verlet Step
	Args:
	f_: The force function (returns Joules/Angstrom)
	a_: The acceleration at current step. (A^2/fs^2)
	x_: Current coordinates
	v_: Velocities
	m_: the mass vector.
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
		while(step < self.maxstep):
			self.t = step*self.dt
			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/8.314
			self.x , self.v, self.a = VelocityVerletstep(self.ForceFunction, self.a, self.x, self.v, self.m, self.dt)
			self.WriteTrajectory()
			step+=1
			LOGGER.info("Step: %i time: %.5f(fs) <KE>(J): %.5f Teff(K): %.5f", step, self.t, self.KE,Teff)
		return
