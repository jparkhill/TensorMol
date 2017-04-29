from Sets import *
from TFManage import *

def VelocityVerletstep(f_, a_, x_, v_, m_, dt_ ):
	""" A Velocity Verlet Step
	Args:
	f_: The force function
	a_: The acceleration at current step.
	x_: Current coordinates
	v_: Velocities
	m_: the mass vector.
	"""
	x = x_ + v_*dt_ + (1./2.)*a_*dt_*dt_

	a = np.einsum("ax,a->ax", f_(x), 1.0/m_)
	v = v_ + (1./2.)*(a_+a)*dt_
	return x,v,a

def KineticEnergy(v_, m_):
	""" The KineticEnergy
	Args:
	v_: Velocities
	m_: the mass vector.
	"""
	return (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_),m_)

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
			self.x , self.v, self.a = VelocityVerletstep(self.ForceFunction, self.a, self.x, self.v, self.m, self.dt)
			self.WriteTrajectory()
			step+=1
			LOGGER.info("Step: %i KE: %.5f ", step, self.KE)
		return
