"""
A United version of SimpleMD
ie: evaluate returns energy, force, charges.
also evaluation requires linear scaling neighbor lists which are managed by class LocalForceField

This is a partial re-write of SimpleMD.py, and so some of the functions there
are repeated here for the sake of independence. Eventually SimpleMD might be depreciated.
"""

from __future__ import absolute_import
from __future__ import print_function
from .Sets import *
from .TFManage import *
from .Neighbors import *
from .Electrostatics import *
from .QuasiNewtonTools import *
from .SimpleMD import *

class LocalForceField:
	"""
	Maps x => Energy, Force, Charge in linear time.
	Handles whatever neighborlist updates might be needed.
	"""
	def __init__(self, f_, rng_=5.0, NeedsTriples_=False):
		"""
		Args:
			f_: a routine taking the arguments x
		"""
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

class UnitedVelocityVerlet():
	def __init__(self, Force_, Mol_, name_ ="UnitedMD"):
		"""
		Molecular dynamics

		Args:
			Force_: a LocalForce object
			Mol_: initial molecule.
			PARAMS["MDMaxStep"]: Number of steps to take.
			PARAMS["MDTemp"]: Temperature to initialize or Thermostat to.
			PARAMS["MDdt"]: Timestep.
			PARAMS["MDV0"]: Sort of velocity initialization (None, or "Random")
			PARAMS["MDLogTrajectory"]: Write MD Trajectory.
		Returns:
			Nothing.
		"""
		self.Force = Force_
		VelocityVerlet.__init__(self, None, self.PForce.mol0, name_, self.Force.__call__)
		if (PARAMS["MDThermostat"]=="Nose"):
			self.Tstat = PeriodicNoseThermostat(self.m,self.v)
		else:
			print("Unthermostated Periodic Velocity Verlet.")
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
			print(("per step cost:", time.time() -t ))
		return
