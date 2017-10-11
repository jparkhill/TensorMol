"""
Periodic Monte Carlo.
"""

from __future__ import absolute_import
from __future__ import print_function
from .Sets import *
from .TFManage import *
from .Neighbors import *
from .Electrostatics import *
from .QuasiNewtonTools import *
from .Periodic import *
from .SimpleMD import *
from .PeriodicMD import *

class OnlineEstimator:
	"""
	Simple storage-less Knuth estimator which
	accumulates mean and variance.
	"""
	def __init__(x_):
		self.n = 1
		self.mean = x_*0.
		delta = x_ - self.mean
		self.mean += delta / n
		delta2 = x_ - self.mean
		self.m2 += delta * delta2
	def __call__(self, x_):
		self.n += 1
		delta = x_ - self.mean
		self.mean += delta / n
		delta2 = x_ - self.mean
		self.m2 += delta * delta2
		return self.mean, self.m2/(self.n-1)

class PeriodicMonteCarlo(PeriodicVelocityVerlet):
	def __init__(self, Force_, name_ ="PdicMC"):
		"""

		Args:
			Force_: A PERIODIC energy, force CLASS.
			PMol_: initial molecule.
		Returns:
			Nothing.
		"""
		PeriodicVelocityVerlet.__init__(self, Force_)
		e0, f0 = self.PForce(self.PForce.mol0.coords)
		self.eold = e0
		self.Estat = OnlineEstimator(e0)
		self.RDFold = self.PForce.RDF(self.PForce.mol0.coords)
		self.RDFstat = OnlineEstimator(self.RDFold)
		self.Xstat = OnlineEstimator(self.x)
		self.kbt = KAYBEETEE*(PARAMS["MDTemp"]/300.0) # Hartrees.
		self.Eav = None
		self.dE2 = None
		self.Xav = None
		self.dX2 = None
	def WriteTrajectory(self):
		m=Mol(self.atoms,self.x)
		m.properties["Lattice"]=self.PForce.lattice.lattice.copy()
		m.properties["Time"]=self.t
		m.properties["KineticEnergy"]=self.KE
		m.properties["PotEnergy"]=self.EPot
		m.WriteXYZfile("./results/", "MDTrajectory"+self.name,'a',True)
		return
	def MetropolisHastings(self,x_):
		# Generate a move.
		dx = np.random.normal(scale=0.005,size=self.x.shape())
		edx = self.PForce(x_+dx,DoForce=False)
		PMove = min(1.0,np.exp(-(edx - self.eold)/self.kbt))
		if (np.random.random()<PMove):
			self.x = x_ + dx
			self.eold = edx
			self.RDFold = self.PForce.RDF(self.x)
		self.Eav, self.dE2 = self.Estat(self.eold)
		self.Xav, self.dX2 = self.Xstat(self.x)
		return
	def Prop(self):
		"""
		Propagate Monte Carlo.
		MEEETROOOPPOLLIISSS
		"""
		step = 0
		self.md_log = np.zeros((self.maxstep, 7)) # time Dipoles Energy
		while(step < self.maxstep):
			self.t = step
			self.MetropolisHastings(self.x)
			rdf, rdf2 = self.RDFstat(self.RDFold)
			self.md_log[step,0] = self.t
			self.md_log[step,5] = self.EPot
			if (step%3==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			if (step%500==0):
				np.savetxt("./results/"+"MDLog"+self.name+".txt",self.md_log)
			step+=1
			LOGGER.info("Step: %i <E>(kJ/mol): %.5f <dE2>: %.5f <dX2>: %.5f Rho(g/cm**3): %.5f ", step, self.Eav, self.dE2, self.Xav, np.linalg.norm(self.dX2), self.Density())
			print(("per step cost:", time.time() -t ))
		return
