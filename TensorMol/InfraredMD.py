"""
The Units chosen are Angstrom, Fs and units derived from those.
I convert the force outside from kcal/(mol angstrom) to Joules/(mol angstrom)

This IR trajectory is specially purposed for a force routine which
produces energy, forces and charges. It also does all the pre-optimization and stuff
automatically.

Usage:
IRProp(f,m,"TrajectoryName").Prop()
"""

from SimpleMD import *
from Opt import *

class IRProp(VelocityVerlet):
	def __init__(self,efq_,g0_,name_=str(0)):
		"""A specialized sort of dynamics which is appropriate for obtaining IR spectra at
		Zero temperature. Absorption cross section is given by:
		alpha = frac{4pi^2}{hbar c} omega (1 - Exp[-beta hbar omega]) sigma(omega))
		sigma  = frac{1}{6 pi} mathcal{F} {mu(t)mu(0)}

		Args:
			efq_: a function which yields the energy, force, and charge
			g0_: an initial geometry.
			PARAMS["MDAnnealTF"]: The temperature at which the IR will take place.
		"""
		self.EnergyAndForce = lambda x: efq_(x)[:2]
		self.EnergyForceCharge = lambda x: efq_(x)
		VelocityVerlet.__init__(self, lambda x: efq_(x)[0], g0_, name_, self.EnergyAndForce)
		self.Force = lambda x: efq_(x)[0]
		# Optimize the molecule.
		LOGGER.info("Preparing to do an IR. First Optimizing... ")
		G0Opt = GeomOptimizer(self.EnergyAndForce).Opt(g0_)
		# Anneal to Target Temperature.
		# Take the resulting x,v
		LOGGER.info("Now Annealing... ")
		anneal = Annealer(self.EnergyAndForce, None, G0Opt)
		anneal.Prop()
		self.x = anneal.x.copy()
		self.v = anneal.v.copy()
		self.EPot0 , self.f0, self.q0 = self.EnergyForceCharge(g0_.coords)
		self.EPot = self.EPot0
		self.qs = np.ones(self.m.shape)
		self.Mu0 = np.zeros(3)
		self.mu_his = None
		self.qs = self.q0.copy()
		self.Mu0 = Dipole(self.x, self.qs)
		print self.qs, self.Mu0
		# This can help in case you had a bad initial geometry
		self.MinS = 0
		self.MinE = 0.0
		self.Minx = None

	def VVStep(self):
		"""
		A Velocity Verlet Step

		Returns:
			x: updated positions
			v: updated Velocities
			a: updated accelerations
			e: Energy at midpoint.
		"""
		x = self.x + self.v*self.dt + (1./2.)*self.a*self.dt*self.dt
		e,f_x_, self.qs = self.EnergyForceCharge(x)
		a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/self.m) # m^2/s^2 => A^2/Fs^2
		v = self.v + (1./2.)*(self.a+a)*self.dt
		return x,v,a,e

	def WriteTrajectory(self):
		m=Mol(self.atoms,self.x)
		#m.properties["Time"]=self.t
		#m.properties["KineticEnergy"]=self.KE
		m.properties["Energy"]=self.EPot
		#m.properties["Charges"]=self.qs
		m.WriteXYZfile("./results/", "MDTrajectory"+self.name)
		return

	def Prop(self):
		self.mu_his = np.zeros((self.maxstep, 7)) # time Dipoles Energy
		vhis = np.zeros((self.maxstep,)+self.v.shape) # time Dipoles Energy
		step = 0
		while(step < self.maxstep):
			self.t = step*self.dt
			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/IDEALGASR

			self.Mu = Dipole(self.x, self.qs) - self.Mu0
			self.mu_his[step,0] = self.t
			self.mu_his[step,1:4] = self.Mu.copy()
			self.mu_his[step,4] = self.KE
			self.mu_his[step,5] = self.EPot
			self.mu_his[step,6] = self.KE+self.EPot
			vhis[step] = self.v.copy()

			self.x , self.v, self.a, self.EPot = self.VVStep() # Always Unthermostated
			if (step%50==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			step+=1
			if (step%1000==0):
				np.savetxt("./results/"+"MDLog"+self.name+".txt",self.mu_his)
				WriteDerDipoleCorrelationFunction(self.mu_his[:step],self.name+"MutMu0.txt")

			LOGGER.info("%s Step: %i time: %.1f(fs) <KE>(kJ): %.5f <PotE>(Eh): %.5f <ETot>(kJ/mol): %.5f Teff(K): %.5f Mu: (%f,%f,%f)", self.name, step, self.t, self.KE, self.EPot, self.KE/1000.0+(self.EPot-self.EPot0)*KJPERHARTREE, Teff, self.Mu[0], self.Mu[1], self.Mu[2])
		return
