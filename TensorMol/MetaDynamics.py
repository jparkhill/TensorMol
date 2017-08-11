"""
For enhanced sampling of a PES near its minimum...
We could even use poorly trained networks for this.
"""
from SimpleMD import *
import ElectrostaticsTF
import TFMolInstanceDirect

class MetaDynamics(VelocityVerlet):
	def __init__(self,f_,g0_,name_="MetaMD",EandF_=None):
		"""
		A trajectory which explores chemical space more rapidly
		by droppin' gaussians after a region has been explored for BumpTime
		Requires a thermostat currently uses Nose.
		"""
		VelocityVerlet.__init__(self, f_, g0_, name_, EandF_)
		self.BumpTime = 4.0 # Fs
		self.MaxBumps = 2500
		self.BumpCoords = np.zeros((self.MaxBumps,self.natoms,3))
		self.NBump = 0
		self.Tstat = NoseThermostat(self.m,self.v)
		self.Bumper = TFMolInstanceDirect.BumpHolder(self.natoms, self.MaxBumps)

	def BumpForce(self,x_):
		BE = 0.0
		BF = np.zeros(x_.shape)
		if (self.NBump > 0):
			BE, BF = self.Bumper.Bump(self.BumpCoords.astype(np.float32), x_.astype(np.float32), self.NBump)
		PF = self.ForceFunction(x_)
		if self.NBump > 0:
			BF[0] *= self.m[:,None]
		tmp = PF+JOULEPERHARTREE*BF[0]
		return tmp

	def Bump(self):
		if (self.NBump == self.MaxBumps):
			return
		self.BumpCoords[self.NBump] = self.x
		self.NBump += 1
		LOGGER.info("Bump added!")
		return

	def Prop(self):
		"""
		Propagate VelocityVerlet
		"""
		step = 0
		bumptimer = self.BumpTime
		self.md_log = np.zeros((self.maxstep, 7)) # time Dipoles Energy
		while(step < self.maxstep):
			t = time.time()
			self.t = step*self.dt
			bumptimer -= self.dt
			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/IDEALGASR
			if (PARAMS["MDThermostat"]==None):
				self.x , self.v, self.a, self.EPot = VelocityVerletStep(self.BumpForce, self.a, self.x, self.v, self.m, self.dt, None)
			else:
				self.x , self.v, self.a, self.EPot, self.force = self.Tstat.step(self.BumpForce, self.a, self.x, self.v, self.m, self.dt, None)

			self.md_log[step,0] = self.t
			self.md_log[step,4] = self.KE
			self.md_log[step,5] = self.EPot
			self.md_log[step,6] = self.KE+(self.EPot-self.EPot0)*JOULEPERHARTREE

			if (bumptimer < 0.0):
				self.Bump()
				bumptimer = self.BumpTime

			if (step%3==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			if (step%500==0):
				np.savetxt("./results/"+"MDLog"+self.name+".txt",self.md_log)

			step+=1
			LOGGER.info("Step: %i time: %.1f(fs) <KE>(kJ/mol): %.5f <|a|>(m/s2): %.5f <EPot>(Eh): %.5f <Etot>(kJ/mol): %.5f Teff(K): %.5f", step, self.t, self.KE/1000.0,  np.linalg.norm(self.a) , self.EPot, self.KE/1000.0+self.EPot*KJPERHARTREE, Teff)
			print ("per step cost:", time.time() -t )
		return
