"""
For enhanced sampling of a PES near its minimum...
We could even use poorly trained networks for this.
"""
from SimpleMD import *
import ElectrostaticsTF

class MetaDynamics(VelocityVerlet):
	def __init__(self,f_,g0_,name_="MetaMD",EandF_=None):
		"""
		A trajectory which explores chemical space more rapidly
		by droppin' gaussians after a region has been explored for BumpTime
		This is an elementary version which just biases away
		based on a poissionian process.... Requires a thermostat.
		"""
		VelocityVerlet.__init__(self, f_, g0_, name_, EandF_)
		self.BumpHeight = 0.001 # Hartree.
		self.BumpWidth = 0.005 # Angstrom
		self.BumpTime = 15.0 # Fs
		self.MaxBumps = 600
		self.BumpCoords = np.zeros((self.MaxBumps,self.natoms,3))
		self.NBump = 0
		self.Tstat = NoseThermostat(self.m,self.v)

	def BumpForce(self,x_):
		BF = ElectrostaticsTF.TFBumpForce(self.BumpHeight,self.BumpWidth,self.BumpCoords[:self.NBump],x_)
		PF = self.ForceFunction(x_)
		print BF
		print PF
		tmp = PF+JOULEPERHARTREE*BF
		return tmp

	def Bump(self):
		self.BumpCoords[self.NBump] = self.x
		self.NBump += 1
		LOGGER.info("Bump!")
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
				self.x , self.v, self.a, self.EPot = VelocityVerletstep(self.BumpForce, self.a, self.x, self.v, self.m, self.dt, None)
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
