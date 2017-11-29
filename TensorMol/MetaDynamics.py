"""
For enhanced sampling of a PES near its minimum...
We could even use poorly trained networks for this.
"""
from __future__ import absolute_import
from __future__ import print_function
from .SimpleMD import *
from . import ElectrostaticsTF
from . import TFForces
from . import TFMolInstanceDirect
from . import Statistics

class MetaDynamics(VelocityVerlet):
	def __init__(self,f_,g0_,name_="MetaMD",EandF_=None):
		"""
		A trajectory which explores chemical space more rapidly
		by droppin' gaussians after a region has been explored for BumpTime
		Requires a thermostat and currently uses Nose.

		Args:
			f_: A routine which returns the force.
			g0_: an initial molecule.
			name_: a name for output.
			EandF_: a routine returning the energy and the force.
			PARAMS["BowlK"] : a force constant of an attractive potential.
		"""
		VelocityVerlet.__init__(self, f_, g0_, name_, EandF_)
		self.BumpTime = PARAMS["MetaBumpTime"]
		self.MaxBumps = PARAMS["MetaMaxBumps"]
		self.bump_height = PARAMS["MetaMDBumpHeight"]
		self.bump_width = PARAMS["MetaMDBumpWidth"]
		self.BumpCoords = np.zeros((self.MaxBumps,self.natoms,3))
		self.NBump = 0
		self.DStat = OnlineEstimator(MolEmb.Make_DistMat(self.x))
		self.BowlK = PARAMS["MetaBowlK"]
		if (self.Tstat.name != "Andersen"):
			LOGGER.info("I really recommend you use Andersen Thermostat with Meta-Dynamics.")
		self.Bumper = TFForces.BumpHolder(self.natoms, self.MaxBumps, self.BowlK, self.bump_height, self.bump_width)#,"MR")

	def BumpForce(self,x_):
		BE = 0.0
		BF = np.zeros(x_.shape)
		if (self.NBump > 0):
			BE, BF = self.Bumper.Bump(self.BumpCoords.astype(np.float32), x_.astype(np.float32), self.NBump%self.MaxBumps)
		if (self.EnergyAndForce != None):
			self.RealPot, PF = self.EnergyAndForce(x_)
		else:
			PF = self.ForceFunction(x_)
		if self.NBump > 0:
			BF[0] *= self.m[:,None]
		print(JOULEPERHARTREE*BF[0],PF)
		PF += JOULEPERHARTREE*BF[0]
		PF = RemoveInvariantForce(x_,PF,self.m)
		return BE+self.RealPot, PF

	def Bump(self):
		self.BumpCoords[self.NBump%self.MaxBumps] = self.x
		self.NBump += 1
		LOGGER.info("Bump added!")
		return

	def Prop(self):
		"""
		Propagate VelocityVerlet
		"""
		step = 0
		bumptimer = self.BumpTime
		self.md_log = np.zeros((self.maxstep, 11)) # time Dipoles Energy
		while(step < self.maxstep):
			t = time.time()
			self.t = step*self.dt
			bumptimer -= self.dt
			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/IDEALGASR
			if (PARAMS["MDThermostat"]==None):
				self.x , self.v, self.a, self.EPot = VelocityVerletStep(self.BumpForce, self.a, self.x, self.v, self.m, self.dt, self.BumpForce)
			else:
				self.x , self.v, self.a, self.EPot, self.force = self.Tstat.step(self.BumpForce, self.a, self.x, self.v, self.m, self.dt, self.BumpForce)

			self.md_log[step,0] = self.t
			self.md_log[step,4] = self.KE
			self.md_log[step,5] = self.EPot
			self.md_log[step,6] = self.KE+(self.EPot-self.EPot0)*JOULEPERHARTREE
			self.md_log[step,7] = self.RealPot

			# Add an averager which accumulates RMS distance. Also, wow the width is small.
			Eav, Evar = self.EnergyStat(self.RealPot)
			Dav, Dvar = self.DStat(MolEmb.Make_DistMat(self.x))
			self.md_log[step,8] = Eav
			self.md_log[step,9] = Evar
			self.md_log[step,10] = np.linalg.norm(Dvar)

			if (bumptimer < 0.0):
				self.Bump()
				bumptimer = self.BumpTime

			if (step%3==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			if (step%500==0):
				np.savetxt("./results/"+"MDLog"+self.name+".txt",self.md_log)

			LOGGER.info("Step: %i time: %.1f(fs) KE(kJ/mol): %.5f <|a|>(m/s2): %.5f EPot(Eh): %.5f <EPot(Eh)>: %.5f Etot(kJ/mol): %.5f  <d(D_ij)^2>: %.5f Teff(K): %.5f",
					step, self.t, self.KE/1000.0,  np.linalg.norm(self.a) , self.EPot, Eav, self.md_log[step,6]/1000.0, self.md_log[step,10], Teff)
			print(("per step cost:", time.time() -t ))
			step+=1
		return

class BoxingDynamics(VelocityVerlet):
	def __init__(self,f_,g0_,name_="MetaMD",EandF_=None,BoxingLat0_=np.eye(3),BoxingLatp_=np.eye(3), BoxingT_ = 500.0):
		"""
		A Trajectory which crushes molecules into a box

		Args:
			f_: A routine which returns the force.
			g0_: an initial molecule.
			name_: a name for output.
			EandF_: a routine returning the energy and the force.
			BoxingLat0_:  Box to Start From
			BoxingLatp_": Box to force it into...
			BoxingT: Duration of boxing process (fs)
		"""
		VelocityVerlet.__init__(self, f_, g0_, name_, EandF_)
		self.BoxingLat0 = BoxingLat0_.copy()
		self.BoxingLatp = BoxingLatp_.copy()
		self.boxnow = self.BoxingLat0
		self.BoxingT = BoxingT_
		self.Tstat = NoseThermostat(self.m,self.v)
		self.Boxer = TFForces.BoxHolder(self.natoms)
	def BoxForce(self, x_ ):
		print("self.boxnow", self.boxnow)
		BE, BF = self.Boxer(x_, self.boxnow)
		print("Mass Vector", self.m[:,None])
		BF *= -500.0*JOULEPERHARTREE*(self.m[:,None]/np.sqrt(np.sum(self.m*self.m)))
		print("Bump Energy and Force: ",BE,BF)
		PE, PF = self.EnergyAndForce(x_)
		print("Pot Energy and Force: ",PE,PF)
		return BE+PE,PF+BF
	def Prop(self):
		"""
		Propagate VelocityVerlet

		mindistance_ is a cut off variable.  The box stops crushing if
		it has reached its minimum intermolecular distance.
		"""
		step = 0
		self.md_log = np.zeros((self.maxstep, 7)) # time Dipoles Energy
		while(step < self.maxstep): # || self.BoxingLatp_[0][0] < ?????current distance????):
			t = time.time()
			self.t = step*self.dt

			if (self.t>self.BoxingT):
				print("Exceeded Boxtime\n",self.BoxingLatp)
				self.boxnow = self.BoxingLatp.copy()
			else:
				self.boxnow = ((self.BoxingT-self.t)/(self.BoxingT))*self.BoxingLat0+(1.0-(self.BoxingT-self.t)/(self.BoxingT))*self.BoxingLatp

			print(self.boxnow)
			print(np.min(self.x),np.max(self.x))

			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/IDEALGASR
			self.x , self.v, self.a, self.EPot, self.force = self.Tstat.step(self.BoxForce, self.a, self.x, self.v, self.m, self.dt, self.BoxForce)

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

class BoxedMetaDynamics(VelocityVerlet):
	def __init__(self, EandF_, g0_, name_="MetaMD", Box_=np.array(10.0*np.eye(3))):
		VelocityVerlet.__init__(self, None, g0_, name_, EandF_)
		self.BumpTime = 12000000000000.0 # Fs
		self.MaxBumps = PARAMS["MetaMaxBumps"] # think you want this to be >500k
		self.BumpCoords = np.zeros((self.MaxBumps,self.natoms,3))
		self.NBump = 0
		self.Tstat = NoseThermostat(self.m,self.v)
		self.Boxer = TFForces.BoxHolder(self.natoms)
		self.Box = Box_.copy()
		self.BowlK = 0.0
		self.Bumper = TFForces.BumpHolder(self.natoms, self.MaxBumps, self.BowlK)
	def Bump(self):
		self.BumpCoords[self.NBump%self.MaxBumps] = self.x
		self.NBump += 1
		LOGGER.info("Bump added!")
		return
	def BoxForce(self, x_ ):
		BxE, BxF = self.Boxer(x_, self.Box)
		BxF *= -500.0*JOULEPERHARTREE*(self.m[:,None]/np.sqrt(np.sum(self.m*self.m)))
		PE, PF = self.EnergyAndForce(x_)
		BE = 0.0
		BF = np.zeros(x_.shape)
		if (self.NBump > 0):
			BE, BF = self.Bumper.Bump(self.BumpCoords.astype(np.float32), x_.astype(np.float32), self.NBump%self.MaxBumps)
			BF[0] *= self.m[:,None]
		return BxE+BE+PE,PF+BxF+JOULEPERHARTREE*BF[0]
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
				self.x , self.v, self.a, self.EPot = VelocityVerletStep(self.BoxForce, self.a, self.x, self.v, self.m, self.dt, self.BoxForce)
			else:
				self.x , self.v, self.a, self.EPot, self.force = self.Tstat.step(self.BoxForce, self.a, self.x, self.v, self.m, self.dt, self.BoxForce)

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
				np.savetxt("./results/"+"MDBoxLog"+self.name+".txt",self.md_log)

			step+=1
			LOGGER.info("Step: %i time: %.1f(fs) <KE>(kJ/mol): %.5f <|a|>(m/s2): %.5f <EPot>(Eh): %.5f <Etot>(kJ/mol): %.5f Teff(K): %.5f", step, self.t, self.KE/1000.0,  np.linalg.norm(self.a) , self.EPot, self.KE/1000.0+self.EPot*KJPERHARTREE, Teff)
			print(("per step cost:", time.time() -t ))
		return
