"""
The Units chosen are Angstrom, Fs.
I convert the force outside from kcal/(mol angstrom) to Joules/(mol angstrom)
"""

from __future__ import absolute_import
from __future__ import print_function
from .Sets import *
from .TFManage import *
from .Electrostatics import *
from .QuasiNewtonTools import *

def VelocityVerletStep(f_, a_, x_, v_, m_, dt_, fande_=None):
	"""
	A Velocity Verlet Step

	Args:
		f_: The force function (returns Joules/Angstrom)
		a_: The acceleration at current step. (A^2/fs^2)
		x_: Current coordinates (A)
		v_: Velocities (A/fs)
		m_: the mass vector. (kg)
	Returns:
		x: updated positions
		v: updated Velocities
		a: updated accelerations
		e: Energy at midpoint.
	"""
	x = x_ + v_*dt_ + (1./2.)*a_*dt_*dt_
	e, f_x_ = 0.0, None
	if (fande_==None):
		f_x_ = f_(x)
	else:
		e, f_x_ = fande_(x)
	a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/m_) # m^2/s^2 => A^2/Fs^2
	v = v_ + (1./2.)*(a_+a)*dt_
	return x,v,a,e

def KineticEnergy(v_, m_):
	"""
	The KineticEnergy

	Args:
		The masses are in kg.
		v_: Velocities (A/fs)
		m_: the mass vector. (kg/mol)
	Returns:
		The kinetic energy per atom (kJ/mol)
	"""
	return (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_)*pow(10.0,10.0),m_)/len(m_)

class Thermostat:
	def __init__(self,m_,v_):
		"""
		Velocity Verlet step with a Rescaling Thermostat
		"""
		self.N = len(m_)
		self.m = m_.copy()
		self.T = PARAMS["MDTemp"]
		self.Teff = 0.001
		self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.tau = 30*PARAMS["MDdt"]
		self.name = "Rescaling"
		print("Using ", self.name, " thermostat at ",self.T, " degrees Kelvin")
		self.Rescale(v_)
		return

	def step(self,f_, a_, x_, v_, m_, dt_ , fande_=None):
		x = x_ + v_*dt_ + (1./2.)*a_*dt_*dt_
		e, f_x_ = 0.0, None
		if (fande_==None):
			f_x_ = f_(x)
		else:
			e, f_x_ = fande_(x)
		a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/m_) # m^2/s^2 => A^2/Fs^2
		v = v_ + (1./2.)*(a_+a)*dt_
		self.Teff = (2./3.)*KineticEnergy(v,self.m)/IDEALGASR
		v *= np.sqrt(self.T/self.Teff)
		return x,v,a,e

	def Rescale(self,v_):
		# Do this elementwise otherwise H's blow off.
		for i in range(self.N):
			Teff = (2.0/(3.0*IDEALGASR))*pow(10.0,10.0)*(1./2.)*self.m[i]*np.einsum("i,i",v_[i],v_[i])
			if (Teff != 0.0):
				v_[i] *= np.sqrt(self.T/(Teff))
		return

class NoseThermostat(Thermostat):
	def __init__(self,m_,v_):
		"""
		Velocity Verlet step with a Nose-Hoover Thermostat.
		"""
		self.m = m_.copy()
		self.N = len(m_)
		self.T = PARAMS["MDTemp"]  # Length of NH chain.
		self.eta = 0.0
		self.name = "Nose"
		self.Rescale(v_)
		print("Using ", self.name, " thermostat at ",self.T, " degrees Kelvin")
		return

	def step(self,f_, a_, x_, v_, m_, dt_ , fande_=None, frc_ = True):
		"""
		http://www2.ph.ed.ac.uk/~dmarendu/MVP/MVP03.pdf
		"""
		# Recompute these stepwise in case of variable T.
		self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.tau = 20.0*PARAMS["MDdt"]*self.N
		self.Q = self.kT*self.tau*self.tau

		x = x_ + v_*dt_ + (1./2.)*(a_ - self.eta*v_)*dt_*dt_
		vdto2 = v_ + (1./2.)*(a_ - self.eta*v_)*dt_
		e, f_x_ = 0.0, None
		if (fande_==None):
			f_x_ = f_(x)
		else:
			e, f_x_ = fande_(x)
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

class NosePerParticleThermostat(Thermostat):
	def __init__(self,m_,v_):
		"""
		This
		"""
		self.m = m_.copy()
		self.N = len(m_)
		self.T = PARAMS["MDTemp"]  # Length of NH chain.
		self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.tau = 40.0*PARAMS["MDdt"]
		self.Q = self.kT*self.tau*self.tau
		self.eta = np.zeros(self.N)
		self.name = "NosePerParticle"
		self.Rescale(v_)
		print("Using ", self.name, " thermostat at ",self.T, " degrees Kelvin")
		return
	def step(self,f_, a_, x_, v_, m_, dt_, fande_=None , frc_ = True):
		"""
		http://www2.ph.ed.ac.uk/~dmarendu/MVP/MVP03.pdf
		"""
		x = x_ + v_*dt_ + (1./2.)*(a_ - np.einsum("i,ij->ij",self.eta,v_))*dt_*dt_
		vdto2 = v_ + (1./2.)*(a_ - np.einsum("i,ij->ij",self.eta,v_))*dt_
		e, f_x_ = 0.0, None
		if (fande_==None):
			f_x_ = f_(x)
		else:
			e, f_x_ = fande_(x)
		a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/m_) # m^2/s^2 => A^2/Fs^2
		kes = (1./2.)*np.einsum("i,i->i",np.einsum("ia,ia->i",v_,v_),m_)
		etadto2 = self.eta + (dt_/(2.*self.Q))*(kes - (((3.*self.N+1)/2.))*self.kT)
		kedto2s = (1./2.)*np.einsum("i,i->i",np.einsum("ia,ia->i",vdto2,vdto2),m_)
		self.eta = etadto2 + (dt_/(2.*self.Q))*(kedto2s - (((3.*self.N+1)/2.))*self.kT)
		v = np.einsum("ij,i->ij",(vdto2 + (dt_/2.)*a),1.0/(1 + (dt_/2.)*self.eta))
		if frc_:
			return x,v,a,e,f_x_
		else:
			return x,v,a,e

class NoseChainThermostat(Thermostat):
	def __init__(self,m_,v_):
		"""
		Velocity Verlet step with a Nose-Hoover Chain Thermostat.
		Based on Appendix A of martyna 1996
		http://dx.doi.org/10.1080/00268979600100761

		Args:
			x_: an example of system positions.
			m_: system masses.
			PARAMS["MNHChain"]: depth of the Nose Hoover Chain
			PARAMS["MDTemp"]: Temperature of the Thermostat.
			PARAMS["MDdt"]: Timestep of the dynamics.
		"""
		self.M = PARAMS["MNHChain"] # Depth of NH chain.
		self.N = len(v_) # Number of particles.
		self.Nf = len(v_)*3 # Number of Degrees of freedom.
		self.T = PARAMS["MDTemp"]  # Length of NH chain.
		self.kT = IDEALGASR*pow(10.0,-10.0)*self.T # energy units here are kg (A/fs)^2
		self.GNKT = self.Nf*self.kT
		self.nc = 2 # nc Number of Trotterizations To be increased if Q is large.
		self.ny = 3# nys (eq. 29 of Martyna. number of quadrature points within step.)
		self.wj = np.zeros(self.ny)
		if (self.ny == 3):
			self.wj[0],self.wj[1],self.wj[2] = (1./(2. - np.power(2.,1./3.))),(1.-2.*(1./(2. - np.power(2.,1./3.)))),(1./(2. - np.power(2.,1./3.)))
		elif (self.ny == 5):
			self.wj[0] = (1./(4. - np.power(4.,1./3.)))
			self.wj[1] = (1./(4. - np.power(4.,1./3.)))
			self.wj[2] = 1.-4.*self.wj[0]
			self.wj[3] = (1./(4. - np.power(4.,1./3.)))
			self.wj[4] = (1./(4. - np.power(4.,1./3.)))
		self.tau = 80.0*PARAMS["MDdt"]
		self.dt = PARAMS["MDdt"]
		self.dt2 = self.dt/2.
		self.dt22 = self.dt*self.dt/2.
		self.Qs = None # Chain Masses.
		self.MartynaQs() # assign the chain masses.
		self.eta = np.zeros(self.M) # Chain positions.
		self.Veta = np.zeros(self.M) # Chain velocities
		self.Geta = np.zeros(self.M) # Chain forces
		self.name = "NoseHooverChain"
		self.Rescale(v_)
		print("Using ", self.name, " thermostat at ",self.T, " degrees Kelvin")
		return

	def MartynaQs(self):
		if (self.M==0):
			return
		self.Qs = np.ones(self.M)*self.kT*self.tau*self.tau
		self.Qs[0] = 3.*self.N*self.kT*self.tau*self.tau
		return

	def step(self,f_, a_, x_, v_, m_, dt_ ,fande_=None):
		v = self.IntegrateChain(a_, x_, v_, m_, dt_) # half step the chain.
		# Get KE of the chain.
		print("Energies of the system... ", self.ke(v_,m_), " Teff ", (2./3.)*self.ke(v_,m_)*pow(10.0,10.0)/IDEALGASR/self.N)
		print("Energies along the chain... Desired:", (3./2.)*self.kT)
		for i in range(self.M):
			print(self.Veta[i]*self.Veta[i]*self.Qs[i]/2.)
		v = v + self.dt2*a_
		x = x_ + self.dt*v

		e, f_x_ = 0.0, None
		if (fande_==None):
			f_x_ = f_(x)
		else:
			e, f_x_ = fande_(x)

		a = pow(10.0,-10.0)*np.einsum("ax,a->ax", f_x_, 1.0/m_) # m^2/s^2 => A^2/Fs^2
		v = v + self.dt2*a
		v = self.IntegrateChain(a, x_, v, m_, dt_) # half step the chain.
		return x, v, a, e

	def ke(self,v_,m_):
		return (1./2.)*np.dot(np.einsum("ia,ia->i",v_,v_),m_)

	def IntegrateChain(self, a_, x_, v_, m_, dt_ ):
		"""
		The Nose Hoover chain is twice trotterized in Martyna's subroutine
		So this evolves the chain a half-step, and updates v_
		"""
		if (self.M==0):
			return v_
		ake = self.ke(v_,m_) # in kg (A^2/Fs^2)
		# Update thermostat forces.
		self.Geta[0]  = (2.*ake - self.GNKT) / self.Qs[0]
		scale = 1.0
		for k in range(self.nc):
			for j in range(self.ny):
				# UPDATE THE THERMOSTAT VELOCITIES.
				wdtj2 = (self.wj[j]*self.dt/self.nc)/2.
				wdtj4 = wdtj2/2.
				wdtj8 = wdtj4/2.
				self.Veta[-1] += self.Geta[-1]*wdtj4
				for i in range(self.M-1)[::-1]:
					AA = np.exp(-wdtj8*self.Veta[i+1])
					self.Veta[i] = self.Veta[i]*AA*AA + wdtj4*self.Geta[i]*AA
				# Update the particle velocities.
				AA = np.exp(-wdtj2*self.Veta[1])
				scale *= AA
				self.Geta[0] = (scale*scale*2.0*ake - self.GNKT)/self.Qs[0]
				# Update the Thermostat Positions.
				for i in range(self.M):
					self.eta[i] += self.Veta[i]*wdtj2
				# Update the thermostat velocities
				for i in range(self.M-1):
					AA = np.exp(-wdtj8*self.Veta[i+1])
					self.Veta[i] = self.Veta[i]*AA*AA + wdtj4*self.Geta[i]*AA
					self.Geta[i+1] = (self.Qs[i]*self.Veta[i]*self.Veta[i] - self.kT)/self.Qs[i+1]
				self.Veta[-1] += self.Geta[-1] * wdtj4
		print("eta",self.eta)
		print("Meta",self.Qs)
		print("Veta",self.Veta)
		print("Geta",self.Geta)
		return v_*scale

class VelocityVerlet:
	def __init__(self, f_, g0_, name_ ="", EandF_=None, cellsize_=None):
		"""
		Molecular dynamics

		Args:
			f_: a force routine
			g0_: initial molecule.
			EandF_: An energy,force routine.
			PARAMS["MDMaxStep"]: Number of steps to take.
			PARAMS["MDTemp"]: Temperature to initialize or Thermostat to.
			PARAMS["MDdt"]: Timestep.
			PARAMS["MDV0"]: Sort of velocity initialization (None, or "Random")
			PARAMS["MDLogTrajectory"]: Write MD Trajectory.
		Returns:
			Nothing.
		"""
		self.name = name_
		self.cellsize = cellsize_
		self.maxstep = PARAMS["MDMaxStep"]
		self.T = PARAMS["MDTemp"]
		self.dt = PARAMS["MDdt"]
		self.ForceFunction = f_
		self.EnergyAndForce = EandF_
		self.EPot0 = 0.0
		if (EandF_ != None):
			self.EPot0 , self.f0 = self.EnergyAndForce(g0_.coords)
		self.EPot = self.EPot0
		self.t = 0.0
		self.KE = 0.0
		self.atoms = g0_.atoms.copy()
		self.m = np.array(map(lambda x: ATOMICMASSES[x-1], self.atoms))
		self.natoms = len(self.atoms)
		self.x = g0_.coords.copy()
		self.v = np.zeros(self.x.shape)
		self.a = np.zeros(self.x.shape)
		self.md_log = None

		if (PARAMS["MDV0"]=="Random"):
			np.random.seed()   # reset random seed
			self.v = np.random.randn(*self.x.shape)
			Tstat = Thermostat(self.m, self.v) # Will rescale self.v appropriately.
		self.Tstat = None
		if (PARAMS["MDThermostat"]=="Rescaling"):
			self.Tstat = Thermostat(self.m,self.v)
		elif (PARAMS["MDThermostat"]=="Nose"):
			self.Tstat = NoseThermostat(self.m,self.v)
		elif (PARAMS["MDThermostat"]=="NosePerParticle"):
			self.Tstat = NosePerParticleThermostat(self.m,self.v)
		elif (PARAMS["MDThermostat"]=="NoseHooverChain"):
			self.Tstat = NoseChainThermostat(self.m, self.v)
		else:
			pass #print("Unthermostated Velocity Verlet.")
		return

	def WriteTrajectory(self):
		m=Mol(self.atoms,self.x)
		m.properties["Time"]=self.t
		m.properties["KineticEnergy"]=self.KE
		m.properties["PotEnergy"]=self.EPot
		m.WriteXYZfile("./results/", "MDTrajectory"+self.name)
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
				self.x , self.v, self.a, self.EPot = VelocityVerletStep(self.ForceFunction, self.a, self.x, self.v, self.m, self.dt, self.EnergyAndForce)
			else:
				self.x , self.v, self.a, self.EPot, self.force = self.Tstat.step(self.ForceFunction, self.a, self.x, self.v, self.m, self.dt, self.EnergyAndForce)
			if self.cellsize != None:
				self.x  = np.mod(self.x, self.cellsize)
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

class IRTrajectory(VelocityVerlet):
	def __init__(self,f_,q_,g0_,name_=str(0),v0_=None):
		"""A specialized sort of dynamics which is appropriate for obtaining IR spectra at
		Zero temperature. Absorption cross section is given by:
		alpha = frac{4pi^2}{hbar c} omega (1 - Exp[-beta hbar omega]) sigma(omega))
		sigma  = frac{1}{6 pi} mathcal{F} {mu(t)mu(0)}

		Args:
			f_: a function which yields the energy, force
			q_: a function which yields the charge.
			g0_: an initial geometry.
			PARAMS["MDFieldVec"]
			PARAMS["MDFieldAmp"]
			PARAMS["MDFieldT0"] = 3.0
			PARAMS["MDFieldTau"] = 1.2 #1.2 fs pulse.
			PARAMS["MDFieldFreq"] = 1/1.2 # 700nm light is about 1/1.2 fs.
		"""
		VelocityVerlet.__init__(self, f_, g0_, name_, f_)
		if (v0_ is not None):
			self.v = v0_.copy()
		self.EField = np.zeros(3)
		self.IsOn = False
		self.FieldVec = PARAMS["MDFieldVec"]
		self.FieldAmp = PARAMS["MDFieldAmp"]
		self.FieldFreq = PARAMS["MDFieldFreq"]
		self.Tau = PARAMS["MDFieldTau"]
		self.TOn = PARAMS["MDFieldT0"]
		self.UpdateCharges = PARAMS["MDUpdateCharges"]
		self.EnergyAndForce = f_
		self.EPot0 , self.f0 = self.EnergyAndForce(g0_.coords)
		self.EPot = self.EPot0
		self.ChargeFunction = None
		self.q0 = 0*self.m
		self.qs = np.ones(self.m.shape)
		self.Mu0 = np.zeros(3)
		self.mu_his = None
		if (q_ != None):
			self.ChargeFunction = q_
			self.q0 = self.ChargeFunction(self.x)
			self.qs = self.q0.copy()
			self.Mu0 = Dipole(self.x, self.ChargeFunction(self.x))
		else:
			self.UpdateCharges = False
		# This can help in case you had a bad initial geometry
		self.MinS = 0
		self.MinE = 0.0
		self.Minx = None

	def Pulse(self,t_):
		"""
		delta pulse of duration
		"""
		sin_part = (np.sin(2.0*3.1415*self.FieldFreq*t_))
		exp_part = (1.0/np.sqrt(2.0*3.1415*self.Tau*self.Tau))*(np.exp(-1.0*np.power(t_-self.TOn,2.0)/(2.0*self.Tau*self.Tau)))
		amp = self.FieldAmp*sin_part*exp_part
		if (np.abs(amp) > np.power(10.0,-12.0)):
			return self.FieldVec*amp, True
		else:
			return np.zeros(3), False

	def ForcesWithCharge(self,x_):
		e, FFForce = self.EnergyAndForce(x_)
		if (self.IsOn):
			# ElectricFieldForce Is in units of Hartree/angstrom.
			# and must be converted to kg*Angstrom/(Fs^2)
			ElecForce = 4184.0*ElectricFieldForce(self.qs, self.EField)
			print("Field Free Force", FFForce)
			print("ElecForce Force", ElecForce)
			return e, RemoveInvariantForce(x_, FFForce + ElecForce, self.m)
		else:
			return e, RemoveInvariantForce(x_, FFForce, self.m)

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
#HACK
		vhis = np.zeros((self.maxstep,)+self.v.shape) # time Dipoles Energy
		step = 0
		while(step < self.maxstep):
			self.t = step*self.dt
			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/IDEALGASR

			self.EField, self.IsOn = self.Pulse(self.t)
			if (self.UpdateCharges and not self.IsOn):
				self.qs = self.ChargeFunction(self.x)
			else:
				self.qs = self.q0
			self.Mu = Dipole(self.x, self.qs) - self.Mu0
			self.mu_his[step,0] = self.t
			self.mu_his[step,1:4] = self.Mu.copy()
			self.mu_his[step,4] = self.KE
			self.mu_his[step,5] = self.EPot
			self.mu_his[step,6] = self.KE+self.EPot
			vhis[step] = self.v.copy()

			if (PARAMS["MDThermostat"]==None):
				self.x , self.v, self.a, self.EPot = VelocityVerletStep(None, self.a, self.x, self.v, self.m, self.dt,self.ForcesWithCharge)
			else:
				self.x , self.v, self.a, self.EPot, self.force = self.Tstat.step(None, self.a, self.x, self.v, self.m, self.dt,self.ForcesWithCharge)

			if (PARAMS["MDIrForceMin"] and self.EPot < self.MinE and abs(self.EPot - self.MinE)>0.00005):
				self.MinE = self.EPot
				self.Minx = self.x.copy()
				self.MinS = step
				LOGGER.info(" -- You didn't start from the global minimum -- ")
				LOGGER.info("   -- I'mma set you back to the beginning -- ")
				print(self.x)
				self.Mu0 = Dipole(self.x, self.qs)
				step=0
			if (step%50==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			step+=1
			if (step%1000==0):
				np.savetxt("./results/"+"MDLog"+self.name+".txt",self.mu_his)
			LOGGER.info("%s Step: %i time: %.1f(fs) <KE>(kJ): %.5f <PotE>(Eh): %.5f <ETot>(kJ/mol): %.5f Teff(K): %.5f Mu: (%f,%f,%f)", self.name, step, self.t, self.KE, self.EPot, self.KE/1000.0+(self.EPot-self.EPot0)*KJPERHARTREE, Teff, self.Mu[0], self.Mu[1], self.Mu[2])
		#WriteVelocityAutocorrelations(self.mu_his,vhis)
		return

class Annealer(IRTrajectory):
	def __init__(self,f_,q_,g0_,name_="anneal",AnnealThresh_ = 0.000009):
		PARAMS["MDThermostat"] = None
		#PARAMS["MDV0"] = None
		IRTrajectory.__init__(self, f_, q_, g0_, name_)
		#self.dt = 0.2
		#self.v *= 0.0
		self.AnnealT0 = PARAMS["MDAnnealT0"]
		self.AnnealSteps = PARAMS["MDAnnealSteps"]
		self.MinS = 0
		self.MinE = 0.0
		self.Minx = None
		self.AnnealThresh = AnnealThresh_
		self.Tstat = NoseThermostat(self.m,self.v)
		# The annealing program is 1K => 0K in 500 steps.
		return

	def Prop(self):
		"""
		Propagate VelocityVerlet
		"""
		step = 0
		self.Tstat.T = self.AnnealT0*float(self.AnnealSteps - step)/self.AnnealSteps + pow(10.0,-10.0)
		Teff = PARAMS["MDAnnealT0"]
		print ("Teff", Teff, " MDAnnealTF:", PARAMS["MDAnnealTF"])
		while(step < self.AnnealSteps or abs(Teff -  PARAMS["MDAnnealTF"])>0.1):
			self.t = step*self.dt
			self.KE = KineticEnergy(self.v,self.m)
			#Teff = (2./3.)*self.KE/IDEALGASR

			self.EField, self.IsOn = self.Pulse(self.t)
			if (self.UpdateCharges and not self.IsOn):
				self.qs = self.ChargeFunction(self.x)
			else:
				self.qs = self.q0
			self.Mu = Dipole(self.x, self.qs) - self.Mu0
			# avoid the thermostat blowing up.
			AnnealFrac = float(self.AnnealSteps - step)/self.AnnealSteps
			self.Tstat.T = max(0.1, self.AnnealT0*AnnealFrac + PARAMS["MDAnnealTF"]*(1.0-AnnealFrac) + pow(10.0,-10.0))
			# First 50 steps without any thermostat.
			self.x , self.v, self.a, self.EPot, self.force = self.Tstat.step(self.ForceFunction, self.a, self.x, self.v, self.m, self.dt, self.EnergyAndForce)
			Teff = (2./3.)*self.KE/IDEALGASR
			if (self.EPot < self.MinE and abs(self.EPot - self.MinE)>self.AnnealThresh):
				self.MinE = self.EPot
				self.Minx = self.x.copy()
				self.MinS = step
				LOGGER.info("   -- cycling annealer -- ")
				if (PARAMS["MDAnnealT0"] > PARAMS["MDAnnealTF"]):
					self.AnnealT0 = min(PARAMS["MDAnnealT0"], self.Tstat.T+PARAMS["MDAnnealKickBack"])
				print(self.x)
				self.Mu0 = Dipole(self.x, self.qs)
				step=0

			if (PARAMS["PrintTMTimer"]):
				PrintTMTIMER()
			if (step%7==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			step+=1
			LOGGER.info("%s Step: %i time: %.1f(fs) <KE>(kJ): %.5f <PotE>(Eh): %.5f <ETot>(kJ/mol): %.5f T_eff(K): %.5f T_target(K): %.5f", self.name, step, self.t, self.KE, self.EPot, self.KE/1000.0+(self.EPot-self.EPot)*2625.5, Teff, self.Tstat.T)
		#self.x = self.Minx.copy()
		print("Achieved Minimum energy ", self.MinE, " at step ", step)
		return

class NoEnergyAnnealer(VelocityVerlet):
	def __init__(self,f_,g0_,name_="anneal",AnnealThresh_ = 10.0):
		PARAMS["MDThermostat"] = None
		PARAMS["MDV0"] = None
		VelocityVerlet.__init__(self, f_, g0_, name_)
		self.dt = 0.2
		self.v *= 0.0
		self.AnnealT0 = 20.0
		self.MinS = 0
		self.MinF = 1e10
		self.Minx = None
		self.AnnealSteps = 3000
		self.AnnealThresh = AnnealThresh_
		self.Tstat = NoseThermostat(self.m,self.v)
		# The annealing program is 1K => 0K in 500 steps.
		return

	def Prop(self):
		"""
		Propagate VelocityVerlet
		"""
		step = 0
		while(step < self.AnnealSteps):
			self.t = step*self.dt
			self.KE = KineticEnergy(self.v,self.m)
			Teff = (2./3.)*self.KE/IDEALGASR

			# avoid the thermostat blowing up.
			self.Tstat.T = self.AnnealT0*float(self.AnnealSteps - step)/self.AnnealSteps + pow(10.0,-10.0)
			# First 50 steps without any thermostat.
			self.x , self.v, self.a, self.EPot, self.frc = self.Tstat.step(self.ForceFunction, self.a, self.x, self.v, self.m, self.dt, self.EnergyAndForce, True)

			if (RmsForce(self.frc) < self.MinF and abs(RmsForce(self.frc) - self.MinF)>self.AnnealThresh):
				self.MinF = RmsForce(self.frc)
				self.Minx = self.x.copy()
				self.MinS = step
				LOGGER.info("   -- cycling annealer -- ")
				self.AnnealT0 = self.Tstat.T
				print(self.x)
				step=0

			if (step%7==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			step+=1
			LOGGER.info("%s Step: %i time: %.1f(fs) <KE>(kJ): %.5f <PotE>(Eh): %.5f <ETot>(kJ/mol): %.5f Teff(K): %.5f ", self.name, step, self.t, self.KE, RmsForce(self.frc), self.KE/1000.0+(self.EPot-self.EPot)*2625.5, Teff)
		self.Minx = self.x.copy()
		print("Achieved Minimum energy ", self.MinF, " at step ", step)
		return
