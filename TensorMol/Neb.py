"""
Changes that need to be made:
"""

from Sets import *
from TFManage import *
from DIIS import *
import random
import time

class NudgedElasticBand:
	def __init__(self,tfm_,g0_,g1_):
		"""
		Nudged Elastic band. JCP 113 9978
		Args:
			tfm_: a TFManager.
			g0_: initial molecule.
			g1_: final molecule.
		Returns:
			A reaction path.
		"""
		self.thresh = PARAMS["OptThresh"]
		self.maxstep = PARAMS["OptMaxStep"]
		self.fscale = PARAMS["OptStepSize"]
		self.momentum = PARAMS["OptMomentum"]
		self.momentum_decay = PARAMS["OptMomentumDecay"]
		self.max_opt_step = PARAMS["OptMaxCycles"]
		self.nbeads = PARAMS["NebNumBeads"]
		self.m_max = PARAMS["NebMaxBFGS"]
		self.k = PARAMS["NebK"]
		self.step = self.maxstep
		self.probtype = 0 # 0 = one atom probability, 1 = product of all probabilities for each sample.
		self.tfm = tfm_
		self.atoms = g0_.atoms.copy()
		self.natoms = len(self.atoms)
		self.beads=np.array([(1.-l)*g0_.coords+l*g1_.coords for l in np.linspace(0.,1.,self.nbeads)])
		self.Fs = np.zeros(self.beads.shape) # Real forces.
		self.Ss = np.zeros(self.beads.shape) # Spring Forces.
		self.Ts = np.zeros(self.beads.shape) # Tangents.
		self.Es = np.zeros(self.nbeads)
		self.Rs = np.zeros(self.nbeads) # Distance between beads.
		for i,bead in enumerate(self.beads):
			m=Mol(self.atoms,bead)
			m.WriteXYZfile("./results/", "NebTraj0")
		if (self.tfm!=None):
			self.OType = self.tfm.TData.dig.OType
			print "Optimizer will use ",self.OType, " outputs from tensorflow to optimize."
		return

	def Tangent(self,i):
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		tm1 = self.beads[i] - self.beads[i-1]
		tp1 = self.beads[i+1] - self.beads[i]
		t = tm1 + tp1
		t = t/np.sqrt(np.einsum('ia,ia',t,t))
		return t

	def SpringDeriv(self,i):
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		tmp = self.k*self.nbeads*(2.0*self.beads[i] - self.beads[i+1] - self.beads[i-1])
		return tmp

	def PauliForce(self,i):
		"""
			Try to help-out HerrNet by preventing any type of collapses.
		"""
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		bead = self.beads[i]
		dm = MolEmb.Make_DistMat(bead)
		tore = np.zeros(self.beads[0].shape)
		for i in range(self.natoms):
			for j in range(self.natoms):
				if (i==j):
					continue
				if dm[i,j]<0.5:
					fv =  bead[i]-bead[j]
					fv /= np.linalg.norm(fv)
					tore[i] += (1.0/dm[i,j])*fv
		return tore

	def Parallel(self,v_,t_):
		return t_*(np.einsum("ia,ia",v_,t_))

	def Perpendicular(self,v_,t_):
		return (v_ - t_*(np.einsum("ia,ia",v_,t_)))

	def BeadAngleCosine(self,i):
		v1 = (self.beads[i+1] - self.beads[i])
		v2 = (self.beads[i-1] - self.beads[i])
		return np.einsum('ia,ia',v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

	def CornerPenalty(self,x):
		return 1./(1.+np.exp(-5.0*(x-0.5)))

	def NebForce(self,i):
		"""
		This uses the mixing of Perpendicular spring force
		to reduce kinks
		"""
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		m=Mol(self.atoms,self.beads[i])
		F = self.tfm.EvalRotAvForce(m, RotAv=PARAMS["RotAvOutputs"], Debug=False)
		self.Fs[i] = F.copy()
		t = self.Tangent(i)
		self.Ts[i] = t
		S = -1.0*self.SpringDeriv(i)
		Spara = self.Parallel(S,t)
		self.Ss[i] = Spara
		F = self.Perpendicular(F,t)
		#Sperp = self.CornerPenalty(self.BeadAngleCosine(i))*(self.Perpendicular(S,t))
		# Instead use Wales' DNEB
		Fn = F/np.linalg.norm(F)
		Sperp = self.Perpendicular(self.Perpendicular(S,t),Fn)
		Fneb = self.PauliForce(i)+Spara+Sperp+F
		return Fneb

	def IntegrateEnergy(self):
		"""
		Use the fundamental theorem of line integrals to calculate an energy.
		An interpolated path could improve this a lot.
		"""
		self.Es[0] = 0
		for i in range(1,self.nbeads):
			dR = self.beads[i] - self.beads[i-1]
			dV = -1*(self.Fs[i] + self.Fs[i-1])/2. # midpoint rule.
			self.Es[i] = self.Es[i-1]+np.einsum("ia,ia",dR,dV)

	def HighQualityPES(self,npts_ = 100):
		"""
		Do a high-quality integration of the path and forces.
		"""
		from scipy.interpolate import CubicSpline
		ls = np.linspace(0.,1.,self.nbeads)
		Rint = CubicSpline(self.beads)
		Fint = CubicSpline(self.Fs)
		Es = np.zeros(npts_)
		Es[0] = 0
		ls = np.linspace(0.,1.,npts_)
		for i,l in enumerate(ls):
			if (i==0):
				continue
			else:
				Es[i] = Es[i-1] + np.einsum("ia,ia", Rint(l) - Rint(ls[i-1]), -1.0*Fint(l))
			m=Mol(self.atoms,Rint(l))
			m.properties["Energy"] = Es[i]
			m.properties["Force"] = Fint(l)
			m.WriteXYZfile("./results/", "NebHQTraj")

	def WriteTrajectory(self):
		for i,bead in enumerate(self.beads):
			m=Mol(self.atoms,bead)
			m.WriteXYZfile("./results/", "Bead"+str(i))
		for i,bead in enumerate(self.beads):
			m=Mol(self.atoms,bead)
			m.properties["NormNebForce"]=np.linalg.norm(self.Fs[i])
			m.WriteXYZfile("./results/", "NebTraj")
		return

	def OptNeb(self, filename="Neb",Debug=False):
		"""
		Optimize
		"""
		# Sweeps one at a time
		rmsgrad = np.array([10.0 for i in range(self.nbeads)])
		maxgrad = np.array([10.0 for i in range(self.nbeads)])
		step=0
		forces = np.zeros(self.beads.shape)
		beadFperp = 10.0
		while(np.mean(beadFperp)>self.thresh and step < self.max_opt_step):
			# Update the positions of every bead together.
			old_force = self.momentum_decay*forces
			beadSfs = [np.linalg.norm(self.SpringDeriv(i)) for i in range(1,self.nbeads-1)]
			for i,bead in enumerate(self.beads):
				forces[i] = self.NebForce(i)
			self.beads += self.fscale*forces
			#self.beads = diis.NextStep(self.beads,forces)
			for i,bead in enumerate(self.beads):
				rmsgrad[i] = np.sum(np.linalg.norm(forces[i],axis=1))/forces[i].shape[0]
				maxgrad[i] = np.amax(np.linalg.norm(forces[i],axis=1))

			self.IntegrateEnergy()
			print "Rexn Profile: ", self.Es
			beadFs = [np.linalg.norm(x) for x in self.Fs[1:-1]]
			beadFperp = [np.linalg.norm(self.Perpendicular(self.Fs[i],self.Ts[i])) for i in range(1,self.nbeads-1)]
			beadRs = [np.linalg.norm(self.beads[x+1]-self.beads[x]) for x in range(self.nbeads-1)]
			beadCosines = [self.BeadAngleCosine(i) for i in range(1,self.nbeads-1)]
			print "Frce Profile: ", beadFs
			print "F_|_ Profile: ", beadFperp
			print "SFrc Profile: ", beadSfs
			print "Dist Profile: ", beadRs
			print "BCos Profile: ", beadCosines
			minforce = np.min(beadFs)
				#rmsdisp[i] = np.sum(np.linalg.norm((prev_m.coords-m.coords),axis=1))/m.coords.shape[0]
				#maxdisp[i] = np.amax(np.linalg.norm((prev_m.coords - m.coords), axis=1))
			self.WriteTrajectory()
			step+=1
			LOGGER.info("Step: %i RMS Gradient: %.5f  Max Gradient: %.5f |F_perp| : %.5f |F_spring|: %.5f ", step, np.mean(rmsgrad), np.max(maxgrad),np.mean(beadFperp),np.linalg.norm(self.Ss))
		#self.HighQualityPES()
		return

	def OptNebGLBFGS(self, filename="Neb",Debug=False):
		"""
		Optimize using a globalLBFGS
		Some light code-borrowing from Kun's MBE_Opt.py...
		"""
		# Sweeps one at a time
		rmsgrad = np.array([10.0 for i in range(self.nbeads)])
		maxgrad = np.array([10.0 for i in range(self.nbeads)])
		step=0
		forces = np.zeros(self.beads.shape)
		R_Hist = np.zeros(([self.m_max]+list(self.beads.shape)))
		F_Hist = np.zeros(([self.m_max]+list(self.beads.shape)))
		while(np.mean(rmsgrad)>self.thresh and step < self.max_opt_step):
			# Update the positions of every bead together.
			for i,bead in enumerate(self.beads):
				forces[i] = self.NebForce(i)
			if step < self.m_max:
				R_Hist[step] = self.beads.copy()
				F_Hist[step] = forces.copy()
			else:
				R_Hist = np.roll(R_Hist,-1,axis=0)
				F_Hist = np.roll(R_Hist,-1,axis=0)
				R_Hist[-1] = self.beads.copy()
				F_Hist[-1] = forces.copy()
			# Quasi Newton L-BFGS global step.
			q = forces.copy()
			for i in range(min(self.m_max,step)-1, 0, -1):
				s = R_Hist[i] - R_Hist[i-1]
				y = F_Hist[i] - F_Hist[i-1]
				rho = 1.0/np.einsum("qia,qia",y,s)#y.dot(s)
				a = rho * np.einsum("qia,qia",s,q)#s.dot(q)
				#print "a ",a
				q -= a*y
			if step < 1:
				H=12.0
			else:
				num = min(self.m_max-1,step)
				v1 = (R_Hist[num] - R_Hist[num-1])
				v2 = (F_Hist[num] - F_Hist[num-1])
				H = (np.einsum("qia,qia",v1,v2))/(np.einsum("qia,qia",v2,v2))
				#print "H:", H
			z = H*q
			for i in range (1,min(self.m_max,step)):
				s = (R_Hist[i] - R_Hist[i-1])
				y = (F_Hist[i] - F_Hist[i-1])
				rho = 1.0/np.einsum("qia,qia",y,s)#y.dot(s)
				a=rho*np.einsum("qia,qia",s,q)#s.dot(q)
				beta = rho*np.einsum("qia,qia",y,z)#(force_his[i] - force_his[i-1]).dot(z)
				#print "a-b: ", (a-beta)
				z += s*(a-beta)
			self.beads += self.fscale*z
			for i,bead in enumerate(self.beads):
				rmsgrad[i] = np.sum(np.linalg.norm(forces[i],axis=1))/forces[i].shape[0]
				maxgrad[i] = np.amax(np.linalg.norm(forces[i],axis=1))
			self.IntegrateEnergy()
			print "Rexn Profile: ", self.Es
			beadFperp = [np.linalg.norm(self.Perpendicular(self.Fs[i],self.Ts[i])) for i in range(1,self.nbeads-1)]
			#rmsdisp[i] = np.sum(np.linalg.norm((prev_m.coords-m.coords),axis=1))/m.coords.shape[0]
			#maxdisp[i] = np.amax(np.linalg.norm((prev_m.coords - m.coords), axis=1))
			self.WriteTrajectory()
			step+=1
			LOGGER.info("Step: %i RMS Gradient: %.5f  Max Gradient: %.5f |F_perp| : %.5f ", step, np.mean(rmsgrad), np.max(maxgrad),np.mean(beadFperp))
		#self.HighQualityPES()
		return
