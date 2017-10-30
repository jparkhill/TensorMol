"""
Changes that need to be made:
 - do the conservative minimization of nabla L
 - implement Climbing image NEB
"""

from __future__ import absolute_import
from __future__ import print_function
from .Sets import *
from .TFManage import *
from .DIIS import *
from .QuasiNewtonTools import *
from .BFGS import *
from .DIIS import *
import random
import time

class NudgedElasticBand:
	def __init__(self,f_,g0_,g1_):
		"""
		Nudged Elastic band. JCP 113 9978

		Args:
			f_: an energy, force routine (energy Hartree, Force Kcal/ang.)
			g0_: initial molecule.
			g1_: final molecule.

		Returns:
			A reaction path.
		"""
		self.thresh = PARAMS["OptThresh"]
		self.max_opt_step = PARAMS["OptMaxCycles"]
		self.step=0
		self.TSI = 0 # index of the TS bead.
		self.nbeads = PARAMS["NebNumBeads"]
		self.k = PARAMS["NebK"]
		self.f = f_
		self.atoms = g0_.atoms.copy()
		self.natoms = len(self.atoms)
		self.beads=np.array([(1.-l)*g0_.coords+l*g1_.coords for l in np.linspace(0.,1.,self.nbeads)])
		self.Fs = np.zeros(self.beads.shape) # Real forces.
		self.Ss = np.zeros(self.beads.shape) # Spring Forces.
		self.Ts = np.zeros(self.beads.shape) # Tangents.
		self.Es = np.zeros(self.nbeads) # As Evaluated.
		self.Esi = np.zeros(self.nbeads) # Integrated
		self.Rs = np.zeros(self.nbeads) # Distance between beads.
		self.Solver=None
		if (PARAMS["NebSolver"]=="SD"):
			self.Solver = SteepestDescent(self.WrappedEForce,self.beads)
		elif (PARAMS["NebSolver"]=="BFGS"):
			self.Solver = BFGS_WithLinesearch(self.WrappedEForce,self.beads)
		elif (PARAMS["NebSolver"]=="DIIS"):
			self.Solver = DIIS(self.WrappedEForce, self.beads)
		elif (PARAMS["NebSolver"]=="CG"):
			self.Solver = ConjGradient(self.WrappedEForce,self.beads)
		else:
			raise Exception("Missing Neb Solver")
		for i,bead in enumerate(self.beads):
			m=Mol(self.atoms,bead)
			m.WriteXYZfile("./results/", "NebTraj0")
		return
	def Tangent(self,beads_,i):
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		tm1 = beads_[i] - beads_[i-1]
		tp1 = beads_[i+1] - beads_[i]
		t = tm1 + tp1
		t = t/np.sqrt(np.einsum('ia,ia',t,t))
		return t
	def SpringEnergy(self,beads_):
		tmp = 0.0
		for i in range(self.nbeads-1):
			tmp2 = beads_[i+1] - beads_[i]
			tmp += 0.5*self.k*self.nbeads*np.sum(tmp2*tmp2)
		return tmp
	def SpringDeriv(self,beads_,i):
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		tmp = self.k*self.nbeads*(2.0*beads_[i] - beads_[i+1] - beads_[i-1])
		return tmp
	def Parallel(self,v_,t_):
		return t_*(np.einsum("ia,ia",v_,t_))
	def Perpendicular(self,v_,t_):
		return (v_ - t_*(np.einsum("ia,ia",v_,t_)))
	def BeadAngleCosine(self,beads_,i):
		v1 = (beads_[i+1] - beads_[i])
		v2 = (beads_[i-1] - beads_[i])
		return np.einsum('ia,ia',v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
	def CornerPenalty(self,x):
		return 1./(1.+np.exp(-5.0*(x-0.5)))
	def NebForce(self, beads_, i, DoForce = True):
		"""
		This uses the mixing of Perpendicular spring force
		to reduce kinks
		"""
		if (i==0 or i==(self.nbeads-1)):
			self.Fs[i] = np.zeros(self.beads[0].shape)
			self.Es[i] = self.f(beads_[i],False)
		elif (DoForce):
			self.Es[i], self.Fs[i] = self.f(beads_[i],DoForce)
		else:
			self.Es[i] = self.f(beads_[i],DoForce)
		# Compute the spring part of the energy.
		if (not DoForce):
			return self.Es[i]

		t = self.Tangent(beads_,i)
		self.Ts[i] = t
		S = -1.0*self.SpringDeriv(beads_,i)
		Spara = self.Parallel(S,t)
		self.Ss[i] = Spara
		F = self.Fs[i].copy()
		F = self.Perpendicular(F,t)
		#Sperp = self.CornerPenalty(self.BeadAngleCosine(i))*(self.Perpendicular(S,t))
		# Instead use Wales' DNEB
		if (np.linalg.norm(F) != 0.0):
			Fn = F/np.linalg.norm(F)
		else:
			Fn = F
		Sperp = self.Perpendicular(self.Perpendicular(S,t),Fn)
		#Fneb = self.PauliForce(i)+Spara+Sperp+F
		Fneb = Spara+F#+Sperp
		# If enabled and this is the TS bead, do the climbing image.
		if (PARAMS["NebClimbingImage"] and self.step>10 and i==self.TSI):
			print("Climbing image i", i)
			Fneb = self.Fs[i] + -2.0*np.sum(self.Fs[i]*self.Ts[i])*self.Ts[i]
		return self.Es[i], Fneb
	def WrappedEForce(self, beads_, DoForce=True):
		F = np.zeros(beads_.shape)
		if (DoForce):
			for i,bead in enumerate(beads_):
				#print(DoForce,self.NebForce(bead,i,DoForce))
				self.Es[i], F[i] = self.NebForce(beads_,i,DoForce)
				F[i] = RemoveInvariantForce(bead, F[i], self.atoms)
				F[i] /= JOULEPERHARTREE
			TSE = np.max(self.Es[1:self.nbeads-1])
			self.TSI = np.where(self.Es==TSE)[0][0]
			TE = np.sum(self.Es)+self.SpringEnergy(beads_)
			return TE,F
		else:
			for i,bead in enumerate(beads_):
				self.Es[i] = self.NebForce(beads_,i,DoForce)
			TE = np.sum(self.Es)+self.SpringEnergy(beads_)
			return TE
	def IntegrateEnergy(self):
		"""
		Use the fundamental theorem of line integrals to calculate an energy.
		An interpolated path could improve this a lot.
		"""
		self.Esi[0] = self.Es[0]
		for i in range(1,self.nbeads):
			dR = self.beads[i] - self.beads[i-1]
			dV = -1*(self.Fs[i] + self.Fs[i-1])/2. # midpoint rule.
			self.Esi[i] = self.Esi[i-1]+np.einsum("ia,ia",dR,dV)
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
			m.properties["bead"] = i
			m.properties["Energy"] = self.Es[i]
			m.properties["NormNebForce"]=np.linalg.norm(self.Fs[i])
			m.WriteXYZfile("./results/", "NebTraj")
		return
	def Opt(self, filename="Neb",Debug=False):
		"""
		Optimize the nudged elastic band using the solver that has been selected.
		"""
		# Sweeps one at a time
		self.step=0
		self.Fs = np.ones(self.beads.shape)
		PES = np.zeros((self.max_opt_step, self.nbeads))
		while(self.step < self.max_opt_step and np.sqrt(np.mean(self.Fs*self.Fs))>self.thresh):
			# Update the positions of every bead together.
			self.beads, energy, self.Fs = self.Solver(self.beads)
			PES[self.step] = self.Es.copy()
			self.IntegrateEnergy()
			print("Rexn Profile: ", self.Es)
			beadFs = [np.linalg.norm(x) for x in self.Fs[1:-1]]
			beadFperp = [np.linalg.norm(self.Perpendicular(self.Fs[i],self.Ts[i])) for i in range(1,self.nbeads-1)]
			beadRs = [np.linalg.norm(self.beads[x+1]-self.beads[x]) for x in range(self.nbeads-1)]
			beadCosines = [self.BeadAngleCosine(self.beads,i) for i in range(1,self.nbeads-1)]
			#print("Frce Profile: ", beadFs)
			print("F_|_ Profile: ", beadFperp)
			#print("SFrc Profile: ", beadSfs)
			#print("Dist Profile: ", beadRs)
			#print("BCos Profile: ", beadCosines)
			minforce = np.min(beadFs)
				#rmsdisp[i] = np.sum(np.linalg.norm((prev_m.coords-m.coords),axis=1))/m.coords.shape[0]
				#maxdisp[i] = np.amax(np.linalg.norm((prev_m.coords - m.coords), axis=1))
			if (self.step%10==0):
				self.WriteTrajectory(filename)
			LOGGER.info("Step: %i Objective: %.5f RMS Gradient: %.5f Ea(kcal/mol): %.5f Del-E(kcal/mol): %.5f Max Gradient: %.5f |F_perp| : %.5f |F_spring|: %.5f ", self.step, np.sum(PES[self.step]), np.sqrt(np.mean(self.Fs*self.Fs)), (self.Es[self.TSI]-(self.Es[0]))*KCALPERHARTREE,((self.Es[self.nbeads-1])-(self.Es[0]))*KCALPERHARTREE, np.max(self.Fs),np.mean(beadFperp),np.linalg.norm(self.Ss))
			self.step+=1
		#self.HighQualityPES()
		LOGGER.info("========= Nudged Elastic Band Computation Complete ==========")
		LOGGER.info("Activation Energy: %0.5f",self.Es[self.TSI]-(self.Es[0]))
		LOGGER.info("Enthalpy: %0.5f",(self.Es[self.nbeads-1])-(self.Es[0]))
		np.savetxt("./results/NEB_"+filename+"_Energy.txt",PES)
		return self.beads
