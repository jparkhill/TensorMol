"""
Changes that need to be made:
"""

from Sets import *
from TFManage import *
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
		self.k = PARAMS["NebK"]
		self.step = self.maxstep
		self.probtype = 0 # 0 = one atom probability, 1 = product of all probabilities for each sample.
		self.tfm = tfm_
		self.atoms = g0_.atoms.copy()
		self.natoms = len(self.atoms)
		self.beads=np.array([(1.-l)*g0_.coords+l*g1_.coords for l in np.linspace(0.,1.,self.nbeads)])
		self.Fs = np.zeros(self.beads.shape)
		self.Ss = np.zeros(self.beads.shape)
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
		tmp = (self.nbeads*2.0*self.beads[i] - self.beads[i+1] - self.beads[i-1])
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

	def NebForce(self,i):
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		m=Mol(self.atoms,self.beads[i])
		F = self.tfm.EvalRotAvForce(m, RotAv=PARAMS["RotAvOutputs"], Debug=False)
		t = self.Tangent(i)
		S = -1.0*self.SpringDeriv(i)
		S = self.Parallel(S,t)
		F = self.Perpendicular(F,t)
		self.Ss[i] = S
		self.Fs[i] = F
		Fneb = self.PauliForce(i)+S+F
		return Fneb

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
		traj_hist = [self.beads.copy()]
		forces = np.zeros(self.beads.shape)
		old_forces = np.zeros(self.beads.shape)
		while(np.mean(rmsgrad)>self.thresh and step < self.max_opt_step):
			# Update the positions of every bead together.
			traj_hist.append(self.beads)
			old_force = self.momentum_decay*forces
			for i,bead in enumerate(self.beads):
				forces[i] = self.NebForce(i)
			forces = (1.0-self.momentum)*self.fscale*forces + self.momentum*old_force
			for i,bead in enumerate(self.beads):
				self.beads[i] += forces[i]
				rmsgrad[i] = np.sum(np.linalg.norm(forces[i],axis=1))/forces[i].shape[0]
				maxgrad[i] = np.amax(np.linalg.norm(forces[i],axis=1))
				#rmsdisp[i] = np.sum(np.linalg.norm((prev_m.coords-m.coords),axis=1))/m.coords.shape[0]
				#maxdisp[i] = np.amax(np.linalg.norm((prev_m.coords - m.coords), axis=1))
			self.WriteTrajectory()
			step+=1
			LOGGER.info("Step: %i RMS Gradient: %.5f  Max Gradient: %.5f |F_parallel| : %.5f |F_spring|: %.5f ", step, np.mean(rmsgrad), np.max(maxgrad),np.linalg.norm(self.Fs),np.linalg.norm(self.Ss))
		return
