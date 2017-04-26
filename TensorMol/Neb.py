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
		self.beads=np.array([(1.-l)*g0_.coords+l*g1_.coords for l in np.linspace(0.,1.,self.nbeads)])
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

	def SpringForce(self,i):
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		tmp = (self.beads[i+1] - self.beads[i]) - (self.beads[i] - self.beads[i-1])
		ti = self.Tangent(i)
		fpar = self.k*(np.einsum("ia,ia",tmp,ti))*ti
		return fpar

	def NebForce(self,i):
		if (i==0 or i==(self.nbeads-1)):
			return np.zeros(self.beads[0].shape)
		m=Mol(self.atoms,self.beads[i])
		F = self.tfm.EvalRotAvForce(m, RotAv=PARAMS["RotAvOutputs"], Debug=False)
		t = self.Tangent(i)
		Fneb = self.SpringForce(i)+F-np.einsum("ia,ia",F,t)*F
		return Fneb

	def WriteTrajectory(self):
		for i,bead in enumerate(self.beads):
			m=Mol(self.atoms,bead)
			m.WriteXYZfile("./results/", "Bead"+str(i))
		for i,bead in enumerate(self.beads):
			m=Mol(self.atoms,bead)
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
		while(np.mean(rmsgrad)>self.thresh and step < self.max_opt_step):
			# Update the positions of every bead
			traj_hist.append(self.beads)
			for i,bead in enumerate(self.beads):
				forces[i] = self.NebForce(i)
				self.beads[i] += self.fscale*forces[i]
				rmsgrad[i] = np.sum(np.linalg.norm(forces[i],axis=1))/forces[i].shape[0]
				maxgrad[i] = np.amax(np.linalg.norm(forces[i],axis=1))
				#rmsdisp[i] = np.sum(np.linalg.norm((prev_m.coords-m.coords),axis=1))/m.coords.shape[0]
				#maxdisp[i] = np.amax(np.linalg.norm((prev_m.coords - m.coords), axis=1))
			self.WriteTrajectory()
			step+=1
			LOGGER.info("Step: %i RMS Gradient: %.5f  Max Gradient: %.5f ", step, np.mean(rmsgrad), np.max(maxgrad))
		return
