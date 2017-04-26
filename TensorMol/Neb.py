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
		self.beads=[(1.-l)*g0_.coords+l*g1_.coords for l in np.linspace(0.,1.,self.nbeads)]
		if (self.tfm!=None):
			self.OType = self.tfm.TData.dig.OType
			print "Optimizer will use ",self.OType, " outputs from tensorflow to optimize."
		return

	def Tangent(self,i):
		if (i==0 or i==self.nbeads):
			return np.zeros(self.beads[0].shape)
		tm1 = self.beads[i] - self.beads[i-1]
		tp1 = self.beads[i+1] - self.beads[i]
		t = tm1 + tp1
		t = t/np.sqrt(np.einsum('ia,ia',t,t))
		return t

	def SpringForce(self,i):
		if (i==0 or i==self.nbeads):
			return np.zeros(self.beads[0].shape)
		tmp = (self.beads[i+1] - self.beads[i]) - (self.beads[i] - self.beads[i-1])
		fpar = self.k*(np.einsum("ia,ia",tmp,self.Tangent[i]))*self.Tangent(i)
		return fpar

	def NebForce(self,i):
		if (i==0 or i==self.nbeads):
			return np.zeros(self.beads[0].shape)
		F = self.tfm.EvalRotAvForce(self.beads[i], RotAv=PARAMS["RotAvOutputs"], Debug=False)
		t = self.Tangent(i)
		Fneb = self.SpringForce(i)+F-np.einsum("ia,ia",F,t)*F
		return Fneb

	def OptTFRealForce(self,m, filename="OptLog",Debug=False):
		"""
		Optimize using force output of an atomwise network.
		now also averages over rotations...
		Args:
			m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print "Orig Coords", m.coords
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		while(rmsdisp>self.thresh and step < self.max_opt_step):
			if (PARAMS["RotAvOutputs"]):
				veloc = self.fscale*self.tfm.EvalRotAvForce(m, RotAv=PARAMS["RotAvOutputs"], Debug=False)
			elif (PARAMS["OctahedralAveraging"]):
				veloc = self.fscale*self.tfm.EvalOctAvForce(m, Debug=True)
			else:
				for i in range(m.NAtoms()):
					veloc[i] = self.fscale*self.tfm.evaluate(m,i)
			if (Debug):
				for i in range(m.NAtoms()):
					print "TF veloc: ",m.atoms[i], ":" , veloc[i]
			veloc = veloc - np.average(veloc,axis=0)
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			old_veloc = self.momentum_decay*c_veloc
			#Remove translation.
			prev_m = Mol(m.atoms, m.coords)
			m.coords = m.coords + c_veloc
			rmsgrad = np.sum(np.linalg.norm(veloc,axis=1))/veloc.shape[0]
			maxgrad = np.amax(np.linalg.norm(veloc,axis=1))
			rmsdisp = np.sum(np.linalg.norm((prev_m.coords-m.coords),axis=1))/m.coords.shape[0]
			maxdisp = np.amax(np.linalg.norm((prev_m.coords - m.coords), axis=1))
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			step+=1
			LOGGER.info("Step: %i RMS Disp: %.5f Max Disp: %.5f RMS Gradient: %.5f  Max Gradient: %.5f ", step, rmsdisp, maxdisp, rmsgrad, maxgrad)
		return prev_m
