from __future__ import absolute_import
from __future__ import print_function
from .Opt import *
from .Periodic import *
import random
import time

class PeriodicGeomOptimizer(GeomOptimizer):
	def __init__(self,f_):
		"""
		Periodic Geometry Optimizations.
		Takes a periodic force f_

		Args:
			f_: A PeriodicForce object
		"""
		self.thresh = PARAMS["OptThresh"]
		self.maxstep = PARAMS["OptMaxStep"]
		self.fscale = PARAMS["OptStepSize"]
		self.momentum = PARAMS["OptMomentum"]
		self.momentum_decay = PARAMS["OptMomentumDecay"]
		self.max_opt_step = PARAMS["OptMaxCycles"]
		self.step = self.maxstep
		self.EnergyAndForce = f_
		return
	def Opt(self,m, filename="PdicOptLog",Debug=False):
		"""
		Optimize using An EnergyAndForce Function.

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
		print("Orig Coords", m.coords)
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		Energy = lambda x_: self.EnergyAndForce(x_)[0]
		while( step < self.max_opt_step and rmsgrad > self.thresh):
			prev_m = Mol(m.atoms, m.coords)
			self.EnergyAndForce.LatticeStep(m.coords)
			energy, frc = self.EnergyAndForce(m.coords)
			frc = RemoveInvariantForce(m.coords, frc, m.atoms)
			frc /= JOULEPERHARTREE
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/frc.shape[0]
			m.coords = self.EnergyAndForce.lattice.ModuloLattice(LineSearch(Energy, m.coords, frc))
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/veloc.shape[0]
			print("step: ", step ," energy: ", energy, " rmsgrad ", rmsgrad, " rmsdisp ", rmsdisp)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			Mol(*self.EnergyAndForce.lattice.TessLattice(prev_m.atoms,prev_m.coords,self.EnergyAndForce.maxrng)).WriteXYZfile("./results/", "Tess"+filename)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		print("Final Energy:", Energy(prev_m.coords))
		return prev_m
	def OptWCell(self,m, filename="OptLog", Debug=False):
		"""
		Optimize using An EnergyAndForce Function.

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
		print("Orig Coords", m.coords)
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		Energy = lambda x_: self.EnergyAndForce(x_)[0]
		while( step < self.max_opt_step and rmsgrad > self.thresh):
			prev_m = Mol(m.atoms, m.coords)
			energy, frc = self.EnergyAndForce(m.coords)
			frc = RemoveInvariantForce(m.coords, frc, m.atoms)
			frc /= JOULEPERHARTREE
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/frc.shape[0]
			m.coords = self.EnergyAndForce.lattice.ModuloLattice(LineSearch(Energy, m.coords, frc))
			m.coords = self.EnergyAndForce.LatticeStep(m.coords)
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/veloc.shape[0]
			print("step: ", step ," energy: ", energy, " rmsgrad ", rmsgrad, " rmsdisp ", rmsdisp)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			Mol(*self.EnergyAndForce.lattice.TessLattice(prev_m.atoms,prev_m.coords,self.EnergyAndForce.maxrng)).WriteXYZfile("./results/", "Tess"+filename)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		print("Final Energy:", Energy(prev_m.coords))
		return prev_m

	def Opt_GD(self,m, filename="OptLog",Debug=False):
		"""
		Optimize using An EnergyAndForce Function.

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
		m.WriteXYZfile("./results/", "Initial")
		m.coords=self.EnergyAndForce.lattice.ModuloLattice(m.coords)
		m.WriteXYZfile("./results/", "Modulo")
		prev_m = Mol(m.atoms, m.coords)
		print("Orig Coords", m.coords)
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		energy, frc  = self.EnergyAndForce(m.coords)
		frc = RemoveInvariantForce(m.coords, frc, m.atoms)
		frc /= JOULEPERKCAL
		while( step < self.max_opt_step and rmsgrad > self.thresh):
			prev_m = Mol(m.atoms, m.coords)
			if step == 0:
				old_frc = frc
			energy, frc = self.EnergyAndForce(m.coords)
			frc = RemoveInvariantForce(m.coords, frc, m.atoms)
			frc /= JOULEPERHARTREE
			print(("force:", frc))
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/frc.shape[0]
			frc = (1-self.momentum)*frc + self.momentum*old_frc
			m.coords = self.EnergyAndForce.lattice.ModuloLattice(m.coords + self.fscale*frc)
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/veloc.shape[0]
			print("step: ", step ," energy: ", energy, " rmsgrad ", rmsgrad, " rmsdisp ", rmsdisp)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			Mol(*self.EnergyAndForce.lattice.TessNTimes(prev_m.atoms,prev_m.coords,2)).WriteXYZfile("./results/", "Tess"+filename)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		return prev_m
