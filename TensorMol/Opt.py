"""
We should get all references to TFManage out of this
and just pass a EnergyAndForce Field function.
"""
from __future__ import absolute_import
from __future__ import print_function
from .Sets import *
from .TFManage import *
from .QuasiNewtonTools import *
from .DIIS import *
from .BFGS import *
from .LinearOperations import *
import random
import time

class GeomOptimizer:
	def __init__(self,f_):
		"""
		Geometry optimizations based on NN-PES's etc.

		Args:
			f_: An EnergyForce routine
		"""
		self.thresh = PARAMS["OptThresh"]
		self.maxstep = PARAMS["OptMaxStep"]
		self.fscale = PARAMS["OptStepSize"]
		self.momentum = PARAMS["OptMomentum"]
		self.momentum_decay = PARAMS["OptMomentumDecay"]
		self.max_opt_step = PARAMS["OptMaxCycles"]
		self.step = self.maxstep
		self.EnergyAndForce = f_
		self.m = None
		return

	def WrappedEForce(self,x_,DoForce=True):
		if (DoForce):
			energy, frc = self.EnergyAndForce(x_, DoForce)
			frc = RemoveInvariantForce(x_, frc, self.m.atoms)
			frc /= JOULEPERHARTREE
			return energy, frc
		else:
			energy = self.EnergyAndForce(x_,False)
			return energy

	def Opt(self,m, filename="OptLog",Debug=False):
		"""
		Optimize using An EnergyAndForce Function with conjugate gradients.

		Args:
			m: A distorted molecule to optimize
		"""
		self.m = m
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print("Orig Mol:\n", m)
		CG = ConjGradient(self.WrappedEForce, m.coords)
		while( step < self.max_opt_step and rmsgrad > self.thresh and (rmsdisp > 0.000001 or step<5) ):
			prev_m = Mol(m.atoms, m.coords)
			m.coords, energy, frc = CG(m.coords)
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/m.coords.shape[0]
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/m.coords.shape[0]
			LOGGER.info("step: %i energy: %0.5f rmsgrad: %0.5f rmsdisp: %0.5f ", step , energy, rmsgrad, rmsdisp)
			mol_hist.append(prev_m)
			prev_m.properties["Step"] = step
			prev_m.properties["Energy"] = energy
			prev_m.WriteXYZfile("./results/", filename,'a',True)
			step+=1
		# Checks stability in each cartesian direction.
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False)[0])
		return prev_m

	def Opt_LS(self,m, filename="OptLog",Debug=False):
		"""
		Optimize with Steepest Descent + Line search using An EnergyAndForce Function.

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
			m.coords = LineSearch(Energy, m.coords, frc)
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/veloc.shape[0]
			print("step: ", step ," energy: ", energy, " rmsgrad ", rmsgrad, " rmsdisp ", rmsdisp)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		print("Final Energy:", Energy(prev_m.coords))
		return prev_m

	def Opt_GD(self,m, filename="OptLog",Debug=False):
		"""
		Optimize using steepest descent  and an EnergyAndForce Function.

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
			m.coords = m.coords + self.fscale*frc
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/veloc.shape[0]
			print("step: ", step ," energy: ", energy, " rmsgrad ", rmsgrad, " rmsdisp ", rmsdisp)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		return prev_m

	def Opt_GD_forces_only(self, mol, filename="OptLog", Debug=False):
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
		prev_mol = Mol(mol.atoms, mol.coords)
		print("Orig Coords", mol.coords)
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		veloc=np.zeros(mol.coords.shape)
		old_veloc=np.zeros(mol.coords.shape)
		frc  = self.EnergyAndForce(mol)
		frc = RemoveInvariantForce(mol.coords, frc, mol.atoms)
		while( step < self.max_opt_step and rmsgrad > self.thresh):
			prev_mol = Mol(mol.atoms, mol.coords)
			if step == 0:
				old_frc = frc
			frc = -1.0 * self.EnergyAndForce(mol)
			frc = RemoveInvariantForce(mol.coords, frc, mol.atoms)
			print(("force:", frc))
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/frc.shape[0]
			frc = (1-self.momentum)*frc + self.momentum*old_frc
			mol.coords = mol.coords + self.fscale*frc
			rmsdisp = np.sum(np.linalg.norm(mol.coords-prev_mol.coords,axis=1))/veloc.shape[0]
			print("step: ", step, " rmsgrad ", rmsgrad, " rmsdisp ", rmsdisp)
			mol_hist.append(prev_mol)
			prev_mol.WriteXYZfile("./results/", filename)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_mol.coords = LineSearchCart(Energy, prev_mol.coords)
		return prev_mol

	def wrapped_force(self, mol):
		frc = self.EnergyAndForce(mol)
		frc = RemoveInvariantForce(mol.coords, frc, mol.atoms)
		energy = np.sum(np.linalg.norm(frc,axis=1))/frc.shape[0]
		return energy, frc

	def Opt_CG_forces_only(self, mol, filename="OptLog",Debug=False):
		"""
		Optimize using An EnergyAndForce Function with conjugate gradients.

		Args:
			m: A distorted molecule to optimize
		"""
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		prev_mol = Mol(mol.atoms, mol.coords)
		print("Orig Coords", mol.coords)
		CG = ConjGradient_force(self.wrapped_force, mol)
		while( step < self.max_opt_step and rmsgrad > self.thresh and (rmsdisp > 0.000001 or step<500) ):
			prev_mol = Mol(mol.atoms, mol.coords)
			mol.coords, energy, frc = CG(mol)
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/mol.coords.shape[0]
			rmsdisp = np.sum(np.linalg.norm(mol.coords-prev_mol.coords,axis=1))/mol.coords.shape[0]
			LOGGER.info("step: %i energy: %0.5f rmsgrad: %0.5f rmsdisp: %0.5f ", step , energy, rmsgrad, rmsdisp)
			mol_hist.append(prev_mol)
			prev_mol.properties["Step"] = step
			prev_mol.properties["Energy"] = energy
			prev_mol.WriteXYZfile("./results/", filename,'a',True)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		# print("Final Energy:", self.EnergyAndForce(prev_mol))
		return prev_mol
