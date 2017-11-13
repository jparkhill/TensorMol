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
from . import TFForces
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
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False))
		return prev_m

	def MetaOptimization(self,m, filename="OptLog",Debug=False):
		"""
		This routine weakly repeatedly converges optimizations

		Args:
			m: A distorted molecule to optimize
		"""
		self.m = m
		self.natoms = m.Natoms()
		self.MaxBumps = PARAMS["MetaMaxBumps"] # think you want this to be >500k
		self.BumpCoords = np.zeros((self.MaxBumps,self.natoms,3))
		self.NBump = 0
		self.Boxer = TFForces.BoxHolder(self.natoms)
		self.Box = Box_.copy()
		self.BowlK = 0.0
		self.Bumper = TFForces.BumpHolder(self.natoms, self.MaxBumps, self.BowlK)
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
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False))
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
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False))
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

class MetaOptimizer(GeomOptimizer):
	def __init__(self,f_,m,Box_=False):
		"""
		A Meta-Optimizer performs repeated optimizatons
		Each time converging to a new local minimum,
		and saves the record of minima it reaches.

		Args:
			f_: An EnergyForce routine
		"""
		GeomOptimizer.__init__(self,f_)
		self.m = m
		self.masses = np.array(map(lambda x: ATOMICMASSES[x-1], m.atoms))
		self.natoms = m.NAtoms()
		self.MaxBumps = PARAMS["MetaMaxBumps"] # think you want this to be >500k
		self.BumpCoords = np.zeros((self.MaxBumps,self.natoms,3))
		self.NBump = 0
		self.UseBox = Box_
		self.Boxer = TFForces.BoxHolder(self.natoms)
		self.lastbumpstep = 0
		# just put the atoms in a box the size of their max and min coordinates.
		self.Box =  Box_=np.array((np.max(m.coords)+0.1)*np.eye(3))
		self.BowlK = 0.0
		self.Bumper = TFForces.BumpHolder(self.natoms, self.MaxBumps, self.BowlK, h_=0.5,w_=3.0,Type_="MR")
		return

	def WrappedEForce(self, x_ ,DoForce = True):
		BxE = 0.0
		BxF = np.zeros(x_.shape)
		if (self.UseBox):
			BxE, BxF = self.Boxer(x_, self.Box)
			BxF *= -5.0*JOULEPERHARTREE#*(self.masses[:,None]/np.sqrt(np.sum(self.masses*self.masses)))
		#print("Box Force",np.max(x_),np.max(BxF),BxE)
		PE,PF = None, None
		if (DoForce):
			PE, PF = self.EnergyAndForce(x_, DoForce)
		else:
			PE = self.EnergyAndForce(x_, DoForce)
		BE = 0.0
		BF = np.zeros(x_.shape)
		if (self.NBump > 0):
			BE, BF = self.Bumper.Bump(self.BumpCoords.astype(np.float32), x_.astype(np.float32), self.NBump%self.MaxBumps)
			#print(BF)
			#BF[0] *= self.masses[:,None]
			BE *= -1.0
			BF = JOULEPERHARTREE*BF[0]
		if (DoForce):
			frc = PF+BF+BxE
			frc = RemoveInvariantForce(x_, frc, self.m.atoms)
			frc /= JOULEPERHARTREE
			return BE+PE+BxE,frc
		else:
			return BE+PE+BxE

	def Bump(self,x_):
		self.BumpCoords[self.NBump%self.MaxBumps] = x_
		self.NBump += 1
		LOGGER.info("Bump added!")
		return

	def Opt(self,m, filename="OptLog",Debug=False):
		"""
		This routine weakly repeatedly converges optimizations
		and bumps when convergence is achieved.

		Args:
			m: A distorted molecule to optimize
		"""
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		nlocmin = 0
		prev_m = Mol(m.atoms, m.coords)
		print("Orig Mol:\n", m)
		CG = ConjGradient(self.WrappedEForce, m.coords,0.0005)
		while( step < self.max_opt_step*self.MaxBumps and (rmsgrad > self.thresh or (rmsdisp > 0.00001 or step<5)) ):
			prev_m = Mol(m.atoms, m.coords)
			m.coords, energy, frc = CG(m.coords)
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/m.coords.shape[0]
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/m.coords.shape[0]
			LOGGER.info("step: %i energy: %0.5f rmsgrad: %0.5f rmsdisp: %0.5f ", step , energy, rmsgrad, rmsdisp)
			if (rmsgrad < 0.002 and abs(self.lastbumpstep-step)>3):
				LOGGER.info("Added local minimum...")
				self.lastbumpstep = step
				self.Bump(m.coords)
				# Because there's no gradient at x, give the coordinates a very weak nudge.
				m.coords += np.random.normal(0.0005,shape=(x.coords.shape))
				# This reliably fucks the line search stepsize, so help it out.
				CG.alpha /= 10.0
				prev_m.WriteXYZfile("./results/", filename+"LM"+str(nlocmin),'a',True)
				rmsgrad = 10.0
				nlocmin += 1
			mol_hist.append(prev_m)
			prev_m.properties["Step"] = step
			prev_m.properties["Energy"] = energy
			prev_m.WriteXYZfile("./results/", filename,'a',True)
			step+=1
		# Checks stability in each cartesian direction.
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False))
		return prev_m

class GeometryOptimizer:
	"""
	John: Don't do this! Just modify the old code, or inherit from it!
	This is how garbage is created.
	What exactly was the reason for reduplicating this class? JAP 2017.
	"""
	def __init__(self, force_field):
		"""
		Geometry optimizations based on NN-PES's etc.

		Args:
			force_field: An EnergyForce routine
		"""
		self.thresh = PARAMS["OptThresh"]
		self.maxstep = PARAMS["OptMaxStep"]
		self.fscale = PARAMS["OptStepSize"]
		self.momentum = PARAMS["OptMomentum"]
		self.momentum_decay = PARAMS["OptMomentumDecay"]
		self.max_opt_step = PARAMS["OptMaxCycles"]
		self.step = self.maxstep
		self.force_field = force_field
		return

	def opt_conjugate_gradient(self, mol, filename="OptLog"):
		"""
		Optimize using An EnergyAndForce Function with conjugate gradients.

		Args:
			mol (TensorMol.Mol): a TensorMol molecule object
		"""
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		prev_mol = Mol(mol.atoms, mol.coords)
		print("Orig Mol:\n", mol)
		CG = ConjugateGradientDirect(self.force_field, mol)
		while( step < self.max_opt_step and rmsgrad > self.thresh and (rmsdisp > 0.000001 or step<5) ):
			prev_mol = Mol(mol.atoms, mol.coords)
			mol, energy, forces = CG(mol)
			rmsgrad = np.sum(np.linalg.norm(forces, axis=1)) / mol.coords.shape[0]
			rmsdisp = np.sum(np.linalg.norm(mol.coords - prev_mol.coords, axis=1)) / mol.coords.shape[0]
			LOGGER.info("step: %i energy: %0.5f rmsgrad: %0.5f rmsdisp: %0.5f ", step, energy, rmsgrad, rmsdisp)
			mol_hist.append(prev_mol)
			prev_mol.properties["Step"] = step
			prev_mol.properties["Energy"] = energy
			prev_mol.WriteXYZfile("./results/", filename, 'a', True)
			step+=1
		print("Final Energy:", self.force_field(prev_mol, False))
		return prev_mol

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
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False))
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
