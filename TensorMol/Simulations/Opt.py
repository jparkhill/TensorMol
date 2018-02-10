"""
We should get all references to TFManage out of this
and just pass a EnergyAndForce Field function.
"""
from __future__ import absolute_import
from __future__ import print_function
from ..Containers.Sets import *
from ..TFNetworks.TFManage import *
from ..Math.QuasiNewtonTools import *
from ..Math.DIIS import *
from ..Math.BFGS import *
from ..Math.LinearOperations import *
from ..ForceModels import *
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

	def Opt(self,m_, filename="OptLog",Debug=False):
		"""
		Optimize using An EnergyAndForce Function with conjugate gradients.

		Args:
			m: A distorted molecule to optimize
		"""
		m = Mol(m_.atoms,m_.coords)
		self.m = m
		rmsdisp = 10.0
		rmsgrad = 10.0
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
			LOGGER.info(filename+"step: %i energy: %0.5f rmsgrad: %0.5f rmsdisp: %0.5f ", step , energy, rmsgrad, rmsdisp)
			mol_hist.append(prev_m)
			prev_m.properties["Step"] = step
			prev_m.properties["Energy"] = energy
			prev_m.WriteXYZfile("./results/", filename,'a',True)
			step+=1
		# Checks stability in each cartesian direction.
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False))
		return prev_m

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
		PF += JOULEPERHARTREE*BF[0]
		PF = RemoveInvariantForce(x_,PF,self.m)
		return BE+self.RealPot, PF

	def Bump(self):
		self.BumpCoords[self.NBump%self.MaxBumps] = self.x
		self.NBump += 1
		LOGGER.info("Bump added!")
		return

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

	def OptGD(self,m_, filename="GDOptLog",Debug=False):
		"""
		Optimize using steepest descent  and an EnergyAndForce Function.

		Args:
		        m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		rmsdisp = 10.0
		rmsgrad = 10.0
		step=0
		ndives = 0
		m = Mol(m_.atoms,m_.coords)
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print("Orig Coords", m.coords)
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		energy, old_frc  = self.WrappedEForce(m.coords)
		while( step < self.max_opt_step and rmsgrad > self.thresh):
			prev_m = Mol(m.atoms, m.coords)
			if step > 0:
				old_frc = frc
			energy, frc = self.WrappedEForce(m.coords)
			if (np.sum(frc*old_frc)<0.0):
				old_frc *= 0.0
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/frc.shape[0]
			frc += self.momentum*old_frc
			m.coords = m.coords + self.fscale*frc
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/m.coords.shape[0]
			LOGGER.info(filename+"step: %i energy: %0.5f rmsgrad: %0.5f rmsdisp: %0.5f ", step , energy, rmsgrad, rmsdisp)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		return prev_m

class MetaOptimizer(GeomOptimizer):
	def __init__(self,f_,m,StopAfter_=20,Box_=False,OnlyHev_=True):
		"""
		A Meta-Optimizer performs nested optimization.
		The outer loop has a bump potential to find new initial geometries.
		the inner loop digs down to new minima.
		it saves the record of minima it reaches.
		The default parameters are tuned to find nearby reactions.
		Conformational search should probably be done with internal coordinates
		not distance matrices.

		Args:
			f_: An EnergyForce routine
			m: a molecules
			StopAfter_: Look for this many nearby minima.
			Box_: whether to use a box.
			OnlyHev_: whether to only bump heavy atom bonds.
		"""
		GeomOptimizer.__init__(self,f_)
		self.thresh = PARAMS["OptThresh"]*5.0
		self.StopAfter = StopAfter_
		self.OnlyHev = OnlyHev_
		self.m = m
		self.fscale = 0.3
		self.momentum = 0.3
		self.thresh = 0.005
		self.masses = np.array(map(lambda x: ATOMICMASSES[x-1], m.atoms))
		self.natoms = m.NAtoms()
		self.MaxBumps = PARAMS["MetaMaxBumps"] # think you want this to be >500k
		self.BumpCoords = np.zeros((self.MaxBumps,self.natoms,3))
		self.MinimaCoords = np.zeros((self.StopAfter,self.natoms,3))
		self.NMinima = 0
		self.NBump = 0
		self.UseBox = Box_
		self.Boxer = TFForces.BoxHolder(self.natoms)
		self.lastbumpstep = 0
		# just put the atoms in a box the size of their max and min coordinates.
		self.Box =  Box_=np.array((np.max(m.coords)+0.1)*np.eye(3))
		self.BowlK = 0.01
		#self.Bumper = TFForces.BumpHolder(self.natoms, self.MaxBumps, self.BowlK, h_=1.0, w_=1.2,Type_="MR")
		self.Bumper = TFForces.BumpHolder(self.natoms, self.MaxBumps, self.BowlK, h_=0.5, w_=0.6,Type_="MR")
		return

	def WrappedBumpedEForce(self, x_ ,DoForce = True, DoBump=True):
		PE,PF = None, None
		if (DoForce):
			PE, PF = self.EnergyAndForce(x_, DoForce)
			if (not DoBump):
				return PE,PF
		else:
			PE = self.EnergyAndForce(x_, DoForce)
			if (not DoBump):
				return PE
		BxE = 0.0
		BxF = np.zeros(x_.shape)
		if (self.UseBox):
			BxE, BxF = self.Boxer(x_, self.Box)
			BxF *= -5.0*JOULEPERHARTREE#*(self.masses[:,None]/np.sqrt(np.sum(self.masses*self.masses)))
		#print("Box Force",np.max(x_),np.max(BxF),BxE)
		BE = 0.0
		BF = np.zeros(x_.shape)
		if (self.NBump > 0):
			BE, BF = self.Bumper.Bump(self.BumpCoords.astype(np.float32), x_.astype(np.float32), self.NBump%self.MaxBumps)
			BF = JOULEPERHARTREE*BF[0]
			if (self.OnlyHev):
				for i in range(self.m.NAtoms()):
					if (self.m.atoms[i]==1):
						BF[i] *= 0.0
		if (DoForce):
			frc = PF+BF+BxE
			frc = RemoveInvariantForce(x_, frc, self.m.atoms)
			frc /= JOULEPERHARTREE
			rmsgrad = np.sum(np.linalg.norm(PF,axis=1))/PF.shape[0]
			rmsgradb = np.sum(np.linalg.norm(BF,axis=1))/PF.shape[0]
			print(rmsgradb,rmsgrad)

			return BE+PE+BxE,frc
		else:
			return BE+PE+BxE

	def Bump(self,x_):
		self.BumpCoords[self.NBump%self.MaxBumps] = x_
		self.NBump += 1
		LOGGER.info("Bump added!")
		return

	def MetaOptCG(self,m, filename="MetaOptLog",Debug=False, SearchConfs_=False):
		"""
		Optimize using An EnergyAndForce Function with conjugate gradients.

		Args:
			m: A distorted molecule to optimize
		"""
		self.m = m
		rmsdisp = 10.0
		rmsgrad = 10.0
		step=0
		mol_hist = []
		ndives=0
		prev_m = Mol(m.atoms, m.coords)
		print("Orig Mol:\n", m)
		BM = m.BondMatrix()
		CG = ConjGradient(self.WrappedBumpedEForce, m.coords)
		while(step < self.max_opt_step):
			while( step < self.max_opt_step and rmsgrad > self.thresh and (rmsdisp > 0.000001 or step<5) ):
				prev_m = Mol(m.atoms, m.coords)
				m.coords, energy, frc = CG(m.coords)
				rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/m.coords.shape[0]
				rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/m.coords.shape[0]
				LOGGER.info(filename+"step: %i energy: %0.5f rmsgrad: %0.5f rmsdisp: %0.5f ", step , energy, rmsgrad, rmsdisp)
				mol_hist.append(prev_m)
				prev_m.properties["Step"] = step
				prev_m.properties["Energy"] = energy
				prev_m.WriteXYZfile("./results/", filename,'a',True)
				step+=1
			self.Bump(m.coords)
			m.Distort(0.01)
			if ((BM != prev_m.BondMatrix()).any() or SearchConfs_):
				d = self.Opt(prev_m,"Dive"+str(ndives))
				BM = prev_m.BondMatrix()
				self.AppendIfNew( d )
				self.Bump(d.coords)
				ndives += 1
			rmsdisp = 10.0
			rmsgrad = 10.0
			step=0
			PARAMS["GSSearchAlpha"]=0.1
			CG = ConjGradient(self.WrappedBumpedEForce, m.coords)
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False))
		return prev_m

	def MetaOpt(self,m_=None, filename="MetaOptLog",Debug=False, SearchConfs_=False):
		"""
		Optimize using steepest descent  and an EnergyAndForce Function.

		Args:
		        m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		rmsdisp = 10.0
		rmsgrad = 10.0
		step=0
		ndives = 0
		m = Mol(self.m.atoms,self.m.coords)
		if (m_ != None):
			m = Mol(m_.atoms,m_.coords)
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print("Orig Coords", m.coords)
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		energy, old_frc  = self.WrappedBumpedEForce(m.coords)
		BM = m.BondMatrix()
		while(self.NMinima < self.StopAfter):
			while( step < self.max_opt_step and rmsgrad > self.thresh):
				prev_m = Mol(m.atoms, m.coords)
				if step > 0:
					old_frc = frc
				energy, frc = self.WrappedBumpedEForce(m.coords)
				if (np.sum(frc*old_frc)<0.0):
					old_frc *= 0.0
				rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/frc.shape[0]
				frc += self.momentum*old_frc
				m.coords = m.coords + self.fscale*frc
				rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/m.coords.shape[0]
				LOGGER.info(filename+"step: %i energy: %0.5f rmsgrad: %0.5f rmsdisp: %0.5f ", step , energy, rmsgrad, rmsdisp)
				mol_hist.append(prev_m)
				prev_m.WriteXYZfile("./results/", filename)
				step+=1
			self.Bump(m.coords)
			m.Distort(0.01)
			if ((BM != prev_m.BondMatrix()).any() or SearchConfs_):
				d = self.OptGD(prev_m,"Dive"+str(ndives))
				BM = prev_m.BondMatrix()
				self.AppendIfNew( d )
				self.Bump(d.coords)
				ndives += 1
			rmsdisp = 10.0
			rmsgrad = 10.0
			step=0
			PARAMS["GSSearchAlpha"]=0.1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		return self.MinimaCoords

	def AppendIfNew(self,m):
		overlaps = []
		if (self.NMinima==0):
			print("New Configuration!")
			m.WriteXYZfile("./results/","NewMin"+str(self.NMinima))
			self.MinimaCoords[self.NMinima] = m.coords
			self.NMinima += 1
			self.Bump(m.coords)
			return
		for i in range(self.NMinima):
			mdm = MolEmb.Make_DistMat(self.MinimaCoords[i])
			odm = MolEmb.Make_DistMat(m.coords)
			tmp = (mdm-odm)
			overlaps.append(np.sqrt(np.sum(tmp*tmp)/(mdm.shape[0]*mdm.shape[0])))
		if (min(overlaps) > self.thresh):
			print("New Configuration!")
			m.WriteXYZfile("./results/","NewMin"+str(self.NMinima))
			self.MinimaCoords[self.NMinima] = m.coords.copy()
			self.NMinima += 1
			self.Bump(m.coords)
		else:
			print("Overlaps", overlaps)
		return
