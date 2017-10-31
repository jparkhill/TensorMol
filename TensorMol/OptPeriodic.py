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
		PARAMS["OptLatticeStep"] = 0.050
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print("Orig Mol:\n", m)
		def WrappedEForce(x_,DoForce=True):
			if (DoForce):
				energy, frc = self.EnergyAndForce(x_, DoForce)
				frc = RemoveInvariantForce(x_, frc, m.atoms)
				frc /= JOULEPERHARTREE
				return energy, frc
			else:
				energy = self.EnergyAndForce(x_,False)
				return energy
		CG = ConjGradient(WrappedEForce, m.coords)
		Density = self.EnergyAndForce.Density()
		while( step < self.max_opt_step and rmsgrad > self.thresh and rmsdisp > 0.0001):
			prev_m = Mol(m.atoms, m.coords)
			m.coords, energy, frc = CG(m.coords)
			rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/m.coords.shape[0]
			rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/m.coords.shape[0]
			m.coords = self.EnergyAndForce.lattice.ModuloLattice(m.coords)
			print("step: ", step ," energy: ", energy," density: ", Density, " rmsgrad ", rmsgrad, " rmsdisp ", rmsdisp)
			mol_hist.append(prev_m)
			prev_m.properties['Lattice']=self.EnergyAndForce.lattice.lattice.copy()
			prev_m.WriteXYZfile("./results/", filename,'a',True)
			Mol(*self.EnergyAndForce.lattice.TessNTimes(prev_m.atoms,prev_m.coords,2)).WriteXYZfile("./results/", "Tess"+filename,'a',wprop=True)
			step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		print("Final Energy:", self.EnergyAndForce(prev_m.coords,False))
		self.EnergyAndForce.Save(prev_m.coords,"FinalPeriodicOpt")
		self.EnergyAndForce.mol0.coords = prev_m.coords.copy()
		return prev_m

	def OptToDensity(self, m, rho_target=1.0, filename="PdicOptLog",Debug=False):
		"""
		Optimize using An EnergyAndForce Function.
		Squeeze the lattice gently such that the atoms achieve a target density.

		Args:
			m: A distorted molecule to optimize
			rho_target: Target density (g/cm**3)
		"""
		# Sweeps one at a time
		PARAMS["OptLatticeStep"] = 0.050
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
		Density = self.EnergyAndForce.Density()
		def WrappedEForce(x_,DoForce=True):
			if (DoForce):
				energy, frc = self.EnergyAndForce(x_, DoForce)
				frc = RemoveInvariantForce(x_, frc, m.atoms)
				frc /= JOULEPERHARTREE
				return energy, frc
			else:
				energy = self.EnergyAndForce(x_,False)
				return energy
		CG = ConjGradient(WrappedEForce, m.coords)
		while (abs(Density-rho_target) > 0.001):
			step = 0
			Density = self.EnergyAndForce.Density()
			fac = rho_target/Density
			oldlat = self.EnergyAndForce.lattice.lattice.copy()
			newlat = 0.65*oldlat + 0.35*(oldlat*pow(1.0/fac,1.0/3.))
			m.coords = self.EnergyAndForce.AdjustLattice(m.coords,oldlat,newlat)
			self.EnergyAndForce.ReLattice(newlat)
			rmsgrad = 10.0
			rmsdisp = 10.0
			while( step < self.max_opt_step and rmsgrad > self.thresh and rmsdisp > 0.0001 ):
				prev_m = Mol(m.atoms, m.coords)
				m.coords, energy, frc = CG(m.coords)
				rmsgrad = np.sum(np.linalg.norm(frc,axis=1))/veloc.shape[0]
				rmsdisp = np.sum(np.linalg.norm(m.coords-prev_m.coords,axis=1))/veloc.shape[0]
				m.coords = self.EnergyAndForce.lattice.ModuloLattice(m.coords)
				print("step: ", step ," energy: ", energy," density: ", Density, " rmsgrad ", rmsgrad, " rmsdisp ", rmsdisp)
				mol_hist.append(prev_m)
				prev_m.properties['Lattice']=self.EnergyAndForce.lattice.lattice.copy()
				prev_m.WriteXYZfile("./results/", filename,'a',True)
				Mol(*self.EnergyAndForce.lattice.TessNTimes(prev_m.atoms,prev_m.coords,2)).WriteXYZfile("./results/", "Tess"+filename,'a',wprop=True)
				step+=1
		# Checks stability in each cartesian direction.
		#prev_m.coords = LineSearchCart(Energy, prev_m.coords)
		print("Final Energy:", Energy(prev_m.coords))
		self.EnergyAndForce.Save(prev_m.coords,"FinalPeriodicOpt")
		self.EnergyAndForce.mol0.coords = prev_m.coords.copy()
		return prev_m

def Solvate(f_,m1_,m2_,shells_=3,rho_target=0.87):
	"""
	Takes a PeriodicForce which includes an initial molecule, and solvates it with
	shells_ of M2. and optimized to a desired density.
	"""
	xmn1 = min(np.min(PF_.mol0.coords),np.min(m2_.coords))
	xmx1 = max(np.max(PF_.mol0.coords),np.max(m2_.coords))
	dx = xmx1-xmn1
	lat = Lattice(np.array([[dx,0.,0.],[0.,dx,0.],[0.,0.,dx]]))
	m = lat.CenteredInLattice(m2_)
	nat = m1_.NAtoms()
	natsolv = m.NAtoms()
	sa,sc = lat.TessNTimes(m.atoms,m.coords,2*shells_+1)
	# cell index of the center.
	ccell = shells_*pow(2*shells_+1,2)+shells_*pow(2*shells_+1,1)+shells_
	# Replace the center cell with the solute.
	sa = np.concatenate([m1_.atoms,sa[:ccell*natsolv],sa[(ccell+1)*natsolv:]])
	sc = np.concatenate([m1_.coords-np.mean(m1_,axis=0)[np.newaxis,:],sa[:ccell*natsolv,:],sa[(ccell+1)*natsolv:,:]],axis=0)
	m = Mol(sa,sc)
	m.WriteXYZfile("./results","solvated")
	m.coords -= np.min(m.coords)
	xmx = np.max(m.coords)
	PF = PeriodicForce(m,np.array([[xmx,0.,0.],[0.,xmx,0.],[0.,0.,xmx]]))
	PF.BindForce(f_, 15.0)
	opt = PeriodicGeomOptimizer(PF)
	msolv = opt.OptToDensity(m,rho_target)
	return msolv
