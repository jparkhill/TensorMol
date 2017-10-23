"""
TODO:
	Systematic comparison of BFGS vs CG etc.
	Consistent solver organization & interface. (CG,BFGS,DIIS etc. )
"""
from __future__ import absolute_import
from .Sets import *
from .TFManage import *
from .PhysicalData import *

class SteepestDescent:
	def __init__(self, ForceAndEnergy_,x0_):
		"""
		The desired interface for a solver in tensormol.

		Args:
			ForceAndEnergy_: a routine which returns energy, force.
			x0_: a initial vector
		"""
		self.step = 0
		self.x0=x0_.copy()
		if (len(self.x0.shape)==2):
			self.natom = self.x0.shape[0]
		else:
			self.natom = self.x0.shape[0]*self.x0.shape[1]
		self.EForce = ForceAndEnergy_ # Used for line-search.
		return
	def __call__(self, new_vec_):
		"""
		Iterate BFGS

		Args:
			new_vec_: Point at which to minimize gradients
		Returns:
			Next point, energy, and gradient.
		"""
		e,g = self.EForce(new_vec_)
		self.step += 1
		return new_vec_ + 0.01*g, e, g

class BFGS(SteepestDescent):
	def __init__(self, ForceAndEnergy_,x0_):
		"""
		Simplest Possible minimizing BFGS

		Args:
			ForceAndEnergy_: a routine which returns energy, force.
			x0_: a initial vector
		"""
		self.m_max = PARAMS["MaxBFGS"]
		self.step = 0
		self.x0=x0_.copy()
		if (len(self.x0.shape)==2):
			self.natom = self.x0.shape[0]
		else:
			self.natom = self.x0.shape[0]*self.x0.shape[1]
		self.EForce = ForceAndEnergy_ # Used for line-search.
		self.R_Hist = np.zeros(([self.m_max]+list(self.x0.shape)))
		self.F_Hist = np.zeros(([self.m_max]+list(self.x0.shape)))
		return
	def BFGSstep(self, new_vec_, new_residual_):
		if self.step < self.m_max:
			self.R_Hist[self.step] = new_vec_.copy()
			self.F_Hist[self.step] = new_residual_.copy()
		else:
			self.R_Hist = np.roll(self.R_Hist,-1,axis=0)
			self.F_Hist = np.roll(self.F_Hist,-1,axis=0)
			self.R_Hist[-1] = new_vec_.copy()
			self.F_Hist[-1] = new_residual_.copy()
		# Quasi Newton L-BFGS global step.
		q = new_residual_.copy()
		for i in range(min(self.m_max,self.step)-1, 0, -1):
			s = self.R_Hist[i] - self.R_Hist[i-1]
			y = self.F_Hist[i] - self.F_Hist[i-1]
			rho = 1.0/np.sum(y*s)
			a = rho * np.sum(s*q)
			#print "a ",a
			q -= a*y
		if self.step < 1:
			H=1.0
		else:
			num = min(self.m_max-1,self.step)
			v1 = (self.R_Hist[num] - self.R_Hist[num-1])
			v2 = (self.F_Hist[num] - self.F_Hist[num-1])
			H = np.sum(v1*v2)/np.sum(v2*v2)
			#print "H:", H
		z = H*q
		for i in range (1,min(self.m_max,self.step)):
			s = (self.R_Hist[i] - self.R_Hist[i-1])
			y = (self.F_Hist[i] - self.F_Hist[i-1])
			rho = 1.0/np.sum(y*s)
			a=rho*np.sum(s*q)
			beta = rho*np.sum(y*z)
			#print "a-b: ", (a-beta)
			z += s*(a-beta)
		self.step += 1
		return z
	def __call__(self, new_vec_):
		"""
		Iterate BFGS

		Args:
			new_vec_: Point at which to minimize gradients
		Returns:
			Next point, energy, and gradient.
		"""
		e,g = self.EForce(new_vec_)
		z = self.BFGSstep(new_vec_, g)
		return new_vec_ - 0.005*z, e, g

class BFGS_WithLinesearch(BFGS):
	def __init__(self, ForceAndEnergy_, x0_ ):
		"""
		Simplest Possible BFGS

		Args:
			ForceAndEnergy_: a routine which returns energy, force.
			x0_: a initial vector
		"""
		BFGS.__init__(self,ForceAndEnergy_,x0_)
		self.alpha = PARAMS["GSSearchAlpha"]
		self.Energy = lambda x: self.EForce(x,False)
		return
	def LineSearch(self, x0_, p_, thresh = 0.0001):
		'''
		golden section search to find the minimum of f on [a,b]

		Args:
			f_: a function which returns energy.
			x0_: Origin of the search.
			p_: search direction.

		Returns:
			x: coordinates which minimize along this search direction.
		'''
		k=0
		rmsdist = 10.0
		a = x0_
		b = x0_ + self.alpha*p_
		c = b - (b - a) / GOLDENRATIO
		d = a + (b - a) / GOLDENRATIO
		fa = self.Energy(a)
		fb = self.Energy(b)
		fc = self.Energy(c)
		fd = self.Energy(d)
		while (rmsdist > thresh):
			if (fa < fc and fa < fd and fa < fb):
				#print fa,fc,fd,fb
				#print RmsForce(fpa), RmsForce(fpc), RmsForce(fpd), RmsForce(fpb)
				print("Line Search: Overstep")
				if (self.alpha > 0.00001):
					self.alpha /= 1.71
				else:
					print("Keeping step")
					return a
				a = x0_
				b = x0_ + self.alpha*p_
				c = b - (b - a) / GOLDENRATIO
				d = a + (b - a) / GOLDENRATIO
				fa = self.Energy(a)
				fb = self.Energy(b)
				fc = self.Energy(c)
				fd = self.Energy(d)
			elif (fb < fc and fb < fd and fb < fa):
				#print fa,fc,fd,fb
				#print RmsForce(fpa), RmsForce(fpc), RmsForce(fpd), RmsForce(fpb)
				print("Line Search: Understep")
				if (self.alpha < 100.0):
					self.alpha *= 1.7
				a = x0_
				b = x0_ + self.alpha*p_
				c = b - (b - a) / GOLDENRATIO
				d = a + (b - a) / GOLDENRATIO
				fa = self.Energy(a)
				fb = self.Energy(b)
				fc = self.Energy(c)
				fd = self.Energy(d)
			elif fc < fd:
				b = d
				c = b - (b - a) / GOLDENRATIO
				d = a + (b - a) / GOLDENRATIO
				fb = fd
				fc = self.Energy(c)
				fd = self.Energy(d)
			else:
				a = c
				c = b - (b - a) / GOLDENRATIO
				d = a + (b - a) / GOLDENRATIO
				fa = fc
				fc = self.Energy(c)
				fd = self.Energy(d)
			rmsdist = np.sum(np.linalg.norm(a-b,axis=1))/self.natom
			k+=1
		return (b + a) / 2
	def __call__(self, new_vec_):
		"""
		Iterate BFGS

		Args:
			new_vec_: Point at which to minimize gradients
		Returns:
			Next point, energy, and gradient.
		"""
		e,g = self.EForce(new_vec_)
		z = self.BFGSstep(new_vec_, g)
		new_vec = self.LineSearch(new_vec_, z)
		return new_vec, e, g
