from Sets import *
from TFManage import *
from PhysicalData import *
from QuasiNewtonTools import *

def RmsForce(f_):
	return np.sum(np.linalg.norm(f_,axis=1))/f_.shape[0]

class BFGS:
	def __init__(self, m_, ForceAndEnergy_):
		"""
		Simplest Possible BFGS
		"""
		self.m_max = PARAMS["OptMaxBFGS"]
		self.step = 0
		self.m = m_
		self.f = ForceAndEnergy_ # Used for line-search.
		self.R_Hist = np.zeros(([self.m_max]+list(self.m.coords.shape)))
		self.F_Hist = np.zeros(([self.m_max]+list(self.m.coords.shape)))
		return

	def NextStep(self, new_vec_, new_residual_):
		if self.step < self.m_max:
			R_Hist[step] = new_vec_.copy()
			F_Hist[step] = new_residual_.copy()
		else:
			R_Hist = np.roll(R_Hist,-1,axis=0)
			F_Hist = np.roll(R_Hist,-1,axis=0)
			R_Hist[-1] = new_vec_.copy()
			F_Hist[-1] = new_residual_.copy()
		# Quasi Newton L-BFGS global step.
		q = new_residual_.copy()
		for i in range(min(self.m_max,step)-1, 0, -1):
			s = R_Hist[i] - R_Hist[i-1]
			y = F_Hist[i] - F_Hist[i-1]
			rho = 1.0/np.einsum("ia,ia",y,s)#y.dot(s)
			a = rho * np.einsum("ia,ia",s,q)#s.dot(q)
			#print "a ",a
			q -= a*y
		if step < 1:
			H=1.0
		else:
			num = min(self.m_max-1,step)
			v1 = (R_Hist[num] - R_Hist[num-1])
			v2 = (F_Hist[num] - F_Hist[num-1])
			H = (np.einsum("ia,ia",v1,v2))/(np.einsum("ia,ia",v2,v2))
			#print "H:", H
		z = H*q
		for i in range (1,min(self.m_max,step)):
			s = (R_Hist[i] - R_Hist[i-1])
			y = (F_Hist[i] - F_Hist[i-1])
			rho = 1.0/np.einsum("ia,ia",y,s)#y.dot(s)
			a=rho*np.einsum("ia,ia",s,q)#s.dot(q)
			beta = rho*np.einsum("ia,ia",y,z)#(force_his[i] - force_his[i-1]).dot(z)
			#print "a-b: ", (a-beta)
			z += s*(a-beta)
		return self.LineSearch(self.f, new_vec_, z)
