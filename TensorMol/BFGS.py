from Sets import *
from TFManage import *
from PhysicalData import *

def DiagHess(f_,x_,f_x_,eps_=0.0005):
	tore=np.zeros(x_.shape)
	x_t = x_.copy()
	it = np.nditer(x_, flags=['multi_index'])
	while not it.finished:
		x_t = x_.copy()
		x_t[it.multi_index] += eps_
		tore[it.multi_index] = ((f_(x_t) - f_x_)/eps_)[it.multi_index]
		it.iternext()
	return tore

def FdiffGradient(f_, x_, eps_=0.0001):
	"""
	Computes a finite difference gradient at x_ for debugging purposes.
	"""
	tore=np.zeros(x_.shape)
	x_t = x_.copy()
	f_x_ = f_(x_)
	it = np.nditer(x_, flags=['multi_index'])
	while not it.finished:
		x_t = x_.copy()
		x_t[it.multi_index] += eps_
		tore[it.multi_index] = ((f_(x_t) - f_x_)/eps_)
		it.iternext()
	return tore

def LineSearch(f_, x0_, p_):
	'''
	golden section search to find the minimum of f on [a,b]
	Args:
	f_: a function which returns energy.
	x0_: Origin of the search.
	p_: search direction.
	'''
	k=0
	alpha = 0.100
	thresh = 0.0001
	rmsdist = 10.0
	a = x0_
	b = x0_ + alpha*p_
	c = b - (b - a) / GOLDENRATIO
	d = a + (b - a) / GOLDENRATIO
	while (rmsdist > thresh):
		fa = f_(a)
		fb = f_(b)
		fc = f_(c)
		fd = f_(d)
		if (fa < fc and fa < fd):
			print "Line Search: Understep"
		elif (fb < fc and fb < fd):
			print "Line Search: Overstep"
		if fc < fd:
			b = d
		else:
			a = c
		rmsdist = np.sum(np.linalg.norm(a-b,axis=1))/a.shape[0]
		# we recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
		c = b - (b - a) / GOLDENRATIO
		d = a + (b - a) / GOLDENRATIO
		k+=1
	return (b + a) / 2

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
