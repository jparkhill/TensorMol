from Sets import *
from TFManage import *
from PhysicalData import *

def RmsForce(f_):
	return np.sum(np.linalg.norm(f_,axis=1))/f_.shape[0]

def DiagHess(f_,x_,eps_=0.0005):
	"""
	Args:
		f_ returns -1*gradient.
		x_ a guess_
	"""
	tore=np.zeros(x_.shape)
	x_t = x_.copy()
	f_x_ = f_(x_)
	it = np.nditer(x_, flags=['multi_index'])
	while not it.finished:
		x_t = x_.copy()
		x_t[it.multi_index] += eps_
		tore[it.multi_index] = ((f_(x_t) - f_x_)/eps_)[it.multi_index]
		it.iternext()
	return tore

def FdiffGradient(f_, x_, eps_=0.0001):
	"""
	Computes a finite difference gradient of a single or multi-valued function
	at x_ for debugging purposes.
	"""
	x_t = x_.copy()
	f_x_ = f_(x_)
	outshape = x_.shape+f_x_.shape
	tore=np.zeros(outshape)
	it = np.nditer(x_, flags=['multi_index'])
	while not it.finished:
		x_t = x_.copy()
		x_t[it.multi_index] += eps_
		tore[it.multi_index] = ((f_(x_t) - f_x_)/eps_)
		it.iternext()
	return tore

def FdiffHessian(f_, x_, eps_=0.0001):
	"""
	Computes a finite difference hessian of a single or multi-valued function
	at x_ for debugging purposes.
	"""
	x_t = x_.copy()
	f_x_ = f_(x_)
	outshape = x_.shape+x_.shape+f_x_.shape
	tore=np.zeros(outshape)
	iti = np.nditer(x_, flags=['multi_index'])
	tmpshape = x_.shape+f_x_.shape
	tmpfs = np.zeros(tmpshape)
	while not iti.finished:
		xi_t = x_.copy()
		xi_t[iti.multi_index] += eps_
		tmpfs[iti.multi_index]  = f_(xi_t)
		iti.iternext()
	iti = np.nditer(x_, flags=['multi_index'])
	itj = np.nditer(x_, flags=['multi_index'])
	while not iti.finished:
		xi_t = x_.copy()
		xi_t[iti.multi_index] += eps_
		while not itj.finished:
			xij_t = xi_t.copy()
			xij_t[itj.multi_index] += eps_
			tore[iti.multi_index][itj.multi_index] = ((f_(xij_t)-tmpfs[iti.multi_index]-tmpfs[itj.multi_index]+f_x_)/eps_/eps_)
			itj.iternext()
		iti.iternext()
	return tore

def FDiffNormalModes(f_, x_, m_):
	"""
	Perform a finite difference normal mode analysis. 
	"""

def LineSearch(f_, x0_, p_):
	'''
	golden section search to find the minimum of f on [a,b]
	Args:
	f_: a function which returns energy.
	x0_: Origin of the search.
	p_: search direction.
	'''
	k=0
	thresh = 0.00001
	rmsdist = 10.0
	a = x0_
	b = x0_ + PARAMS["GSSearchAlpha"]*p_
	c = b - (b - a) / GOLDENRATIO
	d = a + (b - a) / GOLDENRATIO
	fa = f_(a)
	fb = f_(b)
	fc = f_(c)
	fd = f_(d)
	while (rmsdist > thresh):
		if (fa < fc and fa < fd and fa < fb):
			#print fa,fc,fd,fb
			#print RmsForce(fpa), RmsForce(fpc), RmsForce(fpd), RmsForce(fpb)
			print "Line Search: Overstep"
			PARAMS["GSSearchAlpha"] /= 2.0
			a = x0_
			b = x0_ + PARAMS["GSSearchAlpha"]*p_
			c = b - (b - a) / GOLDENRATIO
			d = a + (b - a) / GOLDENRATIO
			fa = f_(a)
			fb = f_(b)
			fc = f_(c)
			fd = f_(d)
		elif (fb < fc and fb < fd and fb < fa):
			#print fa,fc,fd,fb
			#print RmsForce(fpa), RmsForce(fpc), RmsForce(fpd), RmsForce(fpb)
			print "Line Search: Understep"
			PARAMS["GSSearchAlpha"] *= 2.0
			a = x0_
			b = x0_ + PARAMS["GSSearchAlpha"]*p_
			c = b - (b - a) / GOLDENRATIO
			d = a + (b - a) / GOLDENRATIO
			fa = f_(a)
			fb = f_(b)
			fc = f_(c)
			fd = f_(d)
		elif fc < fd:
			b = d
			c = b - (b - a) / GOLDENRATIO
			d = a + (b - a) / GOLDENRATIO
			fb = fd.copy()
			fc = f_(c)
			fd = f_(d)
		else:
			a = c
			c = b - (b - a) / GOLDENRATIO
			d = a + (b - a) / GOLDENRATIO
			fa = fc.copy()
			fc = f_(c)
			fd = f_(d)
		rmsdist = np.sum(np.linalg.norm(a-b,axis=1))/a.shape[0]
		k+=1
	return (b + a) / 2

def RemoveInvariantForce(x_,f_,m_):
	"""
	Removes center of mass motion and torque from f_, and returns the invariant bits.
	"""
	if (PARAMS["RemoveInvariant"]==False):
		return f_
	#print x_, f_ , m_
	fnet = np.sum(f_,axis=0)
	# Remove COM force.
	fnew_ = f_ - (np.einsum("m,f->mf",m_,fnet)/np.sum(m_))
	torque = np.sum(np.cross(x_,fnew_),axis=0)
	#print torque
	# Compute inertia tensor
	I = np.zeros((3,3))
	for i in range(len(m_)):
		I[0,0] += m_[i]*(x_[i,1]*x_[i,1]+x_[i,2]*x_[i,2])
		I[1,1] += m_[i]*(x_[i,0]*x_[i,0]+x_[i,2]*x_[i,2])
		I[2,2] += m_[i]*(x_[i,1]*x_[i,1]+x_[i,0]*x_[i,0])
		I[0,1] -= m_[i]*(x_[i,0]*x_[i,1])
		I[0,2] -= m_[i]*(x_[i,0]*x_[i,2])
		I[1,2] -= m_[i]*(x_[i,1]*x_[i,2])
	I[1,0] = I[0,1]
	I[2,0] = I[0,2]
	I[2,1] = I[1,2]
	Iinv = PseudoInverse(I)
	#print "Inertia tensor", I
	#print "Inverse Inertia tensor", Iinv
	# Compute angular acceleration  = torque/I
	dwdt = np.dot(Iinv,torque)
	#print "Angular acceleration", dwdt
	# Compute the force correction.
	fcorr = np.zeros(f_.shape)
	for i in range(len(m_)):
		fcorr[i,0] += m_[i]*(-1.0*dwdt[2]*x_[i,1] + dwdt[1]*x_[i,2])
		fcorr[i,1] += m_[i]*(dwdt[2]*x_[i,0] - dwdt[0]*x_[i,2])
		fcorr[i,2] += m_[i]*(-1.0*dwdt[1]*x_[i,0] + dwdt[0]*x_[i,1])
	return fnew_ - fcorr
