from Sets import *
from TFManage import *
from PhysicalData import *

def RmsForce(f_):
	return np.sum(np.linalg.norm(f_,axis=1))/f_.shape[0]

def InertiaTensor(x_,m_):
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
	return I

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

def CoordinateScan(f_, x_, eps_=0.06, num_=30):
	# Writes a plaintext file containing scans of each coordinate.
	samps = np.logspace(0.0,eps_,num_)-1.0
	samps = np.concatenate((-1*samps[::-1][:-1],samps),axis=0)
	iti = np.nditer(x_, flags=['multi_index'])
	tore = np.zeros(x_.shape+(len(samps),2))
	ci = 0
	while not iti.finished:
		for i,d in enumerate(samps):
			x_t = x_.copy()
			x_t[iti.multi_index] += d
			tore[iti.multi_index][i,0]=d
			tore[iti.multi_index][i,1]=f_(x_t)
		np.savetxt("./results/CoordScan"+str(ci)+".txt",tore[iti.multi_index])
		ci += 1
		iti.iternext()


def FdiffHessian(f_, x_, eps_=0.0001, mode_ = "central", grad_ = None):
	"""
	Computes a finite difference hessian of a single or multi-valued function
	at x_ for debugging purposes.
	Args:
		f_ : objective function of x_
		x_: point at which derivative is taken.
		eps_: finite difference step
		mode_: forward, central, or gradient Differences
		grad_: a gradient function if available.
	"""
	x_t = x_.copy()
	f_x_ = f_(x_)
	outshape = x_.shape+x_.shape+f_x_.shape
	tore=np.zeros(outshape)
	if (mode_ == "gradient" and grad_ != None):
		tmpshape = x_.shape+x_.shape+f_x_.shape
		tmpp = np.zeros(tmpshape)
		tmpm = np.zeros(tmpshape)
		iti = np.nditer(x_, flags=['multi_index'])
		while not iti.finished:
			xi_t = x_.copy()
			xi_t[iti.multi_index] += eps_
			xmi_t = x_.copy()
			xmi_t[iti.multi_index] -= eps_
			tmpp[iti.multi_index]  = grad_(xi_t).copy()
			tmpm[iti.multi_index]  = grad_(xmi_t).copy()
			iti.iternext()
		iti = np.nditer(x_, flags=['multi_index'])
		while not iti.finished:
			itj = np.nditer(x_, flags=['multi_index'])
			while not itj.finished:
				gpjci = tmpp[itj.multi_index][iti.multi_index]
				gmjci = tmpm[itj.multi_index][iti.multi_index]
				gpicj = tmpp[iti.multi_index][itj.multi_index]
				gmicj = tmpm[iti.multi_index][itj.multi_index]
				tore[iti.multi_index][itj.multi_index] = ((gpjci-gmjci)/(4.0*eps_))+((gpicj-gmicj)/(4.0*eps_))
				itj.iternext()
			iti.iternext()
	elif (mode_ == "forward"):
		tmpshape = x_.shape+f_x_.shape
		tmpfs = np.zeros(tmpshape)
		iti = np.nditer(x_, flags=['multi_index'])
		while not iti.finished:
			xi_t = x_.copy()
			xi_t[iti.multi_index] += eps_
			tmpfs[iti.multi_index]  = f_(xi_t).copy()
			print iti.multi_index,tmpfs[iti.multi_index]
			iti.iternext()
		iti = np.nditer(x_, flags=['multi_index'])
		while not iti.finished:
			xi_t = x_.copy()
			xi_t[iti.multi_index] += eps_
			itj = np.nditer(x_, flags=['multi_index'])
			while not itj.finished:
				xij_t = xi_t.copy()
				xij_t[itj.multi_index] += eps_
				tore[iti.multi_index][itj.multi_index] = ((f_(xij_t)-tmpfs[iti.multi_index]-tmpfs[itj.multi_index]+f_x_)/eps_/eps_)
				itj.iternext()
			iti.iternext()
	elif (mode_ == "central"):
		iti = np.nditer(x_, flags=['multi_index'])
		while not iti.finished:
			xi_t = x_.copy()
			xi_t[iti.multi_index] += eps_
			xmi_t = x_.copy()
			xmi_t[iti.multi_index] -= eps_
			itj = np.nditer(x_, flags=['multi_index'])
			while not itj.finished:

				xpipj_t = xi_t.copy()
				xpipj_t[itj.multi_index] += eps_
				xpimj_t = xi_t.copy()
				xpimj_t[itj.multi_index] -= eps_

				xmipj_t = xmi_t.copy()
				xmipj_t[itj.multi_index] += eps_
				xmimj_t = xmi_t.copy()
				xmimj_t[itj.multi_index] -= eps_

				tore[iti.multi_index][itj.multi_index] = (f_(xpipj_t)-f_(xpimj_t)-f_(xmipj_t)+f_(xmimj_t))/(4.0*eps_*eps_)
				itj.iternext()
			iti.iternext()
	return tore

def HarmonicSpectra(f_, x_, m_, grad_=None):
	"""
	Perform a finite difference normal mode analysis
	of a molecule. basically implements http://gaussian.com/vib/
	f_: Energies in Hartree.
	x_: Coordinates (A)
	m_: masses (kg/mol)
	grad_: forces in Hartree/angstrom if available.
	"""
	n = m_.shape[0]
	n3 = 3*n
	cHess = FdiffHessian(f_, x_, 0.04)
	cHess = cHess.reshape((n3,n3))
	# Reshape it to flatten the cartesian parts.
	print "Hess (central):", cHess
	fHess = FdiffHessian(f_,x_,0.04,"forward")
	fHess = fHess.reshape((n3,n3))
	print "Hess (forward):", fHess
	if (grad_ != None):
		gHess = FdiffHessian(f_,x_,0.04,"gradient",grad_)
		gHess = gHess.reshape((n3,n3))
		print "Hess (gradient):", gHess

	Hess = (fHess).copy()
	# Convert from A^-2 to bohr^-2 and
	Hess /= (BOHRPERA*BOHRPERA)
	m = m_.copy()
	m /= MASSOFELECTRON # convert to atomic units of mass.
	# Mass weight.
	for i,mi in enumerate(m):
		Hess[i*n3:(i+1)*n3, i*n3:(i+1)*n3] /= np.sqrt(mi*mi)
		for j,mj in enumerate(m):
			if (i != j):
				Hess[i*n3:(i+1)*n3, j*n3:(j+1)*n3] /= np.sqrt(mi*mj)
	# Get the vibrational spectrum and normal modes.
	w,v = np.linalg.eig(Hess)
	for l in w:
		print "Energy (cm**-1): ", l*WAVENUMBERPERHARTREE

	print "--"

	Hess = (cHess).copy()
	# Convert from A^-2 to bohr^-2 and
	Hess /= (BOHRPERA*BOHRPERA)
	m = m_.copy()
	m /= MASSOFELECTRON # convert to atomic units of mass.
	# Mass weight.
	for i,mi in enumerate(m):
		Hess[i*n3:(i+1)*n3, i*n3:(i+1)*n3] /= np.sqrt(mi*mi)
		for j,mj in enumerate(m):
			if (i != j):
				Hess[i*n3:(i+1)*n3, j*n3:(j+1)*n3] /= np.sqrt(mi*mj)
	# Get the vibrational spectrum and normal modes.
	w,v = np.linalg.eig(Hess)
	for l in w:
		print "Energy (cm**-1): ", l*WAVENUMBERPERHARTREE

	print "--"

	Hess = (gHess).copy()
	# Convert from A^-2 to bohr^-2 and
	Hess /= (BOHRPERA*BOHRPERA)
	m = m_.copy()
	m /= MASSOFELECTRON # convert to atomic units of mass.
	# Mass weight.
	for i,mi in enumerate(m):
		Hess[i*n3:(i+1)*n3, i*n3:(i+1)*n3] /= np.sqrt(mi*mi)
		for j,mj in enumerate(m):
			if (i != j):
				Hess[i*n3:(i+1)*n3, j*n3:(j+1)*n3] /= np.sqrt(mi*mj)
	# Get the vibrational spectrum and normal modes.
	w,v = np.linalg.eig(Hess)
	for l in w:
		print "Energy (cm**-1): ", l*WAVENUMBERPERHARTREE

	print "--"

	Hess = (fHess+cHess)/2.0
	# Convert from A^-2 to bohr^-2 and
	Hess /= (BOHRPERA*BOHRPERA)
	m = m_.copy()
	m /= MASSOFELECTRON # convert to atomic units of mass.
	# Mass weight.
	for i,mi in enumerate(m):
		Hess[i*n3:(i+1)*n3, i*n3:(i+1)*n3] /= np.sqrt(mi*mi)
		for j,mj in enumerate(m):
			if (i != j):
				Hess[i*n3:(i+1)*n3, j*n3:(j+1)*n3] /= np.sqrt(mi*mj)
	# Get the vibrational spectrum and normal modes.
	w,v = np.linalg.eig(Hess)
	for l in w:
		print "Energy (cm**-1): ", l*WAVENUMBERPERHARTREE
	# Construct internal coordinates.
	D = np.zeros((n3,n3))
	return

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
			PARAMS["GSSearchAlpha"] /= 2.03
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
	I = InertiaTensor(x_,m_)
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
