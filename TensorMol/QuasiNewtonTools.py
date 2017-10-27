"""
Routines which help do differential analysis and Newtonian Mechanics
"""

from __future__ import absolute_import
from __future__ import print_function
from .Sets import *
from .TFManage import *
from .PhysicalData import *

def RmsForce(f_):
	return np.mean(np.linalg.norm(f_,axis=1))
def CenterOfMass(x_,m_):
	return (np.einsum("m,mx->x",m_,x_)/np.sum(m_))
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
def CoordinateScan(f_, x_, name_="", eps_=0.03, num_=15):
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
		np.savetxt("./results/CoordScan"+name_+str(ci)+".txt",tore[iti.multi_index])
		ci += 1
		iti.iternext()
def FdiffHessian(f_, x_, eps_=0.001, mode_ = "forward", grad_ = None):
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
			print(iti.multi_index,tmpfs[iti.multi_index])
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
def FourPointHessQuad(f):
	"""
	f is a 4x4xOutshape
	sampling eps*[-2, -1, 1, 2]
	"""
	term1 = -63.0*(f[2,0]+f[3,1]+f[0,2]+f[1,3])
	term2 =  63.0*(f[1,0]+f[0,1]+f[2,3]+f[3,2])
	term3 =  44.0*(f[3,0]+f[0,3]-f[0,0]-f[3,3])
	term4 =  74.0*(f[1,1]+f[2,2]-f[2,1]-f[1,2])
	return (term1+term2+term3+term4)/600.0
def DirectedFdiffHessian(f_, x_, dirs_, eps_=0.01):
	"""
	Four-Point Hessian quadrature along dirs_ directions.

	Args:
		dirs_ : a set of directions having x_'s shape
	Returns:
		d^2 f/ (d dirs_i, d dirs_j)
	"""
	f_x_ = f_(x_)
	N = dirs_.shape[0]
	tore = np.zeros((N,N))
	for i in range(N):
		for j in range(i,N):
			samps = np.zeros((4,4)+f_x_.shape)
			for ic,di in enumerate([-2.,-1.,1.,2.]):
				for jc,dj in enumerate([-2.,-1.,1.,2.]):
					samps[ic,jc] = f_(x_+dirs_[i]*eps_*di+dirs_[j]*eps_*dj)
			tore[i,j] = FourPointHessQuad(samps)/eps_/eps_
			tore[j,i] = tore[i,j]
	return tore
def InternalCoordinates(x_,m):
	"""
	Generates a set of internal (ie: rot-trans free)
	vectors spanning coordinates for asymmetric mols.
	If you are doing a diatomic, use a quantum chemistry package
	"""
	if (len(m)<=2):
		print(m)
		raise Exception("No Diatomics")
	COM0 = CenterOfMass(x_,m)
	xc_  = x_ - COM0
	I = InertiaTensor(xc_,m)
	Ip,X = np.linalg.eig(I)
	Ip0 = Ip.copy()
	n = len(m)
	n3 = 3*n
	# Generate the 6 rotations and translations.
	D = np.zeros((6,n3))
	MWC = np.zeros((n3,n3))
	for i in range(n3):
		if (i%3==0):
			D[0,i] = np.sqrt(m[int(i/3)])
		elif (i%3==1):
			D[1,i] = np.sqrt(m[int(i/3)])
		elif (i%3==2):
			D[2,i] = np.sqrt(m[int(i/3)])
	for i in range(n):
		Px = np.dot(xc_[i],X[:,0])
		Py = np.dot(xc_[i],X[:,1])
		Pz = np.dot(xc_[i],X[:,2])
		for j in range(3):
			D[3,i*3+j] = (Py*X[2,j]-Pz*X[1,j])/np.sqrt(m[i])
			D[4,i*3+j] = (Pz*X[0,j]-Px*X[2,j])/np.sqrt(m[i])
			D[5,i*3+j] = (Px*X[1,j]-Py*X[0,j])/np.sqrt(m[i])
			MWC[i*3+j,i*3+j] = np.sqrt(m[i])
	S = PairOrthogonalize(D,MWC) # Returns normalized Coords.
	nint = S.shape[0]
	print("3N, Number of Internal Coordinates: ", n3 , nint)
	return S

def HarmonicSpectra(f_, x_, at_, grad_=None, eps_ = 0.001, WriteNM_=False, Mu_ = None):
	"""
	Perform a finite difference normal mode analysis
	of a molecule. basically implements http://gaussian.com/vib/

	Args:
		f_: Energies in Hartree.
		x_: Coordinates (A)
		at_: element type of each atom
		grad_: forces in Hartree/angstrom if available. (unused)
		eps_: finite difference step
		WriteNM_: Whether to write the normal modes to readable files
		Mu_: A dipole field routine for intensities.

	Returns:
		Frequencies in wavenumbers and Normal modes (cart)
	"""
	LOGGER.info("Harmonic Analysis")
	n = x_.shape[0]
	n3 = 3*n
	m_ = np.array(map(lambda x: ATOMICMASSESAMU[x-1]*ELECTRONPERPROTONMASS, at_.tolist()))
	Crds = InternalCoordinates(x_,m_) #invbasis X cart
	#Crds=np.eye(n3).reshape((n3,n,3))
	#print("En?",f_(x_))
	if 0:
		Hess = DirectedFdiffHessian(f_, x_, Crds.reshape((len(Crds),n,3)))
		print("Hess (Internal):", Hess)
		# Transform the invariant hessian into cartesian coordinates.
		cHess = np.dot(Crds.T,np.dot(Hess,Crds))
	else:
		cHess = FdiffHessian(f_, x_,0.0005).reshape((n3,n3))
	cHess /= (BOHRPERA*BOHRPERA)
	print("Hess (Cart):", cHess)
	# Mass weight the invariant hessian in cartesian coordinate
	for i,mi in enumerate(m_):
		cHess[i*3:(i+1)*3, i*3:(i+1)*3] /= np.sqrt(mi*mi)
		for j,mj in enumerate(m_):
			if (i != j):
				cHess[i*3:(i+1)*3, j*3:(j+1)*3] /= np.sqrt(mi*mj)
	# Get the vibrational spectrum and normal modes.
	u,s,v = np.linalg.svd(cHess)
	for l in s:
		print("Central Energy (cm**-1): ", np.sign(l)*np.sqrt(l)*WAVENUMBERPERHARTREE)
	print("--")
	# Get the actual normal modes, for visualization sake.
	w,v = np.linalg.eigh(cHess)
	v = v.real
	wave = np.sign(w)*np.sqrt(abs(w))*WAVENUMBERPERHARTREE
	print("N3, shape v",n3,v.shape)
	if (WriteNM_):
		for i in range(3*n):
			nm = np.zeros(3*n)
		        for j,mi in enumerate(m_):
				nm[3*j:3*(j+1)] = v[3*j:3*(j+1),i]/np.sqrt(mi) 
			nm = nm.reshape((n,3))
			# Take finite difference derivative of mu(Q) and return the <dmu/dQ, dmu/dQ>
			step = 0.005
			dmudq = (Mu_(x_+step*nm)-Mu_(x_))/step
			print("|f| (UNITS????) ",np.dot(dmudq,dmudq.T))
			for alpha in np.append(np.linspace(-.1,.1,30),np.linspace(.1,-.1,30)):
				mdisp = Mol(at_, x_+alpha*nm)
				mdisp.WriteXYZfile("./results/","NormalMode_"+str(i))
	return wave, v

def HarmonicSpectraWithProjection(f_, x_, at_, grad_=None, eps_ = 0.001, WriteNM_=False, Mu_ = None):
	"""
	Perform a finite difference normal mode analysis
	of a molecule. basically implements http://gaussian.com/vib/

	Args:
		f_: Energies in Hartree.
		x_: Coordinates (A)
		at_: element type of each atom
		grad_: forces in Hartree/angstrom if available. (unused)
		eps_: finite difference step
		WriteNM_: Whether to write the normal modes to readable files
		Mu_: A dipole field routine for intensities.

	Returns:
		Frequencies in wavenumbers and Normal modes (cart)
	"""
	LOGGER.info("Harmonic Analysis")
	n = x_.shape[0]
	n3 = 3*n
	m_ = np.array(map(lambda x: ATOMICMASSESAMU[x-1]*ELECTRONPERPROTONMASS, at_.tolist()))
	Crds = InternalCoordinates(x_,m_) #invbasis X cart flatten.
	cHess = FdiffHessian(f_, x_,0.0005).reshape((n3,n3))
	cHess /= (BOHRPERA*BOHRPERA)
	print("Hess (Cart):", cHess)
	# Mass weight the invariant hessian in cartesian coordinate
	for i,mi in enumerate(m_):
		cHess[i*3:(i+1)*3, i*3:(i+1)*3] /= np.sqrt(mi*mi)
		for j,mj in enumerate(m_):
			if (i != j):
				cHess[i*3:(i+1)*3, j*3:(j+1)*3] /= np.sqrt(mi*mj)
	# Get the vibrational spectrum and normal modes.
	# pHess = np.einsum('ab,cb->ac',np.einsum('ij,jk->ik',Crds,cHess),Crds)
	s,v = np.linalg.eigh(Hess)
	for l in s:
		print("Central Energy (cm**-1): ", np.sign(l)*np.sqrt(l)*WAVENUMBERPERHARTREE)
	print("--")
	# Get the actual normal modes, for visualization sake.
	v = v.real
	wave = np.sign(s)*np.sqrt(abs(s))*WAVENUMBERPERHARTREE
	print("N3, shape v",n3,v.shape)
	if (WriteNM_):
		for i in range(3*n):
			nm = v[:,i].reshape((n,3))
			nm *= np.sqrt(m_[:,np.newaxis]).T
			tmp = nm.reshape((x_.shape[0],3))
			# Take finite difference derivative of mu(Q) and return the <dmu/dQ, dmu/dQ>
			step = 0.01
			dmudq = (Mu_(x_+step*tmp)-Mu_(x_))/step
			print("|f| (UNITS????) ",np.dot(dmudq,dmudq.T))
			for alpha in np.append(np.linspace(-.1,.1,30),np.linspace(.1,-.1,30)):
				mdisp = Mol(at_, x_+alpha*tmp)
				mdisp.WriteXYZfile("./results/","NormalMode_"+str(i))
	return wave, v

class ConjGradient:
	def __init__(self,f_,x0_):
		"""
		Args:
			f_ : an energy, force routine.
			x0_: initial point.
			p_: initial search direction.
		"""
		self.EForce = f_
		self.Energy = lambda x: self.EForce(x,False)
		self.x0 = x0_.copy()
		self.xold = x0_.copy()
		self.e, self.gold  = self.EForce(x0_)
		self.s = self.gold.copy()
		self.alpha = PARAMS["GSSearchAlpha"]
		return
	def BetaPR(self,g):
		betapr = np.sum((g)*(g - self.gold))/(np.sum(self.gold*self.gold))
		self.gold = g.copy()
		return max(0,betapr)
	def __call__(self,x0):
		"""
		Iterate Conjugate Gradient.

		Args:
			x0: Point at which to minimize gradients
		Returns:
			Next point, energy, and gradient.
		"""
		e,g = self.EForce(x0)
		beta_n = self.BetaPR(g)
		self.s = g + beta_n*self.s
		self.xold = self.LineSearch(x0,self.s)
		return self.xold, e, g
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
			rmsdist = np.sum(np.linalg.norm(a-b,axis=1))/a.shape[0]
			k+=1
		return (b + a) / 2

class ConjGradient_force:
	def __init__(self, f_, mol):
		"""
		Args:
			f_ : an energy, force routine.
			x0_: initial point.
			p_: initial search direction.
		"""
		self.EForce = f_
		self.Energy = lambda x: self.EForce(x)
		self.x0 = mol.coords.copy()
		self.xold = mol.coords.copy()
		self.e, self.gold = self.EForce(mol)
		self.s = self.gold.copy()
		self.alpha = PARAMS["GSSearchAlpha"]
		return

	def BetaPR(self,g):
		betapr = np.sum((g)*(g - self.gold))/(np.sum(self.gold*self.gold))
		self.gold = g.copy()
		return max(0,betapr)

	def __call__(self, mol):
		"""
		Iterate Conjugate Gradient.

		Args:
			x0: Point at which to minimize gradients
		Returns:
			Next point, energy, and gradient.
		"""
		e,g = self.EForce(mol)
		beta_n = self.BetaPR(g)
		self.s = g + beta_n*self.s
		self.xold = self.LineSearch(mol,self.s)
		return self.xold, e, g

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
		b = Mol(x0_.atoms, x0_.coords + self.alpha*p_)
		c = Mol(x0_.atoms, b.coords - (b.coords - a.coords) / GOLDENRATIO)
		d = Mol(x0_.atoms, a.coords + (b.coords - a.coords) / GOLDENRATIO)
		fa, ffa = self.Energy(a)
		fb, ffb = self.Energy(b)
		fc, ffc = self.Energy(c)
		fd, ffd = self.Energy(d)
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
				b = Mol(x0_.atoms, x0_.coords + self.alpha*p_)
				c = Mol(x0_.atoms, b.coords - (b.coords - a.coords) / GOLDENRATIO)
				d = Mol(x0_.atoms, a.coords + (b.coords - a.coords) / GOLDENRATIO)
				fa, ffa = self.Energy(a)
				fb, ffb = self.Energy(b)
				fc, ffc = self.Energy(c)
				fd, ffd = self.Energy(d)
			elif (fb < fc and fb < fd and fb < fa):
				#print fa,fc,fd,fb
				#print RmsForce(fpa), RmsForce(fpc), RmsForce(fpd), RmsForce(fpb)
				print("Line Search: Understep")
				if (self.alpha < 100.0):
					self.alpha *= 1.7
				a = x0_
				b = Mol(x0_.atoms, x0_.coords + self.alpha*p_)
				c = Mol(x0_.atoms, b.coords - (b.coords - a.coords) / GOLDENRATIO)
				d = Mol(x0_.atoms, a.coords + (b.coords - a.coords) / GOLDENRATIO)
				fa, ffa = self.Energy(a)
				fb, ffb = self.Energy(b)
				fc, ffc = self.Energy(c)
				fd, ffd = self.Energy(d)
			elif fc < fd:
				b = d
				c = Mol(x0_.atoms, b.coords - (b.coords - a.coords) / GOLDENRATIO)
				d = Mol(x0_.atoms, a.coords + (b.coords - a.coords) / GOLDENRATIO)
				fb = fd
				fc, ffc = self.Energy(c)
				fd, ffd = self.Energy(d)
			else:
				a = c
				c = Mol(x0_.atoms, b.coords - (b.coords - a.coords) / GOLDENRATIO)
				d = Mol(x0_.atoms, a.coords + (b.coords - a.coords) / GOLDENRATIO)
				fa = fc
				fc, fcc = self.Energy(c)
				fd, ffd = self.Energy(d)
			rmsdist = np.sum(np.linalg.norm(a.coords-b.coords,axis=1))/a.coords.shape[0]
			k+=1
		return (b.coords + a.coords) / 2

def LineSearch(f_, x0_, p_, thresh = 0.0001):
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
			print("Line Search: Overstep")
			if (PARAMS["GSSearchAlpha"] > 0.00001):
				PARAMS["GSSearchAlpha"] /= 1.71
			else:
				print("Keeping step")
				return a
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
			print("Line Search: Understep")
			if (PARAMS["GSSearchAlpha"] < 10.0):
				PARAMS["GSSearchAlpha"] *= 1.7
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

def LineSearchCart(f_, x0_):
	""" A line search in each cartesian direction. """
	x_ = x0_.copy()
	iti = np.nditer(x_, flags=['multi_index'])
	while not iti.finished:
		x_ = x0_.copy()
		x_[iti.multi_index] -= 0.05
		p=np.zeros(x0_.shape)
		p[iti.multi_index] += 0.05
		x_= LineSearch(f_,x_,p)
		iti.iternext()
	return x_

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
