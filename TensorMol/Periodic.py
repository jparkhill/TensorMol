"""
No symmetry but general unit cells supported.
Maintenance of the unit cell, etc. are handled by PeriodicForce.
Only linear scaling forces with energy are supported.
"""
from __future__ import absolute_import
from __future__ import print_function
from .Neighbors import *
from .Electrostatics import *
from .SimpleMD import *
# from MolEmb import Make_DistMask

class Lattice:
	def __init__(self, latvec_):
		"""
		Build a periodic lattice

		Args:
			latvec_: A 3x3 tensor of lattice vectors.
		"""
		self.lattice = latvec_.copy()
		self.latticeCenter = (self.lattice[0]+self.lattice[1]+self.lattice[2])/2.0
		self.latticeMinDiameter = 2.0*min([np.linalg.norm(self.lattice[0]-self.latticeCenter),np.linalg.norm(self.lattice[1]-self.latticeCenter),np.linalg.norm(self.lattice[2]-self.latticeCenter)])
		self.lp = np.array([[0.,0.,0.],self.lattice[0].tolist(),self.lattice[1].tolist(),self.lattice[2].tolist(),(self.lattice[0]+self.lattice[1]).tolist(),(self.lattice[0]+self.lattice[2]).tolist(),(self.lattice[1]+self.lattice[2]).tolist(),(self.lattice[0]+self.lattice[1]+self.lattice[2]).tolist()])
		self.ntess = 1 # number of shells over which to tesselate.
		self.facenormals = self.LatticeNormals()
		return
	def LatticeNormals(self):
		lp = self.lp
		fn = np.zeros((6,3))
		fn[0] = np.cross(lp[1]-lp[0],lp[2]-lp[0]) # face 012
		fn[1] = np.cross(lp[1]-lp[0],lp[3]-lp[0]) # face 013
		fn[2] = np.cross(lp[2]-lp[0],lp[3]-lp[0]) # face 023
		fn[3] = np.cross(lp[4]-lp[-1],lp[5]-lp[-1])
		fn[4] = np.cross(lp[6]-lp[-1],lp[5]-lp[-1])
		fn[5] = np.cross(lp[6]-lp[-1],lp[4]-lp[-1])
		# Normalize them.
		fn /= np.sqrt(np.sum(fn*fn,axis=1))[:,np.newaxis]
		return fn
	def InRangeOfLatNormals(self,pt,rng_):
		for i in range(6):
			if (i<3):
				if (np.abs(np.sum(self.facenormals[i]*(pt - self.lp[0]))) < rng_):
					return True
			else:
				if (np.abs(np.sum(self.facenormals[i]*(pt - self.lp[7]))) < rng_):
					return True
	def CenteredInLattice(self, mol):
		m=Mol(mol.atoms,self.ModuloLattice(mol.coords - mol.Center() + self.latticeCenter))
		m.properties["Lattice"] = self.lattice.copy()
		return m
	def InLat(self,crds):
		"""
		Express coordinates (atom X 3 cart)
		In units of the lattice vectors.
		"""
		latmet = MatrixPower(np.dot(self.lattice, self.lattice.T),-1)
		return 	np.dot(crds,np.dot(self.lattice.T,latmet))
	def FromLat(self,crds):
		"""
		Express coordinates (atom X 3 lat)
		In cartesian units.
		"""
		return np.dot(crds,self.lattice)
	def ModuloLattice(self, crds):
		"""
		Transports all coordinates into the primitive cell.

		Args:
			crds: a natom X 3 ndarray of atomic coordinates.
		Returns:
			crds modulo the primitive lattice.
		"""
		tmp = self.InLat(crds)
		fpart = np.fmod(tmp,1.0)
		revs=np.where(fpart < 0.0)
		fpart[revs] = 1.0 + fpart[revs]
		return self.FromLat(fpart)
	def TessNTimes(self, atoms_, coords_, ntess_):
		"""
		Enlarges a molecule to allow for accurate calculation of a short-ranged force

		Args:
			mol_: a molecule.
			rng_: minimum distance from center covered by the tesselation. (Angstrom)

		Returns:
			An enlarged molecule where the real coordinates preceed 'fake' periodic images.
		"""
		ntess = ntess_
		natom = atoms_.shape[0]
		nimages = pow(ntess_,3)
		#print("Doing",nimages,"images... of ",natom)
		newAtoms = np.zeros(nimages*natom,dtype=np.uint8)
		newCoords = np.zeros((nimages*natom,3))
		newAtoms[:natom] = atoms_
		newCoords[:natom,:3] = coords_
		ind = 1
		for i in range(ntess):
			for j in range(ntess):
				for k in range(ntess):
					if (i==0 and j==0 and k ==0):
						continue
					newAtoms[ind*natom:(ind+1)*natom] = atoms_
					newCoords[ind*natom:(ind+1)*natom,:] = coords_ + i*self.lattice[0] + j*self.lattice[1] + k*self.lattice[2]
					#print(i,j,k,ind,nimages)
					ind = ind + 1
		return newAtoms, newCoords
	def TessLattice(self, atoms_, coords_, rng_):
		"""
		Enlarges a molecule to allow for accurate calculation of a short-ranged force

		Args:
			mol_: a molecule.
			rng_: minimum distance from center covered by the tesselation. (Angstrom)

		Returns:
			An enlarged molecule where the real coordinates preceed 'fake' periodic images.
		"""
		if (rng_ > self.latticeMinDiameter):
			self.ntess = int(rng_/self.latticeMinDiameter)+1
			#print("Doing",self.ntess,"tesselations...")
			#print(rng_,self.latticeMinDiameter)
		natom = atoms_.shape[0]
		nimages = pow(2*self.ntess+1,3)
		#print("Doing",nimages,"images... of ",natom)
		newAtoms = np.zeros(nimages*natom,dtype=np.uint8)
		newCoords = np.zeros((nimages*natom,3))
		newAtoms[:natom] = atoms_
		newCoords[:natom,:3] = coords_
		ind = 1
		for i in range(-self.ntess,self.ntess+1):
			for j in range(-self.ntess,self.ntess+1):
				for k in range(-self.ntess,self.ntess+1):
					if (i==0 and j==0 and k ==0):
						continue
					newAtoms[ind*natom:(ind+1)*natom] = atoms_
					newCoords[ind*natom:(ind+1)*natom,:] = coords_ + i*self.lattice[0] + j*self.lattice[1] + k*self.lattice[2]
					ind = ind + 1
		# Now pare that down where the images are too far from the edges of the lattice.
		return newAtoms, newCoords
		Atoms = np.zeros(nimages*natom,dtype=np.uint8)
		Coords = np.zeros((nimages*natom,3))
		ind=natom
		Coords[:natom] = newCoords[:natom].copy()
		Atoms[:natom] = newAtoms[:natom].copy()
		for j in range(natom,natom*nimages):
			if(self.InRangeOfLatNormals(newCoords[j],rng_)):
				Coords[ind] = newCoords[j]
				Atoms[ind] = newAtoms[j]
				ind = ind + 1
		#print("tes sparsity",float(ind)/(natom*nimages))
		return Atoms[:ind], Coords[:ind]

class LocalForce:
	def __init__(self, f_, rng_=5.0, NeedsTriples_=False):
		self.range = rng_
		self.func=f_
		self.NeedsTriples = NeedsTriples_
		return
	def __call__(self, z, x, NZ, DoForce = True):
		"""
		Generic call to a linear scaling local force.

		Args:
			z: atomic number vector
			x: atoms X 3 coordinate vector.
			NZ: pair or triples matrix. (NZP X 2)
		returns:
			energy number, and force vector with same shape as x.
		"""
		tmp = self.func(z, x, NZ, DoForce)
		return tmp

class PeriodicForceWithNeighborList:
	def __init__(self, pm_, lat_):
		"""
		A periodic force evaluator. The force consists of two components
		Short-Ranged forces, and long-ranged forces. Short ranged forces are
		evaluated by tesselation. Long-range forces are not supported yet.
		This version manages and passes Neighbor lists.


		Args:
			pm_: a molecule.
			lat_: lattice vectors.
		"""
		self.lattice = Lattice(lat_)
		self.NL = None
		self.mol0 = self.lattice.CenteredInLattice(pm_)
		self.atoms = self.mol0.atoms.copy()
		self.natoms = self.mol0.NAtoms()
		self.natomsReal = pm_.NAtoms()
		self.maxrng = 0.0
		self.LocalForces = []
		self.lastx = np.zeros(pm_.coords.shape)
		self.nlthresh = 0.05 #per-atom Threshold for NL rebuild. (A)
		#self.LongForces = [] Everything is real-space courtesy of DSF.
		return
	def AdjustLattice(self, x_, lat_):
		"""
		Adjusts the lattice and rescales the coordinates of m relative to previous lattice.
		"""
		il = self.lattice.InLat(m.coords)
		self.lattice = Lattice(lat_)
		m.coords = self.lattice.FromLat(il)
		return m
	def BindForce(self, lf_, rng_):
		"""
		Adds a local force to be computed when the PeriodicForce is called.

		Args:
			lf_: a function which takes z,x and returns atom energies, atom forces.
		"""
		self.LocalForces.append(LocalForce(lf_,rng_))
	def __call__(self,x_):
		"""
		Returns the Energy per unit cell and force on all primitive atoms

		Args:
			x_: a primitive geometry of atoms matching self.atoms.
		"""
		# Compute local energy.
		etore = 0.0
		ftore = np.zeros((self.natomsReal,3))
		if (self.maxrng == 0.0):
			self.maxrng = max([f.range for f in self.LocalForces])
		# Tesselate atoms.
		z,x = self.lattice.TessLattice(self.atoms,x_, self.maxrng)
		# Construct NeighborList determine if rebuild is necessary.
		if (self.NL==None):
			NeedsTriples = any([f.NeedsTriples for f in self.LocalForces])
			self.NL = NeighborList(x, DoTriples_=NeedsTriples, DoPerms_=False, ele_=None, alg_=None, sort_=False)
			self.NL.Update(x, self.maxrng, 0.0, None, self.natomsReal)
		rms = np.mean(np.linalg.norm(x_-self.lastx,axis=1))
		if (rms > self.nlthresh):
			self.NL.Update(x, self.maxrng, 0.0, None, self.natomsReal)
			self.lastx = x_.copy()
			#print("NL update",rms)
		else:
			print("No NL update",rms)
		#print(self.NL.pairs)
		# Compute forces and energies.
		for f in self.LocalForces:
			einc, finc = f(z,x,self.NL.pairs)
			etore += np.sum(einc)
			ftore += finc[:self.natomsReal]
		return etore, ftore

class PeriodicForce:
	def __init__(self, pm_, lat_):
		"""
		A periodic force evaluator. The force consists of two components
		Short-Ranged forces, and long-ranged forces. Short ranged forces are
		evaluated by tesselation. Long-range forces are not supported yet.
		This version manages and passes Neighbor lists.


		Args:
			pm_: a molecule.
			lat_: lattice vectors.
		"""
		self.lattice = Lattice(lat_)
		self.NL = None
		self.mol0 = self.lattice.CenteredInLattice(pm_)
		self.atoms = self.mol0.atoms.copy()
		self.natoms = self.mol0.NAtoms()
		self.natomsReal = pm_.NAtoms()
		self.maxrng = 0.0
		self.LocalForces = []
		self.lastx = np.zeros(pm_.coords.shape)
		self.nlthresh = 0.05 #per-atom Threshold for NL rebuild. (A)
		#self.LongForces = [] Everything is real-space courtesy of DSF.
		return
	def ReLattice(self,lat_):
		self.lattice = Lattice(lat_)
		return
	def Density(self):
		"""
		Returns the density in g/cm**3 of the bulk.
		"""
		m = np.array(map(lambda x: ATOMICMASSES[x-1], self.mol0.atoms))
		latvol = np.linalg.det(self.lattice.lattice) # in A**3
		return (np.sum(m)/AVOCONST)/(latvol*pow(10, -24))
	def AdjustLattice(self, x_, lat0_, latp_):
		"""
		rescales the coordinates of m relative to previous lattice.
		"""
		latmet = MatrixPower(np.dot(lat0_, lat0_.T),-1)
		inlat = np.dot(x_,np.dot(lat0_.T,latmet))
		return np.dot(inlat, latp_)
	def LatticeStep(self,x_):
		"""
		Displace all lattice coordinates by dlat.
		Relattice if the energy decreases.
		"""
		xx = x_.copy()
		e,f = self.__call__(xx)
		ifstepped = True
		ifsteppedoverall = False
		dlat = PARAMS["OptLatticeStep"]
		while (ifstepped):
			ifstepped = False
			for i in range(3):
				for j in range(3):
					tmp = self.lattice.lattice.copy()
					tmp[i,j] += dlat
					latt = Lattice(tmp)
					xtmp = latt.ModuloLattice(xx)
					z,x = latt.TessLattice(self.atoms,xtmp, self.maxrng)
					et,ft = (self.LocalForces[-1])(z,x,self.natomsReal)
					if (et < e and abs(e-et) > 0.00001):
						e = et
						self.ReLattice(tmp)
						xx = xtmp
						ifstepped=True
						ifsteppedoverall=True
						print("LatStep: ",e,self.lattice.lattice)
						Mol(z,x).WriteXYZfile("./results","LatOpt")
					tmp = self.lattice.lattice.copy()
					tmp[i,j] -= dlat
					latt = Lattice(tmp)
					xtmp = latt.ModuloLattice(xx)
					z,x = latt.TessLattice(self.atoms, xtmp, self.maxrng)
					et,ft = (self.LocalForces[-1])(z,x,self.natomsReal)
					if (et < e and abs(e-et) > 0.00001):
						e = et
						self.ReLattice(tmp)
						xx = xtmp
						ifstepped=True
						ifsteppedoverall=True
						print("LatStep: ",e,self.lattice.lattice)
						Mol(z,x).WriteXYZfile("./results","LatOpt")
		if (not ifsteppedoverall and PARAMS["OptLatticeStep"] > 0.001):
			PARAMS["OptLatticeStep"] = PARAMS["OptLatticeStep"]/2.0
		return xx
	def Save(self,x_,name_ = "PMol"):
		m=Mol(self.atoms,x_)
		m.properties["Lattice"] = np.array_str(self.lattice.lattice.flatten())
		m.WriteXYZfile("./results/", name_, 'w', True)
	def BindForce(self, lf_, rng_):
		"""
		Adds a local force to be computed when the PeriodicForce is called.

		Args:
			lf_: a function which takes z,x and returns atom energies, atom forces.
		"""
		self.LocalForces.append(LocalForce(lf_,rng_))
	def __call__(self,x_,DoForce = True):
		"""
		Returns the Energy per unit cell and force on all primitive atoms

		Args:
			x_: a primitive geometry of atoms matching self.atoms.
		"""
		# Compute local energy.
		etore = 0.0
		ftore = np.zeros((self.natomsReal,3))
		if (self.maxrng == 0.0):
			self.maxrng = max([f.range for f in self.LocalForces])
		# Tesselate atoms.
		z,x = self.lattice.TessLattice(self.atoms,self.lattice.ModuloLattice(x_), self.maxrng)
		# Compute forces and energies.
		for f in self.LocalForces:
			if (DoForce):
				einc, finc = f(z,x,self.natomsReal)
				etore += np.sum(einc)
				ftore += finc[:self.natomsReal]
			else:
				einc = f(z,x,self.natomsReal,DoForce)
				etore += np.sum(einc)
		return etore, ftore
	def TestGradient(self,x_):
		"""
		Travel along a gradient direction.
		Subsample to examine how integrable the forces are versus
		the energy along this path.
		"""
		e0,g0 = self.__call__(x_)
		g0 /= JOULEPERHARTREE
		efunc = lambda x: self.__call__(x)[0]
		print("Magnitude of g", np.linalg.norm(g0))
		print("g",g0)
		#print("FDiff g", FdiffGradient(efunc,x_))
		xt = x_.copy()
		es = np.zeros(40)
		gs = np.zeros((40,g0.shape[0],g0.shape[1]))
		for i,d in enumerate(range(-20,20)):
			dx = d*0.01*g0
			#print("dx", dx)
			xt = x_ + dx
			es[i], gs[i] = self.__call__(xt)
			gs[i] /= JOULEPERHARTREE
			print("es ", es[i], i, np.sqrt(np.sum(dx*dx)) , np.sum(gs[i]*g0), np.sum(g0*g0))
	def RDF(self,x_,z1=8,z2=8,rng=10.0,dx = 0.02,name_="RDF.txt"):
		zt,xt = self.lattice.TessLattice(self.atoms, x_ , rng)
		ri = np.arange(0.0,rng,dx)
		ni = np.zeros(ri.shape)
		nav = 0.0
		for i in range(self.natoms):
			if (zt[i]==z1):
				nav += 1.0
				for j in range(zt.shape[0]):
					if i==j:
						continue
					elif (zt[j] == z2):
						d = np.linalg.norm(xt[j]-xt[i])
						ni[int(d/dx):] += 1
		#np.savetxt("./results/"+name_+"ni.txt",ni)
		# Estimate the effective density by the number contained in the outermost sphere.
		ni /= nav
		density = ni[-1]/(4.18879*ri[-1]*ri[-1]*ri[-1])
		# Now take the derivative by central differences
		x2gi = np.gradient(ni / (12.56637*density),dx)
		# finally divide by x**2 and moving average it.
		gi = x2gi/(ri*ri)
		gi[0] = 0.0
		gi = MovingAverage(gi,20)
		return gi
		#np.savetxt("./results/"+name_+".txt",gi)
