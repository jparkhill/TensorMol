from Util import *
import numpy as np
import random, math
import MolEmb

class Mol:
	""" Provides a general purpose molecule"""
	def __init__(self, atoms_ =  None, coords_ = None):
		if (atoms_!=None):
			self.atoms = atoms_.copy()
		else:
			self.atoms = np.zeros(1,dtype=np.uint8)
		if (coords_!=None):
			self.coords = coords_.copy()
		else:
			self.coords=np.zeros(shape=(1,1),dtype=np.float)
		self.properties = {}
		self.name=None
		#things below here are sometimes populated if it is useful.
		self.PESSamples = [] # a list of tuples (atom, new coordinates, energy) for storage.
		self.ecoords = None # equilibrium coordinates.
		self.DistMatrix = None # a list of equilbrium distances, for GO-models.
		return

	def AtomTypes(self):
		return np.unique(self.atoms)

	def NEles(self):
		return len(self.AtomTypes())

	def IsIsomer(self,other):
		return np.array_equals(np.sort(self.atoms),np.sort(other.atoms))

	def NAtoms(self):
		return self.atoms.shape[0]

	def AtomTypes(self):
		return np.unique(self.atoms)

	def NEles(self):
		return len(self.AtomTypes())

	def NumOfAtomsE(self, e):
		return sum( [1 if at==e else 0 for at in self.atoms ] )

	def Calculate_Atomization(self):
		self.atomization = self.properties["roomT_H"]
		for i in range (0, self.atoms.shape[0]):
			self.atomization = self.atomization - ele_roomT_H[self.atoms[i]]
		return

	def AtomsWithin(self,rad, pt):
		# Returns indices of atoms within radius of point.
		dists = map(lambda x: np.linalg.norm(x-pt),self.coords)
		return [i for i in range(self.NAtoms()) if dists[i]<rad]

	def Rotate(self, axis, ang, origin=np.array([0.0, 0.0, 0.0])):
		"""
		Rotate atomic coordinates and forces if present.
		Args:
			axis: vector for rotation axis
			ang: radians of rotation
			origin: origin of rotation axis.
		"""
		rm = RotationMatrix_v2()
		crds = np.copy(self.coords)
		crds -= origin
		for i in range(len(self.coords)):
			self.coords[i] = np.dot(rm,crds[i])
		if ("forces" in self.properties.keys()):
			# Must also rotate the force vectors
			old_endpoints = crds+self.properties["forces"]
			new_forces = np.zeros(crds.shape)
			for i in range(len(self.coords)):
				new_endpoint = np.dot(rm,old_endpoints[i])
				new_forces[i] = new_endpoint - self.coords[i]
			self.properties["forces"] = new_forces
		self.coords += origin

	def Transform(self,ltransf,center=np.array([0.0,0.0,0.0])):
		crds=np.copy(self.coords)
		for i in range(len(self.coords)):
			self.coords[i] = np.dot(ltransf,crds[i]-center) + center

	def AtomsWithin(self, SensRadius, coord):
		''' Returns atoms within the sensory radius in sorted order. '''
		satoms=np.arange(0,self.NAtoms())
		diffs= self.coords-coord
		dists= np.power(np.sum(diffs*diffs,axis=1),0.5)
		idx=np.argsort(dists)
		mxidx = len(idx)
		for i in range(self.NAtoms()):
			if (dists[idx[i]] >= SensRadius):
				mxidx=i
				break
		return idx[:mxidx]

	def Distort(self,disp=0.38,movechance=.20):
		''' Randomly distort my coords, but save eq. coords first '''
		self.BuildDistanceMatrix()
		e0= self.GoEnergy(self.coords)
		for i in range(0, self.atoms.shape[0]):
			for j in range(0, 3):
				if (random.uniform(0, 1)<movechance):
					#only accept collisionless moves.
					accepted = False
					maxiter = 100
					while (not accepted and maxiter>0):
						tmp = self.coords
						tmp[i,j] += np.random.normal(0.0, disp)
						# mindist = None
						# if (self.DistMatrix != None):
						# 	if((self.GoEnergy(tmp)-e0) < 0.005):
						# 		#print "LJE: ", self.LJEnergy(tmp)
						# 		#print self.coords
						# 		accepted = True
						# 		self.coords = tmp
						# else:
						mindist = np.min([ np.linalg.norm(tmp[i,:]-tmp[k,:]) if i!=k else 1.0 for k in range(self.NAtoms()) ])
						if (mindist>0.35):
							accepted = True
							self.coords = tmp
						maxiter=maxiter-1

	def Read_Gaussian_Output(self, path, filename, set_name):
		try:
			f = open(path, "r+")
			lines = f.readlines()
			print path
			for i in range (0, len(lines)):
				if "Multiplicity" in lines[i]:
					atoms = []
					coords = []
					for j in range (i+1, len(lines)):
						if lines[j].split():
							atoms.append( AtomicNumber(lines[j].split()[0]))
							coords.append([float(lines[j].split()[1]), float(lines[j].split()[2]), float(lines[j].split()[3])])
						else:
							self.atoms = np.asarray(atoms)
							self.coords = np.asarray(coords)
							break
				if "SCF Done:"  in lines[i]:
					self.properties["energy"]= float(lines[i].split()[4])
				if "Total nuclear spin-spin coupling J (Hz):" in lines[i]:
					self.J_coupling = np.zeros((self.NAtoms(), self.NAtoms()))
					number_per_line  = len(lines[i+1].split())
					block_num = 0
					for j in range (i+1, len(lines)):
						if "D" in lines[j] and "End of" not in lines[j]:
							for k in range (1, len(lines[j].split())):
								J_value = list(lines[j].split()[k])
								J_value[J_value.index("D")]="E"
                                                        	J_value="".join(J_value)
								self.J_coupling[int(lines[j].split()[0])-1][number_per_line * (block_num-1) + k -1] = float(J_value)
						elif "End of" in lines[j]:
							break
						else:
							block_num += 1
			for i in range (0, self.NAtoms()):
				for j in range (i+1, self.NAtoms()):
					self.J_coupling[i][j] = self.J_coupling[j][i]

		except Exception as Ex:
			print "Read Failed.", Ex
			raise Ex
		return

	def ReadGDB9(self,path,filename, set_name):
		try:
			f=open(path,"r")
			lines=f.readlines()
			natoms=int(lines[0])
			self.set_name = set_name
			self.name = filename[0:-4]
			self.atoms.resize((natoms))
			self.coords.resize((natoms,3))
			try:
				self.properties["energy"] = float((lines[1].split())[12])
				self.properties["roomT_H"] = float((lines[1].split())[14])
			except:
				pass
			for i in range(natoms):
				line = lines[i+2].split()
				self.atoms[i]=AtomicNumber(line[0])
				try:
					self.coords[i,0]=float(line[1])
				except:
					self.coords[i,0]=scitodeci(line[1])
				try:
					self.coords[i,1]=float(line[2])
				except:
					self.coords[i,1]=scitodeci(line[2])
				try:
					self.coords[i,2]=float(line[3])
				except:
					self.coords[i,2]=scitodeci(line[3])
			f.close()
		except Exception as Ex:
			print "Read Failed.", Ex
			raise Ex
		if (("energy" in self.properties) and ("roomT_H" in self.properties)):
			self.Calculate_Atomization()
		return

	def FromXYZString(self,string):
		lines = string.split("\n")
		natoms=int(lines[0])
		if (len(lines[1].split())>1):
			try:
				self.properties["energy"]=float(lines[1].split()[1])
			except:
				pass
		self.atoms.resize((natoms))
		self.coords.resize((natoms,3))
		for i in range(natoms):
			line = lines[i+2].split()
			if len(line)==0:
				return
			self.atoms[i]=AtomicNumber(line[0])
			try:
				self.coords[i,0]=float(line[1])
			except:
				self.coords[i,0]=scitodeci(line[1])
			try:
				self.coords[i,1]=float(line[2])
			except:
				self.coords[i,1]=scitodeci(line[2])
			try:
				self.coords[i,2]=float(line[3])
			except:
				self.coords[i,2]=scitodeci(line[3])
		return

	def WriteXYZfile(self, fpath=".", fname="mol", mode="a"):
		if not os.path.exists(os.path.dirname(fpath+"/"+fname+".xyz")):
			try:
				os.makedirs(os.path.dirname(fpath+"/"+fname+".xyz"))
			except OSError as exc:
				if exc.errno != errno.EEXIST:
					raise
		with open(fpath+"/"+fname+".xyz", mode) as f:
			natom = self.atoms.shape[0]
			if ("energy" in self.properties.keys()):
				f.write(str(natom)+"\nComment: "+str(self.properties["energy"])+"\n")
			else:
				f.write(str(natom)+"\nComment:\n")
			for i in range (0, natom):
				atom_name =  atoi.keys()[atoi.values().index(self.atoms[i])]
				f.write(atom_name+"   "+str(self.coords[i][0])+ "  "+str(self.coords[i][1])+ "  "+str(self.coords[i][2])+"\n")

	def NEle(self):
		return np.sum(self.atoms)

	def XYZtoGridIndex(self, xyz, ngrids = 250,padding = 2.0):
		Max = (self.coords).max() + padding
                Min = (self.coords).min() - padding
		binsize = (Max-Min)/float(ngrids-1)
		x_index = math.floor((xyz[0]-Min)/binsize)
		y_index = math.floor((xyz[1]-Min)/binsize)
		z_index = math.floor((xyz[2]-Min)/binsize)
		#index=int(x_index+y_index*ngrids+z_index*ngrids*ngrids)
		return x_index, y_index, z_index

	def MolDots(self, ngrids = 250 , padding =2.0, width = 2):
		grids = self.MolGrids()
		for i in range (0, self.atoms.shape[0]):
			x_index, y_index, z_index = self.XYZtoGridIndex(self.coords[i])
			for m in range (-width, width):
				for n in range (-width, width):
					for k in range (-width, width):
						index = (x_index)+m + (y_index+n)*ngrids + (z_index+k)*ngrids*ngrids
						grids[index] = atoc[self.atoms[i]]
		return grids

	def Center(self):
		''' Returns the center of atom'''
		return np.average(self.coords,axis=0)

	def rms(self, m):
		err  = 0.0
		for i in range (0, (self.coords).shape[0]):
			err += (np.sum((m.coords[i] - self.coords[i])**2))**0.5
		return err/float((self.coords).shape[0])

	def MolGrids(self, ngrids = 250):
		grids = np.zeros((ngrids, ngrids, ngrids), dtype=np.uint8)
		grids = grids.reshape(ngrids**3)   #kind of ugly, but lets keep it for now
		return grids

	def SpanningGrid(self,num=250,pad=4.):
		''' Returns a regular grid the molecule fits into '''
		xmin=np.min(self.coords[:,0])-pad
		xmax=np.max(self.coords[:,0])+pad
		ymin=np.min(self.coords[:,1])-pad
		ymax=np.max(self.coords[:,1])+pad
		zmin=np.min(self.coords[:,2])-pad
		zmax=np.max(self.coords[:,2])+pad
		grids = np.mgrid[xmin:xmax:num*1j, ymin:ymax:num*1j, zmin:zmax:num*1j]
		grids = grids.transpose()
		grids = grids.reshape((grids.shape[0]*grids.shape[1]*grids.shape[2], grids.shape[3]))
		return grids, (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

	def AddPointstoMolDots(self, grids, points, value, ngrids =250):  # points: x,y,z,prob    prob is in (0,1)
		points = points.reshape((-1,3))  # flat it
		value = value.reshape(points.shape[0]) # flat it
		value = value/value.max()
		for i in range (0, points.shape[0]):
			x_index, y_index, z_index = self.XYZtoGridIndex(points[i])
			index = x_index + y_index*ngrids + z_index*ngrids*ngrids
			if grids[index] <  int(value[i]*250):
				grids[index] = int(value[i]*250)
		return grids

	def MakeStoichDict(self):
		dict = {}
		for i in self.AtomTypes():
			dict[i] = self.NumOfAtomsE(i)
		self.stoich = dict
		return

	def SortAtoms(self):
		""" First sorts by element, then sorts by distance to the center of the molecule
			This improves alignment. """
		order = np.argsort(self.atoms)
		self.atoms = self.atoms[order]
		self.coords = self.coords[order,:]
		self.coords = self.coords - self.Center()
		self.ElementBounds = [[0,0] for i in range(self.NEles())]
		for e, ele in enumerate(self.AtomTypes()):
			inblock=False
			for i in range(0, self.NAtoms()):
				if (not inblock and self.atoms[i]==ele):
					self.ElementBounds[e][0] = i
					inblock=True
				elif (inblock and (self.atoms[i]!=ele or i==self.NAtoms()-1)):
					self.ElementBounds[e][1] = i
					inblock=False
					break
		for e in range(self.NEles()):
			blk = self.coords[self.ElementBounds[e][0]:self.ElementBounds[e][1],:].copy()
			dists = np.sqrt(np.sum(blk*blk,axis=1))
			inds = np.argsort(dists)
			self.coords[self.ElementBounds[e][0]:self.ElementBounds[e][1],:] = blk[inds]
		return

	def WriteInterpolation(self,b,n=0):
		for i in range(10): # Check the interpolation.
			m=Mol(self.atoms,self.coords*((9.-i)/9.)+b.coords*((i)/9.))
			m.WriteXYZfile(PARAMS["results_dir"], "Interp"+str(n))

	def AlignAtoms(self, m):
		""" So looking at some interpolations I figured out why this wasn't working The problem was the outside can get permuted and then it can't be fixed by pairwise permutations because it takes all-atom moves to drag the system through itself Ie: local minima

			The solution is to force the crystal to have roughly the right orientation by minimizing position differences in a greedy way, then fixing the local structure once they are all roughly in the right place.

			This now MOVES BOTH THE MOLECULES assignments, but works.
			"""
		assert self.NAtoms() == m.NAtoms(), "Number of atoms do not match"
		if (self.Center()-m.Center()).all() != 0:
			m.coords += self.Center() - m.Center()
		self.SortAtoms()
		m.SortAtoms()
		# Greedy assignment
		for e in range(self.NEles()):
			mones = range(self.ElementBounds[e][0],self.ElementBounds[e][1])
			mtwos = range(self.ElementBounds[e][0],self.ElementBounds[e][1])
			assignedmones=[]
			assignedmtwos=[]
			for b in mtwos:
				acs = self.coords[mones]
				tmp = acs - m.coords[b]
				best = np.argsort(np.sqrt(np.sum(tmp*tmp,axis=1)))[0]
				#print "Matching ", m.coords[b]," to ", self.coords[mones[best]]
				#print "Matching ", b," to ", mones[best]
				assignedmtwos.append(b)
				assignedmones.append(mones[best])
				mones = complement(mones,assignedmones)
			self.coords[mtwos] = self.coords[assignedmones]
			m.coords[mtwos] = m.coords[assignedmtwos]

		self.DistMatrix = MolEmb.Make_DistMat(self.coords)
		m.DistMatrix = MolEmb.Make_DistMat(m.coords)
		diff = np.linalg.norm(self.DistMatrix - m.DistMatrix)
		tmp_coords=m.coords.copy()
		tmp_dm = MolEmb.Make_DistMat(tmp_coords)
		k = 0
		steps = 1
		while (k < 2):
			for i in range(m.NAtoms()):
				for j in range(i+1,m.NAtoms()):
					if m.atoms[i] != m.atoms[j]:
						continue
					ir = tmp_dm[i].copy() - self.DistMatrix[i]
					jr = tmp_dm[j].copy() - self.DistMatrix[j]
					irp = tmp_dm[j].copy()
					irp[i], irp[j] = irp[j], irp[i]
					jrp = tmp_dm[i].copy()
					jrp[i], jrp[j] = jrp[j], jrp[i]
					irp -= self.DistMatrix[i]
					jrp -= self.DistMatrix[j]
					if (np.linalg.norm(irp)+np.linalg.norm(jrp) < np.linalg.norm(ir)+np.linalg.norm(jr)):
						k = 0
						perm=range(m.NAtoms())
						perm[i] = j
						perm[j] = i
						tmp_coords=tmp_coords[perm]
						tmp_dm = MolEmb.Make_DistMat(tmp_coords)
						print np.linalg.norm(self.DistMatrix - tmp_dm)
						steps = steps+1
				print i
			k+=1
		m.coords=tmp_coords.copy()
		print "best",tmp_coords
		print "self",self.coords
		self.WriteInterpolation(Mol(self.atoms,tmp_coords),9999)
		return

# ---------------------------------------------------------------
#  Functions related to energy models and sampling.
# ---------------------------------------------------------------

	def BuildDistanceMatrix(self):
		self.DistMatrix = MolEmb.Make_DistMat(self.coords)

	def GoEnergy(self,x):
		''' The GO potential enforces equilibrium bond lengths. This is the lennard jones soft version'''
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		xmat = np.array(x).reshape(self.NAtoms(),3)
		newd = MolEmb.Make_DistMat(xmat)
		newd -= self.DistMatrix
		newd = newd*newd
		return PARAMS["GoK"]*np.sum(newd)

	def GoEnergyAfterAtomMove(self,s,ii):
		''' The GO potential enforces equilibrium bond lengths. '''
		raise Exception("Depreciated.")

	def GoForce(self, at_=-1):
		'''
			The GO potential enforces equilibrium bond lengths, and this is the force of that potential.
			Args: at_ an atom index, if at_ = -1 it returns an array for each atom.
		'''
		return PARAMS["GoK"]*MolEmb.Make_GoForce(self.coords,self.DistMatrix,at_)

	def GoForceLocal(self, at_=-1):
		''' The GO potential enforces equilibrium bond lengths, and this is the force of that potential.
			A MUCH FASTER VERSION OF THIS ROUTINE IS NOW AVAILABLE, see MolEmb::Make_Go
		'''
		return PARAMS["GoK"]*MolEmb.Make_GoForceLocal(self.coords,self.DistMatrix,at_)

	def NumericGoHessian(self):
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		disp=0.001
		hess=np.zeros((self.NAtoms()*3,self.NAtoms()*3))
		for i in range(self.NAtoms()):
			for j in range(self.NAtoms()):
				for ip in range(3):
					for jp in range(3):
						if (j*3+jp >= i*3+ip):
							tmp = self.coords.flatten()
							tmp[i*3+ip] += disp
							tmp[j*3+jp] += disp
							f1 = self.GoEnergy(tmp)
							tmp = self.coords.flatten()
							tmp[i*3+ip] += disp
							tmp[j*3+jp] -= disp
							f2 = self.GoEnergy(tmp)
							tmp = self.coords.flatten()
							tmp[i*3+ip] -= disp
							tmp[j*3+jp] += disp
							f3 = self.GoEnergy(tmp)
							tmp = self.coords.flatten()
							tmp[i*3+ip] -= disp
							tmp[j*3+jp] -= disp
							f4 = self.GoEnergy(tmp)
							hess[i*3+ip,j*3+jp] = (f1-f2-f3+f4)/(4.0*disp*disp)
		return (hess+hess.T-np.diag(np.diag(hess)))

	def GoHessian(self):
		return PARAMS["GoK"]*MolEmb.Make_GoHess(self.coords,self.DistMatrix)

	def ScanNormalModes(self,npts=11,disp=0.2):
		"These modes are normal"
		self.BuildDistanceMatrix()
		hess = self.GoHessian()
		w,v = np.linalg.eig(hess)
		thresh = pow(10.0,-6.0)
		numincl = np.sum([1 if abs(w[i])>thresh else 0 for i in range(len(w))])
		tore = np.zeros((numincl,npts,self.NAtoms(),3))
		nout = 0
		for a in range(self.NAtoms()):
			for ap in range(3):
				if (abs(w[a*3+ap])<thresh):
					continue
				tmp = v[:,a*3+ap]/np.linalg.norm(v[:,a*3+ap])
				eigv = np.reshape(tmp,(self.NAtoms(),3))
				for d in range(npts):
					tore[nout,d,:,:] = (self.coords+disp*(self.NAtoms()*(d-npts/2.0+0.37)/npts)*eigv).real
					#print disp*(self.NAtoms()*(d-npts/2.0+0.37)/npts)*eigv
					#print d, self.GoEnergy(tore[nout,d,:,:].flatten())#, PARAMS["GoK"]*MolEmb.Make_GoForce(tore[nout,d,:,:],self.DistMatrix,-1)
				nout = nout+1
		return tore

	def SoftCutGoForce(self, cutdist=6):
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		forces = np.zeros((self.NAtoms(),3))
		for i in range(len(self.coords)):
			forces[i]=self.SoftCutGoForceOneAtom(i, cutdist)
		return forces

	def SoftCutGoForceOneAtom(self, at_, cutdist=6):
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		forces = np.zeros(3)
		for j in range (len(self.coords)):
			u = self.coords[j]-self.coords[at_]
			dj = np.linalg.norm(u)
			if (dj != 0.0):
				u = u/np.linalg.norm(u)
				forces += (0.5*(dj-self.DistMatrix[at_,j])*u)*ErfSoftCut(cutdist-1, 0.5,dj)
				print j,forces
		return forces

	def SoftCutGoForceOneAtomGrids(self, samples, at_, cutdist=6):
		if (self.DistMatrix==None):
				print "Build DistMatrix"
				raise Exception("dmat")
		forces = np.zeros(samples.shape[0],3)
		for i in range (0, forces.shape[0]):
			for j in range (len(self.coords)):
				if j!=at_:
					u = self.coords[j]-samples[i]
					dj = np.linalg.norm(u)
					if (dj != 0.0):
						u = u/np.linalg.norm(u)
						forces[i] += (0.5*(dj-self.DistMatrix[at_,j])*u)*ErfSoftCut(cutdist-1, 0.5,dj)
		return forces

	def GoForce_Scan(self, maxstep, ngrid):
		#scan near by regime and return the samllest force
		forces = np.zeros((self.NAtoms(),3))
		TmpForce = np.zeros((self.NAtoms(), ngrid*ngrid*ngrid,3),dtype=np.float)
		for i in range (0, self.NAtoms()):
			print "Atom: ", i
			save_i = self.coords[i].copy()
			samps=MakeUniform(self.coords[i],maxstep,ngrid)
			for m in range (0, samps.shape[0]):
				self.coords[i] = samps[m].copy()
	        	        for j in range(len(self.coords)):
                                # compute force on i due to all j's
                	                u = self.coords[j]-samps[m]
                        	        dij = np.linalg.norm(u)
                                	if (dij != 0.0):
                                        	u = u/np.linalg.norm(u)
                               		TmpForce[i][m] += 0.5*(dij-self.DistMatrix[i,j])*u
			self.coords[i] = save_i.copy()
			TmpAbsForce = (TmpForce[i,:,0]**2+TmpForce[i,:,1]**2+TmpForce[i,:,2]**2)**0.5
			forces[i] = samps[np.argmin(TmpAbsForce)]
		return forces

	def EnergyAfterAtomMove(self,s,i,Type="GO"):
		if (Type=="GO"):
			return self.GoEnergyAfterAtomMove(s,i)
		else:
			raise Exception("Unknown Energy")

	def PySCFEnergyAfterAtomMove(self,s,i):
		disp = np.linalg.norm(s-self.coords[i])
		mol = gto.Mole()
		pyscfatomstring=""
		for j in range(len(self.atoms)):
			if(i==j):
				pyscfatomstring=pyscfatomstring+str(self.atoms[j])+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+(";" if j!= len(self.atoms)-1 else "")
			else:
				pyscfatomstring=pyscfatomstring+str(self.atoms[j])+" "+str(self.coords[j,0])+" "+str(self.coords[j,1])+" "+str(self.coords[j,2])+(";" if j!= len(self.atoms)-1 else "")
		mol.atom = pyscfatomstring
		mol.basis = '6-31G'
		mol.verbose = 0
		try:
			mol.build()
			en=0.0
			if (disp>0.08 or self.NEle()%2 == 1):
				mf = dft.UKS(mol)
				mf.xc = 'PBE'
				en=mf.kernel()
				#en=scf.UHF(mol).scf()
			else:
				mf = dft.RKS(mol)
				mf.xc = 'PBE'
				en=mf.kernel()
				#en=scf.RHF(mol).scf()
			self.PESSamples.append([i,s,en])
			return en
		except Exception as Ex:
			print "PYSCF Calculation error... :",Ex
			print "Mol.atom:", mol.atom
			print "Pyscf string:", pyscfatomstring
			return 10.0
			#raise Ex
		return 0.0

	#Most parameters are unneccesary.
	def OverlapEmbeddings(self, d1, coords, d2 , d3 ,  d4 , d5, i, d6):#(self,coord,i):
		return np.array([GRIDS.EmbedAtom(self,j,i) for j in coords])

	def GoMeanProbForce(self):
		forces = np.zeros(shape=(self.NAtoms(),3))
		for ii in range(self.NAtoms()):
			Ps = self.POfAtomMoves(GRIDS.MyGrid(),ii)
			print "SAMPLE CENTER:", self.coords[ii]
			forces[ii] = np.dot(samps.T,Ps)
			print "Disp CENTER:", Pc
		return forces

	def GoDisp(self,ii,Print=False):
		'''
			Generates a Go-potential for atom i on a uniform grid of 4A with 50 pts/direction
			And fits that go potential with the H@0 basis centered at the same point
			In practice 9 (1A) gaussians separated on a 1A grid around the sensory point appears to work for moderate distortions.
		'''
		Ps = self.POfAtomMoves(GRIDS.MyGrid(),ii)
		return np.array([np.dot(GRIDS.MyGrid().T,Ps)])

	def FitGoProb(self,ii,Print=False):
		'''
			Generates a Go-potential for atom i on a uniform grid of 4A with 50 pts/direction
			And fits that go potential with the H@0 basis centered at the same point
			In practice 9 (1A) gaussians separated on a 1A grid around the sensory point appears to work for moderate distortions.
		'''
		Ps = self.POfAtomMoves(GRIDS.MyGrid(),ii)
		Pc = np.dot(GRIDS.MyGrid().T,Ps)
		if (Print):
			print "Desired Displacement", Pc  # should equal the point for a Go-Model at equilibrium
		V=GRIDS.Vectorize(Ps)#,True)
		out = np.zeros(shape=(1,GRIDS.NGau3+3))
		out[0,:GRIDS.NGau3]+=V
		out[0,GRIDS.NGau3:]+=Pc
		return out

	def UseGoProb(self,ii,inputs):
		'''
			The opposite of the routine above. It takes the digested probability vectors and uses it to calculate desired new positions.
		'''
		#print "Inputs", inputs
		pdisp=inputs[-3:]
		#print "Current Pos and Predicted displacement: ", self.coords[ii], pdisp
		#Pr = GRIDS.Rasterize(inputs[:GRIDS.NGau3])
		#Pr /= np.sum(Pr)
		#p=np.dot(GRIDS.MyGrid().T,Pr)
		#print "Element Type", self.atoms[ii]
		#print "fit disp: ", p
		#print "Using Disp:", pdisp
		#self.FitGoProb(ii,True)
		return pdisp

	def RunPySCFWithCoords(self,samps,i):
		# The samps are new xyz coords for atom i
		# do some fast model chemistry... gah they aren't fast enough.
		if (len(samps)>40):
			print "sampling ",len(samps)," points about atom ",i,"..."
		return np.array([self.PySCFEnergyAfterAtomMove(s,i) for s in samps])

	def EnergiesOfAtomMoves(self,samps,i):
		return np.array([self.energyAfterAtomMove(s,i) for s in samps])

	def POfAtomMoves(self,samps,i):
		''' Arguments are given relative to the coordinate of i'''
		if (self.DistMatrix==None):
			raise Exception("BuildDMat")
		Es=np.zeros(samps.shape[0],dtype=np.float64)
		MolEmb.Make_Go(samps+self.coords[i],self.DistMatrix,Es,self.coords,i)
		Es=np.nan_to_num(Es)
		Es=Es-np.min(Es)
		Ps = np.exp(-1.0*Es/KAYBEETEE)
		Ps=np.nan_to_num(Ps)
		Z = np.sum(Ps)
		return Ps/Z

	def Force_from_xyz(self, path):
		"""
		Reads the forces from the comment line in the md_dataset,
		and if no forces exist sets them to zero. Switched on by
		has_force=True in the ReadGDB9Unpacked routine
		"""
		try:
			f = open(path, 'r')
			lines = f.readlines()
			natoms = int(lines[0])
			forces=np.zeros((natoms,3))
			try:
				read_forces = ((lines[1].strip().split(';'))[1]).replace("],[", ",").replace("[","").replace("]","").split(",")
			except:
				self.properties['forces'] = forces
				pass
				return
			if read_forces:
				for j in range(natoms):
					for k in range(3):
						forces[j,k] = float(read_forces[j*3+k])
			self.properties['forces'] = forces
		except Exception as Ex:
			print "Read Failed.", Ex
