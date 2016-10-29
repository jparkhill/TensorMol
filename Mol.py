from Util import *
import numpy as np
import random

class Mol:
	""" Provides a general purpose molecule"""
	def __init__(self, atoms_ =  None, coords_ = None):
		if (atoms_!=None):
			self.atoms = atoms_
		else: 
			self.atoms = np.zeros(1,dtype=np.uint8)
		if (coords_!=None):
			self.coords = coords_
		else:
			self.coords=np.zeros(shape=(1,1),dtype=np.float)
		self.properties = {"MW":0}
		self.name=None
		#things below here are sometimes populated if it is useful.
		self.PESSamples = [] # a list of tuples (atom, new coordinates, energy) for storage.
		self.ecoords = None # equilibrium coordinates.
		self.DistMatrix = None # a list of equilbrium distances, for GO-models.
		return

	def IsIsomer(self,other):
		return np.array_equals(np.sort(self.atoms),np.sort(other.atoms))
			
	def NAtoms(self):
		return self.atoms.shape[0]

	def AtomsWithin(self,rad, pt):
		# Returns indices of atoms within radius of point.
		dists = map(lambda x: np.linalg.norm(x-pt),self.coords)
		return [i for i in range(self.NAtoms()) if dists[i]<rad]

	def NumOfAtomsE(self, at):
		return sum( [1 if e==at else 0 for e in self.atoms ] )

	def Rotate(self,axis,ang):
		rm=RotationMatrix(axis,ang)
		crds=np.copy(self.coords)
		for i in range(len(self.coords)):
			self.coords[i] = np.dot(rm,crds[i])

	def MoveToCenter(self):
		first_atom = (self.coords[0]).copy()
		for i in range (0, self.NAtoms()):
			self.coords[i] = self.coords[i] - first_atom

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

	def WriteXYZfile(self, fpath=".", fname="mol"):
		if (os.path.isfile(fpath+"/"+fname+".xyz")):
			f = open(fpath+"/"+fname+".xyz","a")
		else:
			f = open(fpath+"/"+fname+".xyz","w")
		natom = self.atoms.shape[0]
		f.write(str(natom)+"\n\n")
		for i in range (0, natom):
			atom_name =  atoi.keys()[atoi.values().index(self.atoms[i])]
			f.write(atom_name+"   "+str(self.coords[i][0])+ "  "+str(self.coords[i][1])+ "  "+str(self.coords[i][2])+"\n")	
		f.write("\n\n")	
		f.close()

	def Distort(self,disp=0.4,movechance=.85):
		''' Randomly distort my coords, but save eq. coords first '''
		self.BuildDistanceMatrix()
		for i in range (0, self.atoms.shape[0]):
			for j in range (0, 3):
				if (random.uniform(0, 1)<movechance):
					self.coords[i,j] = self.coords[i,j] + disp*random.uniform(-1, 1)

	def AtomTypes(self):
		return np.unique(self.atoms)

	def ReadGDB9(self,path):
		try:
			f=open(path,"r")
			lines=f.readlines()
			natoms=int(lines[0])
			self.atoms.resize((natoms))
			self.coords.resize((natoms,3))
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
		return

	def FromXYZString(self,string):
		lines = string.split("\n")
		natoms=int(lines[1])
		self.atoms.resize((natoms))
		self.coords.resize((natoms,3))
		for i in range(natoms):
			line = lines[i+3].split()
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

	def NEle(self):
		return np.sum(self.atoms)

	def XYZtoGridIndex(self, xyz, ngrids = 250,padding = 2.0):
		Max = (self.coords).max() + padding
                Min = (self.coords).min() - padding
		binsize = (Max-Min)/float(ngrids-1)
		x_index = math.floor((xyz[0]-Min)/binsize)
		y_index = math.floor((xyz[1]-Min)/binsize)
		z_index = math.floor((xyz[2]-Min)/binsize)
#		index=int(x_index+y_index*ngrids+z_index*ngrids*ngrids)
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

# ---------------------------------------------------------------
#  Functions related to energy models and sampling.
# ---------------------------------------------------------------

	def BuildDistanceMatrix(self):
		self.ecoords = np.copy(self.coords)
		self.DistMatrix = np.zeros(shape=(len(self.coords),len(self.coords)),dtype=np.float)
		for i in range(len(self.coords)):
			for j in range(i+1,len(self.coords)):
				self.DistMatrix[i,j] = np.linalg.norm(self.coords[i]-self.coords[j])
		self.DistMatrix += self.DistMatrix.T

	def GoEnergyAfterAtomMove(self,s,ii):
		''' The GO potential enforces equilibrium bond lengths. '''
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		newd = np.copy(self.DistMatrix)
		for i in range(len(self.coords)):
			newd[ii,i] = np.linalg.norm(self.coords[i]-s)
			newd[i,ii] = newd[ii,i]
		newd -= self.DistMatrix
		newd = newd*newd
		return 0.0625*np.sum(newd)

	def GoForce(self):
		''' The GO potential enforces equilibrium bond lengths, and this is the force of that potential.
			A MUCH FASTER VERSION OF THIS ROUTINE IS NOW AVAILABLE, see MolEmb::Make_Go
		'''
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		forces = np.zeros((self.NAtoms(),3))
		for i in range(len(self.coords)):
			for j in range(len(self.coords)):
				# compute force on i due to all j's
				u = self.coords[j]-self.coords[i]
				dij = np.linalg.norm(u)
				if (dij != 0.0):
					u = u/np.linalg.norm(u)
				forces[i] += 0.5*(dij-self.DistMatrix[i,j])*u
		return forces

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
		return np.array([self.EnergyAfterAtomMove(s,i) for s in samps])

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
		
