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
		
		self.mbe_order = MBE_ORDER
		self.mbe_frags=dict()    # list of  frag of each order N, dic['N'=list of frags]
		self.mbe_frags_deri=dict()
		self.mbe_permute_frags=dict() # list of all the permuted frags
		self.mbe_frags_energy=dict()  # MBE energy of each order N, dic['N'= E_N]
		self.energy=None
		self.mbe_energy=dict()   # sum of MBE energy up to order N, dic['N'=E_sum]
		self.mbe_deri =None
		self.nn_energy=None
		self.qchem_data_path = None
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

	def Transform(self,ltransf,center=np.array([0.0,0.0,0.0])):
		crds=np.copy(self.coords)
		for i in range(len(self.coords)):
			self.coords[i] = np.dot(ltransf,crds[i]-center) + center

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

	def GoForce(self, at_=-1):
		''' The GO potential enforces equilibrium bond lengths, and this is the force of that potential.
			A MUCH FASTER VERSION OF THIS ROUTINE IS NOW AVAILABLE, see MolEmb::Make_Go
		'''
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		if (at_!=-1):
			forces = np.zeros((1,3))
			for j in range(len(self.coords)):
				# compute force on i due to all j's
				u = self.coords[j]-self.coords[at_]
				dij = np.linalg.norm(u)
				if (dij != 0.0):
					u = u/np.linalg.norm(u)
				forces[0] += 0.5*(dij-self.DistMatrix[at_,j])*u
			return forces
		else:
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

## ----------------------------------------------
## MBE routines:
## ----------------------------------------------

	def Reset_Frags(self):
		self.mbe_frags=dict()    # list of  frag of each order N, dic['N'=list of frags]
		self.mbe_frags_deri=dict()
		self.mbe_permute_frags=dict() # list of all the permuted frags
		self.mbe_frags_energy=dict()  # MBE energy of each order N, dic['N'= E_N]
		self.energy=None
		self.mbe_energy=dict()   # sum of MBE energy up to order N, dic['N'=E_sum]
		self.mbe_deri =None
		self.nn_energy=None
		return

	def AtomName(self, i):
		return atoi.keys()[atoi.values().index(self.atoms[i])]

	def AllAtomNames(self):
		names=[]
		for i in range (0, self.atoms.shape[0]):
			names.append(atoi.keys()[atoi.values().index(self.atoms[i])])
		return names


	def Generate_All_MBE_term(self,  atom_group=1, cutoff=10, center_atom=0):
		for i in range (1, self.mbe_order+1):
			self.Generate_MBE_term(i, atom_group, cutoff, center_atom)
		return  0

	def Generate_MBE_term(self, order,  atom_group=1, cutoff=10, center_atom=0):
		if order in self.mbe_frags.keys():
			print ("MBE order", order, "already generated..skipping..")
			return 0
		if (self.coords).shape[0]%atom_group!=0:
			raise Exception("check number of group size")
		else:
			ngroup = (self.coords).shape[0]/atom_group
		xyz=((self.coords).reshape((ngroup, atom_group, -1))).copy()     # cluster/molecule needs to be arranged with molecule/sub_molecule
		ele=((self.atoms).reshape((ngroup, atom_group))).copy()
		mbe_terms=[]
		mbe_terms_num=0
		mbe_dist=[]
		atomlist=list(range(0,ngroup))
		if order < 1 :
			raise Exception("MBE Order Should be Positive")
		else:	
			time_log = time.time()
			print ("generating the combinations for order: ", order)
			combinations=list(itertools.combinations(atomlist,order))
			print ("finished..takes", time_log-time.time(),"second")
		time_now=time.time()
		flag = np.zeros(1)
		max_case = 10000000   #  set max cases for debug  
		for i in range (0, len(combinations)):
			term = list(combinations[i])
			pairs=list(itertools.combinations(term, 2))	
			saveindex=[]
			dist = [10000000]*len(pairs)
#			flag = 1
#			for j in range (0, len(pairs)):
#				m=pairs[j][0]
#				n=pairs[j][1]
#				#dist[j] = np.linalg.norm(xyz[m]-xyz[n])
#				dist[j]=((xyz[m][center_atom][0]-xyz[n][center_atom][0])**2+(xyz[m][center_atom][1]-xyz[n][center_atom][1])**2+(xyz[m][center_atom][2]-xyz[n][center_atom][2])**2)**0.5
#				if dist[j] > cutoff:
#					flag = 0
#					break
#			if flag == 1:
			flag[0]=1
			npairs=len(pairs)
			code="""
			for (int j=0; j<npairs; j++) {
				int m = pairs[j][0];
				int n = pairs[j][1];
				dist[j] = sqrt(pow(xyz[m*atom_group*3+center_atom*3+0]-xyz[n*atom_group*3+center_atom*3+0],2)+pow(xyz[m*atom_group*3+center_atom*3+1]-xyz[n*atom_group*3+center_atom*3+1],2)+pow(xyz[m*atom_group*3+center_atom*3+2]-xyz[n*atom_group*3+center_atom*3+2],2));
				if (float(dist[j]) > cutoff) {
					flag[0] = 0;
					break;
				}
			}
			
			"""
			res = inline(code, ['pairs','npairs','center_atom','dist','xyz','flag','cutoff','atom_group'],headers=['<math.h>','<iostream>'], compiler='gcc')
			if flag[0]==1:  # end of weave
				if mbe_terms_num%100==0:
					print mbe_terms_num, time.time()-time_now
					time_now= time.time()
				mbe_terms_num += 1
				mbe_terms.append(term)
				mbe_dist.append(dist)
				if mbe_terms_num >=  max_case:   # just for generating training case
					break;
		mbe_frags = []
		for i in range (0, mbe_terms_num):
			tmp_atom = np.zeros(order*atom_group) 
			tmp_coord = np.zeros((order*atom_group, 3))  
			for j in range (0, order):
				tmp_atom[atom_group*j:atom_group*(j+1)] = ele[mbe_terms[i][j]]
				tmp_coord[atom_group*j:atom_group*(j+1)] = xyz[mbe_terms[i][j]]
			tmp_mol = Frag(tmp_atom, tmp_coord, mbe_terms[i], mbe_dist[i], atom_group)
			mbe_frags.append(tmp_mol)
		self.mbe_frags[order]=mbe_frags
		print "generated {:10d} terms for order {:d}".format(len(mbe_frags), order)
		del combinations[:]
		del combinations
		return mbe_frags

	def Calculate_Frag_Energy(self, order, method="pyscf"):
		if order in self.mbe_frags_energy.keys():
			print ("MBE order", order, "already calculated..skipping..")
			return 0
		mbe_frags_energy = 0.0
		fragnum=0
		time_log=time.time()
		print "length of order ", order, ":",len(self.mbe_frags[order])
		if method == "qchem":
			order_path = self.qchem_data_path+"/"+str(order)
			if not os.path.isdir(order_path):
				os.mkdir(order_path)
			os.chdir(order_path)
			for frag in self.mbe_frags[order]:  # just for generating the training set..
				fragnum += 1
				print "working on frag:", fragnum
				frag.Write_Qchem_Frag_MBE_Input_All(fragnum)
			os.chdir("../../../../")
   		elif method == "pyscf":
			for frag in self.mbe_frags[order]:  # just for generating the training set..
				fragnum +=1
				print "doing the ",fragnum
				frag.PySCF_Frag_MBE_Energy_All()
				frag.Set_Frag_MBE_Energy()
				mbe_frags_energy += frag.frag_mbe_energy
				print "Finished, spent ", time.time()-time_log," seconds"
				time_log = time.time()
			self.mbe_frags_energy[order] = mbe_frags_energy
		else:
			raise Exception("unknow ab-initio software!")		
		return 0

	def Get_Qchem_Frag_Energy(self, order):
		fragnum = 0
		path = self.qchem_data_path+"/"+str(order)
		mbe_frags_energy = 0.0
		for frag in self.mbe_frags[order]:
			fragnum += 1
			frag.Get_Qchem_Frag_MBE_Energy_All(fragnum, path)
			print "working on molecule:", self.name," frag:",fragnum, " order:",order
			frag.Set_Frag_MBE_Energy()
			mbe_frags_energy += frag.frag_mbe_energy
			#if order==2:
		#		print frag.frag_mbe_energy, frag.dist[0]
		self.mbe_frags_energy[order] = mbe_frags_energy
		return
 
	def Get_All_Qchem_Frag_Energy(self):
		for i in range (1, 3):  # set to up to 2nd order for debug sake
		#for i in range (1, self.mbe_order+1): 
			#print "getting the qchem energy for MBE order", i
			self.Get_Qchem_Frag_Energy(i)
		return 
	
	def Calculate_All_Frag_Energy(self, method="pyscf"):  # we ignore the 1st order for He here
		if method == "qchem":
			if not os.path.isdir("./qchem"):
                                os.mkdir("./qchem")
			if not os.path.isdir("./qchem"+"/"+self.set_name):
                                os.mkdir("./qchem"+"/"+self.set_name)
                        self.qchem_data_path="./qchem"+"/"+self.set_name+"/"+self.name
			if not os.path.isdir(self.qchem_data_path):
                                os.mkdir(self.qchem_data_path)
		for i in range (1, self.mbe_order+1):
			print "calculating for MBE order", i
			self.Calculate_Frag_Energy(i, method)
		if method == "qchem":
			self.Write_Qchem_Submit_Script()
		#print "mbe_frags_energy", self.mbe_frags_energy
		return 

	def Write_Qchem_Submit_Script(self):     # this is for submitting the jobs on notre dame crc
		if not os.path.isdir("./qchem"):
			os.mkdir("./qchem")
			if not os.path.isdir("./qchem"+"/"+self.set_name):
				os.mkdir("./qchem"+"/"+self.set_name)
                self.qchem_data_path="./qchem"+"/"+self.set_name+"/"+self.name
		if not os.path.isdir(self.qchem_data_path):
			os.mkdir(self.qchem_data_path)
		os.chdir(self.qchem_data_path)
		for i in range (1, self.mbe_order+1):
			num_frag = len(self.mbe_frags[i])
			for j in range (1, i+1):
				index=nCr(i, j)
				for k in range (1, index+1):
					submit_file = open("qchem_order_"+str(i)+"_suborder_"+str(j)+"_index_"+str(k)+".sub","w+")
					lines = Submit_Script_Lines(order=str(i), sub_order =str(j), index=str(k), mincase = str(1), maxcase = str(num_frag), name = "MBE_"+str(i)+"_"+str(j)+"_"+str(index), ncore = str(4), queue="long")
					submit_file.write(lines) 
					submit_file.close()
		os.chdir("../../../")
		return 

	def Set_MBE_Energy(self):
		for i in range (1, self.mbe_order+1): 
			self.mbe_energy[i] = 0.0
			for j in range (1, i+1):
				self.mbe_energy[i] += self.mbe_frags_energy[j]
		return 0.0

	def MBE(self,  atom_group=1, cutoff=10, center_atom=0):
		self.Generate_All_MBE_term(atom_group, cutoff, center_atom)
		self.Calculate_All_Frag_Energy()
		self.Set_MBE_Energy()
		print self.mbe_frags_energy
		return 0

	def PySCF_Energy(self):
		mol = gto.Mole()
		pyscfatomstring=""
		for j in range(len(self.atoms)):
			s = self.coords[j]
			pyscfatomstring=pyscfatomstring+str(self.AtomName(j))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+(";" if j!= len(self.atoms)-1 else "")
		mol.atom = pyscfatomstring
		mol.basis = 'cc-pvqz'
		mol.verbose = 0
		try:
			mol.build()
			mf=scf.RHF(mol)
			hf_en = mf.kernel()
			mp2 = mp.MP2(mf)
			mp2_en = mp2.kernel()
			en = hf_en + mp2_en[0]
			self.energy = en
			return en
		except Exception as Ex:
				print "PYSCF Calculation error... :",Ex
				print "Mol.atom:", mol.atom
				print "Pyscf string:", pyscfatomstring
				return 0.0
				#raise Ex
		return 0.0

	def Get_Permute_Frags(self, indis=[0]):
		self.mbe_permute_frags=dict()
		for order in self.mbe_frags.keys():
		   if order <= 2:  # for debug purpose
			self.mbe_permute_frags[order]=list()
			for frags in self.mbe_frags[order]:
				self.mbe_permute_frags[order] += frags.Permute_Frag( indis  )
			print "length of permuted frags:", len(self.mbe_permute_frags[order]),"order:", order
		return

	def Set_Frag_Force_with_Order(self, cm_deri, nn_deri, order):
		self.mbe_frags_deri[order]=np.zeros((self.NAtoms(),3))
		atom_group = self.mbe_frags[order][0].atom_group  # get the number of  atoms per group by looking at the frags.
		for i in range (0, len(self.mbe_frags[order])):
			deri = self.mbe_frags[order][i].Frag_Force(cm_deri[i], nn_deri[i])
			deri = deri.reshape((order, deri.shape[0]/order, -1))
			index_list = self.mbe_frags[order][i].index
			for j in range (0,  len(index_list)):
				self.mbe_frags_deri[order][index_list[j]*atom_group:(index_list[j]+1)*atom_group] += deri[j]
		return 

	def Set_MBE_Force(self):
		self.mbe_deri = np.zeros((self.NAtoms(), 3))
		for order in range (1, self.mbe_order+1): # we ignore the 1st order term since we are dealing with helium, debug
			if order in self.mbe_frags_deri.keys():
				self.mbe_deri += self.mbe_frags_deri[order]
		return self.mbe_deri
	
class Frag(Mol):
        """ Provides a MBE frag of  general purpose molecule"""
        def __init__(self, atoms_ =  None, coords_ = None, index_=None, dist_=None, atom_group_=1):
		Mol.__init__(self, atoms_, coords_)
		self.atom_group = atom_group_
		self.FragOrder = self.coords.shape[0]/self.atom_group
		if (index_!=None):
			self.index = index_
		else:
			self.index = None
		if (dist_!=None):
			self.dist = dist_
		else:
			self.dist = None
		self.frag_mbe_energies=dict()
		self.frag_mbe_energy = None
		self.frag_energy = None
		self.permute_index = range (0, self.FragOrder)
		self.permute_sub_index = None	
		return
		
	def PySCF_Frag_MBE_Energy(self,order):   # calculate the MBE of order N of each frag 
		inner_index = range(0, self.FragOrder) 
		real_frag_index=list(itertools.combinations(inner_index,order))
		ghost_frag_index=[]
		for i in range (0, len(real_frag_index)):
			ghost_frag_index.append(list(set(inner_index)-set(real_frag_index[i])))

		i =0	
		while(i< len(real_frag_index)):
	#	for i in range (0, len(real_frag_index)):
			pyscfatomstring=""
			mol = gto.Mole()
			for j in range (0, order):
				for k in range (0, self.atom_group):
					s = self.coords[real_frag_index[i][j]*self.atom_group+k]
					pyscfatomstring=pyscfatomstring+str(self.AtomName(real_frag_index[i][j]*self.atom_group+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+";"	
			for j in range (0, self.FragOrder - order):
				for k in range (0, self.atom_group):
					s = self.coords[ghost_frag_index[i][j]*self.atom_group+k]
					pyscfatomstring=pyscfatomstring+"GHOST"+str(j*self.atom_group+k)+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+";" 
			pyscfatomstring=pyscfatomstring[:-1]+"  "
			mol.atom =pyscfatomstring			
		
			mol.basis ={}
			ele_set = list(set(self.AllAtomNames()))
			for ele in ele_set:
				mol.basis[str(ele)]="cc-pvqz"

			for j in range (0, self.FragOrder - order):
				for k in range (0, self.atom_group):
					atom_type = self.AtomName(ghost_frag_index[i][j]*self.atom_group+k)
					mol.basis['GHOST'+str(j*self.atom_group+k)]=gto.basis.load('cc-pvqz',str(atom_type))
			mol.verbose=0
			try:
				print "doing case ", i 
				time_log = time.time()
				mol.build()
				mf=scf.RHF(mol)
				hf_en = mf.kernel()
				mp2 = mp.MP2(mf)
				mp2_en = mp2.kernel()
				en = hf_en + mp2_en[0]
				#print "hf_en", hf_en, "mp2_en", mp2_en[0], " en", en	
				self.frag_mbe_energies[LtoS(real_frag_index[i])]=en
				print ("pyscf time..", time.time()-time_log)
				i = i+1
				gc.collect()
			except Exception as Ex:
				print "PYSCF Calculation error... :",Ex
				print "Mol.atom:", mol.atom
				print "Pyscf string:", pyscfatomstring
		return 0

	def Get_Qchem_Frag_MBE_Energy(self, order, path):
		#print "path:", path, "order:", order
		onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		#print "onlyfiles:", onlyfiles, "path", path, "order", order
		for outfile_name in onlyfiles:
			if ( outfile_name[-4:]!='.out' ):
					continue
			outfile = open(path+"/"+outfile_name,"r+")
			outfile_lines = outfile.readlines()
			key = None
			rimp2 = None
			for line in outfile_lines:
				if "!" in line:
					key = line[1:-1]
					continue
				if "non-Brillouin singles" in line:
					nonB_single = float(line.split()[3])
					continue
				if "RIMP2         total energy" in line:
					rimp2 = float(line.split()[4])
					continue
				if "fatal error" in line:
					print "fata error!"
			if nonB_single != 0.0:
				print "Warning: non-Brillouin singles do not equal to zero, non-Brillouin singles=",nonB_single,path,outfile_name
			if key!=None and rimp2!=None:
				#print "key:", key, "length:", len(key)
				self.frag_mbe_energies[key] = rimp2
			else:
				print "Qchem Calculation error on ",path,outfile_name
				raise Exception("Qchem Error")
		return 


	def Write_Qchem_Frag_MBE_Input(self,order):   # calculate the MBE of order N of each frag 
		inner_index = range(0, self.FragOrder)
		real_frag_index=list(itertools.combinations(inner_index,order))
		ghost_frag_index=[]
		for i in range (0, len(real_frag_index)):
			ghost_frag_index.append(list(set(inner_index)-set(real_frag_index[i])))
		i =0
		while(i< len(real_frag_index)):
			qchemstring="$molecule\n0  1\n"
			for j in range (0, order):
				for k in range (0, self.atom_group):
					s = self.coords[real_frag_index[i][j]*self.atom_group+k]
					qchemstring+=str(self.AtomName(real_frag_index[i][j]*self.atom_group+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+"\n"
			for j in range (0, self.FragOrder - order):
				for k in range (0, self.atom_group):
					s = self.coords[ghost_frag_index[i][j]*self.atom_group+k]
					qchemstring+="@"+str(self.AtomName(ghost_frag_index[i][j]*self.atom_group+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+"\n"
			qchemstring += "$end\n"
			qchemstring += "!"+LtoS(real_frag_index[i])+"\n"
			qchemstring += Qchem_RIMP2_Block
			qchem_input=open(str(i+1)+".in","w+")
			qchem_input.write(qchemstring)
			qchem_input.close()
			i = i+1
		gc.collect()
		return

	def Write_Qchem_Frag_MBE_Input_All(self, fragnum):
		if not os.path.isdir(str(fragnum)):
			os.mkdir(str(fragnum))
		os.chdir(str(fragnum))
		for i in range (0, self.FragOrder):
			if not os.path.isdir(str(i+1)):
				os.mkdir(str(i+1))
			os.chdir(str(i+1))
			self.Write_Qchem_Frag_MBE_Input(i+1)
			os.chdir("..")
		os.chdir("..")
		return 

	def Get_Qchem_Frag_MBE_Energy_All(self, fragnum, path):
		if not os.path.isdir(path+"/"+str(fragnum)):
			raise Exception(path+"/"+str(fragnum),"is not calculated")
		oldpath = path
		for i in range (0, self.FragOrder):
			path = oldpath+"/"+str(fragnum)+"/"+str(i+1)
			self.Get_Qchem_Frag_MBE_Energy(i+1, path)
		return

	def PySCF_Frag_MBE_Energy_All(self):
		for i in range (0, self.FragOrder):
			self.PySCF_Frag_MBE_Energy(i+1)
		return  0

	def Set_Frag_MBE_Energy(self):
		self.frag_mbe_energy =  self.Frag_MBE_Energy()
		self.frag_energy = self.frag_mbe_energies[LtoS(self.permute_index)]
		print " self.frag_energy : ",  self.frag_energy
		prod = 1
		for i in self.dist:
			prod = i*prod
		#print "self.frag_mbe_energy", self.frag_mbe_energy
		return 0

	def Frag_MBE_Energy(self,  index=None):     # Get MBE energy recursively 
		if index==None:
			index=range(0, self.FragOrder)
		order = len(index)
		if order==0:
			return 0
		energy = self.frag_mbe_energies[LtoS(index)] 
		for i in range (0, order):
			sub_index = list(itertools.combinations(index, i))
			for j in range (0, len(sub_index)):
				try:
					energy=energy-self.Frag_MBE_Energy( sub_index[j])
				except Exception as Ex:
					print "missing frag energy, error", Ex
		return  energy

	def CopyTo(self, target):
		target.FragOrder = self.FragOrder
		target.frag_mbe_energies=self.frag_mbe_energies
		target.frag_mbe_energy = self.frag_mbe_energy
		target.permute_index = self.permute_index

	def Permute_Frag_by_Index(self, index, indis=[0]):
		new_frags=list()		
		inner_index = Binominal_Combination(indis, self.FragOrder)
		#print "inner_index",inner_index
		for sub_index in inner_index: 
			new_frag = Frag( atoms_ =  self.atoms, coords_ = self.coords, index_= self.index, dist_=self.dist, atom_group_=self.atom_group)
			self.CopyTo(new_frag)
			new_frag.permute_index = index	
			new_frag.permute_sub_index = sub_index
			new_frag.coords=new_frag.coords.reshape((new_frag.FragOrder, new_frag.atom_group,  -1))
			new_frag.coords = new_frag.coords[new_frag.permute_index]
			new_frag.atoms = new_frag.atoms.reshape((new_frag.FragOrder, new_frag.atom_group))
			new_frag.atoms = new_frag.atoms[new_frag.permute_index]
			for group in range (0, new_frag.FragOrder):
				new_frag.coords[group][sorted(sub_index[group*len(indis):(group+1)*len(indis)])] = new_frag.coords[group][sub_index[group*len(indis):(group+1)*len(indis)]]
				new_frag.atoms[group][sorted(sub_index[group*len(indis):(group+1)*len(indis)])] = new_frag.atoms[group][sub_index[group*len(indis):(group+1)*len(indis)]] 	
			new_frag.coords = new_frag.coords.reshape((new_frag.FragOrder*new_frag.atom_group, -1))
			new_frag.atoms = new_frag.atoms.reshape(new_frag.FragOrder*new_frag.atom_group)
			#print "coords:", new_frag.coords, "atom:",new_frag.atoms
			new_frags.append(new_frag)
		# needs some code that fix the keys in frag_mbe_energies[LtoS(index)] after permutation in futher.  KY
		return new_frags

	def Permute_Frag(self, indis = [0]):
		permuted_frags=[]
		indexs=list(itertools.permutations(range(0, self.FragOrder)))
		for index in indexs:
			permuted_frags += self.Permute_Frag_by_Index(list(index), indis)
			#print permuted_frags[-1].atoms, permuted_frags[-1].coords
		return permuted_frags 

	def Frag_Force(self, cm_deri, nn_deri):
		return self.Combine_CM_NN_Deri(cm_deri, nn_deri)	

	def Combine_CM_NN_Deri(self, cm_deri, nn_deri):
		natom = self.NAtoms()
		frag_deri = np.zeros((natom, 3))
		for i in range (0, natom):  ## debug, this is for not including the diagnol
			for j in range (0, natom):  # debug, this is for not including the diagnol
				if j >= i:
					cm_dx = cm_deri[i][j][0]
					cm_dy = cm_deri[i][j][1]
					cm_dz = cm_deri[i][j][2] 
					nn_deri_index = i*(natom+natom-i-1)/2 + (j-i-1) # debug, this is for not including the diagnol
					#nn_deri_index = i*(natom+natom-i+1)/2 + (j-i)  # debug, this is for including the diagnol in the CM
					nn_dcm = nn_deri[nn_deri_index]
				else:
					cm_dx = cm_deri[j][i][3]
					cm_dy = cm_deri[j][i][4]
					cm_dz = cm_deri[j][i][5]
					nn_deri_index = j*(natom+natom-j-1)/2 + (i-j-1)  #debug , this is for not including the diangol
					#nn_deri_index = j*(natom+natom-j+1)/2 + (i-j)    # debug, this is for including the diagnoal in the CM
					nn_dcm = nn_deri[nn_deri_index]
				frag_deri[i][0] += nn_dcm * cm_dx
				frag_deri[i][1] += nn_dcm * cm_dy
				frag_deri[i][2] += nn_dcm * cm_dz
		return frag_deri
