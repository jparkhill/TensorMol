from Util import *
import numpy as np
import random, math
import MolEmb
import networkx as nx

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
		self.set_name = None
		self.num_atom_connected = None # connected  atoms of each atom 
		#things below here are sometimes populated if it is useful.
		self.PESSamples = [] # a list of tuples (atom, new coordinates, energy) for storage.
		self.ecoords = None # equilibrium coordinates.
		self.DistMatrix = None # a list of equilbrium distances, for GO-models.
		self.LJE = None #Lennard-Jones Well-Depths.
		self.GoK = 0.05
		self.energy=None
		self.roomT_H = None
		self.atomization = None
		self.zpe = None # zero point energy
		self.nn_energy=None
		self.qchem_data_path = None
		self.vdw = None
		self.smiles= None
		return

	
	def Num_of_Heavy_Atom(self):
		num = 0
		for i in range (0, self.NAtoms()):
			if self.atoms[i] != 1:
				num += 1
		return num

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
		self.atomization = self.roomT_H
		for i in range (0, self.atoms.shape[0]):
			self.atomization = self.atomization - ele_roomT_H[self.atoms[i]]
			self.energy = self.energy - ele_U[self.atoms[i]]
		return

        def Calculate_vdw(self):
                c = 0.38088 
                self.vdw = 0.0
                s6 = S6['B3LYP']
                for i in range (0, self.NAtoms()):
			atom1 = self.atoms[i]
                        for j in range (i+1, self.NAtoms()):
				atom2 = self.atoms[j]
                                self.vdw += -s6*c*((C6_coff[atom1]*C6_coff[atom2])**0.5)/(self.DistMatrix[i][j])**6 * (1.0/(1.0+6.0*(self.DistMatrix[i][j]/(atomic_vdw_radius[atom1]+atomic_vdw_radius[atom2]))**-12))
                return 



	def AtomsWithin(self,rad, pt):
		# Returns indices of atoms within radius of point.
		dists = map(lambda x: np.linalg.norm(x-pt),self.coords)
		return [i for i in range(self.NAtoms()) if dists[i]<rad]

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
			self.name = filename
			print "name:", self.name
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
					self.energy = float(lines[i].split()[4])
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
			return False
			#raise Ex
		return True

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
				self.internal = float((lines[1].split())[12])
				self.roomT_H = float((lines[1].split())[14])
				self.zpe = float((lines[1].split())[11])
				self.energy = self.internal - self.zpe
				self.smiles = lines[-2].split()[0]
				print "smiles:", self.smiles
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
		if (self.energy!=None and self.roomT_H!=None):
			self.Calculate_Atomization()
		return

	def FromXYZString(self,string, set_name = None):
		lines = string.split("\n")
		self.set_name = set_name
		natoms=int(lines[0])
		self.name = lines[1] #debug
		print "working on mol: ", self.name
		if (len(lines[1].split())>1):
			try:
				self.energy=float(lines[1].split()[1])
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
		if self.energy:
			for i in range (0, self.atoms.shape[0]):
                        	self.energy = self.energy - ele_U[self.atoms[i]]
			#print "after self.energy:", self.energy
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
			f.write(str(natom)+"\nComment:\n")
			for i in range (0, natom):
				atom_name =  atoi.keys()[atoi.values().index(self.atoms[i])]
				f.write(atom_name+"   "+str(self.coords[i][0])+ "  "+str(self.coords[i][1])+ "  "+str(self.coords[i][2])+"\n")

	def WriteSmiles(self, fpath=".", fname="gdb9_smiles", mode = "a"):
		if not os.path.exists(os.path.dirname(fpath+"/"+fname+".dat")):
                        try:
                                os.makedirs(os.path.dirname(fpath+"/"+fname+".dat"))
                        except OSError as exc:
                                if exc.errno != errno.EEXIST:
                                        raise
                with open(fpath+"/"+fname+".dat", mode) as f:
                        f.write(self.name+ "  "+ self.smiles+"\n")
			f.close()
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

	def RotateX(self):
		self.coords[:,1] = self.Center()[1] + np.cos(np.pi)*(self.coords[:,1]-self.Center()[1]) - np.sin(np.pi)*(self.coords[:,2]-self.Center()[2])
		self.coords[:,2] = self.Center()[2] + np.sin(np.pi)*(self.coords[:,1]-self.Center()[1]) + np.cos(np.pi)*(self.coords[:,2]-self.Center()[2])

	def WriteInterpolation(self,b,n=0):
		for i in range(10): # Check the interpolation.
			m=Mol(self.atoms,self.coords*((9.-i)/9.)+b.coords*((i)/9.))
			m.WriteXYZfile("./results/", "Interp"+str(n))

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
		import MolEmb
		self.DistMatrix = MolEmb.Make_DistMat(self.coords)
		self.LJEFromDist()

	def LJEFromDist(self):
		" Assigns lennard jones depth matrix "
		self.LJE = np.zeros((len(self.coords),len(self.coords)))
		self.LJE += 0.1
		return
		for i in range(len(self.coords)):
			for j in range(i+1,len(self.coords)):
				if (self.DistMatrix[i,j] < 2.8): # is covalent
					if (self.atoms[i]==6 and self.atoms[j]==6):
						if ( self.DistMatrix[i,j] <1.3):
							self.LJE[i,j] = 0.319558 # Bond energies in hartree
						elif ( self.DistMatrix[i,j]<1.44):
							self.LJE[i,j] = 0.23386
						else:
							self.LJE[i,j] = 0.132546
					elif ((self.atoms[i]==1 and self.atoms[j]==6) or (self.atoms[i]==6 and self.atoms[j]==1)):
						self.LJE[i,j] = 0.157
					elif ((self.atoms[i]==1 and self.atoms[j]==7) or (self.atoms[i]==7 and self.atoms[j]==1)):
						self.LJE[i,j] = 0.148924
					elif ((self.atoms[i]==1 and self.atoms[j]==8) or (self.atoms[i]==8 and self.atoms[j]==1)):
						self.LJE[i,j] = 0.139402
					elif ((self.atoms[i]==6 and self.atoms[j]==7) or (self.atoms[i]==7 and self.atoms[j]==6)):
						self.LJE[i,j] = 0.0559894
					elif ((self.atoms[i]==6 and self.atoms[j]==8) or (self.atoms[i]==8 and self.atoms[j]==6)):
						self.LJE[i,j] = 0.0544658
					elif (self.atoms[i]==8 and self.atoms[j]==8):
						if( self.DistMatrix[i,j]<1.40):
							self.LJE[i,j] = 0.189678
						else:
							self.LJE[i,j] = 0.0552276
					elif (self.atoms[i]==7 and self.atoms[j]==7):
						if ( self.DistMatrix[i,j] <1.2):
							self.LJE[i,j] = 0.359932 # Bond energies in hartree
						elif ( self.DistMatrix[i,j]<1.4):
							self.LJE[i,j] = 0.23386
						else:
							self.LJE[i,j] = 0.0552276
					else:
						self.LJE[i,j] = 0.1
				else:
					self.LJE[i,j] = 0.005 # Non covalent interactions
		self.LJE += self.LJE.T

	def LJEnergy(self,x):
		''' The GO potential enforces equilibrium bond lengths with Lennard Jones Forces.'''
		xmat = np.array(x).reshape(self.NAtoms(),3)
		dmat = MolEmb.Make_DistMat(xmat)
		np.fill_diagonal(dmat,1.0)
		term2 = np.power(self.DistMatrix/dmat,6.0)
		term1 = np.power(term2,2.0)
		return np.sum(self.LJE*(term1-2.0*term2))

	def GoEnergy(self,x):
		''' The GO potential enforces equilibrium bond lengths. This is the lennard jones soft version'''
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		xmat = np.array(x).reshape(self.NAtoms(),3)
		newd = MolEmb.Make_DistMat(xmat)
		newd -= self.DistMatrix
		newd = newd*newd
		return self.GoK*np.sum(newd)

	def GoEnergyAfterAtomMove(self,s,ii):
		''' The GO potential enforces equilibrium bond lengths. '''
		raise Exception("Depreciated.")

	def GoForce(self, at_=-1):
		'''
			The GO potential enforces equilibrium bond lengths, and this is the force of that potential.
			Args: at_ an atom index, if at_ = -1 it returns an array for each atom. 
		'''
		return self.GoK*MolEmb.Make_GoForce(self.coords,self.DistMatrix,at_)

	def GoForceLocal(self, at_=-1):
		''' The GO potential enforces equilibrium bond lengths, and this is the force of that potential.
			A MUCH FASTER VERSION OF THIS ROUTINE IS NOW AVAILABLE, see MolEmb::Make_Go
		'''
		return self.GoK*MolEmb.Make_GoForceLocal(self.coords,self.DistMatrix,at_)

	def LJForce(self, at_=-1):
		''' The GO potential enforces equilibrium bond lengths, and this is the force of that potential.
			A MUCH FASTER VERSION OF THIS ROUTINE IS NOW AVAILABLE, see MolEmb::Make_Go
		'''
		return MolEmb.Make_LJForce(self.coords,self.DistMatrix,self.LJE,at_)

	def NumericLJForce(self):
		disp = 0.00000001
		frc = np.zeros((self.NAtoms(),3))
		for i in range(self.NAtoms()):
			for ip in range(3):
				tmp = self.coords
				tmp[i,ip] += disp
				e1 = self.LJEnergy(tmp)
				tmp = self.coords
				tmp[i,ip] -= disp
				e2 = self.LJEnergy(tmp)
				frc[i,ip] = (e1-e2)/(2.0*disp)
		return frc

	def NumericLJHessDiag(self):
		if (self.DistMatrix==None):
			print "Build DistMatrix"
			raise Exception("dmat")
		disp=0.001
		hessd=np.zeros((self.NAtoms(),3))
		for i in range(self.NAtoms()):
			for ip in range(3):
				tmp = self.coords.flatten()
				tmp[i*3+ip] += disp
				tmp[i*3+ip] += disp
				f1 = self.LJEnergy(tmp)
				tmp = self.coords.flatten()
				tmp[i*3+ip] += disp
				tmp[i*3+ip] -= disp
				f2 = self.LJEnergy(tmp)
				tmp = self.coords.flatten()
				tmp[i*3+ip] -= disp
				tmp[i*3+ip] += disp
				f3 = self.LJEnergy(tmp)
				tmp = self.coords.flatten()
				tmp[i*3+ip] -= disp
				tmp[i*3+ip] -= disp
				f4 = self.LJEnergy(tmp)
				hessd[i, ip] = (f1-f2-f3+f4)/(4.0*disp*disp)
		return hessd

	def NumericLJHessian(self):
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
							f1 = self.LJEnergy(tmp)
							tmp = self.coords.flatten()
							tmp[i*3+ip] += disp
							tmp[j*3+jp] -= disp
							f2 = self.LJEnergy(tmp)
							tmp = self.coords.flatten()
							tmp[i*3+ip] -= disp
							tmp[j*3+jp] += disp
							f3 = self.LJEnergy(tmp)
							tmp = self.coords.flatten()
							tmp[i*3+ip] -= disp
							tmp[j*3+jp] -= disp
							f4 = self.LJEnergy(tmp)
							hess[i*3+ip,j*3+jp] = (f1-f2-f3+f4)/(4.0*disp*disp)
		return (hess+hess.T-np.diag(np.diag(hess)))

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
		return self.GoK*MolEmb.Make_GoHess(self.coords,self.DistMatrix)

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
					tore[nout,d,:,:] = self.coords+disp*(self.NAtoms()*(d-npts/2.0+0.37)/npts)*eigv
					#print disp*(self.NAtoms()*(d-npts/2.0+0.37)/npts)*eigv
					#print d, self.GoEnergy(tore[nout,d,:,:].flatten())#, self.GoK*MolEmb.Make_GoForce(tore[nout,d,:,:],self.DistMatrix,-1)
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


	def AtomName(self, i):
		return atoi.keys()[atoi.values().index(self.atoms[i])]



	def AllAtomNames(self):
		names=[]
		for i in range (0, self.atoms.shape[0]):
			names.append(atoi.keys()[atoi.values().index(self.atoms[i])])
		return names

	def Set_Qchem_Data_Path(self):
		self.qchem_data_path="./qchem"+"/"+self.set_name+"/"+self.name
		return


	def PySCF_Energy(self, basis_='cc-pvqz'):
		mol = gto.Mole()
		pyscfatomstring=""
		for j in range(len(self.atoms)):
			s = self.coords[j]
			pyscfatomstring=pyscfatomstring+str(self.AtomName(j))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+(";" if j!= len(self.atoms)-1 else "")
		mol.atom = pyscfatomstring
		mol.basis = basis_
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
		return 



class Frag_of_Mol(Mol):
	def __init__(self, atoms_=None, coords_=None):
		Mol.__init__(self, atoms_, coords_)
		self.undefined_bond_type =  None # whether the dangling bond can be connected  to H or not
		self.undefined_bonds = None  # capture the undefined bonds of each atom


        def FromXYZString(self,string, set_name = None):
		self.set_name = set_name
                lines = string.split("\n")
                natoms=int(lines[0])
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
		import ast
		try:
			self.undefined_bonds = ast.literal_eval(lines[1][lines[1].index("{"):lines[1].index("}")+1])
			if "type" in self.undefined_bonds.keys():
				self.undefined_bond_type = self.undefined_bonds["type"]
			else:
				self.undefined_bond_type = "any"
		except:
			self.name = lines[1] #debug
			self.undefined_bonds = {}
			self.undefined_bond_type = "any"
                return


	def Make_AtomNodes(self):
                atom_nodes = []
                for i in range (0, self.NAtoms()):
			if i in self.undefined_bonds.keys():
                        	atom_nodes.append(AtomNode(self.atoms[i], i,  self.undefined_bond_type, self.undefined_bonds[i]))
			else:
				atom_nodes.append(AtomNode(self.atoms[i], i, self.undefined_bond_type))
                self.atom_nodes = atom_nodes
		return


