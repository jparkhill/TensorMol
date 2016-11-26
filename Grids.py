import numpy as np
import random
from pyscf import scf
from pyscf import gto
from pyscf import dft
import math
from math import pi as Pi

def MakeUniform(point,disp,num):
	''' Uniform Grids of dim numxnumxnum around a point'''
	grids = np.mgrid[-disp:disp:num*1j, -disp:disp:num*1j, -disp:disp:num*1j]
	grids = grids.transpose()
	grids = grids.reshape((grids.shape[0]*grids.shape[1]*grids.shape[2], grids.shape[3]))
	return point+grids

def GridstoRaw(grids, ngrids=250, save_name="mol", save_path ="./densities/"):
	#print "Writing Grid Mx, Mn, Std, Sum ", np.max(grids),np.min(grids),np.std(grids),np.sum(grids)
	mgrids = np.copy(grids)
	mgrids *= (254/np.max(grids))
	mgrids = np.array(mgrids, dtype=np.uint8)
	#print np.bincount(mgrids)
	#print "Writing Grid Mx, Mn, Std, Sum ", np.max(mgrids),np.min(mgrids),np.std(mgrids),np.sum(mgrids)
	print "Saving density to:",save_path+save_name+".raw"
	f = open(save_path+save_name+".raw", "wb")
	f.write(bytes(np.array([ngrids,ngrids,ngrids],dtype=np.uint8).tostring())+bytes(mgrids.tostring()))
	f.close()

def MatrixPower(A,p,PrintCondition=False):
	''' Raise a Hermitian Matrix to a possibly fractional power. '''
	#w,v=np.linalg.eig(A)
	# Use SVD
	u,s,v = np.linalg.svd(A)
	if (PrintCondition):
		print "MatrixPower: Minimal Eigenvalue =", np.min(s)
	for i in range(len(s)):
		if (abs(s[i]) < np.power(10.0,-14.0)):
			s[i] = np.power(10.0,-14.0)
	#print("Matrixpower?",np.dot(np.dot(v,np.diag(w)),v.T), A)
	#return np.dot(np.dot(v,np.diag(np.power(w,p))),v.T)
	return np.dot(u,np.dot(np.diag(np.power(s,p)),v))

def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/np.linalg.norm(axis)
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def ReflectionMatrix(axis1,axis2):
	axis1 = np.asarray(axis1)
	axis2 = np.asarray(axis2)
	a1=axis1/np.linalg.norm(axis1)
	a2=axis2/np.linalg.norm(axis2)
	unitNormal=np.cross(a1,a2)
	return np.eye(3) - 2.0*np.outer(unitNormal,unitNormal)

def OctahedralOperations():
	''' 
		Transformation matrices for symmetries of an octahedral shape.
		Far from the complete set but enough for debugging and seeing if it helps.
	'''
	Ident=[np.eye(3)]
	FaceRotations=[RotationMatrix([1,0,0], Pi/2.0),RotationMatrix([0,1,0], Pi/2.0),RotationMatrix([0,0,1], Pi/2.0),RotationMatrix([-1,0,0], Pi/2.0),RotationMatrix([0,-1,0], Pi/2.0),RotationMatrix([0,0,-1], Pi/2.0)]
	FaceRotations2=[RotationMatrix([1,0,0], Pi),RotationMatrix([0,1,0], Pi),RotationMatrix([0,0,1], Pi),RotationMatrix([-1,0,0], Pi),RotationMatrix([0,-1,0], Pi),RotationMatrix([0,0,-1], Pi)]
	FaceRotations3=[RotationMatrix([1,0,0], 3.0*Pi/2.0),RotationMatrix([0,1,0], 3.0*Pi/2.0),RotationMatrix([0,0,1], 3.0*Pi/2.0),RotationMatrix([-1,0,0], 3.0*Pi/2.0),RotationMatrix([0,-1,0], 3.0*Pi/2.0),RotationMatrix([0,0,-1], 3.0*Pi/2.0)]
	CornerRotations=[RotationMatrix([1,1,1], 2.0*Pi/3.0),RotationMatrix([-1,1,1], 2.0*Pi/3.0),RotationMatrix([-1,-1,1], 2.0*Pi/3.0),RotationMatrix([-1,-1,-1], 2.0*Pi/3.0),RotationMatrix([-1,1,-1], 2.0*Pi/3.0),RotationMatrix([1,-1,-1], 2.0*Pi/3.0),RotationMatrix([1,1,-1], 2.0*Pi/3.0),RotationMatrix([1,-1,1], 2.0*Pi/3.0)]
	CornerRotations2=[RotationMatrix([1,1,1], 4.0*Pi/3.0),RotationMatrix([-1,1,1], 4.0*Pi/3.0),RotationMatrix([-1,-1,1], 4.0*Pi/3.0),RotationMatrix([-1,-1,-1], 4.0*Pi/3.0),RotationMatrix([-1,1,-1], 4.0*Pi/3.0),RotationMatrix([1,-1,-1], 4.0*Pi/3.0),RotationMatrix([1,1,-1], 4.0*Pi/3.0),RotationMatrix([1,-1,1], 4.0*Pi/3.0)]
	EdgeRotations=[RotationMatrix([1,1,0], Pi),RotationMatrix([1,0,1], Pi),RotationMatrix([0,1,1], Pi)]
	EdgeReflections=[ReflectionMatrix([1,0,0],[0,1,0]),ReflectionMatrix([0,0,1],[0,1,0]),ReflectionMatrix([1,0,0],[0,0,1])]
	return FaceRotations+FaceRotations2+FaceRotations3+CornerRotations+CornerRotations2+EdgeRotations+EdgeReflections

#
# The H@0 atom is for fitting the potential near equilibrium and it's small...
#
ATOM_BASIS={'H@0': gto.basis.parse('''
H    S
	  15.0					1.0
		''')}
TOTAL_SENSORY_BASIS={'C': gto.basis.parse('''
C    S
	  1.0					1.0000000
C    S
	  0.5					1.0000000
C    S
	  0.1					1.0000000
C    S
	  0.02					1.0000000
C    S
	  0.005					1.0000000
C    P
	  1.0					1.0000000
C    P
	  0.5					1.0000000
C    P
	  0.1					1.0000000
C    P
	  0.02					1.0000000
C    P
	  0.005					1.0000000
C    D
	  0.5					1.0000000
C    D
	  0.05					1.0000000
C    D
	  0.01					1.0000000
C    D
	  0.004					1.0000000
C    F
	  0.2					1.0000000
C    F
	  0.02					1.0000000
C    F
	  0.004					1.0000000
C    G
	  0.1					1.0000000
C    G
	  0.02					1.0000000
C    G
	  0.004					1.0000000
C    H
	  0.1					1.0000000
C    H
	  0.02					1.0000000
C    H
	  0.004					1.0000000
C    I
	  0.02					1.0000000
C    I
	  0.004					1.0000000
	  '''),'H@0': gto.basis.parse('''
H    S
	  1.5					1.0
		'''),'H@1': gto.basis.parse('''
H    S
	  3              1.0
		'''),'H@2': gto.basis.parse('''
H    S
	  2.6				1.0
		'''),'H@3': gto.basis.parse('''
H    S
	  2.34					1.0
		'''),'H@4': gto.basis.parse('''
H    S
	  2				1.0
		'''),'H@5': gto.basis.parse('''
H    S
	  1.7				1.0
		'''),'H@6': gto.basis.parse('''
H    S
	  1.3					1.0
		'''),'H@7': gto.basis.parse('''
H    S
	  1					1.0
		'''),'H@8': gto.basis.parse('''
H    S
	  .6666             1.0
		'''),'H@9': gto.basis.parse('''
H    S
	  .3333             1.0
		'''),'N@0': gto.basis.parse('''
H    S
	  1.0             -1.0
		''')
		}

class Grids:
	""" 
		Precomputes and stores orthogonalized grids for embedding molecules and potentials.
		self,NGau_=6, GridRange_=0.6, NPts_=45 is a good choice for a 1 Angstrom potential fit
		
		Also now has a few
	"""
	def __init__(self,NGau_=6, GridRange_=0.8, NPts_=40):
		# Coulomb Fitting parameters
		self.Spherical = False
		# Cartesian Embedding parameters.
		self.GridRange = GridRange_
		self.NGau=NGau_ # Number of gaussians in each direction.
		self.NPts=NPts_ # Number of gaussians in each direction.
		self.NGau3=NGau_*NGau_*NGau_ # Number of gaussians in each direction.
		self.NPts3=NPts_*NPts_*NPts_
		self.OBFs=None # matrix of orthogonal basis functions stored on grid, once.
		self.GauGrid=None
		self.Grid=None
		self.dx = 0.0
		self.dy = 0.0
		self.dz = 0.0
		
		self.SH_S = None
		self.SH_Sinv = None
		self.SH_C = None
		
		#
		# These are for embedding molecular environments.
		#
		self.SenseRange = 3.5
		self.NSense = self.NGau
		if (self.Spherical):
			self.SenseRange = 12.0
			self.NSense = None
		self.SenseGrid = None
		self.SenseS = None
		self.SenseSinv = None
		self.Isometries = None
		self.InvIsometries = None
		self.IsometryRelabelings = None
		return

	def	MyGrid(self):
		if (self.Grid==None):
			self.Populate()
		return self.Grid

	def	BasisRelabelingUnderTransformation(self,trans,i):
		v=self.GauGrid[i]
		vp=np.dot(trans,v)
		dgr = np.array(map(lambda x: np.linalg.norm(x-vp),self.GauGrid))
		poss = np.where(dgr<0.000001)[0][0]
		return poss

	def BuildIsometries(self):
		self.Isometries = OctahedralOperations()
		self.InvIsometries = OctahedralOperations()
		self.IsometryRelabelings = np.zeros(shape=(len(self.Isometries),len(self.GauGrid)),dtype=np.int64)
		for i in range(len(self.Isometries)):
			op=self.Isometries[i]
			i0=range(len(self.GauGrid))
			self.InvIsometries[i] = np.linalg.inv(op)
			self.IsometryRelabelings[i]=np.array(map(lambda x: self.BasisRelabelingUnderTransformation(op,x), i0))
			# print ip
			# Check that it's an isometry.
			if (i0 != sorted(self.IsometryRelabelings[i])):
				print "Not an isometry :( ", i0,self.IsometryRelabelings[i]
				raise Exception("Bad Isometry")
		return

	def Populate(self):
		print "Populating Grids... "
		#
		# Populate output Bases
		#
		self.GauGrid = MakeUniform([0,0,0],self.GridRange*.8,self.NGau)
		self.Grid = MakeUniform([0,0,0],self.GridRange,self.NPts)
		self.dx=(np.max(self.Grid[:,0])-np.min(self.Grid[:,0]))/self.NPts*1.889725989
		self.dy=(np.max(self.Grid[:,1])-np.min(self.Grid[:,1]))/self.NPts*1.889725989
		self.dz=(np.max(self.Grid[:,2])-np.min(self.Grid[:,2]))/self.NPts*1.889725989
		import MolEmb
		self.SH_S = MolEmb.Overlap_SH()
		self.SH_Sinv = MatrixPower(self.SH_S,-1.0,True)
		self.SH_C = MatrixPower(self.SH_S,-0.5)
		return
		
		mol = gto.Mole()
		mol.atom = ''.join(["H@0 "+str(self.GauGrid[iii,0])+" "+str(self.GauGrid[iii,1])+" "+str(self.GauGrid[iii,2])+";" for iii in range(len(self.GauGrid))])[:-1]
		if (self.NGau3%2==0):
			mol.spin = 0
		else:
			mol.spin = 1
		if (ATOM_BASIS == None):
			raise("missing ATOM_BASIS")
		mol.basis = ATOM_BASIS
		try:
			mol.build()
		except Exception as Ex:
			print mol.atom
			raise Ex
		# All this shit could be Pre-Computed...
		# Really any grid could be used.
		orbs=gto.eval_gto('GTOval_sph',mol._atm,mol._bas,mol._env,self.Grid*1.889725989)
		nbas=orbs.shape[1]
		if (nbas!=self.NGau3):
			raise Exception("insanity")
		S=mol.intor('cint1e_ovlp_sph')
		#S = np.zeros(shape=(nbas,nbas))
		#for i in range(nbas):
		#	for j in range(nbas):
		#		S[i,j] += np.sum(orbs[:,i]*orbs[:,j])
		C = MatrixPower(S,-0.5)
		if (0):
			for i in range(nbas):
				CM = np.dot(self.Grid.T,orbs[:,i])
				print "Centers of Mass, i", np.dot(self.Grid.T,orbs[:,i]*orbs[:,i])*self.dx*self.dy*self.dz, self.GauGrid[i]
				Rsq = np.array(map(np.linalg.norm,self.Grid-CM))
				print "Rsq of Mass, i", np.sqrt(np.dot(Rsq,orbs[:,i]*orbs[:,i]))*self.dx*self.dy*self.dz
				for j in range(nbas):
					print "Normalization of grid i.", np.sum(orbs[:,i]*orbs[:,j])*self.dx*self.dy*self.dz
		self.OBFs = np.zeros(shape=(self.NGau3,self.NPts3))
		for i in range(nbas):
			for j in range(nbas):
				self.OBFs[i,:] += (C.T[i,j]*orbs[:,j]).T
		# Populate Sensory bases.
		self.PopulateSense()
		if (not self.Spherical):
			self.BuildIsometries()
			print "Using ", len(self.Isometries), " isometries."
		print "Grid storage cost: ",self.OBFs.size*64/1024/1024, "Mb"
		#for i in range(nbas):
		#	GridstoRaw(orbs[:,i]*orbs[:,i],self.NPts,"BF"+str(i))
		#	GridstoRaw(self.OBFs[i,:]*self.OBFs[i,:],self.NPts,"OBF"+str(i))
		# Quickly check orthonormality.
		#for i in range(nbas):
		#		for j in range(nbas):
		#			print np.dot(self.OBFs[i],self.OBFs[j])*self.dx*self.dy*self.dz
		#		print ""

	def PopulateSense(self):
		mol = gto.Mole()
		if (self.Spherical):
			mol.atom ="C 0.0 0.0 0.0"
			mol.spin = 0
		else:
			self.SenseGrid = MakeUniform([0,0,0],self.SenseRange,self.NSense)
			#pyscfatomstring="C "+str(p[0])+" "+str(p[1])+" "+str(p[2])+";"
			mol.atom = ''.join(["H@0 "+str(self.SenseGrid[iii,0])+" "+str(self.SenseGrid[iii,1])+" "+str(self.SenseGrid[iii,2])+";" for iii in range(len(self.SenseGrid))])
			na = self.NSense
			if (na%2 == 0):
				mol.spin = 0
			else:
				mol.spin = 1
		if (TOTAL_SENSORY_BASIS == None):
			raise("missing sensory basis")
		mol.basis = TOTAL_SENSORY_BASIS
		nsaos = 0
		try:
			mol.build()
		except Exception as Ex:
			print mol.atom, mol.basis
			raise Ex
		#nbas = gto.nao_nr(mol)
		#print nbas
		self.SenseS = mol.intor('cint1e_ovlp_sph',shls_slice=(0,mol.nbas,0,mol.nbas))
		self.NSense = self.SenseS.shape[0]
		self.SenseSinv = MatrixPower(self.SenseS,-1.0)
		return

	def Vectorize(self,input, QualityOfFit=False):
		''' 
		Input is rasterized volume information, 
		output is a vector of NGau3 coefficients fitting that volume. 
		
		The underlying grid is assumed to be MyGrid(), although it can be scaled.
		'''
		if self.Grid==None:
			self.Populate()
		CM = np.dot(self.Grid.T,input)
		if ((self.GridRange-np.max(CM))/self.GridRange < 0.2):
			print "Warning... GridRange ", ((self.GridRange-np.max(CM))/self.GridRange)
		output = np.tensordot(self.OBFs,np.power(input,0.5),axes=[[1],[0]])*self.dx*self.dy*self.dz
		if (QualityOfFit and np.linalg.norm(input)!=0.0):
			GridstoRaw(input,self.NPts,"Input")
			print "Coefs", output
			tmp = self.Rasterize(output)
			GridstoRaw(tmp,self.NPts,"Output")
			print "Sum of Input and reconstruction", np.sum(input), np.sum(tmp)
			print "Average of Input and reconstruction", np.average(input), np.average(tmp)
			print "Max of Input and reconstruction", np.max(input), np.max(tmp)
			print "relative norm of difference:", np.linalg.norm(tmp-input)/np.linalg.norm(input)
			GridstoRaw(input-tmp,self.NPts,"Diff")
			tmp /= np.sum(tmp)
			print "Centers of Mass, in", np.dot(self.Grid.T,input)," and out ", np.dot(self.Grid.T,tmp)
			Rsq = np.array(map(np.linalg.norm,self.Grid-CM))
			print "Variance of In", np.dot(Rsq.T,input)
			print "Variance of Out", np.dot(Rsq.T,tmp)
		return output

	def Rasterize(self,inp):
		if (self.Spherical):
			grd = MakeUniform([0,0,0],self.SenseRange, self.NPts)
			orbs = self.SenseOnGrid([0.0,0.0,0.0],grd)
			if (len(inp)!=self.NSense):
				raise Exception("Bad input dim.")
			return np.tensordot(inp,orbs,axes=[[0],[1]])
		else :
			if (len(inp)!=self.NGau3):
				raise Exception("Bad input dim.")
			return np.power(np.tensordot(inp,self.OBFs,axes=[[0],[0]]),2.0)

	def CenterOfP(self,POnGrid,AGrid=None):
		if (len(POnGrid)!=self.NPts3):
			raise Exception("Bad input dim.")
		if (AGrid==None):
			return np.array([np.dot(self.MyGrid().T,POnGrid)])[0]
		else:
			return np.array([np.dot(AGrid.T,POnGrid)])[0]

	def SenseOnGrid(self,p,grd_):
		mol = gto.Mole()
		mol.atom =''
		if (not self.Spherical):
			tmpgrid = self.SenseGrid + p
			mol.atom = ''.join(["H@0 "+str(tmpgrid[iii,0])+" "+str(tmpgrid[iii,1])+" "+str(tmpgrid[iii,2])+";" for iii in range(len(tmpgrid))])
		else:
			mol.atom = ''.join("C "+str(p[0])+" "+str(p[1])+" "+str(p[2])+";")
		mol.spin = 0
		if (TOTAL_SENSORY_BASIS == None): 
			raise("missing sensory basis")
		mol.basis = TOTAL_SENSORY_BASIS
		mol.build()
		return gto.eval_gto('GTOval_sph',mol._atm,mol._bas,mol._env,grd_*1.889725989)

	def VecToRaw(self,inp,Nm_="VecToRaw"):
		GridstoRaw(self.Rasterize(inp),self.NPts,Nm_)

	def VdwDensity(self,m,p=[0.0, 0.0, 0.0],ngrid=150,Nm_="Atoms",tag=None):
		samps, vol = m.SpanningGrid(ngrid,2)
		print "Grid ranges (A):",np.max(samps[:,0]),np.min(samps[:,0])
		print "Grid ranges (A):",np.max(samps[:,1]),np.min(samps[:,1])
		print "Grid ranges (A):",np.max(samps[:,2]),np.min(samps[:,2])
		# Make the atom densities.
		Ps = self.MolDensity(samps,m,p,tag)
		GridstoRaw(Ps,ngrid,Nm_)
		return samps

	def MolDensity(self,samps,m,p=[0.0,0.0,0.0],tag=None):
		Ps = np.zeros(len(samps))
		mol = gto.Mole()
		pyscfatomstring=''
		if (tag==None):
			for j in range(len(m.atoms)):
				pyscfatomstring=pyscfatomstring+"H@"+str(m.atoms[j])+" "+str(m.coords[j,0])+" "+str(m.coords[j,1])+" "+str(m.coords[j,2])+(";" if j!= len(m.atoms)-1 else "")
		else:
			for j in range(len(m.atoms)):
				if (j == tag):
					print "Tagging atom", j
					pyscfatomstring=pyscfatomstring+"N@0"+" "+str(m.coords[j,0])+" "+str(m.coords[j,1])+" "+str(m.coords[j,2])+(";" if j!= len(m.atoms)-1 else "")
				else:
					pyscfatomstring=pyscfatomstring+"H@"+str(m.atoms[j])+" "+str(m.coords[j,0])+" "+str(m.coords[j,1])+" "+str(m.coords[j,2])+(";" if j!= len(m.atoms)-1 else "")
		mol.atom = pyscfatomstring
		mol.basis = TOTAL_SENSORY_BASIS
		mol.verbose = 0
		if (len(m.atoms)%2 == 0):
			mol.spin = 0
		else:
			mol.spin = 1
		try:
			mol.build()
		except Exception as Ex:
			print mol.atom, mol.basis, m.atoms, m.coords
			raise Ex
		return np.sum(gto.eval_gto('GTOval_sph',mol._atm,mol._bas,mol._env,samps*1.889725989),axis=1)

	def AtomEmbedAtomCentered(self,samps,m,p,i=-1):
		mol = gto.Mole()
		MaxEmbedded = 30
		SensedAtoms = [a for a in m.AtomsWithin(30.0,p) if a != i]
		if (len(SensedAtoms)>MaxEmbedded):
			SensedAtoms=SensedAtoms[:MaxEmbedded]
		if (len(SensedAtoms)==0):
			raise Exception("NoAtomsInSensoryRadius")
		mol.atom="C "+str(p[0])+" "+str(p[1])+" "+str(p[2])+";"
		na=0
		for j in SensedAtoms:
			mol.atom=mol.atom+"H@"+str(m.atoms[j])+" "+str(m.coords[j,0])+" "+str(m.coords[j,1])+" "+str(m.coords[j,2])+(";" if j!=SensedAtoms[-1] else "")
			na=na+1
		#print mol.atom
		#print self.atoms
		#print self.coords
		#print "Basis Atom",[mol.bas_atom(i) for i in range(mol.nbas)]
		if (na%2 == 0):
			mol.spin = 0
		else:
			mol.spin = 1
		if (TOTAL_SENSORY_BASIS == None): 
			raise("missing sensory basis")
		mol.basis = TOTAL_SENSORY_BASIS
		nsaos = 0
		try:
			mol.build()
		except Exception as Ex:
			print mol.atom, mol.basis, m.atoms, m.coords, SensedAtoms, p
			raise Ex
		nsaos = gto.nao_nr_range(mol,0,mol.atom_nshells(0))[1]
		nbas = gto.nao_nr(mol)
		print "nAtoms: ",m.NAtoms()," nsaos: ", nsaos, " nbas ", nbas
		S = mol.intor('cint1e_ovlp_sph',shls_slice=(0,mol.atom_nshells(0),0,mol.atom_nshells(0)))
		Sinv = MatrixPower(S,-1.0)
		SBFs = gto.eval_gto('GTOval_sph',mol._atm,mol._bas,mol._env,samps*1.889725989,comp=1,shls_slice=(0,mol.atom_nshells(0)))
		print "SBFs.shape", SBFs.shape
		Cs = mol.intor('cint1e_ovlp_sph',shls_slice=(0,mol.atom_nshells(0),mol.atom_nshells(0),mol.nbas))
		print "Cs.shape", Cs.shape
	#	for i in range(len(Cs[0])):
	#		tmp = np.dot(SBFs,np.dot(Sinv,Cs[:,i]))
	#		GridstoRaw(tmp*tmp,150,"Atoms"+str(i))
	#	exit(0)
		Sd = np.sum(np.dot(SBFs,np.dot(Sinv,Cs)),axis=1)
		print "Sd.shape", Sd.shape
		return Sd

	def TestGridGauEmbedding(self,samps,m,p,i):
		mol = gto.Mole()
		MaxEmbedded = 15
		SensedAtoms = [a for a in m.AtomsWithin(10.0,p) if a != i]
		if (len(SensedAtoms)>MaxEmbedded):
			SensedAtoms=SensedAtoms[:MaxEmbedded]
		if (len(SensedAtoms)==0):
			raise Exception("NoAtomsInSensoryRadius")

		GauGrid = MakeUniform([0,0,0],3.5,self.NGau)
		#pyscfatomstring="C "+str(p[0])+" "+str(p[1])+" "+str(p[2])+";"
		mol.atom = ''.join(["H@0 "+str(GauGrid[iii,0])+" "+str(GauGrid[iii,1])+" "+str(GauGrid[iii,2])+";" for iii in range(len(GauGrid))])

		na = len(GauGrid)
		#na=0
		for j in SensedAtoms:
			mol.atom=mol.atom+"H@"+str(m.atoms[j])+" "+str(m.coords[j,0])+" "+str(m.coords[j,1])+" "+str(m.coords[j,2])+(";" if j!=SensedAtoms[-1] else "")
			na=na+1
		#print mol.atom
		#print self.atoms
		#print self.coords
		#print "Basis Atom",[mol.bas_atom(i) for i in range(mol.nbas)]

		if (na%2 == 0):
			mol.spin = 0
		else:
			mol.spin = 1
		if (TOTAL_SENSORY_BASIS == None): 
			raise("missing sensory basis")
		mol.basis = TOTAL_SENSORY_BASIS
		nsaos = 0

		try:
			mol.build()
		except Exception as Ex:
			print mol.atom, mol.basis, m.atoms, m.coords, SensedAtoms, p
			raise Ex
		
		nsaos = len(GauGrid)
		nbas = gto.nao_nr(mol)
		print "nAtoms: ",m.NAtoms()," nsaos: ", nsaos, " nbas ", nbas
		S = mol.intor('cint1e_ovlp_sph',shls_slice=(0,nsaos,0,nsaos))
		Sinv = MatrixPower(S,-1.0)
		SBFs = gto.eval_gto('GTOval_sph',mol._atm,mol._bas,mol._env,samps*1.889725989,comp=1,shls_slice=(0,nsaos))
		print "SBFs.shape", SBFs.shape
		Cs = mol.intor('cint1e_ovlp_sph',shls_slice=(0,nsaos,nsaos,mol.nbas))
		print "Cs.shape", Cs.shape
		Sd = np.sum(np.dot(SBFs,np.dot(Sinv,Cs)),axis=1)
		print "Sd.shape", Sd.shape
		return Sd

	def EmbedAtom(self,m,p,i=-1):
		''' 
			Returns coefficents embedding the environment of atom i, centered at point p.
		'''
		mol = gto.Mole()
		MaxEmbedded = 15
		SensedAtoms = None
		if (i != -1):
			SensedAtoms = [a for a in m.AtomsWithin(10.0,p) if a != i]
		else:
			SensedAtoms = [a for a in m.AtomsWithin(10.0,p)]
		if (len(SensedAtoms)>MaxEmbedded):
			SensedAtoms=SensedAtoms[:MaxEmbedded]
		if (len(SensedAtoms)==0):
			raise Exception("NoAtomsInSensoryRadius")
		mol.atom =''
		if (not self.Spherical):
			tmpgrid = self.SenseGrid + p
			mol.atom = ''.join(["H@0 "+str(tmpgrid[iii,0])+" "+str(tmpgrid[iii,1])+" "+str(tmpgrid[iii,2])+";" for iii in range(len(tmpgrid))])
		else:
			mol.atom = ''.join("C "+str(p[0])+" "+str(p[1])+" "+str(p[2])+";")
		na = 0
		if (not self.Spherical):
			na = self.NSense
		for j in SensedAtoms:
			mol.atom=mol.atom+"H@"+str(m.atoms[j])+" "+str(m.coords[j,0])+" "+str(m.coords[j,1])+" "+str(m.coords[j,2])+(";" if j!=SensedAtoms[-1] else "")
			na=na+1
		if (na%2 == 0):
			mol.spin = 0
		else:
			mol.spin = 1
		if (TOTAL_SENSORY_BASIS == None): 
			raise("missing sensory basis")
		mol.basis = TOTAL_SENSORY_BASIS
		nsaos = 0
		try:
			mol.build()
		except Exception as Ex:
			print mol.atom, mol.basis, m.atoms, m.coords, SensedAtoms, p
			raise Ex
		nsaos = len(self.SenseSinv)
		if (self.Spherical):
			nsaos = mol.atom_nshells(0)
		Cs = mol.intor('cint1e_ovlp_sph',shls_slice=(0,nsaos,nsaos,mol.nbas))
		return np.sum(np.dot(self.SenseSinv,Cs),axis=1)

	def TestSense(self,m,p=[0.0, 0.0, 0.0],ngrid=150,Nm_="Atoms"):
		samps, vol = m.SpanningGrid(ngrid,2)
		# Make the atom densities.
		Ps = self.MolDensity(samps,m,p)
		GridstoRaw(Ps,ngrid,Nm_)
		for i in range(m.NAtoms()):
			Pe = self.TestGridGauEmbedding(samps,m,p,i)
			GridstoRaw(Pe,ngrid,Nm_+str(i))

	def TransformGrid(self,aGrid,ALinearOperator):
		return np.array([np.dot(ALinearOperator,pt) for pt in aGrid])

	def ExpandIsometries(self,inputs,outputs):
		ncase=inputs.shape[0]
		niso=self.NIso()
		if (len(outputs)!=ncase):
			raise Exception("Nonsense inputs")
		ins=[ncase*niso]+(list(inputs.shape)[1:])
		ous=[ncase*niso]+(list(outputs.shape)[1:])
		newins=np.zeros(shape=ins)
		newout=np.zeros(shape=ous)
		#for i in range(ncase):
		#	for j in range(niso):
		#		newins[i*niso+j] = inputs[i][self.IsometryRelabelings[j]]
		#		newout[i*niso+j] = np.dot(self.InvIsometries[j],outputs[i])
		for j in range(niso):
			newins[j*ncase:(j+1)*ncase]=inputs[:,self.IsometryRelabelings[j]]
			newout[j*ncase:(j+1)*ncase]=np.tensordot(outputs,self.InvIsometries[j],axes=[[1],[1]])
		#for i in range(ncase):
		#	for j in range(niso):
		#		newins[i*niso+j] = inputs[i][self.IsometryRelabelings[j]]
		#		newout[i*niso+j] = np.dot(self.InvIsometries[j],outputs[i])
		return newins,newout

	def	NIso(self):
		return len(self.Isometries)

	def TestIsometries(self,m,p=[0.0, 0.0, 0.0],ngrid=150):
		''' Tests that isometries of the basis match up to isometries of the fit. '''
		samps, vol = m.SpanningGrid(ngrid,2)
		# Make the atom densities.
		Ps = self.MolDensity(samps,m,p)
		GridstoRaw(Ps,ngrid,"Atoms")
		Cs = self.EmbedAtom(m,p,-1)
		Pe = self.Rasterize(Cs)
		GridstoRaw(Pe,self.NPts,"Atoms0")
		CoP=self.CenterOfP(Pe)
		print "COM:",CoP
		for i in range(len(self.IsometryRelabelings)):
			PCs = self.Rasterize(Cs[self.IsometryRelabelings[i]])
			print "Transformed",np.dot(self.InvIsometries[i],CoP)
			print "COM:",self.CenterOfP(PCs)
			#GridstoRaw(PCs,self.NPts,"Atoms"+str(i+1))







