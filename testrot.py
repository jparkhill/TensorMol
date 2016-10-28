from Util import * 
from Mol import * 
from pyscf import gto

ch3xyz = """
7

C          0.00000        0.00000        0.00000
H          0.52881        0.16105        0.93595
H          0.20514        0.82402       -0.67859
H          0.73455       -0.93140       -0.44955
H         -1.06851       -0.05371        0.19215
O         -1.06851       -0.05371        3.19215
N         -2.06851       -3.05371        3.19215
"""

#mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
#coords = np.random.random((100,3))  # 100 random points
#ao_value = gto.eval_gto("GTOval_sph", mol._atm, mol._bas, mol._env, coords)
#print(ao_value.shape)

# place our embedding at the central C atom. 
# rotate the molecule and test the resulting embedding. 
m=Mol()
m.FromXYZString(ch3xyz)
#print m.coords
S=m.OverlapAtPoint([0.0,0.0,0.0],0)
print(S[0])
print S
w,v=np.linalg.eig(S)
idx = w.argsort()[::-1]
print "Spectrum", w[idx]

nat=m.NAtoms()
nbas=len(S)
numsense = nbas - (nat-1) #Number of sensory AOs
numenv =(nat-1)
print "NBas: ", len(S)
print "NEnv: ", len(S)-numsense
C = MatrixPower(S[:numsense,:numsense],-0.5)
print "Symmetric Orthogonalization?", np.dot(np.dot(C,S[:numsense,:numsense]),C)

# The first 17 basis functions should be the "sense atom."
# The following should be the "Environment atoms."

#Construct a uniform grid to represent the environmental density, and compare with projection onto the sense atoms.
samps, vol = m.SpanningGrid(100,8)
print "Grid Edges: ", np.max(samps[:,0]),np.max(samps[:,1]),np.max(samps[:,2])
print "Grid Edges: ", np.min(samps[:,0]),np.min(samps[:,1]),np.min(samps[:,2])
dx=(np.max(samps[:,0])-np.min(samps[:,0]))/100*1.8
dy=(np.max(samps[:,1])-np.min(samps[:,1]))/100*1.8
dz=(np.max(samps[:,2])-np.min(samps[:,2]))/100*1.8
print "dx,dy,dz (au)", dx,dy,dz
print "Grid Volume (au): ", dx*dy*dz*np.power(100.0,3)
print samps.shape #, samps[0:101]
# Dump the AO's onto the grid.
grids=m.BasisSample(0,samps) # samplesXao
ogrids = np.copy(grids[:,:numsense])
ogrids *= 0.0
dgrids = grids*grids
envgrid = np.copy(grids[:,0])
envgrid *= 0.0
print "basis sample shape: ", dgrids.shape
if (1):
	for i in range(nbas):
		gridt = np.copy(dgrids[:,i])
		print "Normalization of grid i.", np.sum(gridt)*dx*dy*dz, np.max(gridt), np.min(gridt) # Yeah. that's normalized.
		if (i>=numsense):
			envgrid += gridt
			#gridt *= 254.0/np.max(gridt)  #Prints all the AOs.
			#m.GridstoRaw(gridt,100,"O"+str(i))

Sn=np.zeros(S.shape)
for i in range(numsense):
	for j in range(numsense):
		Sn[i,j] = np.sum(grids[:,i]*grids[:,j]*dx*dy*dz)

Cn = MatrixPower(Sn,-0.5)
print "SOnAOs ", Cn

# make the symmetrically orthogonalized Sensory AO's
# off the numerical overlaps
for i in range(numsense):
	ogrids[:,i] *= 0.0
	for j in range(numsense):
		ogrids[:,i] += Cn.T[i,j]*grids[:,j]
	tmp = ogrids[:,i]*ogrids[:,i]
	print "Normalization of SAO i.", np.sum(tmp)*dx*dy*dz, np.max(tmp), np.min(tmp) # Yeah. that's normalized.
	#tmp *= 254.0/np.max(tmp)
	#m.GridstoRaw(tmp,100,"SAO"+str(i))

# works perfectly if the overlap is numerical
#for i in range(numsense):
#	for j in range(numsense):
#		print "Numerical overlap: ",i,j,np.sum(ogrids[:,i]*ogrids[:,j])*dx*dy*dz

CjAO = np.sum(S[:numsense,numsense:],axis=1) #np.zeros((numsense,numenv))
Cj = np.zeros(numsense)

if(1):
	print "Normalization of envgrid", np.sum(envgrid)*dx*dy*dz, np.max(envgrid), np.min(envgrid)
	envgrid *= 254.0/np.max(envgrid)
	m.GridstoRaw(envgrid,100,"env")

	# Compare that with the basis set expansion of the density given by the sensory atoms.
	for i in range(numsense):
		for j in range(numsense):
			Cj[i] += Cn.T[i,j]*CjAO[j]
	
	expd = np.copy(grids[:,0])
	expd *= 0.0
	tmp = np.copy(grids[:,0])
	tmp *= 0.0
	for i in range(0,numsense):
		tmp +=Cj[i]*ogrids[:,i]
	expd = np.power(tmp,2.0)
	print "Normalization of Exp.", np.sum(expd)*dx*dy*dz, np.max(expd), np.min(expd) # Yeah. that's normalized.
	expd *= 254.0/np.max(expd)
	m.GridstoRaw(expd,100,"exp")

exit(0)

# Bi-spectrum?
c=np.sum(S,axis=0)
Bm=np.outer(c,c)
w,v=np.linalg.eig(Bm)
idx = w.argsort()[::-1]
print "BiSpectrum", w[idx]


m.Rotate([0,0,1.0],6.28*0.75)
#print m.coords
m.FromXYZString(ch3xyz)
S=m.OverlapAtPoint([0.0,0.0,0.0],1)
w,v=np.linalg.eig(S)
idx = w.argsort()[::-1]
print "Spectrum", w[idx]

c=np.sum(S,axis=0)
Bm=np.outer(c,c)
print Bm
w,v=np.linalg.eig(Bm)
idx = w.argsort()[::-1]
print "BiSpectrum", w[idx]
