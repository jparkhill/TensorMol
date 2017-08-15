"""
Linear algebra operations and coordinate transformations.
"""
import numpy as np
import random
import math
from math import pi as Pi

def PseudoInverse(mat_):
	U, s, V = np.linalg.svd(mat_) #M=u * np.diag(s) * v,
	for i in range(len(s)):
		if (s[i]!=0.0 and abs(s[i])>0.0000001):
			s[i]=1.0/s[i]
		else:
			s[i] = 0.0
	return np.dot(np.dot(U,np.diag(s)),V)

# Simple vectorized coordinate transformations.
def SphereToCart(arg_):
	r = arg_[0]
	theta = arg_[1]
	phi = arg_[2]
	x = r*np.sin(theta)*np.cos(phi)
	y = r*np.sin(theta)*np.sin(phi)
	z = r*np.cos(theta)
	return np.array([x,y,z])

def CartToSphere(arg_):
	x = arg_[0]
	y = arg_[1]
	z = arg_[2]
	r = np.sqrt(x*x+y*y+z*z)
	theta = np.arccos(z/r)
	phi = np.arctan2(y,x)
	return np.array([r,theta,phi])

def SchmidtStep(xs,y_):
	"""
	return y - projection of y onto all xs normalized.
	Args:
		xs: orthonormal row vectors
		y: another row vector.
	"""
	y = y_.copy()
	for i in range(xs.shape[0]):
		y -= np.dot(xs[i],y)*xs[i]/np.dot(xs[i],xs[i])
	ntmp = np.dot(y,y)
	return ntmp, y

def Normalize(x_):
	return x_/np.sqrt(np.dot(x_,x_))

def PairOrthogonalize(x,y):
	"""
	Does a Graham-Schmidt
	The assumption here is that y is square and full-rank
	and x has smaller and full rank. Returns rank(y)-rank(x) row vectors
	which are all normalized and orthogonal to x.
	"""
	ny = y.shape[0]
	nx = x.shape[0]
	dim = y.shape[1]
	if (x.shape[1] != y.shape[1]):
		raise Exception("Dim mismatch")
	# Orthogonalize
	Ox = np.zeros((nx+ny,dim))
	Ox[0] = Normalize(x[0])
	Orank = 1
	for i in range(1,nx):
		ntmp, tmp = SchmidtStep(Ox[:Orank],x[i])
		if (ntmp > pow(10.0,-12.0)):
			Ox[Orank] = tmp/np.sqrt(ntmp)
			Orank += 1
	LastXVec = Orank
	for i in range(ny):
		ntmp, tmp = SchmidtStep(Ox[:Orank],y[i])
		if (ntmp > pow(10.0,-12.0)):
			Ox[Orank] = tmp/np.sqrt(ntmp)
			Orank += 1
	return Ox[LastXVec:Orank]#Ox[:Orank]#

def SphereToCartV(arg_):
	return np.array(map(SphereToCart,arg_))

def CartToSphereV(arg_):
	return np.array(map(CartToSphere,arg_))

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

def RotationMatrix_v2(randnums=None, deflection=1.0):
	"""
	Creates a uniformly random rotation matrix
	Args:
		randnums: theta, phi, and z for rotation, if None then chosen uniformly random
		deflection: magnitude of rotation, 0 is no rotation, 1 is completely random rotation, inbetween are perturbations
	"""
	if randnums is None:
		randnums = np.random.uniform(size=(3,))
	theta, phi, z = randnums[0]*2.0*deflection*np.pi, randnums[1]*2.0*np.pi, randnums[2]*2.0*deflection
	r = np.sqrt(z)
	v = np.array([np.sin(phi)*r, np.cos(phi)*r, np.sqrt(2.0-z)])
	R = np.array(((np.cos(theta),np.sin(theta),0.),(-np.sin(theta),np.cos(theta),0.),(0.,0.,1.)))
	M = (np.outer(v,v) - np.eye(3)).dot(R)
	return M

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
	return Ident+FaceRotations+FaceRotations2+FaceRotations3+CornerRotations+CornerRotations2+EdgeRotations+EdgeReflections
