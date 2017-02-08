import numpy as np
import random
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
