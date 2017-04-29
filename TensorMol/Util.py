from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle as pickle
import gc, random, os, sys, re, atexit
import math, time, itertools, warnings
from math import pi as Pi
import scipy.special
from scipy.weave import inline
#from collections import defaultdict
#from collections import Counter
from TensorMol.TMParams import *

warnings.simplefilter(action = "ignore", category = FutureWarning)
#
# GLOBALS
#	Any global variables of the code must be put here, and must be in all caps.
#	Global variables are almost never acceptable except in these few cases

# PARAMETERS
#  TODO: Migrate these to PARAMS
PARAMS = TMParams()
LOGGER = TMLogger(PARAMS["results_dir"])
MAX_ATOMIC_NUMBER = 10
MBE_ORDER = 2
# Derived Quantities and useful things.
N_CORES = 1
HAS_PYSCF = False
HAS_EMB = False
HAS_TF = False
GRIDS = None
HAS_GRIDS=True
# KUN PLEASE MAKE THESE ALL CAPS FOLLOWING OUR CONVENTION.
ELEHEATFORM = {1:-0.497912, 6:-37.844411, 7:-54.581501, 8:-75.062219, 9:-99.716370}     # ref: https://figshare.com/articles/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643
atoi = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'Si':23,'V':24,'Cr':25,'Br':35, 'Cs':55, 'Pb':82}
atoc = {1: 40, 6: 100, 7: 150, 8: 200, 9:240}
bond_length_thresh = {"HH": 1.5, "HC": 1.5, "HN": 1.5, "HO": 1.5, "CC":2.0, "CN":2.0, "CO": 2.0, "NN":2.0, "NO":2.0, "OO":2.0 }
atomic_radius = {1:53.0, 2:31.0, 3:167.0, 4:112.0, 5:87.0, 6:67.0, 7:56.0, 8:48.0, 9:42.0, 10:38.0, 11:190.0, 12:145.0, 13:118.0, 14:111.0, 15:98.0, 16:88.0, 17:79.0, 18:71.0} # units in pm, ref: https://en.wikipedia.org/wiki/Atomic_radius
atomic_radius_2 = {1:25.0, 3:145.0, 4:105.0, 5:85.0, 6:70.0, 7:65.0, 8:60.0, 9:50.0, 11:180.0, 12:150.0, 13:125.0, 14:110.0, 15:100.0, 16:100.0, 17:100.0} # units in pm, ref: https://en.wikipedia.org/wiki/Atomic_radius
atomic_raidus_cho = {1:0.328, 6:0.754, 8:0.630} # roughly statisfy mp2 cc-pvtz equilibrium carbohydrate bonds.
BOHRPERA = 1.889725989
KCALPERJOULE = 4183.9953
AVOCONST = 6.02*np.power(10.0,23.0)
# These are in Kilograms.
ATOMICMASSES = 0.000999977*np.array([1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 15.9994, 18.9984032, 20.1791, 22.98976928, 24.3050, 26.9815386, 28.0855, 30.973762, 32.065, 35.453, 39.948, 39.0983, 40.078, 44.955912, 47.867, 50.9415,51.9961,54.938045,55.845,58.933195,58.6934,63.546,65.38,69.723,72.63,74.92160,78.96,79.904,83.798,85.4678,87.62,88.90585,91.224,92.90638,95.96,98.,101.07,102.90550,106.42,107.8682,112.411,114.818,118.710,121.760,127.60,126.90447,131.293,132.9054519,137.327,138.90547,140.116,140.90765,144.242,145.,150.36,151.964,157.25,158.92535,162.500,164.93032,167.259,168.93421,173.054,174.9668,178.49,180.94788,183.84,186.207,190.23,192.217,195.084,196.966569,200.59,204.3833,207.2,208.98040,209.,210.,222.,223.,226.,227.,232.03806,231.03586,238.02891,237.,244.,243.,247.,247.,251.,252.,257.])
Qchem_RIMP2_Block = "$rem\n   jobtype   sp\n   method   rimp2\n   MAX_SCF_CYCLES  200\n   basis   cc-pvtz\n   aux_basis rimp2-cc-pvtz\n   symmetry   false\n   INCFOCK 0\n   thresh 12\n   SCF_CONVERGENCE 12\n$end\n"
#
# -- begin Environment set up.
#

TMBanner()
LOGGER.info("Searching for Installed Optional Packages...")
try:
	from pyscf import scf
	from pyscf import gto
	from pyscf import dft
	from pyscf import mp
	HAS_PYSCF = True
	LOGGER.debug("Pyscf has been found")
except Exception as Ex:
	LOGGER.info("Pyscf is not installed -- no ab-initio sampling")
	pass

try:
	import MolEmb
	HAS_EMB = True
	LOGGER.debug("MolEmb has been found, Orthogonalizing Radial Basis.")
	S = MolEmb.Overlap_SH(PARAMS)
	from TensorMol.LinearOperations import MatrixPower
	SOrth = MatrixPower(S,-1./2)
	PARAMS["GauSHSm12"] = SOrth
	S_Rad = MolEmb.Overlap_RBF(PARAMS)
	S_RadOrth = MatrixPower(S_Rad,-1./2)
	PARAMS["SRBF"] = S_RadOrth
	# THIS SHOULD BE IMPLEMENTED TOO.
	#PARAMS["GauInvSm12"] = MatrixPower(S,-1./2)
except Exception as Ex:
	print("MolEmb is not installed. Please cd C_API; sudo python setup.py install",Ex)
	pass

try:
	import tensorflow as tf
	LOGGER.debug("Tensorflow version "+tf.__version__+" has been found")
	HAS_TF = True
except:
	LOGGER.info("Tensorflow not Installed, very limited functionality")
	pass

try:
	import multiprocessing
	N_CORES=multiprocessing.cpu_count()
	LOGGER.debug("Found "+str(N_CORES)+" CPUs to thread over... ")
except:
	LOGGER.info("Only a single CPU, :( did you lose a war?")
	pass
LOGGER.debug("TensorMol ready...")

LOGGER.debug("TMPARAMS----------")
LOGGER.debug(PARAMS)
LOGGER.debug("TMPARAMS~~~~~~~~~~")

TOTAL_SENSORY_BASIS=None
SENSORY_BASIS=None
if (HAS_PYSCF and HAS_GRIDS):
	from TensorMol.Grids import *
	GRIDS = Grids()
	GRIDS.Populate()
print("--------------------------")
#
# -- end Environment set up.
#

@atexit.register
def exitTensorMol():
	LOGGER.info("~ Adios Homeshake ~")

def complement(a,b):
	return [i for i in a if b.count(i)==0]

def scitodeci(sci):
	tmp=re.search(r'(\d+\.?\d+)\*\^(-?\d+)',sci)
	return float(tmp.group(1))*pow(10,float(tmp.group(2)))

def AtomicNumber(Symb):
	try:
		return atoi[Symb]
	except Exception as Ex:
		raise Exception("Unknown Atom")
	return 0

def AtomicSymbol(number):
	try:
		return atoi.keys()[atoi.values().index(number)]
	except Exception as Ex:
		raise Exception("Unknown Atom")
	return 0

def SignStep(S):
	if (S<0.5):
		return -1.0
	else:
		return 1.0

# Choose random samples near point...
def PointsNear(point,NPts,Dist):
	disps=Dist*0.2*np.abs(np.log(np.random.rand(NPts,3)))
	signs=signstep(np.random.random((NPts,3)))
	return (disps*signs)+point

def SamplingFunc_v2(S, maxdisp):    ## with sampling function f(x)=M/(x+1)^2+N; f(0)=maxdisp,f(maxdisp)=0; when maxdisp =5.0, 38 % lie in (0, 0.1)
	M = -((-1 - 2*maxdisp - maxdisp*maxdisp)/(2 + maxdisp))
	N = ((-1 - 2*maxdisp - maxdisp*maxdisp)/(2 + maxdisp)) + maxdisp
	return M/(S+1.0)**2 + N

def LtoS(l):
	s=""
	for i in l:
		s+=str(i)+" "
	return s

def ErfSoftCut(dist, width, x):
	return (1-scipy.special.erf(1.0/width*(x-dist)))/2.0

def CosSoftCut(dist, x):
	if x > dist:
		return 0
	else:
		return 0.5*(math.cos(math.pi*x/dist)+1.0)
	return

def nCr(n, r):
	f = math.factorial
	return int(f(n)/f(r)/f(n-r))

def Submit_Script_Lines(order=str(3), sub_order =str(1), index=str(1), mincase = str(0), maxcase = str(1000), name = "MBE", ncore = str(4), queue="long"):
	lines = "#!/bin/csh\n"+"# Submit a job for 8  processors\n"+"#$ -N "+name+"\n#$ -t "+mincase+"-"+maxcase+":1\n"+"#$ -pe smp "+ncore+"\n"+"#$ -r n\n"+"#$ -q "+queue+"\n\n\n"
	lines += "module load gcc/5.2.0\nsetenv  QC /afs/crc.nd.edu/group/parkhill/qchem85\nsetenv  QCAUX /afs/crc.nd.edu/group/parkhill/QCAUX_1022\nsetenv  QCPLATFORM LINUX_Ix86\n\n\n"
	lines += "/afs/crc.nd.edu/group/parkhill/qchem85/bin/qchem  -nt "+ncore+"   "+str(order)+"/"+"${SGE_TASK_ID}/"+sub_order+"/"+index+".in  "+str(order)+"/"+"${SGE_TASK_ID}/"+sub_order+"/"+index+".out\n\nrm MBE*.o*"
	return lines

def Binominal_Combination(indis=[0,1,2], group=3):
	if (group==1):
		index=list(itertools.permutations(indis))
		new_index =[]
		for i in range (0, len(index)):
			new_index.append(list(index[i]))
		return new_index
	else:
		index=list(itertools.permutations(indis))
		new_index=[]
		for sub_list in Binominal_Combination(indis, group-1):
			for sub_index in index:
				new_index.append(list(sub_list)+list(sub_index))
		return new_index

def NormMatrices(mat1, mat2):
	#Only faster due to subtraction of matrices being done in C_API, possibly remove
	assert mat1.shape == mat2.shape, "Shape of matrices must match to calculate the norm"
	return MolEmb.Norm_Matrices(mat1, mat2)

def String_To_Atoms(s=""):
	l = list(s)
	atom_l = []
	tmp = ""
	for i, c in enumerate(l):
		if  ord('A') <= ord(c) <= ord('Z'):
			tmp=c
		else:
			tmp += c
		if i==len(l)-1:
			atom_l.append(tmp)
		elif ord('A') <= ord(l[i+1]) <= ord('Z'):
			atom_l.append(tmp)
		else:
			continue
	return atom_l

def iter_product(args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield list(prod)

def Subset(A, B): # check whether B is subset of A
	checked_index = []
	found = 0
	for value in B:
		for i in range (0, len(A)):
			if value==A[i] and i not in checked_index:
				checked_index.append(i)
				found += 1
				break
	if found == len(B):
		return True
	else:
		return False

def Setdiff(A, B): # return the element of A that not included in B
	diff = []
	checked_index = []
	for value in A:
		found = 0
		for i in range (0, len(B)):
			if value == B[i] and i not in checked_index:
				found = 1
				checked_index.append(i)
				break
		if found == 0:
			diff.append(value)
	return diff

def Pair_In_List(l, pairs): # check whether l contain pair
	for pair in pairs:
		if Subset(l, pair):
			return True
	return False


signstep = np.vectorize(SignStep)
samplingfunc_v2 = np.vectorize(SamplingFunc_v2)
