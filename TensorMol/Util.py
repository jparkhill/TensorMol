from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gc, random, os, sys, re, atexit
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle
import random, math, time, itertools, warnings
from math import pi as Pi
import scipy.special
#from collections import defaultdict
#from collections import Counter
from TensorMol.TMParams import *
TMBanner()
from TensorMol.PhysicalData import *
warnings.simplefilter(action = "ignore", category = FutureWarning)
#
# GLOBALS
#	Any global variables of the code must be put here, and must be in all caps.
#	Global variables are almost never acceptable except in these few cases

# PARAMETERS
#  TODO: Migrate these to PARAMS
PARAMS = TMParams()
LOGGER = TMLogger(PARAMS["log_dir"])
MAX_ATOMIC_NUMBER = 10
# Derived Quantities and useful things.
N_CORES = 1
HAS_PYSCF = False
HAS_EMB = False
HAS_TF = False
GRIDS = None
HAS_GRIDS=False
Qchem_RIMP2_Block = "$rem\n   jobtype   sp\n   method   rimp2\n   MAX_SCF_CYCLES  200\n   basis   cc-pvtz\n   aux_basis rimp2-cc-pvtz\n   symmetry   false\n   INCFOCK 0\n   thresh 12\n   SCF_CONVERGENCE 12\n$end\n"
#
# -- begin Environment set up.
#

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

#
# All the garbage below here needs to be removed, organized
# and the perpretrators genetalia scattered to the four winds.
#

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

def LtoS(l):
	s=""
	for i in l:
		s+=str(i)+" "
	return s

def ErfSoftCut(dist, width, x):
	return (1-scipy.special.erf(1.0/width*(x-dist)))/2.0

def nCr(n, r):
	f = math.factorial
	return int(f(n)/f(r)/f(n-r))

signstep = np.vectorize(SignStep)
