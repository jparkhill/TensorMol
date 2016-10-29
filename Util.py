#
# Catch-all for useful little snippets that don't need organizing. 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os,sys,pickle,re
import math
import time
from math import pi as Pi
import scipy.special
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

#
# GLOBALS
#	Any global variables of the code must be put here, and must be in all caps.
#	Global variables are almost never acceptable except in these few cases
#

MAX_ATOMIC_NUMBER = 10
HAS_PYSCF = False
HAS_EMB = False
HAS_TF = False
atoi = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'Si':23,'V':24,'Cr':25}
atoc = {1: 40, 6: 100, 7: 150, 8: 200, 9:240}
KAYBEETEE = 0.000950048 # At 300K
BOHRPERA = 1.889725989
GRIDS = None

#
# -- begin Environment set up.
#
print("--------------------------\n")
print("         /\\______________")
print("      __/  \\   \\_________")
print("    _/  \\   \\            ")
print("___/\_TensorMol_0.0______")
print("   \\_/\\______  __________")
print("     \\/      \\/          ")
print("      \\______/\\__________\n")
print("--------------------------")
print("By using this software you accept the terms of the GNU public license in ")
print("COPYING, and agree to attribute the use of this software in publications as: \n")
print("K.Yao, J. Herr, J. Parkhill. TensorMol0.0 (2016)")
print("Depending on Usage, please also acknowledge, TensorFlow, PySCF, or your training sets.")
print("--------------------------")
print("Searching for Installed Optional Packages...")
try:
	from pyscf import scf
	from pyscf import gto
	from pyscf import dft
	HAS_PYSCF = True
	print("Pyscf has been found")
except Exception as Ex:
	print("Pyscf is not installed -- no ab-initio sampling",Ex)
	pass

try:
	import MolEmb
	HAS_EMB = True
	print("MolEmb has been found")
except:
	print("MolEmb is not installed. Please cd C_API; sudo python setup.py install")
	pass

try:
	import tensorflow as tf 
	HAS_TF = True
	print("Tensorflow has been found")
except:
	print("Tensorflow not Installed, very limited functionality")
	pass
print("TensorMol ready...")

TOTAL_SENSORY_BASIS=None
SENSORY_BASIS=None
if (HAS_PYSCF):
	from Grids import *
	GRIDS = Grids()
	GRIDS.Populate()
print("--------------------------")
#
# -- end Environment set up.
#

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

def ErfSoftCut(dist, width, x):
	return (1-scipy.special.erf(1.0/width*(x-dist)))/2.0	

signstep = np.vectorize(SignStep)
samplingfunc_v2 = np.vectorize(SamplingFunc_v2)
