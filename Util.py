#
# Catch-all for useful little snippets that don't need organizing. 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import random
import numpy as np
import os,sys,pickle,re
import math
import time
from math import pi as Pi
import scipy.special
import itertools
import warnings
from scipy.weave import inline
from collections import defaultdict
from collections import Counter
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
MBE_ORDER = 2
Qchem_RIMP2_Block = "$rem\n   jobtype   sp\n   method   rimp2\n   MAX_SCF_CYCLES  200\n   basis   cc-pvtz\n   aux_basis rimp2-cc-pvtz\n   symmetry   false\n   INCFOCK 0\n   thresh 12\n   SCF_CONVERGENCE 12\n$end\n"

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
	from pyscf import mp
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
	

signstep = np.vectorize(SignStep)
samplingfunc_v2 = np.vectorize(SamplingFunc_v2)
