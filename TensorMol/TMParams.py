"""
PARAMETER CONVENTION:
- It's okay to have parameters which you pass to functions used to perform a test if you don't change them often.
- It's NOT okay to put default parameters in __init__() and change them all the time.
- These params should be added to a logfile of results so that we can systematically see how our approximations are doing.
"""
import logging, time, os
import numpy as np

class TMParams(dict):
	def __init__(self, *args, **kwargs ):
		myparam = kwargs.pop('myparam', '')
		dict.__init__(self, *args, **kwargs )
		self["GIT_REVISION"] = os.popen("git rev-parse --short HEAD").read()
		self["check_level"] = 1 # whether to test the consistency of several things...
		# Parameters of MolEmb
		self["RBFS"] = np.array([[0.1, 0.156787], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [1.3, 1.3], [2.2,
			2.4], [4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]])
		self["SRBF"] = np.zeros((self["RBFS"].shape[0],self["RBFS"].shape[0]))
		self["ORBFS"] = np.zeros((self["RBFS"].shape[0],self["RBFS"].shape[0]))
		self["SH_LMAX"]=6
		self["SH_NRAD"]=10
		self["SH_ORTH"]=1
		self["SH_MAXNR"]=self["RBFS"].shape[0]
		# SET GENERATION parameters
		self["MAX_ATOMIC_NUMBER"] = 10
		self["MBE_ORDER"] = 2
		self["RotateSet"] = 0
		self["TransformSet"] = 1
		self["NModePts"] = 10
		self["NDistorts"] = 5
		self["GoK"] = 0.05
		self["dig_ngrid"] = 20
		self["dig_SamplingType"]="Smooth"
		self["BlurRadius"] = 0.05
		self["Classify"] = False # Whether to use a classifier histogram scheme rather than normal output.
		# DATA usage parameters
		self["InNormRoutine"] = None
		self["OutNormRoutine"] = "MeanStd" 
		self["batch_size"] = 8000
		self["MxTimePerElement"] = 36000
		self["MxMemPerElement"]=16000 # Max Array for an element in MB
		self["ChopTo"] = None
		self["results_dir"] = "./results/"
		self["RotAvOutputs"] = 0 # Rotational averaging of force outputs.
		self["OctahedralAveraging"] = 1 # Octahedrally Average Outputs
		# Training Parameters
		self["learning_rate"] = 0.001
		self["momentum"] = 0.9
		self["max_steps"] = 1000
		self["test_freq"] = 5 
		self["hidden1"] = 512
		self["hidden2"] = 512
		self["hidden3"] = 512
		# Garbage we're putting here for now.
		self["Qchem_RIMP2_Block"] = "$rem\n   jobtype   sp\n   method   rimp2\n   MAX_SCF_CYCLES  200\n   basis   cc-pvtz\n   aux_basis rimp2-cc-pvtz\n   symmetry   false\n   INCFOCK 0\n   thresh 12\n   SCF_CONVERGENCE 12\n$end\n"

def __str__(self):
	tore=""
	for k in self.keys():
		tore = tore+k+":"+str(self[k])+"\n"
	return tore

def TMBanner():
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
	print("K.Yao, J. E. Herr, J. Parkhill. TensorMol 0.0 (2016)")
	print("Depending on Usage, please also acknowledge, TensorFlow, PySCF, or your training sets.")
	print("--------------------------")

def TMLogger(path_):
	tore=logging.getLogger('TensorMol')
	tore.setLevel(logging.DEBUG)
	fh = logging.FileHandler(filename=path_+time.ctime()+'.log')
	fh.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	fformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	pformatter = logging.Formatter('%(message)s')
	fh.setFormatter(fformatter)
	ch.setFormatter(pformatter)
	tore.addHandler(fh)
	tore.addHandler(ch)
	return tore
