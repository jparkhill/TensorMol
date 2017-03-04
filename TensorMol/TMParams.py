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
		self["MAX_ATOMIC_NUMBER"] = 10
		# Parameters of MolEmb
		self["RBFS"] = np.array([[0.1, 0.156787], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [1.3, 1.3], [2.2,
			2.4], [4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]])
		#self["RBFS"] = np.array([[0.99438638, 0.41676905], [0.59320762, 0.26405542], [0.44094295, 0.4394062],
  		#						[0.61691555, 0.61625545], [1.20242598, 2.17935545]])
		self["ERBFS"] = np.zeros((self["MAX_ATOMIC_NUMBER"],self["RBFS"].shape[0],self["RBFS"].shape[0])) # element specific version.
		self["SRBF"] = np.zeros((self["RBFS"].shape[0],self["RBFS"].shape[0]))
		self["ORBFS"] = np.zeros((self["RBFS"].shape[0],self["RBFS"].shape[0]))
		self["ANES"] = np.array([1.0, 1.0, 1.0, 1.0]) #Atomic Number Encoding, only for C, H, N, and O for now
		self["SH_LMAX"]=2
		self["SH_NRAD"]=5
		self["SH_ORTH"]=1
		self["SH_MAXNR"]=self["RBFS"].shape[0]
		# SET GENERATION parameters
		self["MBE_ORDER"] = 2
		self["KAYBEETEE"] = 0.000950048 # At 300K
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
		self["RandomizeData"] = False
		self["batch_size"] = 8000
		self["MxTimePerElement"] = 36000
		self["MxMemPerElement"]=16000 # Max Array for an element in MB
		self["ChopTo"] = None
		self["RotAvOutputs"] = 1 # Rotational averaging of force outputs.
		self["OctahedralAveraging"] = 0 # Octahedrally Average Outputs
		# Training Parameters
		self["learning_rate"] = 0.001
		self["momentum"] = 0.9
		self["max_steps"] = 500
		self["test_freq"] = 50
		self["hidden1"] = 512
		self["hidden2"] = 512
		self["hidden3"] = 512
		#paths
		self["results_dir"] = "./results/"
		self["dens_dir"] = "./densities/"

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
    #Delete Jupyter notebook root logger handler
    logger = logging.getLogger()
    logger.handlers = []
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
    print "Built Logger... "
    return tore
