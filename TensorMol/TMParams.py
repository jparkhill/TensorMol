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
		self["RBFS"] = np.array([[0.26649229, 0.86693935], [0.48411375, 0.72556564], [0.72194098, 0.09265219],[0.95801627, 0.10751769], [0.99667822, 1.20433031], [2.15205854, 2.34423998],[4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]])
		self["ANES"] = np.array([0.52392538, 1., 1., 1., 1., 2.50880292, 2.76668503, 1.95700163])
		self["SRBF"] = np.zeros((self["RBFS"].shape[0],self["RBFS"].shape[0]))
		#self["ANES"] = np.array([0.50068655, 1., 1., 1., 1., 1.12237954, 0.90361766, 1.06592739])
		#self["ANES"] = np.array([1., 1., 1., 1., 1., 4., 3., 2.])
		self["SH_LMAX"]=4
		self["SH_NRAD"]=10
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
		self["RandomizeData"] = True
		self["batch_size"] = 8000
		self["MxTimePerElement"] = 36000
		self["MxMemPerElement"]=16000 # Max Array for an element in MB
		self["ChopTo"] = None
		self["RotAvOutputs"] = 20 # Rotational averaging of force outputs.
		self["OctahedralAveraging"] = 0 # Octahedrally Average Outputs
		# Opt Parameters
		self["OptMaxCycles"]=400
		self["OptThresh"]=0.0002
		self["OptMaxStep"]=0.1
		self["OptStepSize"] = 0.002
		self["OptMomentum"] = 0.0
		self["OptMomentumDecay"] = 0.8
		self["OptPrintLvl"] = 1
		self["NebNumBeads"] = 25
		self["NebK"] = 2.0
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
	print("--------------------------")
	print("    "+unichr(0x1350)+unichr(0x2107)+unichr(0x2115)+unichr(0x405)+unichr(0x29be)+unichr(0x2c64)+'-'+unichr(0x164f)+unichr(0x29be)+unichr(0x2112)+"  0.0")
	print("--------------------------")
	print("By using this software you accept the terms of the GNU public license in ")
	print("COPYING, and agree to attribute the use of this software in publications as: \n")
	print("K.Yao, J. E. Herr, J. Parkhill. TensorMol 0.0 (2016)")
	print("--------------------------")

def TMLogger(path_):
	#Delete Jupyter notebook root logger handler
	logger = logging.getLogger()
	logger.handlers = []
	tore=logging.getLogger('TensorMol')
	tore.setLevel(logging.DEBUG)
	# Check path and make if it doesn't exist...
	if not os.path.exists(path_):
		os.makedirs(path_)
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
