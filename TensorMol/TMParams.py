"""
PARAMETER CONVENTION:
- It's okay to have parameters which you pass to functions used to perform a test if you don't change them often.
- It's NOT okay to put default parameters in __init__() and change them all the time.
- These params should be added to a logfile of results so that we can systematically see how our approximations are doing.
"""
import logging, time, os
from math import pi as Pi
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
		self["AN1_r_Rc"] = 4.6
		self["AN1_a_Rc"] = 3.1
		self["AN1_eta"] = 4.0
		self["AN1_zeta"] = 8.0
		self["AN1_num_r_Rs"] = 32
		self["AN1_num_a_Rs"] = 8
		self["AN1_num_a_As"] = 8
		self["AN1_r_Rs"] = np.array([ self["AN1_r_Rc"]*i/self["AN1_num_r_Rs"] for i in range (0, self["AN1_num_r_Rs"])])
		self["AN1_a_Rs"] = np.array([ self["AN1_a_Rc"]*i/self["AN1_num_a_Rs"] for i in range (0, self["AN1_num_a_Rs"])])
		self["AN1_a_As"] = np.array([ 2.0*Pi*i/self["AN1_num_a_As"] for i in range (0, self["AN1_num_a_As"])])
		# SET GENERATION parameters
		self["RotateSet"] = 0
		self["TransformSet"] = 1
		self["NModePts"] = 10
		self["NDistorts"] = 5
		self["GoK"] = 0.05
		self["dig_ngrid"] = 20
		self["dig_SamplingType"]="Smooth"
		self["BlurRadius"] = 0.05
		self["Classify"] = False # Whether to use a classifier histogram scheme rather than normal output.
		# MBE PARAMS
		self["MBE_ORDER"] = 4
		# DATA usage parameters
		self["InNormRoutine"] = None
		self["OutNormRoutine"] = None
		self["RandomizeData"] = False
		self["batch_size"] = 8000
		self["MxTimePerElement"] = 36000
		self["MxMemPerElement"]=16000 # Max Array for an element in MB
		self["ChopTo"] = None
		self["RotAvOutputs"] = 1 # Rotational averaging of force outputs.
		self["OctahedralAveraging"] = 0 # Octahedrally Average Outputs
		# Opt Parameters
		self["OptMaxCycles"]=1000
		self["OptThresh"]=0.0004
		self["OptMaxStep"]=0.1
		self["OptStepSize"] = 0.004
		self["OptMomentum"] = 0.0
		self["OptMomentumDecay"] = 0.8
		self["OptPrintLvl"] = 1
		self["OptMaxBFGS"] = 7
		self["NebNumBeads"] = 10
		self["NebK"] = 0.01
		self["NebMaxBFGS"] = 12
		self["DiisSize"] = 20
		# MD Parameters
		self["MDMaxStep"] = 20000
		self["MDdt"] = 0.2 # In fs.
		self["MDTemp"] = 300.0
		self["MDV0"] = "Random"
		self["MDThermostat"] = None # None, "Rescaling", "Nose", "NoseHooverChain"
		self["MDLogTrajectory"] = True
		self["MDLogVelocity"] = False
		# MD applied pulse parameters
		self["MDFieldVec"] = np.array([1.0,0.0,0.0])
		self["MDFieldAmp"] = 0.001
		self["MDFieldFreq"] = 1.0/1.2
		self["MDFieldTau"] = 1.2
		self["MDFieldT0"] = 3.0
		# Training Parameters
		self["learning_rate"] = 0.001
		self["momentum"] = 0.9
		self["max_steps"] = 1500
		self["test_freq"] = 10
		self["hidden1"] = 512
		self["hidden2"] = 512
		self["hidden3"] = 512
		# parameters of electrostatic embedding
		self["EEOn"] = True # Whether to calculate/read in the required data at all...
		self["EEVdw"] = True # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EEOrder"] = 2 # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EEdr"] = 1.0 # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EECutoff"] = 5.0 # switch between 0 and 1/r occurs at Angstroms.
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
