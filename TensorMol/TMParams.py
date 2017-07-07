import logging, time, os
from math import pi as Pi
import numpy as np
import tensorflow as tf

class TMParams(dict):
	def __init__(self, *args, **kwargs ):
		myparam = kwargs.pop('myparam', '')
		dict.__init__(self, *args, **kwargs )
		self["GIT_REVISION"] = os.popen("git rev-parse --short HEAD").read()
		self["CheckLevel"] = 1 # whether to test the consistency of several things...
		self["MAX_ATOMIC_NUMBER"] = 10
		# Parameters of MolEmb
		self["RBFS"] = np.array([[0.24666382, 0.37026093], [0.42773663, 0.47058503], [0.5780647, 0.47249905], [0.63062578, 0.60452219],
		 						[1.30332807, 1.2604625], [2.2, 2.4], [4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]])
		self["ANES"] = np.array([0.96763427, 1., 1., 1., 1., 2.14952757, 1.95145955, 2.01797792])
		self["SRBF"] = np.zeros((self["RBFS"].shape[0],self["RBFS"].shape[0]))
		self["SH_LMAX"]=4
		self["SH_NRAD"]=7
		self["SH_ORTH"]=1
		self["SH_MAXNR"]=self["RBFS"].shape[0]
		self["AN1_r_Rc"] = 4.6  # orgin ANI1 set
		self["AN1_a_Rc"] = 3.1  # orgin ANI1 set
		self["AN1_eta"] = 4.0
		self["AN1_zeta"] = 8.0
		#self["AN1_num_r_Rs"] = 40
		#self["AN1_num_a_Rs"] = 10
		#self["AN1_num_a_As"] = 10
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
		self["Embedded_Charge_Order"] = 2
		self["MBE_ORDER"] = 3
		# Training Parameters
		self["NeuronType"] = "relu"
		self["tf_prec"] = "tf.float32"
		self["learning_rate"] = 0.001
		self["momentum"] = 0.9
		self["max_steps"] = 1001
		self["batch_size"] = 1000
		self["test_freq"] = 50
		self["HiddenLayers"] = [512, 512, 512]
		self["hidden1"] = 512
		self["hidden2"] = 512
		self["hidden3"] = 512
		self["GradWeight"] = 0.01
		self["TestRatio"] = 0.2
		# DATA usage parameters
		self["InNormRoutine"] = None
		self["OutNormRoutine"] = None
		self["RandomizeData"] = True
		self["MxTimePerElement"] = 36000
		self["MxMemPerElement"]=16000 # Max Array for an element in MB
		self["ChopTo"] = None
		self["RotAvOutputs"] = 1 # Rotational averaging of force outputs.
		self["OctahedralAveraging"] = 0 # Octahedrally Average Outputs
		# Opt Parameters
		self["OptMaxCycles"]=10000
		self["OptThresh"]=0.0001
		self["OptMaxStep"]=0.1
		self["OptStepSize"] = 0.0005
		self["OptMomentum"] = 0.0
		self["OptMomentumDecay"] = 0.8
		self["OptPrintLvl"] = 1
		self["OptMaxBFGS"] = 7
		self["GSSearchAlpha"] = 0.005
		self["NebNumBeads"] = 10
		self["NebK"] = 0.01
		self["NebMaxBFGS"] = 12
		self["DiisSize"] = 20
		self["RemoveInvariant"] = True
		# Periodic Parameters, only cubic supported.
		self["CellWidth"] = 15.0 # Angstrom.
		# MD Parameters
		self["MDMaxStep"] = 20000
		self["MDdt"] = 0.2 # In fs.
		self["MDTemp"] = 300.0
		self["MDV0"] = "Random"
		self["MDThermostat"] = None # None, "Rescaling", "Nose", "NoseHooverChain"
		self["MDLogTrajectory"] = True
		self["MDUpdateCharges"] = True
		self["MDIrForceMin"] = False
		self["MDAnnealT0"] = 20.0
		self["MDAnnealSteps"] = 500
		# MD applied pulse parameters
		self["MDFieldVec"] = np.array([1.0,0.0,0.0])
		self["MDFieldAmp"] = 0.0
		self["MDFieldFreq"] = 1.0/1.2
		self["MDFieldTau"] = 1.2
		self["MDFieldT0"] = 3.0
		# parameters of electrostatic embedding
		self["EEOn"] = True # Whether to calculate/read in the required data at all...
		self["EESwitchFunc"] = "Cos" # options are Cosine, and Tanh.
		self["EEVdw"] = True # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EEOrder"] = 2 # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EEdr"] = 1.0 # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EECutoff"] = 5.0 # switch between 0 and 1/r occurs at Angstroms.
		#paths
		self["results_dir"] = "./results/"
		self["dens_dir"] = "./densities/"
		self["log_dir"] = "./logs/"
		# Garbage we're putting here for now.
		self["Qchem_RIMP2_Block"] = "$rem\n   jobtype   sp\n   method   rimp2\n   MAX_SCF_CYCLES  200\n   basis   cc-pvtz\n   aux_basis rimp2-cc-pvtz\n   symmetry   false\n   INCFOCK 0\n   thresh 12\n   SCF_CONVERGENCE 12\n$end\n"

	def __str__(self):
		tore=""
		for k in self.keys():
			tore = tore+k+":"+str(self[k])+"\n"
		return tore

def TMBanner():
	print("--------------------------")
	print("    "+unichr(0x1350)+unichr(0x2107)+unichr(0x2115)+unichr(0x405)+unichr(0x29be)+unichr(0x2c64)+'-'+unichr(0x164f)+unichr(0x29be)+unichr(0x2112)+"  0.1")
	print("--------------------------")
	print("By using this software you accept the terms of the GNU public license in ")
	print("COPYING, and agree to attribute the use of this software in publications as: \n")
	print("K.Yao, J. E. Herr, D. Toth, J. Parkhill. TensorMol 0.1 (2016)")
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
	print("Built Logger...")
	return tore
