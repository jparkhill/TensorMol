from __future__ import absolute_import
from __future__ import print_function
import logging, time, os, sys
from math import pi as Pi
import numpy as np
import tensorflow as tf

class TMParams(dict):
	def __init__(self, *args, **kwargs ):
		myparam = kwargs.pop('myparam', '')
		dict.__init__(self, *args, **kwargs )
		self["GIT_REVISION"] = os.popen("git rev-parse --short HEAD").read()
		self["CheckLevel"] = 1 # whether to test the consistency of several things...
		self["PrintTMTimer"] = False # whether to emit timing messages.
		self["MAX_ATOMIC_NUMBER"] = 10
		# Parameters of MolEmb
		self["RBFS"] = np.array([[0.35, 0.35], [0.70, 0.35], [1.05, 0.35], [1.40, 0.35], [1.75, 0.35], [2.10, 0.35], [2.45, 0.35],
									[2.80, 0.35], [3.15, 0.35], [3.50, 0.35], [3.85, 0.35], [4.20, 0.35], [4.55, 0.35], [4.90, 0.35]])
		self["ANES"] = np.array([2.20, 1.0, 1.0, 1.0, 1.0, 2.55, 3.04, 3.44]) #pauling electronegativity
		self["SRBF"] = np.zeros((self["RBFS"].shape[0],self["RBFS"].shape[0]))
		self["SH_LMAX"]=4
		self["SH_NRAD"]=14
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
		self["GradScalar"] = 1.0
		self["DipoleScalar"] = 1.0
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
		self["MonitorSet"] = None
		self["NetNameSuffix"] = ""
		self["NeuronType"] = "relu"
		self["tf_prec"] = "tf.float64" # Do not change this to 32.
		self["learning_rate"] = 0.001
		self["learning_rate_dipole"] = 0.0001
		self["learning_rate_energy"] = 0.00001
		self["momentum"] = 0.9
		self["max_steps"] = 1001
		self["batch_size"] = 1000
		self["test_freq"] = 10
		self["HiddenLayers"] = [200, 200, 200]
		self["hidden1"] = 512
		self["hidden2"] = 512
		self["hidden3"] = 512
		self["GradWeight"] = 0.01
		self["TestRatio"] = 0.2
		self["Profiling"] = False
		self["max_checkpoints"] = 1
		self["KeepProb"] = 0.7
		self["weight_decay"] = 0.001
		self["ConvFilter"] = [32, 64]
		self["ConvKernelSize"] = [[8,1],[4,1]]
		self["ConvStrides"] = [[8,1],[4,1]]
		self["sigmoid_alpha"] = 100.0
		self["EnergyScalar"] = 1.0
		self["GradScalar"] = 1.0/20.0
		self["DipoleScaler"]=1.0
		# DATA usage parameters
		self["InNormRoutine"] = None
		self["OutNormRoutine"] = None
		self["RandomizeData"] = True
		self["MxTimePerElement"] = 36000
		self["MxMemPerElement"]=16000 # Max Array for an element in MB
		self["ChopTo"] = None
		self["RotAvOutputs"] = 1 # Rotational averaging of force outputs.
		self["OctahedralAveraging"] = 0 # Octahedrally Average Outputs
		self["train_gradients"] = True
		self["train_dipole"] = True
		self["train_rotation"] = True
		# Opt Parameters
		self["OptMaxCycles"]=50
		self["OptThresh"]=0.0001
		self["OptMaxStep"]=0.1
		self["OptStepSize"] = 0.1
		self["OptMomentum"] = 0.0
		self["OptMomentumDecay"] = 0.8
		self["OptPrintLvl"] = 1
		self["OptLatticeStep"] = 0.050
		self["GSSearchAlpha"] = 0.001
		self["SDStep"] = 0.05
		self["MaxBFGS"] = 7
		self["NebSolver"] = "Verlet"
		self["NebNumBeads"] = 18 # It's important to have enough beads.
		self["NebK"] = 0.07
		self["NebKMax"] = 1.0
		self["NebClimbingImage"] = True
		self["DiisSize"] = 20
		self["RemoveInvariant"] = True
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
		self["MDAnnealTF"] = 300.0
		self["MDAnnealKickBack"] = 1.0
		self["MDAnnealSteps"] = 1000
		# MD applied pulse parameters
		self["MDFieldVec"] = np.array([1.0,0.0,0.0])
		self["MDFieldAmp"] = 0.0
		self["MDFieldFreq"] = 1.0/1.2
		self["MDFieldTau"] = 1.2
		self["MDFieldT0"] = 3.0
		# Metadynamics parameters
		self["MetaBumpTime"] = 15.0
		self["MetaBowlK"] = 0.0
		self["MetaMaxBumps"] = 500
		self["MetaMDBumpHeight"] = 0.05
		self["MetaMDBumpWidth"] = 0.1
		# parameters of electrostatic embedding
		self["AddEcc"] = True
		self["OPR12"] = "Poly" # Poly = Polynomial cutoff or Damped-Shifted-Force
		self["Poly_Width"] = 4.6
		self["Elu_Width"] = 4.6
		self["EEOn"] = True # Whether to calculate/read in the required data at all...
		self["EESwitchFunc"] = "CosLR" # options are Cosine, and Tanh.
		self["EEVdw"] = True # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EEOrder"] = 2 # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EEdr"] = 1.0 # 1/r => 0.5*(Tanh[(r - EECutoff)/EEdr] + 1)/r
		self["EECutoff"] = 5.0 #switch between 0 and 1/r occurs at Angstroms.
		self["EECutoffOn"] = 4.4 # switch on between 0 and 1/r occurs at Angstroms.
		self["EECutoffOff"] = 15.0 # switch off between 0 and 1/r occurs at Angstroms.
		self["Erf_Width"] = 0.2
		self["DSFAlpha"] = 0.18
		#paths
		self["sets_dir"] = "./datasets/"
		self["results_dir"] = "./results/"
		self["dens_dir"] = "./densities/"
		self["log_dir"] = "./logs/"
		self["networks_directory"] = "./networks"
		# Garbage we're putting here for now.
		self["Qchem_RIMP2_Block"] = "$rem\n   jobtype   sp\n   method   rimp2\n   MAX_SCF_CYCLES  200\n   basis   cc-pvtz\n   aux_basis rimp2-cc-pvtz\n   symmetry   false\n   INCFOCK 0\n   thresh 12\n   SCF_CONVERGENCE 12\n$end\n"
		np.set_printoptions(formatter={'float': '{: .8f}'.format}) #Set pretty printing for numpy arrays

	def __str__(self):
		tore=""
		for k in self.keys():
			tore = tore+k+":"+str(self[k])+"\n"
		return tore

def TMBanner():
	print("--------------------------")
	try:
		if sys.version_info[0] < 3:
			print(("    "+unichr(0x1350)+unichr(0x2107)+unichr(0x2115)+unichr(0x405)+unichr(0x29be)+unichr(0x2c64)+'-'+unichr(0x164f)+unichr(0x29be)+unichr(0x2112)+"  0.1"))
		else:
			print(("    "+chr(0x1350)+chr(0x2107)+chr(0x2115)+chr(0x405)+chr(0x29be)+chr(0x2c64)+'-'+chr(0x164f)+chr(0x29be)+chr(0x2112)+"  0.1"))
	except:
		pass
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
	fh = logging.FileHandler(filename=path_+time.strftime("%a_%b_%d_%H.%M.%S_%Y")+'.log')
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
