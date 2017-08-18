from __future__ import absolute_import
import memory_util
memory_util.vlog(1)
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from TensorMol.ElectrostaticsTF import *
from TensorMol.NN_MBE import *

a = MSet("chemspider9_metady_force")
a.Load()
for i in range(141776):
	a.mols.pop()
TreatedAtoms = a.AtomTypes()
PARAMS["learning_rate"] = 0.00001
PARAMS["momentum"] = 0.95
PARAMS["max_steps"] = 101
PARAMS["batch_size"] =  40   # 40 the max min-batch size it can go without memory error for training
PARAMS["test_freq"] = 2
PARAMS["tf_prec"] = "tf.float64"
PARAMS["GradScaler"] = 1.0
PARAMS["DipoleScaler"]=1.0
PARAMS["NeuronType"] = "relu"
PARAMS["HiddenLayers"] = [1000, 1000, 1000]
PARAMS["EECutoff"] = 15.0
PARAMS["EECutoffOn"] = 7.0
PARAMS["Erf_Width"] = 0.4
#PARAMS["AN1_r_Rc"] = 8.0
#PARAMS["AN1_num_r_Rs"] = 64
PARAMS["EECutoffOff"] = 15.0
PARAMS["learning_rate_dipole"] = 0.0001
PARAMS["learning_rate_energy"] = 0.00001
PARAMS["SwitchEpoch"] = 10
d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
tset = TensorMolData_BP_Direct_EE(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True) # Initialize TensorMolData that contain the training data fo
#tset = TensorMolData_BP_Multipole_2_Direct(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = False)
manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode") # Initialzie a manager than manage the training of neural network.
#manager=TFMolManage("",tset,False,"Dipole_BP_2_Direct")
PARAMS["Profiling"] = 1
with memory_util.capture_stderr() as stderr:
	manager.Train(1)
memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)
