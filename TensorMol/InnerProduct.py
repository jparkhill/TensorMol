# Creates inner product of Q Chem and TensorMol vectors
from TensorMol import *
import numpy as np 
nm = PullFreqData()
a = MSet("david_test.xyz")
a.ReadXYZ("david_test")
manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)
m = a.mols[6]
EnergyField = lambda x: manager.Eval_BPForceSingle(Mol(m.atoms,x),True)[0]
masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms))
w,v = HarmonicSpectra(EnergyField, m.coords, masses)
v = v.real 
print v.shape
IP = np.zeros(v.shape)
for i in range(v.shape[0]):
    for j in range(v.shape[0]):
        IP[i, j] = np.dot(v[:,i], v[:, j])
print IP