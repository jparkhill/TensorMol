# Creates inner product of  TensorMol vectors
from TensorMol import *
import numpy as np 
nm = PullFreqData()
a = MSet("david_test.xyz")
a.ReadXYZ("david_test")
manager= TFMolManage("Mol_uneq_chemspider_ANI1_Sym_fc_sqdiff_BP_1" , None, False, RandomTData_=False, Trainable_=False)
m = a.mols[7]
masses = np.array(list(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms)))
resized_masses = np.transpose(np.resize(masses,(3,masses.shape[0]))).reshape((-1))
print "resized_masses:", resized_masses
nm = nm.reshape((33,-1))
IP = np.zeros((33,33))
for i in range(nm.shape[0]):
    for j in range(nm.shape[0]):
        IP[i, j] = np.dot(nm[i,:], nm[j, :]*np.sqrt(resized_masses))
print IP



print "m.Natoms()", m.NAtoms(), 3*m.NAtoms()
EnergyField = lambda x: manager.Eval_BPForceSingle(Mol(m.atoms,x),True)[0]
w,v = HarmonicSpectra(EnergyField, m.coords, masses)
v = v.real 
print "v:", v
print "nm:",nm
print "3n:", m.NAtoms()*3
print "v.shape:", v.shape
print "nm.shape:", nm.shape
for i in range(3*m.NAtoms()): 
	nm = v[:,i].reshape((m.NAtoms(),3))
	# nm *= np.sqrt(np.array([map(lambda x: ATOMICMASSESAMU[x-1],m.atoms)])).T
	for alpha in np.append(np.linspace(-.1,.1,30),np.linspace(.1,-.1,30)):
		mdisp = Mol(m.atoms,m.coords+alpha*nm)
		mdisp.WriteXYZfile("./results/","NormalMode_"+str(i))


#for i in range(v.shape[0]):
#    for j in range(v.shape[0]):
#        IP[i, j] = np.dot(v[:,i], v[:, j])
#print IP

nm = nm.reshape((33,-1))
IP = np.zeros((39,39))
for i in range(v.shape[0]):
    for j in range(nm.shape[0]):
        IP[i, j] = np.dot(v[:,i], nm[j, :])#*np.sqrt(resized_masses))
print IP
np.savetxt("IPmat.dat",IP)
