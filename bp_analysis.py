import pickle
import numpy  as np

result = pickle.load(open("test_result_cleaned_connectedbond_angle.dat", "rb"))


nn = result['nn']
acc = result['acc']
#print "nn",nn

nmol = len(nn)
mol = np.zeros((nmol,2))

mol[:,0] = acc
mol[:,1] = nn

diff = mol[:,0] - mol[:,1]
print diff
print "MAE:", np.mean(abs(diff))

print "MSE", (np.sum((diff)**2)/diff.shape[0])**0.5

length = result['length']
atoms = result['atoms']

hartreetokjmol = 2625.5
for i in range (0, len(atoms)):
	natom = len(atoms[i])
	print "natom:", natom
	tmp =np.zeros((natom,2))
	tmp[:,0] = length[i]
	tmp[:,1] = np.asarray(atoms[i])*hartreetokjmol
	np.savetxt("bond_"+str(i)+"_connectedbond_angle.dat", tmp)
