#
# Optimization algorithms
#

from Sets import *
from TFMolManage import *
from Mol import *

class NN_MBE:
	def __init__(self,tfm_=None):
		self.nn_mbe = dict()
		if tfm_ != None:
			for order in tfm_:
				print tfm_[order]
				self.nn_mbe[order] = TFMolManage(tfm_[order], None, False)
		return 


	def NN_Energy(self, mol):
		mol.Generate_All_MBE_term(atom_group=3, cutoff=6, center_atom=0)  # one needs to change the variable here 
		nn_energy = 0.0
		for i in range (1, mol.mbe_order+1):
			nn_energy += self.nn_mbe[i].Eval_Mol(mol)
		mol.Set_MBE_Force()
		mol.nn_energy = nn_energy
		print "coords of mol:", mol.coords
		print "force of mol:", mol.mbe_deri
		print "energy of mol:", nn_energy
		return 

