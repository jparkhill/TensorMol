#
# Optimization algorithms
#

from Sets import *
from TFMolManage import *
from Mol import *
from Electrostatics import *

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
		print "force of mol:", mol.properties["mbe_deri"]
		print "energy of mol:", nn_energy
		return 


class NN_MBE_BF:
        def __init__(self,tfm_=None, dipole_tfm_=None):
		self.mbe_order = PARAMS["MBE_ORDER"]
                self.nn_mbe = tfm_
		self.nn_dipole_mbe = dipole_tfm_
                return

        def NN_Energy(self, mol):
		s = MSet()
		for order in range (1, self.mbe_order+1):
			s.mols += mol.mbe_frags[order]
		energies =  np.asarray(self.nn_mbe.Eval_BPEnergy(s))
		pointer = 0
		for order in range (1, self.mbe_order+1):
			mol.frag_energy_sum[order] = np.sum(energies[pointer:pointer+len(mol.mbe_frags[order])])
			pointer += len(mol.mbe_frags[order])
		mol.MBE_Energy()
                return

	def NN_Dipole(self, mol):
                s = MSet()
                for order in range (1, self.mbe_order+1):
                        s.mols += mol.mbe_frags[order]
                dipoles, charges =  self.nn_dipole_mbe.Eval_BPDipole_2(s)
                #print "dipole:", dipoles
                pointer = 0
                for order in range (1, self.mbe_order+1):
                        mol.frag_dipole_sum[order] = np.sum(dipoles[pointer:pointer+len(mol.mbe_frags[order])], axis=0)
                        pointer += len(mol.mbe_frags[order])
		#print "mol.frag_dipole_sum[order] ", mol.frag_dipole_sum 
		mol.MBE_Dipole()
                return

	def NN_Energy_Force(self, mol):
                s = MSet()
                for order in range (1, self.mbe_order+1):
                        s.mols += mol.mbe_frags[order]
                energies, forces =  self.nn_mbe.Eval_BPForceSet(s)
		energies = np.asarray(energies)
                pointer = 0
                for order in range (1, self.mbe_order+1):
			mol.frag_force_sum[order] = np.zeros((mol.NAtoms(),3))
			for i, mol_frag in enumerate(mol.mbe_frags[order]):
				mol.frag_force_sum[order][mol_frag.properties["mbe_atom_index"]] += forces[pointer+i]
			#print "energy of frags in order", order
			#print energies[pointer:pointer+len(mol.mbe_frags[order])]
                        mol.frag_energy_sum[order] = np.sum(energies[pointer:pointer+len(mol.mbe_frags[order])])
                        pointer += len(mol.mbe_frags[order])
                mol.MBE_Energy()
		mol.MBE_Force()
		#print mol.properties['mbe_deri']
                return mol.nn_energy, mol.nn_force

	def NN_Charge(self, mol):
		s = MSet()
                for order in range (1, self.mbe_order+1):
                        s.mols += mol.mbe_frags[order]
                dipoles, charges =  self.nn_dipole_mbe.Eval_BPDipole_2(s)
		pointer = 0
		for order in range(1, self.mbe_order+1):
			mol.frag_charge_sum[order] = np.zeros(mol.NAtoms())
			for i, mol_frag in enumerate(mol.mbe_frags[order]):
				mol.frag_charge_sum[order][mol_frag.properties["mbe_atom_index"]] += charges[pointer+i]
			#print "charge for order", order, mol.frag_charge_sum[order]
                        pointer += len(mol.mbe_frags[order])
		mol.MBE_Charge()
		#print mol.mbe_charge
		return	mol.nn_charge
