"""
Kun: please move your work to this separate file to keep stuff organized. 
"""

from Util import *
import numpy as np
import random, math
import Mol

class FragableMol(Mol):
	""" Provides a molecule which can be fragmented """
	def __init__(self, atoms_ =  None, coords_ = None):
		Mol.__init__(self,atoms_,coords_)
		self.mbe_order = MBE_ORDER
		self.frag_list = []    # list of [{"atom":.., "charge":..},{"atom":.., "charge":..},{"atom":.., "charge":..}]
		self.type_of_frags = []  # store the type of frag (1st order) in the self.mbe_frags:  [1,1,1 (H2O), 2,2,2(Na),3,3,3(Cl)]
		self.type_of_frags_dict = {}
		# I'm glad you're commenting but the above is unacceptably redundant ... -JAP
		# at the very least you should make a derived class FragmentableMol or something that gets all this shite
		self.atoms_of_frags = [] # store the index of atoms of each frag
		self.mbe_frags=dict()    # list of  frag of each order N, dic['N'=list of frags]
		self.mbe_frags_deri=dict()
		self.mbe_permute_frags=dict() # list of all the permuted frags
		self.mbe_frags_energy=dict()  # MBE energy of each order N, dic['N'= E_N]
		self.mbe_energy=dict()   # sum of MBE energy up to order N, dic['N'=E_sum]
		self.mob_all_frags = dict() # dictionary that stores all the necessary frags in mol format  for MOB dic[LtoS([list])] = mol
		return

	def Make_AtomNodes(self):
		atom_nodes = []
		for i in range (0, self.NAtoms()):
			atom_nodes.append(AtomNode(self.atoms[i], i))
		self.properties["atom_nodes"] = atom_nodes

	def Connect_AtomNodes(self):
		dist_mat = MolEmb.Make_DistMat(self.coords)
		for i in range (0, self.NAtoms()):
			for j in range (i+1, self.NAtoms()):
				dist = dist_mat[i][j]
				atom_pair=[self.atoms[i], self.atoms[j]]
				atom_pair.sort()
				bond_name = self.AtomName_From_List(atom_pair)
				if dist <= bond_length_thresh[bond_name]:
					(self.properties["atom_nodes"][i]).Append(self.properties["atom_nodes"][j])
					(self.properties["atom_nodes"][j]).Append(self.properties["atom_nodes"][i])
		return

	def Make_Mol_Graph(self):
		self.Make_AtomNodes()
		self.Connect_AtomNodes()
		return

	def Bonds_Between_All(self):
		H_bonds = np.zeros((self.NAtoms(), self.NAtoms()))
		total_bonds = np.zeros((self.NAtoms(), self.NAtoms()))
		memory_total=dict()
                memory_H = dict()
		for atom_1 in range (0, self.NAtoms()):
			for atom_2 in range (0, self.NAtoms()):
				total_bonds[atom_1][atom_2], H_bonds[atom_1][atom_2] = self.Shortest_Path_DP(atom_1, atom_2, self.NAtoms()-1, memory_total, memory_H)
		self.properties["Bonds_Between"] = total_bonds
		self.properties["H_Bonds_Between"] = H_bonds
		return

	def Bonds_Between(self, atom_1, atom_2, ignore_Hbond = False):  # number of bonds between to atoms, ignore H-? or not.
		memory_total=dict()
		memory_H = dict()
		bonds, H_bonds = self.Shortest_Path_DP(atom_1, atom_2, self.NAtoms()-1, memory_total, memory_H)
		return  bonds, H_bonds

	def Shortest_Path_DP(self, a, b, k, memory_total, memory_H):
		index = [a, b]
		index.sort()
		index.append(k)
		index_string = LtoS(index)
		if index_string in memory_total.keys() and index_string in memory_H.keys():
			return  memory_total[index_string], memory_H[index_string]
		elif k == 0 and a!=b:
			memory_total[index_string] = float('inf')
			memory_H[index_string] = float('inf')
			return memory_total[index_string], memory_H[index_string]
		elif a == b:
			memory_total[index_string] = 0
			memory_H[index_string] = 0
			return  memory_total[index_string],  memory_H[index_string]
		else:
			mini_bond  = float('inf')
			mini_H_bond = float('inf')
			save_index = None
			for node in self.properties["atom_nodes"][b].connected_nodes:
				index = node.node_index
				num_bond, H_bond = self.Shortest_Path_DP(a, index, k-1, memory_total, memory_H)
				if num_bond < mini_bond:
					mini_bond = num_bond
					mini_H_bond = H_bond
					save_index = index
			if save_index !=None:
				if self.properties["atom_nodes"][b].node_type == 1 or self.properties["atom_nodes"][save_index].node_type==1 :
					mini_H_bond += 1
			mini_bond = mini_bond + 1
			memory_total[index_string] = mini_bond
			memory_H [index_string] = mini_H_bond
			return mini_bond, mini_H_bond

	def GetNextNode_DFS(self, visited_list, node_stack):
		node = node_stack.pop()
		visited_list.append(node.node_index)
		for next_node in node.connected_nodes:
			if next_node.node_index not in visited_list:
				node_stack.append(next_node)
		return node, visited_list, node_stack

	def DFS(self, head_node):
		node_stack = [head_node]
		visited_list = []
		while(node_stack):   # if node stack is not empty
			node, visited_list, node_stack  = self.GetNextNode_DFS(visited_list, node_stack)
			print "node.node_index", node.node_index,  visited_list

	def DFS_recursive(self, node, visited_list):
		print node.node_index
		visited_list.append(node.node_index)
		for next_node in node.connected_nodes:
			if next_node.node_index not in visited_list:
				visited_list = self.DFS_recursive(next_node, visited_list)
		return visited_list

	def DFS_recursive_all_order(self, node, visited_list, ignored_ele = [1]):
		atom_set = list(set(node.connected_atoms))
		atom_set.sort()
		node_set_index = []
		for i in range (0, len(atom_set)):
			node_set_index.append([])
			for j in range (0, len(node.connected_atoms)):
				if node.connected_atoms[j] == atom_set[i]:
					node_set_index[i].append(j)
		sub_order = []
		for index in node_set_index:
			sub_order.append([])
			if node.connected_atoms[index[0]]in ignored_ele:  # element that do not permute
				sub_order[-1] = [list(index)]
			else:
				sub_order[-1] = [list(x) for x in list(itertools.permutations(index))]
		all_order = [list(x) for x in list(itertools.product(*sub_order))]
		tmp = []
		for order in all_order:
			tmp.append([])
			for l in order:
				tmp[-1] += l
		all_order = list(tmp)

		print node.node_index
		visited_list.append(node.node_index)
		visited_list_save =  list(visited_list)
		for order in all_order:
			connected_nodes = [node.connected_nodes[i] for i in order]
			for next_node in connected_nodes:
				visited_list = list(visited_list_save)
				if next_node.node_index not in visited_list:
					visited_list = self.DFS_recursive(next_node, visited_list)
		return visited_list

	def Find_Frag(self, frag, ignored_ele=[1], frag_head=0, avail_atoms=None):   # ignore all the H for assigment
		if avail_atoms==None:
			avail_atoms = range(0, self.NAtoms())
		frag_head_node = frag.properties["atom_nodes"][frag_head]
		frag_node_stack = [frag_head_node]
                frag_visited_list = []
		all_mol_visited_list = [[]]
                while(frag_node_stack):   # if node stack is not empty
			current_frag_node = frag_node_stack[-1]
			updated_all_mol_visited_list = []
			for mol_visited_list in all_mol_visited_list:
				possible_node = []
				if mol_visited_list ==[]:
                                                possible_node = [self.properties["atom_nodes"][i] for i in avail_atoms]
						for mol_node in possible_node:
							if mol_node.node_index not in mol_visited_list and self.Compare_Node(mol_node, current_frag_node) and self.Check_Connection(mol_node, current_frag_node, mol_visited_list, frag_visited_list):
								updated_all_mol_visited_list.append(mol_visited_list+[mol_node.node_index])
								if mol_node.node_type in ignored_ele:# just once
									break
				else:
					connected_node_index_in_frag = []
                	                for connected_node_in_frag in current_frag_node.connected_nodes:
                        	                if connected_node_in_frag.node_index in frag_visited_list:
							connected_node_index_in_frag.append(frag_visited_list.index(connected_node_in_frag.node_index))
					for connected_node_index in connected_node_index_in_frag:
						connected_node_in_mol = self.properties["atom_nodes"][mol_visited_list[connected_node_index]]
						for target_node in connected_node_in_mol.connected_nodes:
							if target_node.node_index not in mol_visited_list and self.Compare_Node(target_node, current_frag_node) and self.Check_Connection(target_node, current_frag_node, mol_visited_list, frag_visited_list) and target_node.node_index in avail_atoms:
								updated_all_mol_visited_list.append(mol_visited_list+[target_node.node_index])
								if target_node.node_type in ignored_ele:
									break
			all_mol_visited_list = list(updated_all_mol_visited_list)
                        next_frag_node, frag_visited_list, frag_node_stack  = self.GetNextNode_DFS(frag_visited_list, frag_node_stack)
		frags_in_mol = []
		for mol_visited_list in all_mol_visited_list:
			mol_visited_list.sort()
			if mol_visited_list not in frags_in_mol:
				sorted_mol_visited_list = [x for (y, x) in sorted(zip(frag_visited_list,mol_visited_list))]## sort the index order of frags in mol to the same as the frag
				frags_in_mol.append(sorted_mol_visited_list)
		return frags_in_mol

	def Mol_Frag_Index_to_Mol(self, frags_index, capping=True):
		convert_to_mol = []
		for frag_index in frags_index:
			convert_to_mol.append(Mol())
			convert_to_mol[-1].atoms = self.atoms[frag_index].copy()
			convert_to_mol[-1].coords =  self.coords[frag_index].copy()
			if capping:
				cap_atoms, cap_coords = self.Frag_Caps(frag_index)
				convert_to_mol[-1].atoms = np.concatenate((convert_to_mol[-1].atoms, cap_atoms))
				convert_to_mol[-1].coords = np.concatenate((convert_to_mol[-1].coords, cap_coords))
		return convert_to_mol

	def Frag_Caps(self, frag_index, capping_atom = 1):  # 1 stand for H, used H capping by default
		cap_atoms = []
		cap_coords = []
		for atom_index in frag_index:
			for node in self.properties["atom_nodes"][atom_index].connected_nodes:
				if node.node_index not in frag_index:
					node_index = node.node_index
					cap_atoms.append(capping_atom)
					cap_coords.append(self.coords[atom_index]+ (atomic_radius[self.atoms[atom_index]]+atomic_radius[capping_atom])/(atomic_radius[self.atoms[atom_index]]+atomic_radius[self.atoms[node_index]])*(self.coords[node_index] - self.coords[atom_index]))
		cap_atoms = np.array(cap_atoms)
		cap_coords = np.array(cap_coords)
		return cap_atoms, cap_coords

	def Mol_Frag_Index_to_Mol_Old(self, frag, frags_in_mol=None, capping=False):
		convert_to_mol = []
		if frags_in_mol == None:
			frags_in_mol = self.Find_Frag(frag)
		for frag_in_mol in frags_in_mol:
			convert_to_mol.append(Mol())
			convert_to_mol[-1].atoms = self.atoms[frag_in_mol].copy()
			convert_to_mol[-1].coords = self.coords[frag_in_mol].copy()
			if capping:
				caps = []
				for dangling_atom in frag.undefined_bonds.keys():
					if isinstance(dangling_atom, int):
						dangling_index_in_mol = frag_in_mol[dangling_atom]
						for node in self.properties["atom_nodes"][dangling_index_in_mol].connected_nodes:
							node_index = node.node_index
							if node_index not in frag_in_mol:
								convert_to_mol[-1].atoms = np.concatenate((convert_to_mol[-1].atoms,[1])) # use hydrogen capping
								H_coords = self.coords[dangling_index_in_mol] + (atomic_radius_cho[self.atoms[dangling_index_in_mol]]+atomic_radius_cho[1])/(atomic_radius_cho[self.atoms[dangling_index_in_mol]]+atomic_radius_cho[self.atoms[node_index]])*(self.coords[node_index] - self.coords[dangling_index_in_mol]) # use the SMF capping scheme
								convert_to_mol[-1].coords = np.concatenate((convert_to_mol[-1].coords, H_coords.reshape((1,-1))))
			# print convert_to_mol[-1].atoms, convert_to_mol[-1].coords
		return convert_to_mol

	def Frag_Overlaps(self, frags_index_list, order=2):  #   decide the Nth order overlap of fragment
		overlap_list = []
		frag_pair_list = []
		if order < 2:
			raise Exception("Overlap order needs to be  >= 2 ")
		elif order == 2:
			for i in range (0, len(frags_index_list)):
				for j in range (i+1, len(frags_index_list)):
					overlap=list(set(frags_index_list[i]).intersection(frags_index_list[j]))
					if overlap:
						overlap_list.append(overlap)
						frag_pair_list.append([i,j])
			return overlap_list, frag_pair_list
		else:
			old_index_list, old_pair_list = self.Frag_Overlaps(frags_index_list, order-1)
			new_index_list = []
			new_pair_list = []
			for i in range (0, len(old_index_list)):
				for j in range (0, len(frags_index_list)):
					if j not in old_pair_list[i]:
						overlap = list(set(old_index_list[i]).intersection(frags_index_list[j]))
						if overlap:
							tmp_pair = old_pair_list[i]+[j]
							tmp_pair.sort()
							if tmp_pair not in new_pair_list:
								new_index_list.append(overlap)
								new_pair_list.append(tmp_pair)
			return new_index_list, new_pair_list

	def Overlap_Partition(self, frags_list, frag_overlap_list=None, capping=True, Order=8):   # Order should be chosen as the max possible number of frags that has comon overlap
		all_frags_index  = []
		frags_type = []
		all_frags_mol = []
		for i in range (0, len(frags_list)):
			one_type_frags_index = self.Find_Frag(frags_list[i])
			for frag_index in one_type_frags_index:
				all_frags_index.append(frag_index)
				frags_type.append(i)
			one_type_frags_mol = self.Mol_Frag_Index_to_Mol(one_type_frags_index, capping)
			for frag_mol in one_type_frags_mol:
				all_frags_mol.append(frag_mol)

		overlap_index_list = []
		frag_pair_list = []
		for order in range (2, Order+1):
			tmp_index_list, tmp_pair_list = self.Frag_Overlaps(all_frags_index, order)
			overlap_index_list += tmp_index_list
			frag_pair_list += tmp_pair_list
		all_overlaps_mol = []
		overlaps_type = []

		for i, overlap_index in enumerate(overlap_index_list):
			overlap_index.sort()
			tmp_mol = self.Mol_Frag_Index_to_Mol([overlap_index], capping)
			all_overlaps_mol.append(tmp_mol[0])
			overlaps_type.append(i)
	#	if frag_overlap_list !=None :  # already provide possible type of overlaps in advance.
	#		for overlap_index in overlap_index_list:
	#			found = 0
	#			for i, overlap in enumerate(frag_overlap_list):
	#				if len(overlap_index) == overlap.NAtoms() and self.Find_Frag(overlap, avail_atoms = overlap_index):
	#					found = 1
	#					overlaps_type.append(i)
	#					tmp_mol = self.Mol_Frag_Index_to_Mol([overlap_index], capping)
	#					all_overlaps_mol.append(tmp_mol[0])
	#					break   # assuming the overlap can only belong to one kind of overlap fragment
	#			if not found:
	#				print "Warning! Overlap: ", overlap_index," is not found in the provided list"
	#
	#	else:   # determine the type of overlaps after generate, this has not been implemented yet. KY
	#		raise Exception("needs to provide the possible overlaps")

		self.properties["all_frags_index"] = all_frags_index
		self.properties["overlap_index_list"] = overlap_index_list
		self.properties["frags_type"] = frags_type
		self.properties["overlaps_type"] = overlaps_type
		self.properties["all_frags_mol"] = all_frags_mol
		self.properties["all_overlaps_mol"] = all_overlaps_mol
		self.properties["frag_pair_list"] = frag_pair_list
		return	all_frags_index, overlap_index_list, frags_type, overlaps_type, all_frags_mol, all_overlaps_mol

	def MOB_Monomer(self):
		self.properties["mob_monomer_index"] = []
		self.properties["mob_monomer_type"] = []
		for i in range (0, len(self.properties["all_frags_index"])):
			self.properties["mob_monomer_index"].append(self.properties["all_frags_index"][i])
			self.properties["mob_monomer_type"].append(self.properties["frags_type"][i])  # if it is frag,the type is the Nth frag in frag_list
		for i in range (0, len(self.properties["overlap_index_list"])):
			self.properties["mob_monomer_index"].append(self.properties["overlap_index_list"][i])
			self.properties["mob_monomer_type"].append(int(-1-self.properties["overlaps_type"][i]))  # if it is overlap, the type is the (abs(N)-1)th overlap in overlap_list
		for monomer_index in self.properties["mob_monomer_index"]:
			monomer_index.sort()
			harsh_string = LtoS(monomer_index)
			if harsh_string not in self.mob_all_frags.keys():
				self.mob_all_frags[harsh_string] = (self.Mol_Frag_Index_to_Mol([monomer_index],True))[0]
		return

	def MOB_Monomer_Overlap(self):
		self.mob_monomer_overlap_index = []
		self.mob_monomer_overlap_type = []
		for i in range (0, len(self.properties["mob_monomer_index"])):
			for j in range (i+1, len(self.properties["mob_monomer_index"])):
				mob_monomer_overlap_index=list(set(self.properties["mob_monomer_index"][i]).intersection(self.properties["mob_monomer_index"][j]))
				if mob_monomer_overlap_index:
					self.mob_monomer_overlap_index.append(mob_monomer_overlap_index)
					self.mob_monomer_overlap_type.append([i,j])
		for overlap_type in self.mob_monomer_overlap_type:
			overlap_type.sort()
		for overlap_index in self.mob_monomer_overlap_index:
			overlap_index.sort()
                        harsh_string = LtoS(overlap_index)
                        if harsh_string not in self.mob_all_frags.keys():
                                self.mob_all_frags[harsh_string] = (self.Mol_Frag_Index_to_Mol([overlap_index],True))[0]
		return

	def Connected_MOB_Dimer(self):
		self.properties["connected_dimers_index"] = []
		self.properties["connected_dimers_type"] = []
		for i in range (0, len(self.properties["mob_monomer_index"])):
			for j in range (i+1, len(self.properties["mob_monomer_index"])):
				for atom_index in self.properties["mob_monomer_index"][i]:
					is_connected = False
					for node in self.properties["atom_nodes"][atom_index].connected_nodes:
						if node.node_index in self.properties["mob_monomer_index"][j]:
							self.properties["connected_dimers_type"].append([i,j])
							self.properties["connected_dimers_index"].append(list(set(self.properties["mob_monomer_index"][i]+self.properties["mob_monomer_index"][j])))
							is_connected = True
							break
					if is_connected:
						break
		for dimer_type in self.properties["connected_dimers_type"]:
			dimer_type.sort()
		for dimer_index in self.properties["connected_dimers_index"]:
                        dimer_index.sort()
                        harsh_string = LtoS(dimer_index)
                        if harsh_string not in self.mob_all_frags.keys():
                                self.mob_all_frags[harsh_string] = (self.Mol_Frag_Index_to_Mol([dimer_index],True))[0]
		return

	def Calculate_MOB_Frags(self, method='pyscf'):
		if method=="pyscf":
			for key in self.mob_all_frags.keys():
				mol = self.mob_all_frags[key]
				if mol.energy == None:
					mol.PySCF_Energy("cc-pvdz")
		else:
			raise Exception("Other method is not supported yet")

	def MOB_Energy(self):
		Mono_Cp = []
		for i in range (0, len(self.properties["mob_monomer_index"])):
			if self.properties["mob_monomer_type"][i] >= 0:  # monomer is from frag
				Mono_Cp.append(1)
			else:   #momer is from overlap
				for j in self.properties["frag_pair_list"][abs(self.properties["mob_monomer_type"][i])-1]:
					print self.properties["all_frags_index"][j]
				overlap_order = len(self.properties["frag_pair_list"][abs(self.properties["mob_monomer_type"][i])-1])
				Mono_Cp.append(pow(-1, overlap_order-1))
		first_order_energy = 0
		for i in range (0, len(self.properties["mob_monomer_index"])):
			first_order_energy += Mono_Cp[i]*self.mob_all_frags[LtoS(self.properties["mob_monomer_index"][i])].energy
		second_order_energy = 0
		for i in range (0, len(self.properties["connected_dimers_index"])):
			p_index = self.properties["connected_dimers_type"][i][0]
			q_index = self.properties["connected_dimers_type"][i][1]
			Epandq = self.mob_all_frags[LtoS(self.properties["connected_dimers_index"][i])].energy
			Ep = self.mob_all_frags[LtoS(self.properties["mob_monomer_index"][p_index])].energy
			Eq = self.mob_all_frags[LtoS(self.properties["mob_monomer_index"][q_index])].energy
			if self.properties["connected_dimers_type"][i] in self.mob_monomer_overlap_type:  #overlap
				index = self.mob_monomer_overlap_type.index(self.properties["connected_dimers_type"][i])
				Epnotq = self.mob_all_frags[LtoS(self.mob_monomer_overlap_index[index])].energy
			else:
				Epnotq = 0.0  #not overlap
			deltaEpq = Epandq - (Ep + Eq - Epnotq)
			second_order_energy += Mono_Cp[p_index]*Mono_Cp[q_index]*deltaEpq

		self.properties["mob_energy"] = first_order_energy + second_order_energy
		print "MOB_energy", self.properties["mob_energy"]
		return

	def Pick_Not_Allowed_Overlaps(self,  frags_index_list, allowed_overlap_list, overlap_list=None, frag_pair_list = None):   # check whether overlap of frags is allowed
		if overlap_list == None or frag_pair_list == None:
			overlap_list, frag_pair_list = self.Frag_Overlaps(frags_index_list)
		not_allowed_overlap_index = []
		for overlap_index,  overlap in  enumerate(overlap_list):
			for allowed_overlap in allowed_overlap_list:
				if allowed_overlap.NAtoms() != len(overlap) or not self.Find_Frag(allowed_overlap, avail_atoms=overlap):
					not_allowed_overlap_index.append(overlap_index)

		return not_allowed_overlap_index

	def Optimize_Overlap(self, frags_index_list, allowed_overlap_list, not_allowed_overlap_index=None):         #delete the frags that generate the not allowed overlaps
		overlap_list, frag_pair_list = self.Frag_Overlaps(frags_index_list)
		if not_allowed_overlap_index == None:
			not_allowed_overlap_index = self.Pick_Not_Allowed_Overlaps(frags_index_list, allowed_overlap_list, overlap_list, frag_pair_list)
		deleted_frag_list = self.Greedy_Delete_Frag(frags_index_list, frag_pair_list, not_allowed_overlap_index)
		opt_frags_index_list = [ frags_index_list[i] for i in Setdiff(range(0, len(frags_index_list)), deleted_frag_list) ]
		print "Is the new frags complete:", self.Check_Frags_Is_Complete(opt_frags_index_list)
		return

	def Greedy_Delete_Frag(self, frags_index_list, frag_pair_list, not_allowed_overlap_index): # use greedy algorithim to delete frags that generate not allowed overlaps
		not_allowed_frag_pairs = [frag_pair_list[i] for i in not_allowed_overlap_index]
		frag_list = [frag for frag_pairs in not_allowed_frag_pairs for frag in frag_pairs]
		frag_freq = {}
		for frag in frag_list:
			if frag  in frag_freq.keys():
				frag_freq[frag] += 1
			else:
				frag_freq[frag] = 1
		frag_set = list(set(frag_list))
		deleted_frag_list = []
		while(Pair_In_List(frag_set, not_allowed_frag_pairs)):
		 	deleted_frag = max(frag_freq, key=frag_freq.get)
			deleted_frag_list.append(deleted_frag)
			frag_set.pop(frag_set.index(deleted_frag))
			del frag_freq[deleted_frag]
		return	deleted_frag_list

	def Check_Frags_Is_Complete(self, frags_index_list):  # check the frags contains all the heavy atoms
		visited = []
		heavy_atoms = 0
		for frag_index in frags_index_list:
			for atom_index in frag_index:
				if self.atoms[atom_index]!=1 and atom_index not in visited:
					heavy_atoms += 1
					visited.append(atom_index)
		if heavy_atoms == self.Num_of_Heavy_Atom():
			return True
		else:
			return False

	def Num_of_Heavy_Atom(self):
		num = 0
		for i in range (0, self.NAtoms()):
			if self.atoms[i] != 1:
				num += 1
		return num

	def Check_Connection(self, mol_node, frag_node, mol_visited_list, frag_visited_list):  # the connection of mol_node should be the same as frag_node in the list we visited so far.
		mol_node_connection_index_found = []
		for node in mol_node.connected_nodes:
			if node.node_index in mol_visited_list:
				mol_node_connection_index_found.append(mol_visited_list.index(node.node_index))

		frag_node_connection_index_found = []
                for node in frag_node.connected_nodes:
                        if node.node_index in frag_visited_list:
                                frag_node_connection_index_found.append(frag_visited_list.index(node.node_index))

		if set(mol_node_connection_index_found) == set(frag_node_connection_index_found):
			return True
		else:
			return False

	def Compare_Node(self, mol_node, frag_node):
		if mol_node.node_type == frag_node.node_type and mol_node.num_of_bonds == frag_node.num_of_bonds  and Subset(mol_node.connected_atoms, frag_node.connected_atoms):
			if frag_node.undefined_bond_type == "heavy": #  check whether the dangling bond is connected to H in the mol
				if 1 in Setdiff(mol_node.connected_atoms, frag_node.connected_atoms):   # the dangling bond is connected to H
					return False
				else:
					return True
			else:
				return True
		else:
			return False

	def NoOverlapping_Partition(self, frags):
		frag_list = []
		for frag in frags:
			if not frag_list: # empty
				frag_list = list(self.Find_Frag(frag))
			else:
				frag_list += self.Find_Frag(frag)
		memory = dict()
		penalty, opt_frags = self.DP_Partition( range(0, self.NAtoms()),frag_list, memory)
		#penalty, opt_frags = self.Partition( range(0, self.NAtoms()),frag_list)
		print "penalty:", penalty, "opt_frags", opt_frags
		return

	def Partition(self, suffix_atoms, frag_list): # recursive partition
		possible_frags = []
		#print suffix_atoms
		for frag in frag_list:
                        if  Subset(suffix_atoms, frag):   # possible frag contained in the suffix
                                        possible_frags.append(frag)
                if possible_frags:    # continue partitioning
                	mini_penalty = float('inf')
                        opt_frags = []
                        for frag in possible_frags:
                        	 new_suffix_atoms = Setdiff(suffix_atoms, frag)
                                 penalty, prev_frags = self.Partition(new_suffix_atoms, frag_list) # recursive
                                 if penalty < mini_penalty:
                                 	mini_penalty = penalty
                                        opt_frags = list(prev_frags)
                                        opt_frags.append(frag)
                        return [mini_penalty, opt_frags]
             	else:   # could not partition anymore
                	return [(len(suffix_atoms))*(len(suffix_atoms)), [["left"]+suffix_atoms]]  # return penalty and what is left

	def DP_Partition(self, suffix_atoms, frag_list, memory):    # non-overlapping partition using dynamic programming
		#print "suffix_atoms", suffix_atoms
		#print "memory:", memory
		possible_frags = []
		suffix_atoms.sort()
		suffix_atoms_string = LtoS(suffix_atoms)
		if suffix_atoms_string in memory.keys():  # already calculated
			#print "using memory"
			return memory[suffix_atoms_string]
		else:   # not calculated yet
			#print "calculating"
			for frag in frag_list:
				if  Subset(suffix_atoms, frag):   # possible frag contained in the suffix
					possible_frags.append(frag)

			if possible_frags:    # continue partitioning
				mini_penalty = float('inf')
				opt_frags = []
				for frag in possible_frags:
					new_suffix_atoms = Setdiff(suffix_atoms, frag)
					penalty, prev_frags = self.DP_Partition(new_suffix_atoms, frag_list, memory) # recursive
					if penalty < mini_penalty:
						mini_penalty = penalty
						opt_frags = list(prev_frags)
						opt_frags.append(frag)
				memory[suffix_atoms_string] = [mini_penalty, opt_frags] #memorize it
				return [mini_penalty, opt_frags]
			else:	# could not partition anymore
				memory[suffix_atoms_string] = [(len(suffix_atoms))*(len(suffix_atoms)), [["left"]+suffix_atoms]]  # memorize it
				return [(len(suffix_atoms))*(len(suffix_atoms)), [["left"]+suffix_atoms]]  # return penalty and what is left

## ----------------------------------------------
## MBE routines:
## ----------------------------------------------

	def Reset_Frags(self):
		self.mbe_frags=dict()    # list of  frag of each order N, dic['N'=list of frags]
		self.mbe_frags_deri=dict()
		self.mbe_permute_frags=dict() # list of all the permuted frags
		self.mbe_frags_energy=dict()  # MBE energy of each order N, dic['N'= E_N]
		self.properties["energy"]=None
		self.mbe_energy=dict()   # sum of MBE energy up to order N, dic['N'=E_sum]
		self.properties["mbe_deri"] =None
		self.properties["nn_energy"]=None
		return

	def AtomName(self, i):
		return atoi.keys()[atoi.values().index(self.atoms[i])]

	def AtomName_From_List(self, atom_list):
		name = ""
		for i in atom_list:
			name += atoi.keys()[atoi.values().index(i)]
		return name

	def AllAtomNames(self):
		names=[]
		for i in range (0, self.atoms.shape[0]):
			names.append(atoi.keys()[atoi.values().index(self.atoms[i])])
		return names

	def Sort_frag_list(self):
		a=[]
		for dic in self.frag_list:
			a.append(len(dic["atom"]))
		self.frag_list = [x for (y,x) in sorted(zip(a,self.frag_list))]
		self.frag_list.reverse()
		return self.frag_list

	def Generate_All_Pairs(self, pair_list=[]):
		mono = []
		for pair in pair_list:
			for frag in pair['mono']:
				mono.append([atoi[atom] for atom in  String_To_Atoms(frag)])
		tmp = []
		for frag in mono:
			if frag not in tmp:
				tmp.append(frag)
		mono = tmp
		(mono.sort(key=lambda x:len(x)))
		mono.reverse()

		dic_mono = {}
		dic_mono_index = {}
		masked = []
		for frag_atoms in mono:
			frag_name = LtoS(frag_atoms)
			dic_mono[frag_name] = []
			dic_mono_index[frag_name] = []
			num_frag_atoms = len(frag_atoms)
			j = 0
			while (j < self.NAtoms()):
				if j in masked:
					j += 1
				else:
					tmp_list = list(self.atoms[j:j+num_frag_atoms])
					if tmp_list == frag_atoms:
						dic_mono[frag_name].append(self.coords[j:j+num_frag_atoms,:].copy())
						dic_mono_index[frag_name].append(range (j, j+num_frag_atoms))
						masked += range (j, j+num_frag_atoms)
						j += num_frag_atoms
					else:
						j += 1
		happy_atoms = []
		for pair in pair_list:
			happy_atoms = self.PairUp(dic_mono, dic_mono_index, pair, happy_atoms)  #it is amazing that the dictionary is passed by pointer...
		left_atoms = list(set(range (0, self.NAtoms())) - set(happy_atoms))
		sorted_atoms = happy_atoms + left_atoms
		self.atoms = self.atoms[sorted_atoms]
		self.coords = self.coords[sorted_atoms]
		return

	def PairUp(self, dic_mono, dic_mono_index, pair, happy_atoms):  # stable marriage pairing  Ref: https://en.wikipedia.org/wiki/Stable_marriage_problem
		mono_1 = LtoS([atoi[atom] for atom in  String_To_Atoms(pair["mono"][0])])
		mono_2 = LtoS([atoi[atom] for atom in  String_To_Atoms(pair["mono"][1])])

		center_1 = pair["center"][0]
		center_2 = pair["center"][1]

		switched = False
		if len(dic_mono[mono_1]) > len(dic_mono[mono_2]):
			mono_1, mono_2 = mono_2, mono_1
                        center_1, center_2  = center_2, center_1
			switched = True

		mono_1_pair = [-1]*len(dic_mono[mono_1])
		dist_matrix = np.zeros((len(dic_mono[mono_1]), len(dic_mono[mono_2])))
		for i in range (0, len(dic_mono[mono_1])):
			for j in range (0, len(dic_mono[mono_2])):
				dist_matrix[i][j] = np.linalg.norm(dic_mono[mono_1][i][center_1] - dic_mono[mono_2][j][center_2])

		mono_1_prefer = []
		mono_2_prefer = []
		for i in range (0, len(dic_mono[mono_1])):
			s = list(dist_matrix[i])
			mono_1_prefer.append(sorted(range(len(s)), key=lambda k: s[k]))
		for i in range (0, len(dic_mono[mono_2])):
                        s = list(dist_matrix[:,i])
                        mono_2_prefer.append(sorted(range(len(s)), key=lambda k: s[k]))


		mono_1_info = [-1]*len(dic_mono[mono_1]) # -1 means they are not paired, and the number means the Nth most prefered are chosen
		mono_2_info = [-1]*len(dic_mono[mono_2])

		mono_1_history = [0]*len(dic_mono[mono_1]) # history of the man's proposed

		# first round  mono_1 is the man, mono_2 is woman,  num of man > num of woman
		for i in range (0, len(dic_mono[mono_1])):
			target = mono_1_prefer[i][0]
			if i == mono_2_prefer[target][0]:  # Congs! find true lovers
				mono_1_info[i] = 0
				mono_2_info[target] = 0
			mono_1_history[i] += 1

		while (-1 in mono_1_info):
			for i in range (0, len(dic_mono[mono_1])):
				if mono_1_info[i] == -1:
					target = mono_1_prefer[i][mono_1_history[i]] # propose
					if mono_2_info[target] == -1: # met a single woman
						mono_1_info[i] = mono_1_history[i]
						mono_2_info[target] = mono_2_prefer[target].index(i)
						mono_1_history[i] += 1
					elif mono_2_info[target] > mono_2_prefer[target].index(i):   # this man is the better choice than the previous one
						poorguy = mono_2_prefer[target][mono_2_info[target]]
						mono_1_info[poorguy] = -1   # this poor guy is abandoned...
						mono_1_info[i] = mono_1_history[i]
						mono_2_info[target] = mono_2_prefer[target].index(i)
						mono_1_history[i] += 1
					else:
						mono_1_history[i] += 1
						continue
				else:
					continue

		final_pairs = []
		for i in range (0, len(dic_mono[mono_1])):
			final_pairs.append([i, mono_1_prefer[i][mono_1_info[i]]])


		for i in range (0, len(final_pairs)):
			if switched == False:
				#print dic_mono_index[mono_1][final_pairs[i][0]], dic_mono_index[mono_2][final_pairs[i][1]]
				happy_atoms += dic_mono_index[mono_1][final_pairs[i][0]]
				happy_atoms += dic_mono_index[mono_2][final_pairs[i][1]]
			else:
				happy_atoms += dic_mono_index[mono_2][final_pairs[i][1]]
                                happy_atoms += dic_mono_index[mono_1][final_pairs[i][0]]

		indices_1 = [item[0] for item in final_pairs]
		indices_2 = [item[1] for item in final_pairs]

		dic_mono_index[mono_1] = [i for j, i in enumerate(dic_mono_index[mono_1]) if j not in indices_1]
		dic_mono_index[mono_2] = [i for j, i in enumerate(dic_mono_index[mono_2]) if j not in indices_2]
		dic_mono[mono_1] = [i for j, i in enumerate(dic_mono[mono_1]) if j not in indices_1]
                dic_mono[mono_2] = [i for j, i in enumerate(dic_mono[mono_2]) if j not in indices_2]
		#print dic_mono_index[mono_1], dic_mono_index[mono_2], happy_atoms
		return happy_atoms

	def Generate_All_MBE_term_General(self, frag_list=[], cutoff=10, center_atom=[]):
		self.frag_list = frag_list
		#self.Sort_frag_list()  # debug, not sure it is necessary
		if center_atom == []:
			center_atom = [0]*len(frag_list)
                for i in range (1, self.mbe_order+1):
                        self.Generate_MBE_term_General(i, cutoff, center_atom)
                return

	def Generate_MBE_term_General(self, order,  cutoff=10, center_atom=[]):
                if order in self.mbe_frags.keys():
                        print ("MBE order", order, "already generated..skipping..")
                        return
		if order==1:
			self.mbe_frags[order] = []
			masked=[]
			frag_index = 0
			for i, dic in enumerate(self.frag_list):
				self.type_of_frags_dict[i] = []
				frag_atoms = String_To_Atoms(dic["atom"])
				frag_atoms = [atoi[atom] for atom in frag_atoms]
				num_frag_atoms = len(frag_atoms)
				j = 0
				while (j < self.NAtoms()):
					if j in masked:
						j += 1
					else:
						tmp_list = list(self.atoms[j:j+num_frag_atoms])
						if tmp_list == frag_atoms:
							self.atoms_of_frags.append([])
							masked += range (j, j+num_frag_atoms)
						 	self.atoms_of_frags[-1]=range (j, j+num_frag_atoms)
							self.type_of_frags.append(i)
							self.type_of_frags_dict[i].append(frag_index)

							tmp_coord = self.coords[j:j+num_frag_atoms,:].copy()
							tmp_atom  = self.atoms[j:j+num_frag_atoms].copy()
							mbe_terms = [frag_index]
							mbe_dist = None
							atom_group = [num_frag_atoms]
							dic['num_electron'] = sum(list(tmp_atom))-dic['charge']
							frag_type = [dic]
							frag_type_index = [i]
							tmp_mol = Frag(tmp_atom, tmp_coord, mbe_terms, mbe_dist, atom_group, frag_type, frag_type_index, FragOrder_=order)
							self.mbe_frags[order].append(tmp_mol)

							j += num_frag_atoms
							frag_index += 1
							#print self.atoms_of_frags, tmp_list, self.type_of_frags
							#print self.mbe_frags[order][-1].atoms, self.mbe_frags[order][-1].coords, self.mbe_frags[order][-1].index
						else:
							j += 1
		else:
			num_of_each_frag = {}
			frag_list_length = len(self.frag_list)
			frag_list_index = range (0, frag_list_length)
			frag_list_index_list = list(itertools.product(frag_list_index, repeat=order))
			tmp_index_list = []
			for i in range (0, len(frag_list_index_list)):
				tmp_index = list(frag_list_index_list[i])
				tmp_index.sort()
				if tmp_index not in tmp_index_list:
					tmp_index_list.append(tmp_index)
					num_of_each_frag[LtoS(tmp_index)]=0


			self.mbe_frags[order] = []
			mbe_terms=[]
                	mbe_dist=[]
			ngroup = len(self.mbe_frags[1])	#
			atomlist=list(range(0,ngroup))
			time_log = time.time()

                        print ("generating the combinations for order: ", order)
			max_case = 5000

			time_now=time.time()
			for index_list in tmp_index_list:
				frag_case = 0
				sample_index = []
				for i in index_list:
					sample_index.append(self.type_of_frags_dict[i])

				print("begin the most time consuming step: ")
				tmp_time  = time.time()
				sub_combinations = list(itertools.product(*sample_index))
				print ("end of the most time consuming step. time cost:", time.time() - tmp_time)
				shuffle_time = time.time()
				new_begin = random.randint(1,len(sub_combinations)-2)
				sub_combinations = sub_combinations[new_begin:]+sub_combinations[:new_begin] # debug, random shuffle the list, so the pairs are chosen randomly, this is not necessary for generate training cases
				#random.shuffle(sub_combinations)  # debug, random shuffle the list, so the pairs are chosen randomly, this is not necessary for generate training cases
				print  "time to shuffle it", time.time()-shuffle_time
				for i in range (0, len(sub_combinations)):
                        	        term = list(sub_combinations[i])
					if len(list(set(term))) < len(term):
						continue
                        	        pairs=list(itertools.combinations(term, 2))
                        	        saveindex=[]
                        	        dist = [10000000]*len(pairs)
                        	        flag=1
                        	        npairs=len(pairs)
                        	        for j in range (0, npairs):
                        	                #print self.type_of_frags[pairs[j][0]], self.type_of_frags[pairs[j][1]], pairs[j][0], pairs[j][1]
                        	                if self.type_of_frags[pairs[j][0]] == -1 :
                        	                        center_1 = self.Center()
                        	                else:
                        	                        center_1 = self.mbe_frags[1][pairs[j][0]].coords[center_atom[self.type_of_frags[pairs[j][0]]]]

                        	                if self.type_of_frags[pairs[j][1]] == -1 :
                        	                        center_2 = self.Center()
                        	                else:
                        	                        center_2 = self.mbe_frags[1][pairs[j][1]].coords[center_atom[self.type_of_frags[pairs[j][1]]]]
                        	                dist[j] = np.linalg.norm(center_1- center_2)
                        	                if dist[j] > cutoff:
                        	                        flag = 0
                        	                        break
                        	        if flag == 1:   # we find a frag
						if frag_case%100==0:
							print "working on frag:", frag_case, "frag_type:", index_list, " i:", i
						frag_case  += 1
                        	                if  frag_case >=  max_case:   # just for generating training case
                        	                        break;
                        	                mbe_terms.append(term)
                        	                mbe_dist.append(dist)

                        print ("finished..takes", time_log-time.time(),"second")

			mbe_frags = []
			for i in range (0, len(mbe_terms)):
				frag_type = []
				frag_type_index = []
				atom_group = []
				for index in mbe_terms[i]:
					frag_type.append(self.frag_list[self.type_of_frags[index]])
					frag_type_index.append(self.type_of_frags[index])
					atom_group.append(self.mbe_frags[1][index].atoms.shape[0])
				tmp_coord = np.zeros((sum(atom_group), 3))
				tmp_atom = np.zeros(sum(atom_group), dtype=np.uint8)
				pointer = 0
				for j, index in enumerate(mbe_terms[i]):
					tmp_coord[pointer:pointer+atom_group[j],:] = self.mbe_frags[1][index].coords
					tmp_atom[pointer:pointer+atom_group[j]] = self.mbe_frags[1][index].atoms
					pointer += atom_group[j]
				tmp_mol = Frag(tmp_atom, tmp_coord, mbe_terms[i], mbe_dist[i], atom_group, frag_type, frag_type_index, FragOrder_=order)
                                self.mbe_frags[order].append(tmp_mol)
			del sub_combinations
		return

	def Generate_All_MBE_term(self,  atom_group=1, cutoff=10, center_atom=0, max_case=1000000):
		for i in range (1, self.mbe_order+1):
			self.Generate_MBE_term(i, atom_group, cutoff, center_atom, max_case)
		return

	def Generate_MBE_term(self, order,  atom_group=1, cutoff=10, center_atom=0, max_case=1000000):
		if order in self.mbe_frags.keys():
			print ("MBE order", order, "already generated..skipping..")
			return
		if (self.coords).shape[0]%atom_group!=0:
			raise Exception("check number of group size")
		else:
			ngroup = (self.coords).shape[0]/atom_group
		xyz=((self.coords).reshape((ngroup, atom_group, -1))).copy()     # cluster/molecule needs to be arranged with molecule/sub_molecule
		ele=((self.atoms).reshape((ngroup, atom_group))).copy()
		mbe_terms=[]
		mbe_terms_num=0
		mbe_dist=[]
		atomlist=list(range(0,ngroup))
		if order < 1 :
			raise Exception("MBE Order Should be Positive")
		else:
			time_log = time.time()
			print ("generating the combinations for order: ", order)
			combinations=list(itertools.combinations(atomlist,order))
			print ("finished..takes", time_log-time.time(),"second")
		time_now=time.time()
		flag = np.zeros(1)
		for i in range (0, len(combinations)):
			term = list(combinations[i])
			pairs=list(itertools.combinations(term, 2))
			saveindex=[]
			dist = [10000000]*len(pairs)
			#flag = 1
			#for j in range (0, len(pairs)):
			#	m=pairs[j][0]
			#	n=pairs[j][1]
			#	#dist[j] = np.linalg.norm(xyz[m]-xyz[n])
			#	dist[j]=((xyz[m][center_atom][0]-xyz[n][center_atom][0])**2+(xyz[m][center_atom][1]-xyz[n][center_atom][1])**2+(xyz[m][center_atom][2]-xyz[n][center_atom][2])**2)**0.5
			#	if dist[j] > cutoff:
			#		flag = 0
			#		break
			#if flag == 1:
			flag[0]=1
			npairs=len(pairs)
			code="""
			for (int j=0; j<npairs; j++) {
				int m = pairs[j][0];
				int n = pairs[j][1];
				dist[j] = sqrt(pow(xyz[m*atom_group*3+center_atom*3+0]-xyz[n*atom_group*3+center_atom*3+0],2)+pow(xyz[m*atom_group*3+center_atom*3+1]-xyz[n*atom_group*3+center_atom*3+1],2)+pow(xyz[m*atom_group*3+center_atom*3+2]-xyz[n*atom_group*3+center_atom*3+2],2));
				if (float(dist[j]) > cutoff) {
					flag[0] = 0;
					break;
				}
			}

			"""
			res = inline(code, ['pairs','npairs','center_atom','dist','xyz','flag','cutoff','atom_group'],headers=['<math.h>','<iostream>'], compiler='gcc')
			if flag[0]==1:  # end of weave
				if mbe_terms_num%100==0:
					print mbe_terms_num, time.time()-time_now
					time_now= time.time()
				mbe_terms_num += 1
				mbe_terms.append(term)
				mbe_dist.append(dist)
				if mbe_terms_num >=  max_case:   # just for generating training case
					break;
		mbe_frags = []
		for i in range (0, mbe_terms_num):
			tmp_atom = np.zeros(order*atom_group)
			tmp_coord = np.zeros((order*atom_group, 3))
			for j in range (0, order):
				tmp_atom[atom_group*j:atom_group*(j+1)] = ele[mbe_terms[i][j]]
				tmp_coord[atom_group*j:atom_group*(j+1)] = xyz[mbe_terms[i][j]]
			tmp_mol = Frag(tmp_atom, tmp_coord, mbe_terms[i], mbe_dist[i], atom_group)
			mbe_frags.append(tmp_mol)
		self.mbe_frags[order]=mbe_frags
		print "generated {:10d} terms for order {:d}".format(len(mbe_frags), order)
		del combinations[:]
		del combinations
		return mbe_frags

	def Calculate_Frag_Energy_General(self, order, method="pyscf"):
                if order in self.mbe_frags_energy.keys():
                        print ("MBE order", order, "already calculated..skipping..")
                        return 0
                mbe_frags_energy = 0.0
                fragnum=0
                time_log=time.time()
                print "length of order ", order, ":",len(self.mbe_frags[order])
                if method == "qchem":
                        order_path = self.properties["qchem_data_path"]+"/"+str(order)
                        if not os.path.isdir(order_path):
                                os.mkdir(order_path)
                        os.chdir(order_path)
			time0 =time.time()
                        for frag in self.mbe_frags[order]:  # just for generating the training set..
                                fragnum += 1
				if fragnum%100 == 0:
                               		print "working on frag:", fragnum
					print  "total time:", time.time() - time0
					time0 = time.time()
                                frag.Write_Qchem_Frag_MBE_Input_All_General(fragnum)
                        os.chdir("../../../../")
                elif method == "pyscf":
			raise Exception("PyScf for MBE General has not implemented yet, please use qchem")
                else:
                        raise Exception("unknow ab-initio software!")
                return

	def Calculate_Frag_Energy(self, order, method="pyscf"):
		if order in self.mbe_frags_energy.keys():
			print ("MBE order", order, "already calculated..skipping..")
			return 0
		mbe_frags_energy = 0.0
		fragnum=0
		time_log=time.time()
		print "length of order ", order, ":",len(self.mbe_frags[order])
		if method == "qchem":
			order_path = self.properties["qchem_data_path"]+"/"+str(order)
			if not os.path.isdir(order_path):
				os.mkdir(order_path)
			os.chdir(order_path)
			for frag in self.mbe_frags[order]:  # just for generating the training set..
				fragnum += 1
				print "working on frag:", fragnum
				frag.Write_Qchem_Frag_MBE_Input_All(fragnum)
			os.chdir("../../../../")
   		elif method == "pyscf":
			for frag in self.mbe_frags[order]:  # just for generating the training set..
				fragnum +=1
				print "doing the ",fragnum
				frag.PySCF_Frag_MBE_Energy_All()
				frag.Set_Frag_MBE_Energy()
				mbe_frags_energy += frag.frag_mbe_energy
				print "Finished, spent ", time.time()-time_log," seconds"
				time_log = time.time()
			self.mbe_frags_energy[order] = mbe_frags_energy
		else:
			raise Exception("unknow ab-initio software!")
		return

	def Get_Qchem_Frag_Energy(self, order):
		fragnum = 0
		path = self.properties["qchem_data_path"]+"/"+str(order)
		mbe_frags_energy = 0.0
		for frag in self.mbe_frags[order]:
			fragnum += 1
			frag.Get_Qchem_Frag_MBE_Energy_All(fragnum, path)
			print "working on molecule:", self.name," frag:",fragnum, " order:",order
			frag.Set_Frag_MBE_Energy()
			mbe_frags_energy += frag.frag_mbe_energy
			#if order==2:
		#		print frag.frag_mbe_energy, frag.dist[0]
		self.mbe_frags_energy[order] = mbe_frags_energy
		return

	def Get_All_Qchem_Frag_Energy_General(self):
		self.Get_All_Qchem_Frag_Energy()
                return

	def Get_All_Qchem_Frag_Energy(self):
		#for i in range (1, 3):  # set to up to 2nd order for debug sake
		for i in range (1, self.mbe_order+1):
			#print "getting the qchem energy for MBE order", i
			self.Get_Qchem_Frag_Energy(i)
		return

	def Set_Qchem_Data_Path(self):
		self.properties["qchem_data_path"]="./qchem"+"/"+self.set_name+"/"+self.name
		return

	def Calculate_All_Frag_Energy_General(self, method="pyscf"):
                if method == "qchem":
                        if not os.path.isdir("./qchem"):
                                os.mkdir("./qchem")
                        if not os.path.isdir("./qchem"+"/"+self.set_name):
                                os.mkdir("./qchem"+"/"+self.set_name)
                        self.properties["qchem_data_path"]="./qchem"+"/"+self.set_name+"/"+self.name
                        if not os.path.isdir(self.properties["qchem_data_path"]):
                                os.mkdir(self.properties["qchem_data_path"])
                for i in range (1, self.mbe_order+1):
                        print "calculating for MBE order", i
                        self.Calculate_Frag_Energy_General(i, method)
                if method == "qchem":
                        self.Write_Qchem_Submit_Script()
                #print "mbe_frags_energy", self.mbe_frags_energy
                return

	def Calculate_All_Frag_Energy(self, method="pyscf"):  # we ignore the 1st order for He here
		if method == "qchem":
			if not os.path.isdir("./qchem"):
                                os.mkdir("./qchem")
			if not os.path.isdir("./qchem"+"/"+self.set_name):
                                os.mkdir("./qchem"+"/"+self.set_name)
                        self.properties["qchem_data_path"]="./qchem"+"/"+self.set_name+"/"+self.name
			if not os.path.isdir(self.properties["qchem_data_path"]):
                                os.mkdir(self.properties["qchem_data_path"])
		for i in range (1, self.mbe_order+1):
			print "calculating for MBE order", i
			self.Calculate_Frag_Energy(i, method)
		if method == "qchem":
			self.Write_Qchem_Submit_Script()
		#print "mbe_frags_energy", self.mbe_frags_energy
		return

	def Write_Qchem_Submit_Script(self):     # this is for submitting the jobs on notre dame crc
		if not os.path.isdir("./qchem"):
			os.mkdir("./qchem")
			if not os.path.isdir("./qchem"+"/"+self.set_name):
				os.mkdir("./qchem"+"/"+self.set_name)
                self.properties["qchem_data_path"]="./qchem"+"/"+self.set_name+"/"+self.name
		if not os.path.isdir(self.properties["qchem_data_path"]):
			os.mkdir(self.properties["qchem_data_path"])
		os.chdir(self.properties["qchem_data_path"])
		for i in range (1, self.mbe_order+1):
			num_frag = len(self.mbe_frags[i])
			for j in range (1, i+1):
				index=nCr(i, j)
				for k in range (1, index+1):
					submit_file = open("qchem_order_"+str(i)+"_suborder_"+str(j)+"_index_"+str(k)+".sub","w+")
					lines = Submit_Script_Lines(order=str(i), sub_order =str(j), index=str(k), mincase = str(1), maxcase = str(num_frag), name = "MBE_"+str(i)+"_"+str(j)+"_"+str(index), ncore = str(4), queue="long")
					submit_file.write(lines)
					submit_file.close()

		python_submit = open("submit_all.py","w+")
		line = 'import os,sys\n\nfor file in os.listdir("."):\n        if file.endswith(".sub"):\n                cmd = "qsub "+file\n                os.system(cmd)\n'
		python_submit.write(line)
		python_submit.close()
		os.chdir("../../../")
		return

	def Set_MBE_Energy(self):
		for i in range (1, self.mbe_order+1):
			self.mbe_energy[i] = 0.0
			for j in range (1, i+1):
				self.mbe_energy[i] += self.mbe_frags_energy[j]
		return

	def MBE(self,  atom_group=1, cutoff=10, center_atom=0, max_case = 1000000):
		self.Generate_All_MBE_term(atom_group, cutoff, center_atom, max_case)
		self.Calculate_All_Frag_Energy()
		self.Set_MBE_Energy()
		print self.mbe_frags_energy
		return

	def PySCF_Energy(self, basis_='cc-pvqz'):
		mol = gto.Mole()
		pyscfatomstring=""
		for j in range(len(self.atoms)):
			s = self.coords[j]
			pyscfatomstring=pyscfatomstring+str(self.AtomName(j))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+(";" if j!= len(self.atoms)-1 else "")
		mol.atom = pyscfatomstring
		mol.basis = basis_
		mol.verbose = 0
		try:
			mol.build()
			mf=scf.RHF(mol)
			hf_en = mf.kernel()
			mp2 = mp.MP2(mf)
			mp2_en = mp2.kernel()
			en = hf_en + mp2_en[0]
			self.properties["energy"] = en
			return en
		except Exception as Ex:
				print "PYSCF Calculation error... :",Ex
				print "Mol.atom:", mol.atom
				print "Pyscf string:", pyscfatomstring
				return 0.0
				#raise Ex
		return

	def Get_Permute_Frags(self, indis=[0]):
		self.mbe_permute_frags=dict()
		for order in self.mbe_frags.keys():
		#   if order <= 2:  # for debug purpose
			self.mbe_permute_frags[order]=list()
			for frags in self.mbe_frags[order]:
				self.mbe_permute_frags[order] += frags.Permute_Frag( indis  )
			print "length of permuted frags:", len(self.mbe_permute_frags[order]),"order:", order
		return

	def Set_Frag_Force_with_Order(self, cm_deri, nn_deri, order):
		self.mbe_frags_deri[order]=np.zeros((self.NAtoms(),3))
		atom_group = self.mbe_frags[order][0].atom_group  # get the number of  atoms per group by looking at the frags.
		for i in range (0, len(self.mbe_frags[order])):
			deri = self.mbe_frags[order][i].Frag_Force(cm_deri[i], nn_deri[i])
			deri = deri.reshape((order, deri.shape[0]/order, -1))
			index_list = self.mbe_frags[order][i].index
			for j in range (0,  len(index_list)):
				self.mbe_frags_deri[order][index_list[j]*atom_group:(index_list[j]+1)*atom_group] += deri[j]
		return

	def Set_MBE_Force(self):
		self.properties["mbe_deri"] = np.zeros((self.NAtoms(), 3))
		for order in range (1, self.mbe_order+1): # we ignore the 1st order term since we are dealing with helium, debug
			if order in self.mbe_frags_deri.keys():
				self.properties["mbe_deri"] += self.mbe_frags_deri[order]
		return self.properties["mbe_deri"]

class Frag(Mol):
	""" Provides a MBE frag of general purpose molecule"""
	def __init__(self, atoms_ =  None, coords_ = None, index_=None, dist_=None, atom_group_=1, frag_type_=None, frag_type_index_=None, FragOrder_=None):
		Mol.__init__(self, atoms_, coords_)
		self.atom_group = atom_group_
		if FragOrder_==None:
			self.FragOrder = self.coords.shape[0]/self.atom_group
		else:
			self.FragOrder = FragOrder_
		if (index_!=None):
			self.index = index_
		else:
			self.index = None
		if (dist_!=None):
			self.dist = dist_
		else:
			self.dist = None
		if (frag_type_!=None):
			self.frag_type = frag_type_
		else:
			self.frag_type = None
		if (frag_type_!=None):
                        self.frag_type_index = frag_type_index_
                else:
                        self.frag_type_index = None
		self.frag_mbe_energies=dict()
		self.frag_mbe_energy = None
		self.frag_energy = None
		self.permute_index = range (0, self.FragOrder)
		self.permute_sub_index = None
		return

	def PySCF_Frag_MBE_Energy(self,order):   # calculate the MBE of order N of each frag
		inner_index = range(0, self.FragOrder)
		real_frag_index=list(itertools.combinations(inner_index,order))
		ghost_frag_index=[]
		for i in range (0, len(real_frag_index)):
			ghost_frag_index.append(list(set(inner_index)-set(real_frag_index[i])))

		i =0
		while(i< len(real_frag_index)):
			#for i in range (0, len(real_frag_index)):
			pyscfatomstring=""
			mol = gto.Mole()
			for j in range (0, order):
				for k in range (0, self.atom_group):
					s = self.coords[real_frag_index[i][j]*self.atom_group+k]
					pyscfatomstring=pyscfatomstring+str(self.AtomName(real_frag_index[i][j]*self.atom_group+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+";"
			for j in range (0, self.FragOrder - order):
				for k in range (0, self.atom_group):
					s = self.coords[ghost_frag_index[i][j]*self.atom_group+k]
					pyscfatomstring=pyscfatomstring+"GHOST"+str(j*self.atom_group+k)+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+";"
			pyscfatomstring=pyscfatomstring[:-1]+"  "
			mol.atom =pyscfatomstring

			mol.basis ={}
			ele_set = list(set(self.AllAtomNames()))
			for ele in ele_set:
				mol.basis[str(ele)]="cc-pvqz"

			for j in range (0, self.FragOrder - order):
				for k in range (0, self.atom_group):
					atom_type = self.AtomName(ghost_frag_index[i][j]*self.atom_group+k)
					mol.basis['GHOST'+str(j*self.atom_group+k)]=gto.basis.load('cc-pvqz',str(atom_type))
			mol.verbose=0
			try:
				print "doing case ", i
				time_log = time.time()
				mol.build()
				mf=scf.RHF(mol)
				hf_en = mf.kernel()
				mp2 = mp.MP2(mf)
				mp2_en = mp2.kernel()
				en = hf_en + mp2_en[0]
				#print "hf_en", hf_en, "mp2_en", mp2_en[0], " en", en
				self.frag_mbe_energies[LtoS(real_frag_index[i])]=en
				print ("pyscf time..", time.time()-time_log)
				i = i+1
				gc.collect()
			except Exception as Ex:
				print "PYSCF Calculation error... :",Ex
				print "Mol.atom:", mol.atom
				print "Pyscf string:", pyscfatomstring
		return 0

	def Get_Qchem_Frag_MBE_Energy(self, order, path):
		#print "path:", path, "order:", order
		onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		#print "onlyfiles:", onlyfiles, "path", path, "order", order
		for outfile_name in onlyfiles:
			if ( outfile_name[-4:]!='.out' ):
					continue
			outfile = open(path+"/"+outfile_name,"r+")
			outfile_lines = outfile.readlines()
			key = None
			rimp2 = None
			for line in outfile_lines:
				if "!" in line:
					key = line[1:-1]
					continue
				if "non-Brillouin singles" in line:
					nonB_single = float(line.split()[3])
					continue
				if "RIMP2         total energy" in line:
					rimp2 = float(line.split()[4])
					continue
				if "fatal error" in line:
					print "fata error! file:", path+"/"+outfile_name
			if nonB_single != 0.0:
				print "Warning: non-Brillouin singles do not equal to zero, non-Brillouin singles=",nonB_single,path,outfile_name
			if key!=None and rimp2!=None:
				#print "key:", key, "length:", len(key)
				self.frag_mbe_energies[key] = rimp2
			else:
				print "Qchem Calculation error on ",path,outfile_name
				raise Exception("Qchem Error")
		return

	def Write_Qchem_Frag_MBE_Input_General(self,order):   # calculate the MBE of order N of each frag
                inner_index = range(0, self.FragOrder)
                real_frag_index=list(itertools.combinations(inner_index,order))
                ghost_frag_index=[]
                for i in range (0, len(real_frag_index)):
                        ghost_frag_index.append(list(set(inner_index)-set(real_frag_index[i])))
                i =0
                while(i< len(real_frag_index)):
			charge = 0
			num_ele = 0
			for j in range (0, order):
				charge += self.frag_type[real_frag_index[i][j]]["charge"]
				num_ele += self.frag_type[real_frag_index[i][j]]["num_electron"]
			if num_ele%2 == 0:   # here we always prefer the low spin state
				spin = 1
			else:
				spin = 2

                        qchemstring="$molecule\n"+str(charge)+" "+str(spin)+"\n"
                        for j in range (0, order):
				pointer = sum(self.atom_group[:real_frag_index[i][j]])
                                for k in range (0, self.atom_group[real_frag_index[i][j]]):
                                        s = self.coords[pointer+k]
                                        qchemstring+=str(self.AtomName(pointer+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+"\n"
                        for j in range (0, self.FragOrder - order):
				pointer = sum(self.atom_group[:ghost_frag_index[i][j]])
                                for k in range (0, self.atom_group[ghost_frag_index[i][j]]):
                                        s = self.coords[pointer+k]
                                        qchemstring+="@"+str(self.AtomName(pointer+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+"\n"
                        qchemstring += "$end\n"
                        qchemstring += "!"+LtoS(real_frag_index[i])+"\n"
                        qchemstring += Qchem_RIMP2_Block
                        qchem_input=open(str(i+1)+".in","w+")
                        qchem_input.write(qchemstring)
                        qchem_input.close()
                        i = i+1
                #gc.collect()  # speed up the function by 1000 times just deleting this single line!
                return

	def Write_Qchem_Frag_MBE_Input(self,order):   # calculate the MBE of order N of each frag
		inner_index = range(0, self.FragOrder)
		real_frag_index=list(itertools.combinations(inner_index,order))
		ghost_frag_index=[]
		for i in range (0, len(real_frag_index)):
			ghost_frag_index.append(list(set(inner_index)-set(real_frag_index[i])))
		i =0
		while(i< len(real_frag_index)):
			qchemstring="$molecule\n0  1\n"
			for j in range (0, order):
				for k in range (0, self.atom_group):
					s = self.coords[real_frag_index[i][j]*self.atom_group+k]
					qchemstring+=str(self.AtomName(real_frag_index[i][j]*self.atom_group+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+"\n"
			for j in range (0, self.FragOrder - order):
				for k in range (0, self.atom_group):
					s = self.coords[ghost_frag_index[i][j]*self.atom_group+k]
					qchemstring+="@"+str(self.AtomName(ghost_frag_index[i][j]*self.atom_group+k))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+"\n"
			qchemstring += "$end\n"
			qchemstring += "!"+LtoS(real_frag_index[i])+"\n"
			qchemstring += Qchem_RIMP2_Block
			qchem_input=open(str(i+1)+".in","w+")
			qchem_input.write(qchemstring)
			qchem_input.close()
			i = i+1
		gc.collect()
		return

	def Write_Qchem_Frag_MBE_Input_All_General(self, fragnum):
                if not os.path.isdir(str(fragnum)):
                        os.mkdir(str(fragnum))
                os.chdir(str(fragnum))
                for i in range (0, self.FragOrder):
                        if not os.path.isdir(str(i+1)):
                                os.mkdir(str(i+1))
                        os.chdir(str(i+1))
                        self.Write_Qchem_Frag_MBE_Input_General(i+1)
                        os.chdir("..")
                os.chdir("..")
                return

	def Write_Qchem_Frag_MBE_Input_All(self, fragnum):
		if not os.path.isdir(str(fragnum)):
			os.mkdir(str(fragnum))
		os.chdir(str(fragnum))
		for i in range (0, self.FragOrder):
			if not os.path.isdir(str(i+1)):
				os.mkdir(str(i+1))
			os.chdir(str(i+1))
			self.Write_Qchem_Frag_MBE_Input(i+1)
			os.chdir("..")
		os.chdir("..")
		return

	def Get_Qchem_Frag_MBE_Energy_All(self, fragnum, path):
		if not os.path.isdir(path+"/"+str(fragnum)):
			raise Exception(path+"/"+str(fragnum),"is not calculated")
		oldpath = path
		for i in range (0, self.FragOrder):
			path = oldpath+"/"+str(fragnum)+"/"+str(i+1)
			self.Get_Qchem_Frag_MBE_Energy(i+1, path)
		return

	def PySCF_Frag_MBE_Energy_All(self):
		for i in range (0, self.FragOrder):
			self.PySCF_Frag_MBE_Energy(i+1)
		return  0

	def Set_Frag_MBE_Energy(self):
		self.frag_mbe_energy =  self.Frag_MBE_Energy()
		self.frag_energy = self.frag_mbe_energies[LtoS(self.permute_index)]
		print "self.frag_type: ", self.frag_type
		print "self.frag_mbe_energy: ", self.frag_mbe_energy
		#prod = 1
		#for i in self.dist:
		#	prod = i*prod
		#print "self.frag_mbe_energy", self.frag_mbe_energy
		return 0

	def Frag_MBE_Energy(self,  index=None):     # Get MBE energy recursively
		if index==None:
			index=range(0, self.FragOrder)
		order = len(index)
		if order==0:
			return 0
		energy = self.frag_mbe_energies[LtoS(index)]
		for i in range (0, order):
			sub_index = list(itertools.combinations(index, i))
			for j in range (0, len(sub_index)):
				try:
					energy=energy-self.Frag_MBE_Energy( sub_index[j])
				except Exception as Ex:
					print "missing frag energy, error", Ex
		return  energy

	def CopyTo(self, target):
		target.FragOrder = self.FragOrder
		target.frag_mbe_energies=self.frag_mbe_energies
		target.frag_mbe_energy = self.frag_mbe_energy
		target.frag_energy = self.frag_energy
		target.permute_index = self.permute_index

	def Permute_Frag_by_Index(self, index, indis=[0]):
		new_frags=list()
		inner_index = Binominal_Combination(indis, self.FragOrder)
		#print "inner_index",inner_index
		for sub_index in inner_index:
			new_frag = Frag( atoms_ =  self.atoms, coords_ = self.coords, index_= self.index, dist_=self.dist, atom_group_=self.atom_group)
			self.CopyTo(new_frag)
			new_frag.permute_index = index
			new_frag.permute_sub_index = sub_index
			new_frag.coords=new_frag.coords.reshape((new_frag.FragOrder, new_frag.atom_group,  -1))
			new_frag.coords = new_frag.coords[new_frag.permute_index]
			new_frag.atoms = new_frag.atoms.reshape((new_frag.FragOrder, new_frag.atom_group))
			new_frag.atoms = new_frag.atoms[new_frag.permute_index]
			for group in range (0, new_frag.FragOrder):
				new_frag.coords[group][sorted(sub_index[group*len(indis):(group+1)*len(indis)])] = new_frag.coords[group][sub_index[group*len(indis):(group+1)*len(indis)]]
				new_frag.atoms[group][sorted(sub_index[group*len(indis):(group+1)*len(indis)])] = new_frag.atoms[group][sub_index[group*len(indis):(group+1)*len(indis)]]
			new_frag.coords = new_frag.coords.reshape((new_frag.FragOrder*new_frag.atom_group, -1))
			new_frag.atoms = new_frag.atoms.reshape(new_frag.FragOrder*new_frag.atom_group)
			#print "coords:", new_frag.coords, "atom:",new_frag.atoms
			new_frags.append(new_frag)
		# needs some code that fix the keys in frag_mbe_energies[LtoS(index)] after permutation in futher.  KY
		return new_frags

	def Permute_Frag(self, indis = [0]):
		permuted_frags=[]
		indexs=list(itertools.permutations(range(0, self.FragOrder)))
		for index in indexs:
			permuted_frags += self.Permute_Frag_by_Index(list(index), indis)
			#print permuted_frags[-1].atoms, permuted_frags[-1].coords
		return permuted_frags

	def Frag_Force(self, cm_deri, nn_deri):
		return self.Combine_CM_NN_Deri(cm_deri, nn_deri)

	def Combine_CM_NN_Deri(self, cm_deri, nn_deri):
		natom = self.NAtoms()
		frag_deri = np.zeros((natom, 3))
		for i in range (0, natom):  ## debug, this is for not including the diagnol
			for j in range (0, natom):  # debug, this is for not including the diagnol
				if j >= i:
					cm_dx = cm_deri[i][j][0]
					cm_dy = cm_deri[i][j][1]
					cm_dz = cm_deri[i][j][2]
					nn_deri_index = i*(natom+natom-i-1)/2 + (j-i-1) # debug, this is for not including the diagnol
					#nn_deri_index = i*(natom+natom-i+1)/2 + (j-i)  # debug, this is for including the diagnol in the CM
					nn_dcm = nn_deri[nn_deri_index]
				else:
					cm_dx = cm_deri[j][i][3]
					cm_dy = cm_deri[j][i][4]
					cm_dz = cm_deri[j][i][5]
					nn_deri_index = j*(natom+natom-j-1)/2 + (i-j-1)  #debug , this is for not including the diangol
					#nn_deri_index = j*(natom+natom-j+1)/2 + (i-j)    # debug, this is for including the diagnoal in the CM
					nn_dcm = nn_deri[nn_deri_index]
				frag_deri[i][0] += nn_dcm * cm_dx
				frag_deri[i][1] += nn_dcm * cm_dy
				frag_deri[i][2] += nn_dcm * cm_dz
		return frag_deri


class Frag_of_Mol(Mol):
	def __init__(self, atoms_=None, coords_=None):
		Mol.__init__(self, atoms_, coords_)
		self.undefined_bond_type =  None # whether the dangling bond can be connected  to H or not
		self.undefined_bonds = None  # capture the undefined bonds of each atom


        def FromXYZString(self,string):
                lines = string.split("\n")
                natoms=int(lines[0])
                self.atoms.resize((natoms))
                self.coords.resize((natoms,3))
                for i in range(natoms):
                        line = lines[i+2].split()
                        if len(line)==0:
                                return
                        self.atoms[i]=AtomicNumber(line[0])
                        try:
                                self.coords[i,0]=float(line[1])
                        except:
                                self.coords[i,0]=scitodeci(line[1])
                        try:
                                self.coords[i,1]=float(line[2])
                        except:
                                self.coords[i,1]=scitodeci(line[2])
                        try:
                                self.coords[i,2]=float(line[3])
                        except:
                                self.coords[i,2]=scitodeci(line[3])
		import ast
		self.undefined_bonds = ast.literal_eval(lines[1][lines[1].index("{"):lines[1].index("}")+1])
		if "type" in self.undefined_bonds.keys():
			self.undefined_bond_type = self.undefined_bonds["type"]
		else:
			self.undefined_bond_type = "any"
                return

	def Make_AtomNodes(self):
                atom_nodes = []
                for i in range (0, self.NAtoms()):
			if i in self.undefined_bonds.keys():
                        	atom_nodes.append(AtomNode(self.atoms[i], i,  self.undefined_bond_type, self.undefined_bonds[i]))
			else:
				atom_nodes.append(AtomNode(self.atoms[i], i, self.undefined_bond_type))
                self.properties["atom_nodes"] = atom_nodes
		return


class AtomNode:
	""" Treat each atom as a node for the purpose of building the molecule graph """
        def __init__(self, node_type_=None, node_index_=None, undefined_bond_type_="any", undefined_bond_ = 0):
		self.node_type = node_type_
		self.node_index = node_index_
		self.connected_nodes = []
		self.undefined_bond = undefined_bond_
		self.undefined_bond_type = undefined_bond_type_
		self.num_of_bonds = None
		self.connected_atoms = None
		self.Update_Node()
		return

	def Append(self, node):
		self.connected_nodes.append(node)
		self.Update_Node()
		return

	def Num_of_Bonds(self):
		self.num_of_bonds = len(self.connected_nodes)+self.undefined_bond
		return len(self.connected_nodes)+self.undefined_bond

	def Connected_Atoms(self):
		connected_atoms = []
		for node in self.connected_nodes:
			connected_atoms.append(node.node_type)
		self.connected_atoms = connected_atoms
		return connected_atoms

	def Update_Node(self):
		self.Num_of_Bonds()
		self.Connected_Atoms()
		self.connected_nodes = [x for (y, x) in sorted(zip(self.connected_atoms, self.connected_nodes))]
		self.connected_atoms.sort()
		return
