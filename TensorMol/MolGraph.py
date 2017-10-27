from __future__ import absolute_import
from __future__ import print_function
from .Util import *
import numpy as np
import random, math
from .Mol import *
from .PhysicalData import *

def AtomName_From_List(atom_list):
	name = ""
	for i in atom_list:
		name += atoi.keys()[atoi.values().index(i)]
	return name

def Subset(A, B): # check whether B is subset of A
	checked_index = []
	found = 0
	for value in B:
		for i in range (0, len(A)):
			if value==A[i] and i not in checked_index:
				checked_index.append(i)
				found += 1
				break
	if found == len(B):
		return True
	else:
		return False

def Setdiff(A, B): # return the element of A that not included in B
	diff = []
	checked_index = []
	for value in A:
		found = 0
		for i in range (0, len(B)):
			if value == B[i] and i not in checked_index:
				found = 1
				checked_index.append(i)
				break
		if found == 0:
			diff.append(value)
	return diff

class MolGraph(Mol):
	def __init__(self, atoms_ =  np.zeros(1,dtype=np.uint8), coords_ = np.zeros(shape=(1,1),dtype=np.float), bond_length_thresh_ =  None):
		""" graph of a molecule """
		Mol.__init__(self, atoms_, coords_)
		# self.name= self.name+"_graph"
		self.num_atom_connected = None # connected  atoms of each atom
		self.atom_nodes = None
		self.bonds = None  #{connection type, length, atom_index_1, atom_index_2}
		self.bond_type = None # define whether it is a single, double or triple bond
		self.bond_conju = None # whether a bond is in a conjugated system
		self.bond_index = None # the bond index between two atoms
		self.Bonds_Between  = None
		self.H_Bonds_Between = None
		self.nx_mol_graph = None
		self.shortest_path = None
		if not bond_length_thresh_:
			self.bond_length_thresh = bond_length_thresh
		self.Make_Mol_Graph()
		return

	def NAtoms(self):
		return self.atoms.shape[0]

	def NBonds(self):
		return self.bonds.shape[0]

	def AtomTypes(self):
		return np.unique(self.atoms)

	def BondTypes(self):
		return np.unique(self.bonds[:,0]).astype(int)

	def Make_Mol_Graph(self):
		self.Make_AtomNodes()
		self.Connect_AtomNodes()
		#self.Make_Bonds()
		return

	def Find_Bond_Index(self):
		#print "name", self.name, "\n\n\nbonds", self.bonds, " bond_type:", self.bond_type
		self.bond_index = dict()
		for i in range (0, self.NBonds()):
			pair = [int(self.bonds[i][2]), int(self.bonds[i][3])]
			pair.sort()
			self.bond_index[LtoS(pair)] = i
		return

	def Make_AtomNodes(self):
		self.atom_nodes = []
		for i in range (0, self.NAtoms()):
			self.atom_nodes.append(AtomNode(self.atoms[i], i))
		return

	def Connect_AtomNodes(self):
		self.DistMatrix = MolEmb.Make_DistMat(self.coords)
		self.num_atom_connected = []
		for i in range (0, self.NAtoms()):
			for j in range (i+1, self.NAtoms()):
				dist = self.DistMatrix[i][j]
				atom_pair=[self.atoms[i], self.atoms[j]]
				atom_pair.sort()
				bond_name = AtomName_From_List(atom_pair)
				if dist <= self.bond_length_thresh[bond_name]:
					self.atom_nodes[i].Append(self.atom_nodes[j])
					self.atom_nodes[j].Append(self.atom_nodes[i])
		for i in range (0, self.NAtoms()):
			self.num_atom_connected.append(len(self.atom_nodes[i].connected_nodes))
		return

	def Make_Bonds(self):
		self.bonds = []
		visited_pairs = []
		for i in range (0, self.NAtoms()):
			for node in self.atom_nodes[i].connected_nodes:
				j  = node.node_index
				pair_index =  [i, j]
				atom_pair=[self.atoms[i], self.atoms[j]]
				pair_index = [x for (y, x) in sorted(zip(atom_pair, pair_index))]
				if pair_index not in visited_pairs:
					visited_pairs.append(pair_index)
					atom_pair.sort()
					bond_name = AtomName_From_List(atom_pair)
					bond_type = bond_index[bond_name]
					dist = self.DistMatrix[i][j]
					self.bonds.append(np.array([bond_type, dist, pair_index[0], pair_index[1]]))
		self.bonds = np.asarray(self.bonds)
		self.Find_Bond_Index()
		return

	def GetNextNode_DFS(self, visited_list, node_stack):
		node = node_stack.pop()
		visited_list.append(node.node_index)
		for next_node in node.connected_nodes:
			if next_node.node_index not in visited_list and next_node not in node_stack:
				node_stack.append(next_node)
		return node, visited_list, node_stack

	def Calculate_Bond_Type(self):
		self.bond_type = [0 for i in range (0, self.NBonds())]
		left_atoms = range (0, self.NAtoms())
		left_connections = list(self.num_atom_connected)
		left_valance = [ atom_valance[at] for at in self.atoms ]
		bond_of_atom = [[] for i in  range (0, self.NAtoms())] # index of the bonds that the atom are connected
		for i in range (0, self.NBonds()):
			bond_of_atom[int(self.bonds[i][2])].append(i)
			bond_of_atom[int(self.bonds[i][3])].append(i)
		flag = 1
		while (flag == 1):  # finish the easy assigment
			flag  = self.Define_Easy_Bonds(bond_of_atom, left_connections, left_atoms, left_valance)
			if (flag == -1):
				print("error when define bond type..")
				self.bond_type = [-1 for i in range (0, self.NBonds())]
				return
		save_bond_type = list(self.bond_type)
		if left_atoms: # begin try and error
			try_index = bond_of_atom[left_atoms[0]][0]
			for try_type in range (1, left_valance[left_atoms[0]] - left_connections[left_atoms[0]]+2):
				self.bond_type = list(save_bond_type)
				import copy
				cp_bond_of_atom = copy.deepcopy(bond_of_atom)
				cp_left_connections = list(left_connections)
				cp_left_atoms = list(left_atoms)
				cp_left_valance = list(left_valance)
				self.bond_type[try_index] = try_type
				cp_bond_of_atom[left_atoms[0]].pop(0)
				cp_left_connections[left_atoms[0]] -= 1
				cp_left_valance[left_atoms[0]] -= try_type
				other_at = (int(self.bonds[try_index][2]) if left_atoms[0] != int(self.bonds[try_index][2]) else int(self.bonds[try_index][3]))
				cp_bond_of_atom[other_at].pop(cp_bond_of_atom[other_at].index(try_index))
				cp_left_connections[other_at] -= 1
				cp_left_valance[other_at] -= try_type
				flag = 1
				while(flag == 1):
					flag  = self.Define_Easy_Bonds(cp_bond_of_atom, cp_left_connections, cp_left_atoms, cp_left_valance, True)
				if not cp_left_atoms and flag == 0 :
					left_atoms = []
					break
		if  left_atoms or flag != 0  :
			print("error when define bond type..")
			self.bond_type = [-1 for i in range (0, self.NBonds())]
			return
		return

	def Find_Frag(self, frag, ignored_ele=[1], frag_head=0, avail_atoms=None):   # ignore all the H for assigment
		if avail_atoms==None:
			avail_atoms = range(0, self.NAtoms())
		frag_head_node = frag.atom_nodes[frag_head]
		frag_node_stack = [frag_head_node]
		frag_visited_list = []
		all_mol_visited_list = [[]]
		while(frag_node_stack):   # if node stack is not empty
			current_frag_node = frag_node_stack[-1]
			updated_all_mol_visited_list = []
			for mol_visited_list in all_mol_visited_list:
				possible_node = []
				if mol_visited_list ==[]:
					possible_node = [self.atom_nodes[i] for i in avail_atoms]
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
						connected_node_in_mol = self.atom_nodes[mol_visited_list[connected_node_index]]
						for target_node in connected_node_in_mol.connected_nodes:
							if target_node.node_index not in mol_visited_list and self.Compare_Node(target_node, current_frag_node) and self.Check_Connection(target_node, current_frag_node, mol_visited_list, frag_visited_list) and target_node.node_index in avail_atoms:
								updated_all_mol_visited_list.append(mol_visited_list+[target_node.node_index])
								if target_node.node_type in ignored_ele:
									break
			all_mol_visited_list = list(updated_all_mol_visited_list)
			next_frag_node, frag_visited_list, frag_node_stack  = self.GetNextNode_DFS(frag_visited_list, frag_node_stack)
		frags_in_mol = []
		already_included = []
		for mol_visited_list in all_mol_visited_list:
			mol_visited_list.sort()
			if mol_visited_list not in already_included:
				already_included.append(mol_visited_list)
				sorted_mol_visited_list = [x for (y, x) in sorted(zip(frag_visited_list,mol_visited_list))]## sort the index order of frags in mol to the same as the frag
				frags_in_mol.append(sorted_mol_visited_list)
		return frags_in_mol

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

	def IsIsomer(self,other):
		return np.array_equals(np.sort(self.atoms),np.sort(other.atoms))

	def AtomName(self, i):
		return atoi.keys()[atoi.values().index(self.atoms[i])]

	def AllAtomNames(self):
		names=[]
		for i in range (0, self.atoms.shape[0]):
			names.append(atoi.keys()[atoi.values().index(self.atoms[i])])
		return names

class Frag_of_MolGraph(MolGraph):
	def __init__(self, mol_, undefined_bonds_ = None, undefined_bond_type_ = None, bond_length_thresh_ =  None):
		Mol.__init__(self, mol_, bond_length_thresh_)
		self.undefined_bond_type = undefined_bond_type_  # whether the dangling bond can be connected  to H or not
		self.undefined_bonds = undefined_bonds_  # capture the undefined bonds of each atom

	def FromXYZString(self,string, set_name = None):
		self.properties["set_name"] = set_name
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
		try:
			self.undefined_bonds = ast.literal_eval(lines[1][lines[1].index("{"):lines[1].index("}")+1])
			if "type" in self.undefined_bonds.keys():
				self.undefined_bond_type = self.undefined_bonds["type"]
			else:
				self.undefined_bond_type = "any"
		except:
			self.name = lines[1] #debug
			self.undefined_bonds = {}
			self.undefined_bond_type = "any"
		return

	def Make_AtomNodes(self):
		atom_nodes = []
		for i in range (0, self.NAtoms()):
			if i in self.undefined_bonds.keys():
				atom_nodes.append(AtomNode(self.atoms[i], i,  self.undefined_bond_type, self.undefined_bonds[i]))
			else:
				atom_nodes.append(AtomNode(self.atoms[i], i, self.undefined_bond_type))
		self.atom_nodes = atom_nodes
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
