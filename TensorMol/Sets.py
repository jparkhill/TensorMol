
#

from Mol import *
from MolGraph import *
from MolFrag import *
from Util import *
import numpy as np
import os,sys,pickle,re,copy,time

class MSet:
	""" A molecular database which
		provides structures """
	def __init__(self, name_ ="gdb9", path_="./datasets/"):
		self.mols=[]
		self.path=path_
		self.name=name_
		self.suffix=".pdb" #Pickle Database? Poor choice.

	def Save(self):
		print "Saving set to: ", self.path+self.name+self.suffix
		f=open(self.path+self.name+self.suffix,"wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
		return

	def Load(self):
		f = open(self.path+self.name+self.suffix,"rb")
		tmp=pickle.load(f)
		self.__dict__.update(tmp)
		f.close()
		print "Loaded, ", len(self.mols), " molecules "
		print self.NAtoms(), " Atoms total"
		print self.AtomTypes(), " Types "
		return

	def DistortAlongNormals(self, npts=8, random=True, disp=.2):
		'''
		Create a distorted copy of a set
		Args:
			npts: the number of points to sample along the normal mode coordinate.
			random: whether to randomize the order of the new set.
			disp: the maximum displacement of atoms along the mode
		Returns:
			A set containing distorted versions of the original set.
		'''
		print "Making distorted clone of:", self.name
		s = MSet(self.name+"_NEQ")
		ord = range(len(self.mols))
		if(random):
			np.random.seed(int(time.time()))
			ord=np.random.permutation(len(self.mols))
		for j in ord:
			newcoords = self.mols[j].ScanNormalModes(npts,disp)
			for i in range(newcoords.shape[0]): # Loop modes
				for k in range(newcoords.shape[1]): # loop points
					s.mols.append(Mol(self.mols[j].atoms,newcoords[i,k,:,:]))
					s.mols[-1].DistMatrix = self.mols[j].DistMatrix
		return s

	def DistortedClone(self, NDistorts=1, random=True):
			''' Create a distorted copy of a set'''
			print "Making distorted clone of:", self.name
			s = MSet(self.name+"_NEQ")
			ord = range(len(self.mols))
			if(random):
				np.random.seed(int(time.time()))
				ord=np.random.permutation(len(self.mols))
			for j in ord:
				for i in range (0, NDistorts):
					s.mols.append(copy.deepcopy(self.mols[j]))
					s.mols[-1].Distort()
			return s

	def TransformedClone(self, transf_num):
		''' make a linearly transformed copy of a set. '''
		print "Making distorted clone of:", self.name
		s = MSet(self.name+"_transf_"+str(transf_num))
		ord = range(len(self.mols))
		for j in ord:
				s.mols.append(copy.deepcopy(self.mols[j]))
				s.mols[-1].Transform(GRIDS.InvIsometries[transf_num])
		return s

	def NAtoms(self):
		nat=0
		for m in self.mols:
			nat += m.NAtoms()
		return nat


	def AtomTypes(self):
		types = np.array([],dtype=np.uint8)
		for m in self.mols:
			types = np.union1d(types,m.AtomTypes())
		return types


	def ReadGDB9Unpacked(self, path="/Users/johnparkhill/gdb9/"):
		""" Reads the GDB9 dataset as a pickled list of molecules"""
		from os import listdir
		from os.path import isfile, join
		#onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		for file in onlyfiles:
			if ( file[-4:]!='.xyz' ):
					continue
			self.mols.append(Mol())
			self.mols[-1].ReadGDB9(path+file, file, self.name)
		return

	def ReadXYZ(self,filename, xyz_type = 'mol'):
		""" Reads XYZs concatenated into a single separated by \n\n file as a molset """
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].count('Comment:')>0):
				line0=line-1
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(Mol())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]), set_name=self.name)
		return


	def WriteXYZ(self,filename=None):
		if filename == None:
			filename = self.name
		for mol in self.mols:
			mol.WriteXYZfile(self.path,filename)
		return

	def pop(self, ntopop):
		for i in range(ntopop):
			self.mols.pop()
		return

	def OnlyWithElements(self, allowed_eles):
		mols=[]
		for mol in self.mols:
			if set(list(mol.atoms)).issubset(allowed_eles):
				mols.append(mol)
		for i in allowed_eles:
			self.name += "_"+str(i)
		self.mols=mols
		return

	def CutSet(self, allowed_eles):
		mols=[]
		for mol in self.mols:
			if set(list(mol.atoms)).issubset(allowed_eles):
				mols.append(mol)
		for i in allowed_eles:
			self.name += "_"+str(i)
		self.mols=mols
		return

	def AppendSet(self, b):
		self.name = self.name + "_" + b.name
		self.mols = self.mols+b.mols
		return

	def Statistics(self):
		""" Return some energy information about the samples we have... """
		print "Set Statistics----"
		print "Nmol: ", len(self.mols)
		sampfrac = 0.1;
		np.random.seed(int(time.time()))
		ord=np.random.permutation(int(len(self.mols)*sampfrac))
		if len(ord)==0:
			return
		ens = np.zeros(len(ord))
		rmsd = np.zeros(len(ord))
		n=0
		for j in ord:
			if (self.mols[j].energy != None):
				ens[n] = self.mols[j].energy
			else :
				ens[n] = self.mols[j].GoEnergy(self.mols[j].coords.flatten())
				tmp = MolEmb.Make_DistMat(self.mols[j].coords) - self.mols[j].DistMatrix
				rmsd[n] = np.sum(tmp*tmp)/len(self.mols[j].coords)
				n=n+1
		print "Mean and Std. Energy", np.average(ens), np.std(ens)
		print "Energy Histogram", np.histogram(ens, 100)
		print "RMSD Histogram", np.histogram(rmsd, 100)
		return

	def Make_Graphs(self):
		graphs = map(MolGraph, self.mols)
		return graphs
	
	def Clean_GDB9(self):
		s = MSet(self.name+"_cleaned")
		s.path = self.path
		for mol in self.mols:
			if float('inf') in mol.Bonds_Between:
				print "disconnected atoms in mol.. discard"
			elif -1 in mol.bond_type or 0 in mol.bond_type:
				print "allowed bond type in mol... discard"
			else:
				s.mols.append(mol)
		return s

	def Calculate_vdw(self):
		for mol in self.mols:
			mol.Calculate_vdw()
			print "atomization:", mol.atomization, " vdw:", mol.vdw
		return
	
	def WriteSmiles(self):
		for mol in self.mols:
			mol.WriteSmiles()
		return 


class FragableMSet(MSet):
	def __init__(self, name_ ="NaClH2O", path_="./datasets/"):
		MSet.__init__(self, name_, path_)
		return	

	def ReadGDB9Unpacked(self, path="/Users/johnparkhill/gdb9/"):
                """ Reads the GDB9 dataset as a pickled list of molecules"""
                from os import listdir
                from os.path import isfile, join
                #onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
                onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
                for file in onlyfiles:
                        if ( file[-4:]!='.xyz' ):
                                        continue
                        self.mols.append(FragableCluster())
                        self.mols[-1].ReadGDB9(path+file, file, self.name)
                return

        def ReadXYZ(self,filename, xyz_type = 'mol'):
                """ Reads XYZs concatenated into a single separated by \n\n file as a molset """
                f = open(self.path+filename+".xyz","r")
                txts = f.readlines()
                for line in range(len(txts)):
                        if (txts[line].count('Comment:')>0):
                                line0=line-1
                                nlines=int(txts[line0])
                                if xyz_type == 'mol':
                                        self.mols.append(FragableCluster())
                                elif xyz_type == 'frag_of_mol':
                                        self.mols.append(Frag_of_Mol())
                                else:
                                        raise Exception("Unknown Type!")
                                self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]), set_name=self.name)
                return

	def MBE(self,  atom_group=1, cutoff=10, center_atom=0):
		for mol in self.mols:
			mol.MBE(atom_group, cutoff, center_atom)
		return

	def PySCF_Energy(self):
		for mol in self.mols:
			mol.PySCF_Energy()
		return

	def Generate_All_MBE_term(self,  atom_group=1, cutoff=10, center_atom=0, max_case = 1000000):
		for mol in self.mols:
                	mol.Generate_All_MBE_term(atom_group, cutoff, center_atom, max_case)
                return

	def Generate_All_MBE_term_General(self, frag_list=[], cutoff=10, center_atom=0):
		for mol in self.mols:
			mol.Generate_All_MBE_term_General(frag_list, cutoff, center_atom)
		return

	def Calculate_All_Frag_Energy(self, method="pyscf"):
		for mol in self.mols:
			mol.Calculate_All_Frag_Energy(method)
               # 	mol.Set_MBE_Energy()
		return

	def Calculate_All_Frag_Energy_General(self, method="pyscf"):
                for mol in self.mols:
                        mol.Calculate_All_Frag_Energy_General(method)
               #        mol.Set_MBE_Energy()
                return

	def Get_All_Qchem_Frag_Energy(self):
		for mol in self.mols:
			mol.Get_All_Qchem_Frag_Energy()
		return

	def Get_All_Qchem_Frag_Energy_General(self):
                for mol in self.mols:
                        mol.Get_All_Qchem_Frag_Energy_General()
                return

	def Generate_All_Pairs(self, pair_list=[]):
		for mol in self.mols:
			mol.Generate_All_Pairs(pair_list)
		return

	def Get_Permute_Frags(self, indis=[0]):
		for mol in self.mols:
			mol.Get_Permute_Frags(indis)
		return

	def Set_Qchem_Data_Path(self):
		for mol in self.mols:
			mol.Set_Qchem_Data_Path()
		return


class GraphSet:
	def __init__(self, name_ ="gdb9", path_="./datasets/"):
                self.graphs=[]
                self.path=path_
                self.name=name_
                self.suffix=".graph" #Pickle Database? Poor choice.

        def BondTypes(self):
                types = np.array([],dtype=np.uint8)
                for m in self.mols:
                        types = np.union1d(types,m.BondTypes())
                return types

	def NBonds(self):
                nbonds=0
                for m in self.mols:
                        nbonds += m.NBonds()
                return nbonds

        def Save(self):
                print "Saving set to: ", self.path+self.name+self.suffix
                f=open(self.path+self.name+self.suffix,"wb")
                pickle.dump(self.__dict__, f, protocol=1)
                f.close()
                return

        def Load(self):
                f = open(self.path+self.name+self.suffix,"rb")
                tmp=pickle.load(f)
                self.__dict__.update(tmp)
                f.close()
                print "Loaded, ", len(self.mols), " molecules "
                print self.NAtoms(), " Atoms total"
                print self.AtomTypes(), " Types "
                return
	
