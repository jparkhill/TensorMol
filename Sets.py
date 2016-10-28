#
# A molecule set is not a training set.
#

from Mol import *
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
		self.NDistorts = 1
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

	def DistortedClone(self, NDistorts_=1, random=True):
		self.NDistorts = NDistorts_
		print "Making distorted clone of:", self.name
		s = MSet(self.name+"_NEQ")
		ord = range(len(self.mols))
		if(random):
			np.random.seed(time.time()) 
			ord=np.random.permutation(len(self.mols))
		for j in ord: 
			for i in range (0, self.NDistorts):
				s.mols.append(copy.deepcopy(self.mols[j]))
				s.mols[-1].Distort(seed=i)
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
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		for file in onlyfiles:
			if ( file[-4:]!='.xyz' ):
				continue
			self.mols.append(Mol())
			self.mols[-1].ReadGDB9(path+file)
		return

	def ReadXYZ(self,filename):
		""" Reads XYZs concatenated into a single separated by @@@ file as a molset """
		f = open(self.path+filename+".xyz","r")
		txts = f.read()
		for mol in txts.split("@@@")[1:]:
			self.mols.append(Mol())
			self.mols[-1].FromXYZString(mol)
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
	
	def CombineSet(self, b, name_=None):
		if name_ == None:
			self.name = self.name + b.name
		self.mols.append(b.mols) 
		return
			
