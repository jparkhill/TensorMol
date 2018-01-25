"""
Loads and handles basis sets for embeddings
"""

from __future__ import absolute_import
from __future__ import print_function
import os, sys, re, random, math, copy
import numpy as np
import sys
if sys.version_info[0] < 3:
	import pickle
else:
	import _pickle as pickle
from ..Math import LinearOperations, DigestMol, Digest, Opt, Ipecac

class Basis:
	def __init__(self, Name_ = None):
		self.path = "./basis/"
		self.name = Name_
		self.params = None

	def Load(self, filename=None):
		print("Unpickling Basis Set")
		if filename == None:
			filename = self.name
		f = open(self.path+filename+".tmb","rb")
		tmp=pickle.load(f)
		self.__dict__.update(tmp)
		f.close()
		return

	def Save(self):
		if filename == None:
			filename = self.name
		f=open(self.path+self.name+"_"+self.dig.name+".tmb","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

class Basis_GauSH(Basis):
	def __init__(self, Name_ = None):
		Basis.__init__(self, Name_)
		self.type = "GauSH"
		self.RBFS = np.tile(np.array([[0.1, 0.156787], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [1.3, 1.3], [2.2, 2.4],
										[4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]]), (10,1,1))
		return

	def Orthogonalize(self):
		from TensorMol.LinearOperations import MatrixPower
		S_Rad = MolEmb.Overlap_RBFS(PARAMS, self.RBFS)
		self.SRBF = np.zeros((self.RBFS.shape[0],PARAMS["SH_NRAD"],PARAMS["SH_NRAD"]))
		for i in range(S_Rad.shape[0]):
			self.SRBF[i] = MatrixPower(S_Rad[i],-1./2)
