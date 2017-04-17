"""
Loads and handles basis sets for embeddings
"""

from Util import *
from TensorData import *
import os, sys, re, random, math, copy
import numpy as np
import cPickle as pickle
import LinearOperations, DigestMol, Digest, Opt, Ipecac

class Basis:
	def __init__(self, Name_ = None):
		self.path = "./basis/"
		self.name = Name_
		self.params = None

	def Load(self, filename=None):
		print "Unpickling Basis Set"
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
		self.RBFS = np.broadcast_to(np.array([[0.1, 0.156787], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [1.3, 1.3], [2.2, 2.4],
										[4.4, 2.4], [6.6, 2.4], [8.8, 2.4], [11., 2.4], [13.2,2.4], [15.4, 2.4]]), (10,12,2))
		return

	def SetParams(self):
		from TensorMol.LinearOperations import MatrixPower
		self.SRBF = np.zeros(
		S_Rad = MolEmb.Overlap_RBF(PARAMS)
		S_RadOrth = MatrixPower(S_Rad,-1./2)
		PARAMS["SRBF"] = S_RadOrth
