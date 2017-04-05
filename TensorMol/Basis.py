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
	def __init__(self, Name_ = None, Type_ = "GauSH"):
		self.path = "./basis/"
		self.name = Name_
		self.type = Type_
		self.params = None

	def Load(self):
		print "Unpickling Basis Set"
		f = open(self.path+self.name+".tmb","rb")
		tmp=pickle.load(f)
		self.__dict__.update(tmp)
		f.close()
		# self.CheckShapes()
		# print "Training data manager loaded."
		# print "Based on ", len(self.set.mols), " molecules "
		# print "Based on files: ",self.AvailableDataFiles
		# self.QueryAvailable()
		# self.PrintSampleInformation()
		# self.dig.Print()
		return

	def Save(self):
		f=open(self.path+self.name+"_"+self.dig.name+".tmb","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

	def Orthogonalize(self):
