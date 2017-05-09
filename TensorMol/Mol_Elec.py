"""
Routines for calculating dipoles quadropoles, etc, and cutoff electrostatic energies.
"""

from Util import *
import numpy as np
import random, math
import PhysicalData, MolEmb

def WeightedCoordAverage(x_, q_, center_=None):
	""" Dipole relative to center of x_ """
	if (center_==None):
		center_ = np.average(x_,axis=0)
	return np.einsum("ax,a", x_-center_ , q_)

def Dipole(m_):
		if ("dipole" in m_.properties and "charges" in m_.properties):
			print "Qchem, Calc'd", m_.properties["dipole"]*AUPERDEBYE, WeightedCoordAverage(m_.coords*BOHRPERA, m_.properties["charges"], m_.Center())

def ECoulECutoff(m_):
	dm = MolEmb.Make_DistMat(m_.coords)*BOHRPERA
	dm += np.eye(len(m_.coords))
	idm = 1.0/dm
	idm *= (np.ones(len(m_.coords)) - np.eye(len(m_.coords)))
	ECoul = np.dot(m_.properties["charges"],np.dot(idm,m_.properties["charges"]))
	# 1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	OneOverRScreened = 0.5*(np.tanh((dm - PARAMS["EECutoff"]*BOHRPERA)/(PARAMS["EEdr"]*BOHRPERA))+1)*idm
	ECutoff = np.dot(m_.properties["charges"],np.dot(OneOverRScreened, m_.properties["charges"]))
	print ECoul, ECutoff
	return
