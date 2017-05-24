"""
Routines for calculating dipoles quadropoles, etc, and cutoff electrostatic energies in python
See also: ElectrostaticsTF for tensorflow implementations of electrostatics.
"""

from Util import *
import numpy as np
import random, math
import PhysicalData, MolEmb

def WeightedCoordAverage(x_, q_, center_=None):
	""" Dipole relative to center of x_ """
	if (center_== None):
		center_ = np.average(x_,axis=0)
	return np.einsum("ax,a", x_-center_ , q_)

def DipoleDebug(m_):
		if ("dipole" in m_.properties and "charges" in m_.properties):
			print "Qchem, Calc'd", m_.properties["dipole"]*AUPERDEBYE, WeightedCoordAverage(m_.coords*BOHRPERA, m_.properties["charges"], m_.Center())

def Dipole(x_, q_):
	""" Arguments are in A, and elementary charges.  """
 	return WeightedCoordAverage(x_*BOHRPERA, q_)


def ChargeCharge(m1_, m2_):
	"""calcuate  the charge-charge interaction energy between two molecules"""
	cc_energy = 0.0
	for i in range (0, m1_.NAtoms()):
		for j in range (0, m2_.NAtoms()):
			dist = (np.sum(np.square(m1_.coords[i] - m2_.coords[j])))**0.5 * BOHRPERA
			cc_energy += m1_.properties['atom_charges'][i]*m2_.properties['atom_charges'][j]/dist
	return cc_energy


def ElectricFieldForce(q_,E_):
	"""
	Both are received in atomic units.
	The force should be returned in kg(m/s)^2, but I haven't fixed the units yet.
	"""
	return np.einsum("q,x->qx",q_,E_)

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

def WriteDerDipoleCorrelationFunction(MuTraj, name_="MutMu0.txt"):
	"""
	Args:
		time, mux, muy, muz ...
	Returns: \sum_i \langle \dot{\mu_i(t)}\cdot \dot{\mu_i(0)} \rangle
	"""
	dt = MuTraj[0,0] - MuTraj[1,0]
	t0 = np.zeros((MuTraj.shape[0]-1,4))
	for i in range(MuTraj.shape[0]-1):
		t0[i,0] = MuTraj[i,0]
		t0[i,1:4] = (MuTraj[i+1,1:4]- MuTraj[i,1:4])/dt
	n = t0.shape[0]
	tore = np.zeros((n,2))
	for i in range(int(n/4)): # Can't use the whole data....
		tore[i,0] = t0[i,0]
		tore[i,1] = 0.0
		for j in range(n-i):
			tore[i,1] +=  np.dot(t0[i,1:4],t0[i+j,1:4])
		tore[i,1] /= 3.*float(n-i)
	np.savetxt("./results/"+name_,tore)
	return tore

def WriteVelocityAutocorrelations(muhis,vhis):
	"""
	Args:
		Generate velocity autocorrelation functions.
	"""
	dt = muhis[0,0] - muhis[1,0]
	n = muhis.shape[0]
	natom = vhis.shape[1]
	ncorr = int(n/4)
	for atom in range(natom):
		tore = np.zeros((ncorr,2))
		for i in range(ncorr):
			tore[i,0] = muhis[i,0]
			tore[i,1] = 0.0
			for j in range(n-i):
				tore[i,1] +=  np.dot(vhis[i,atom],vhis[i+j,atom])
			tore[i,1] /= 3.*float(n-i)
		np.savetxt("./results/"+"VtV0_"+str(atom)+".txt",tore)
	return


def WriteDipoleCorrelationFunction(t0,t1,t2):
	"""
	Args:
		time, mux, muy, muz ...
	Returns: \sum_i \langle \mu_i(t)\cdot \mu_i(0) \rangle
	"""
	dt = t0[0,0] - t0[1,0]
	n = t0.shape[0]
	tore = np.zeros((n,2))
	for i in range(n):
		tore[i,0] = t0[i,0]
		tore[i,1] = 0.0
		for j in range(n-i):
			tore[i,1] +=  t0[j,1]*t0[i+j,1]+t1[j,2]*t1[i+j,2]+t2[j,2]*t2[i+j,2]
		tore[i,1] /= 3.*float(n-i)
	np.savetxt("./results/"+"MutMu0.txt",tore)
	return tore
