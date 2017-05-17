"""
Routines for running external Ab-Initio packages to get shit out of mol.py
"""
from Util import *
import numpy as np
import random, math, subprocess
import Mol

def PyscfDft(m_,basis_ = '6-31g*',xc_='b3lyp'):
	if (not HAS_PYSCF):
		print "Missing PYSCF"
		return 0.0
	mol = gto.Mole()
	pyscfatomstring=""
	crds = m_.coords.copy()
	crds[abs(crds)<0.0001] *=0.0
	for j in range(len(m_.atoms)):
		pyscfatomstring=pyscfatomstring+str(m_.atoms[j])+" "+str(crds[j,0])+" "+str(crds[j,1])+" "+str(crds[j,2])+(";" if j!= len(m_.atoms)-1 else "")
	mol.atom = pyscfatomstring
	mol.unit = "Angstrom"
	mol.charge = 0
	mol.spin = 0
	mol.basis = basis_
	mol.verbose = 0
	mol.build()
	mf = dft.RKS(mol)
	mf.xc = xc_
	e = mf.kernel()
	return e

def QchemDft(m_,basis_ = '6-31g*',xc_='b3lyp'):
	istring = '$molecule\n0 1 \n'
	crds = m_.coords.copy()
	crds[abs(crds)<0.0000] *=0.0
	for j in range(len(m_.atoms)):
		istring=istring+itoa[m_.atoms[j]]+' '+str(crds[j,0])+' '+str(crds[j,1])+' '+str(crds[j,2])+'\n'
	istring =istring + '$end\n\n$rem\njobtype sp\nbasis '+basis_+'\nexchange '+xc_+'\n thresh 11\n symmetry false\n sym_ignore true\n$end\n'
	#print istring
	f=open('./qchem/tmp.in','w')
	f.write(istring)
	f.close()
	subprocess.call('qchem ./qchem/tmp.in ./qchem/tmp.out'.split(),shell=False)
	f=open('./qchem/tmp.out','r')
	lines = f.readlines()
	f.close()
	for line in lines:
		if line.count(' met')>0:
			return np.array([float(line.split()[1])])[0]
	return np.array([0.0])[0]
