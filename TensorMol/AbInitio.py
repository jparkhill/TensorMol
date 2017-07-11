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

def PullFreqData():
	a = open("/media/sdb1/dtoth/qchem_jobs/new/phenol.out", "r+") #Change file name
	# each time to read correct output file
	f=open("phenol_freq.dat", "w") #Change file name to whatever you want --
	# make sure it's different each time
	lines = a.readlines()
	data = []
	ip = 0
	for i, line in enumerate(lines):
		if line.count("NAtoms") > 0:
			atoms = int(lines[i+1].split()[0])
			break
	nm = np.zeros((3*atoms-6, atoms, 3))
	for i, line in enumerate(lines):
		if "Frequency:" in line:
			freq = [line.split()[1], line.split()[2],line.split()[3]]
			intens = [lines[i+4].split()[2], lines[i+4].split()[3],lines[i+4].split()[4]]
			f.write(freq[0] + "   " + intens[0] + "\n")
			f.write(freq[1] + "   " + intens[1] + "\n")
			f.write(freq[2] + "   " + intens[2] + "\n")
		if "Raman Active" in line:
			for j in range(atoms):
				it = 0
				for k in range(3):
					for l in range(3):
						nm[it+ip,j,l] = float(lines[i+j+2].split()[k*3+l+1])
					it += 1
			ip += 3
			# f.write(nm[0] + "  " + nm)
	np.save("morphine_nm.npy", nm)
	f.close()

	def PySCFMP2Energy(m, basis_='cc-pvqz'):
		mol = gto.Mole()
		pyscfatomstring=""
		for j in range(len(m.atoms)):
			s = m.coords[j]
			pyscfatomstring=pyscfatomstring+str(m.AtomName(j))+" "+str(s[0])+" "+str(s[1])+" "+str(s[2])+(";" if j!= len(m.atoms)-1 else "")
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
			m.properties["energy"] = en
			return en
		except Exception as Ex:
			print "PYSCF Calculation error... :",Ex
			print "Mol.atom:", mol.atom
			print "Pyscf string:", pyscfatomstring
			return 0.0
			#raise Ex
		return
