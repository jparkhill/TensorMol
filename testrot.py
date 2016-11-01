from Util import * 
from Mol import * 
from Sets import *
from Digest import *
from pyscf import gto

ch3xyz = """
7

C          0.00000        0.00000        0.00000
H          0.62881        0.16105        0.93595
H          0.30514        0.82402       -0.67859
H          0.53455       -0.93140       -0.44955
H         -1.06851       -0.05371        0.19215
O         -1.06851       -0.15371        3.19215
N         -2.06851       -3.05371        3.19215
"""

# place our embedding at the central C atom. 
# rotate the molecule and test the resulting embedding. 
m=Mol()
m.FromXYZString(ch3xyz)
m.Distort()
GRIDS.TestIsometries(m)


if (0):
	b=MSet("OptMols")
	b.ReadXYZ("OptMols")
	b.Save()

if (1):
	# 1 - Get molecules into memory
	b=MSet("OptMols")
	b.Load()
	a = b.DistortedClone()
	m = a.mols[0]
	print m.coords
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms, name_="SensoryBasis",OType_ ="Disp")
	ins,outs = d.TrainDigest(m,6)
	print ins.shape, outs.shape
	ncase = ins.shape[0]
	i1 = np.copy(ins[0])
	o1 = np.copy(outs[0])
	eins,eouts = GRIDS.ExpandIsometries(ins,outs)
	m = a.mols[0]
	print
	m.Transform(GRIDS.InvIsometries[2],m.coords[0])
	print m.coords
	ins,outs = d.TrainDigest(m,6)
	print "Expanded 1:",eins.shape,eouts.shape
	print "Expanded 1:",ins.shape,outs.shape

	print "emb diff:",eins[2*ncase]-i1
	print "out diff:",eouts[2*ncase]-o1
	
	print "emb diff:",eins[2*ncase] - ins[0]
	print "out diff:",eouts[2*ncase] - outs[0]

	print "Expanded 1:",eins,eouts
	print "Expanded 1:",ins,outs


