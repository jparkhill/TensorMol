from Util import * 
from Mol import * 
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


if (1):
	# 1 - Get molecules into memory
	b=MSet("OptMols")
	b.Load()
	a = b.DistortedClone()
	m = a.mols[0]
	d = Digester(TreatedAtoms, name_="SensoryBasis",OType_ ="SmoothP")
	ins,outs = d.TrainDigest(m,1)
	eins,eouts = GRIDS.ExpandIsometries(ins,outs)
	
	c = a.TransformedClone(1)
	m = c.mols[0]
	ins,outs = d.TrainDigest(m,1)

	print "Expanded 1:",eins,eouts
	print "Expanded 1:",ins,outs


