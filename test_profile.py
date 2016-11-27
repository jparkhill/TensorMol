from Util import *
from Sets import *
from TensorData import *
from TFManage import *
from Opt import *
import cProfile, pstats, StringIO
pr = cProfile.Profile()

# 1 - Get molecules into memory
#a=MSet("gdb9")
#a.Load()
#b=a.DistortedClone()
#b.Save()

b=MSet("gdb9_NEQ")
b.Load()
# Choose allowed atoms.
TreatedAtoms = b.AtomTypes()
# 2 - Choose Digester
d = Digester(TreatedAtoms, name_="GauSH",OType_ ="Force")
# 4 - Generate training set samples.
tset = TensorData(b,d,None,10) #100s/element
pr.enable()
tset.BuildTrain("gdb9_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.
pr.disable()

s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
