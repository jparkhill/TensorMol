from Util import *
from Sets import *
from TensorData import *
from TFManage import *
from Opt import *
import cProfile, pstats, StringIO
pr = cProfile.Profile()

# 1 - Get molecules into memory
a=MSet("gdb9_NEQ")
a.Load()
# Choose allowed atoms.
TreatedAtoms = a.AtomTypes()
# 2 - Choose Digester
d = Digester(TreatedAtoms, name_="SensoryBasis",OType_ ="SmoothP")
# 4 - Generate training set samples.
tset = TensorData(a,d,None,100) #100s/element
pr.enable()
tset.BuildTrain("gdb9_NEQ",TreatedAtoms) # generates dataset numpy arrays for each atom.
pr.disable()

s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
