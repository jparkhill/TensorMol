"""
PARAMETER CONVENTION:
- It's okay to have parameters which you pass to functions used to perform a test if you don't change them often.
- It's NOT okay to put default parameters in __init__() and change them all the time.
- These params should be added to a logfile of results so that we can systematically see how our approximations are doing.
"""
import logging, time

class TMParams(dict):
    def __init__(self, *args, **kwargs ):
        myparam = kwargs.pop('myparam', '')
        dict.__init__(self, *args, **kwargs )

        # SET GENERATION parameters
        self["MAX_ATOMIC_NUMBER"] = 10
        self["MBE_ORDER"] = 2
        self["NDistort"] = 100
        self["NModePts"] = 20
        self["GoK"] = 0.05

        # DATA usage parameters
        self["NormalizeInputs"] = True
        self["NormalizeOutputs"] = True
        self["batch_size"] = 8000
        self["results_dir"] = "./results/"

        self["learning_rate"] = 0.0001
        self["momentum"] = 0.9
        self["max_steps"] = 10000
        self["hidden1"] = 512
        self["hidden2"] = 512
        self["hidden3"] = 512
        self["Qchem_RIMP2_Block"] = "$rem\n   jobtype   sp\n   method   rimp2\n   MAX_SCF_CYCLES  200\n   basis   cc-pvtz\n   aux_basis rimp2-cc-pvtz\n   symmetry   false\n   INCFOCK 0\n   thresh 12\n   SCF_CONVERGENCE 12\n$end\n"
        # This just sets defaults.

    def __str__():
        tore=""
        for k in self.keys():
            tore = tore +k+":"+str(self[k])
        return tore

def TMBanner():
	print("--------------------------\n")
	print("         /\\______________")
	print("      __/  \\   \\_________")
	print("    _/  \\   \\            ")
	print("___/\_TensorMol_0.0______")
	print("   \\_/\\______  __________")
	print("     \\/      \\/          ")
	print("      \\______/\\__________\n")
	print("--------------------------")
	print("By using this software you accept the terms of the GNU public license in ")
	print("COPYING, and agree to attribute the use of this software in publications as: \n")
	print("K.Yao, J. E. Herr, J. Parkhill. TensorMol0.0 (2016)")
	print("Depending on Usage, please also acknowledge, TensorFlow, PySCF, or your training sets.")
	print("--------------------------")

def TMLogger(path_):
    tore=logging.getLogger('TensorMol')
    tore.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=path_+time.ctime()+'.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    pformatter = logging.Formatter('%(message)s')
    fh.setFormatter(fformatter)
    ch.setFormatter(pformatter)
    tore.addHandler(fh)
    tore.addHandler(ch)
    return tore
