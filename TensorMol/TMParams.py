"""
PARAMETER CONVENTION:
- It's okay to have parameters which you pass to functions used to perform a test if you don't change them often.
- It's NOT okay to put default parameters in __init__() and change them all the time.
- These params should be added to a logfile of results so that we can systematically see how our approximations are doing.
"""

class TMParams(dict):
    def __init__(self, *args, **kwargs ):
		myparam = kwargs.pop('myparam', '')
		dict.__init__(self, *args, **kwargs )
		# SET GENERATION parameters
		self["MAX_ATOMIC_NUMBER"] = 10
		self["MBE_ORDER"] = 2
		self["NDistort"] = 100
		self["NModePts"] = 20

		# DATA usage parameters
		self["NormalizeInputs"] = True
		self["NormalizeOutputs"] = True
		self["batch_size"] = 8000

		self["learning_rate"] = 0.0001
		self["momentum"] = 0.9
		self["max_steps"] = 10000
		self["hidden1"] = 512
		self["hidden2"] = 512
		self["hidden3"] = 512
		self["Qchem_RIMP2_Block"] = "$rem\n   jobtype   sp\n   method   rimp2\n   MAX_SCF_CYCLES  200\n   basis   cc-pvtz\n   aux_basis rimp2-cc-pvtz\n   symmetry   false\n   INCFOCK 0\n   thresh 12\n   SCF_CONVERGENCE 12\n$end\n"
		# This just sets defaults.
