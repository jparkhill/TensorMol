"""
Generates artificial data for H_3, learns a potential for it, tests it in optimizations and whatnot.
"""
from TensorMol import *

# Todo: This default model type isn't doing anything. Change to None and make sure nothing breaks. -JD
def GenerateData(model_="Huckel"):
	"""
	Generate random configurations in a reasonable range.
	and calculate their energies and forces.

    Args:
        model (str): Type of model to be used. If "Morse is supplied, a morse potential will be
                     used to create the dataset (no dipole moments or charge). If anything else
                     is supplied, a Quantum Electrostatic potential will be used to create the
                     dataset, which will include dipole moments and charge. In either case,
                     energy, forces, and gradients are all calculated.
	"""
    # Configuration
	nsamp = 10000 # Number of configurations to sample
	crds = np.random.uniform(4.0,size = (nsamp,3,3)) # Coordinates of the atom
	st = MSet() # Molecule dataset this function generates
    # Todo: The next few lines don't seem to do anything, since the MDL variable gets over-written, and the others
    #       never get used. Keeping them here in case their removal breaks anything. -JD
	MDL = None
	natom = 4 # Number of atoms
	ANS = np.array([3,1,1])

    # Select between the model type.
    # Morse -> Energy / Forces
	if (model_=="Morse"):
		MDL = MorseModel()
    # QuantuMElectrostatic -> Energy / Forces / Dipole / Charges
	else:
		MDL = QuantumElectrostatic()

    # Todo: This loop could be improved by pre-initializing the list, instead of appending to it at the end of
    #       each iteration of the for-loop. Would make the code more readable than working with st.mols[-1] each
    #       time as well. -JD
    # Run an iteration of the simulation and update the dataset
	for s in range(nsamp):
        # If Morse Model (Energy & Forces)
		if (model_=="Morse"):
			st.mols.append(Mol(np.array([1.,1.,1.]),crds[s]))
			en,f = MDL(crds[s])
			st.mols[-1].properties["dipole"] = np.array([0.,0.,0.])
        # If Quantum Electrostatic Model (Energy, Force, Dipole, Charge)
		else:
			st.mols.append(Mol(np.array([3.,1.,1.]),crds[s]))
			en, f, d, q = MDL(crds[s])
			st.mols[-1].properties["dipole"] = d
			st.mols[-1].properties["charges"] = q
		st.mols[-1].properties["energy"] = en
		st.mols[-1].properties["force"] = f
		st.mols[-1].properties["gradients"] = -1.0*f
		st.mols[-1].CalculateAtomization()
	return st

# Todo: This function doesn't seem to ever get used, and this is the only location in the repo where
#       BehlerParinelloDirectGauSH ever gets called. Commenting it out for now, probably will remove
#       in a future verison. Legacy code that never got trimmed? -JD
def TestTraining_John():
   # Global state is controlled by PARAMS
   # PARAMS gets pulled into the TM namespace in TensorMol/__init__.py by importing TensorMol/Utils
   # TensorMol/Utils pulls PARAMS into the namespace from TensorMol/TMParams.py
	PARAMS["train_dipole"] = True
	tset = GenerateData()
	net = BehlerParinelloDirectGauSH(tset)
	net.train()
	return

def TestTraining():
    # Generate Dataset
	a = GenerateData()
	TreatedAtoms = a.AtomTypes()
    # Set some global state
	PARAMS["NetNameSuffix"] = "training_sample"
	PARAMS["learning_rate"] = 0.00001
	PARAMS["momentum"] = 0.95
	PARAMS["max_steps"] = 15 # Train for 5 epochs in total
	PARAMS["batch_size"] =  100
	PARAMS["test_freq"] = 5 # Test for every epoch
	PARAMS["tf_prec"] = "tf.float64" # double precsion
	PARAMS["EnergyScalar"] = 1.0
	PARAMS["GradScalar"] = 1.0/20.0
	PARAMS["NeuronType"] = "sigmoid_with_param" # choose activation function
	PARAMS["sigmoid_alpha"] = 100.0  # activation params
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 1.0] # each layer's keep probability for dropout
    # Create the embedding for the molecule
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
	tset = TensorMolData_BP_Direct_EandG_Release(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
    # Spawn a manager for the TensorFlow instances 
	manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EandG_SymFunction")
	PARAMS['Profiling']=0
    # Train the final model
	manager.Train(1)

# Todo: This doesn't seem to be doing anything? -JD
def TestOpt():
	return

# Todo: This doesn't seem tod o anything either. -JD
def TestMD():
	return

TestTraining()
