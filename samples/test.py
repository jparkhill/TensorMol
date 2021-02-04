"""
Generates artificial data for H_3, learns a potential for it, tests it in optimizations and whatnot.
"""
import TensorMol as tm
import numpy as np

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
    crds = np.random.uniform(low=1.0, high=4.0,size = (nsamp,3,3)) # Coordinates of the atom
    st = tm.MSet() # Molecule dataset this function generates

    # Select between the model type.
    # Morse -> Energy / Forces
    if (model_=="Morse"):
        MDL = tm.MorseModel()
    # QuantuMElectrostatic -> Energy / Forces / Dipole / Charges
    else:
        MDL = tm.QuantumElectrostatic()

    # Todo: This loop could be improved by pre-initializing the list, instead of appending to it at the end of
    #       each iteration of the for-loop. Would make the code more readable than working with st.mols[-1] each
    #       time as well. -JD
    # Run an iteration of the simulation and update the dataset
    for s in range(nsamp):
        # If Morse Model (Energy & Forces)
        if (model_=="Morse"):
            st.mols.append(tm.Mol(np.array([1.,1.,1.]),crds[s]))
            en,f = MDL(crds[s])
            st.mols[-1].properties["dipole"] = np.array([0.,0.,0.])
        # If Quantum Electrostatic Model (Energy, Force, Dipole, Charge)
        else:
            st.mols.append(tm.Mol(np.array([3.,1.,1.]),crds[s]))
            en, f, d, q = MDL(crds[s])
            st.mols[-1].properties["dipole"] = d
            st.mols[-1].properties["charges"] = q
        st.mols[-1].properties["energy"] = en
        st.mols[-1].properties["force"] = f
        st.mols[-1].properties["gradients"] = -1.0*f
        st.mols[-1].CalculateAtomization()
    return st

def TestTraining():
    # Generate Dataset
    a = GenerateData()
    TreatedAtoms = a.AtomTypes()
    # Set some global state
    tm.PARAMS["NetNameSuffix"] = "training_sample"
    tm.PARAMS["learning_rate"] = 0.00001
    tm.PARAMS["momentum"] = 0.95
    tm.PARAMS["max_steps"] = 15 # Train for 5 epochs in total
    tm.PARAMS["batch_size"] =  100
    tm.PARAMS["test_freq"] = 5 # Test for every epoch
    tm.PARAMS["tf_prec"] = "tf.float64" # double precsion
    tm.PARAMS["EnergyScalar"] = 1.0
    tm.PARAMS["GradScalar"] = 1.0/20.0
    tm.PARAMS["NeuronType"] = "sigmoid_with_param" # choose activation function
    tm.PARAMS["sigmoid_alpha"] = 100.0  # activation params
    tm.PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 1.0] # each layer's keep probability for dropout
    # Create the embedding for the molecule
    d = tm.MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
    tset = tm.TensorMolData_BP_Direct_EandG_Release(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
    # Spawn a manager for the TensorFlow instances 
    manager=tm.TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EandG_SymFunction")
    tm.PARAMS['Profiling']=0
    # Train the final model
    manager.Train(1)

TestTraining()

