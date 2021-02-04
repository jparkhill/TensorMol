"""
Generates artificial data for H_3, learns a potential for it, tests it in optimizations and whatnot.
"""
import TensorMol as tm
import numpy as np

# Global configuration for the neural network
configuration = {"NetNameSuffix": "training_sample",
"learning_rate": 0.00001,
"momentum": 0.95,
"max_steps": 15,  # Train for 5 epochs in total
"batch_size": 100,
"test_freq": 5,  # Test for every epoch
"tf_prec": "tf.float64",  # double precsion
"EnergyScalar": 1.0,
"GradScalar": 1.0 / 20.0,
"NeuronType": "sigmoid_with_param",  # choose activation function
"sigmoid_alpha": 100.0,  # activation params
"KeepProb": [1.0, 1.0, 1.0, 1.0],  # each layer's keep probability for dropout
'Profiling': 0
}
tm.PARAMS.update(configuration)

def GenerateData(model_=None):
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
    nsamp = 10000  # Number of configurations to sample
    coords = np.random.uniform(low=1.0, high=4.0, size=(nsamp, 3, 3))  # Generate random 3D coordinates for nsamp atoms

    # Initialize our dataset
    dataset = tm.MSet()
    # Morse -> Energy / Forces
    if model_ == "Morse":
        MDL = tm.MorseModel()
        atomic_numbers = np.array([1, 1, 1], dtype=np.uint8)  # Hydrogen / Hydrogen / Hydrogen
    # QuantumElectrostatic -> Energy / Forces / Dipole / Charges
    else:
        MDL = tm.QuantumElectrostatic()
        atomic_numbers = np.array([3, 1, 1], dtype=np.uint8)  # Lithium / Hydrogen / Hydrogen

    # Populate the dataset with a set of molecules
    dataset.mols = [tm.Mol(atomic_numbers, coords[sample]) for sample in range(nsamp)]

    # Iterate over each molecule in the dataset and calculate properties
    for molecule in range(nsamp):
        # If Morse Model (Energy & Forces & 0-dipole)
        if (model_ == "Morse"):
            energy, force = MDL(coords[molecule])
            dipole = np.array([0., 0., 0.])
        # If Quantum Electrostatic Model (Energy, Force, Dipole, Charge)
        else:
            energy, force, dipole, charge = MDL(coords[molecule])
            dataset.mols[molecule].properties["charges"] = charge
        # Store dipole / energy / force /gradients
        dataset.mols[molecule].properties["dipole"] = dipole
        dataset.mols[molecule].properties["energy"] = energy
        dataset.mols[molecule].properties["force"] = force
        dataset.mols[molecule].properties["gradients"] = -1.0 * force
        dataset.mols[molecule].CalculateAtomization()
    return dataset


def TestTraining():
    # Generate Dataset
    dataset = GenerateData()
    TreatedAtoms = dataset.AtomTypes()

    # Create the embedding for the molecule
    embeddings = tm.MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
    tset = tm.TensorMolData_BP_Direct_EandG_Release(dataset, embeddings, order_=1, num_indis_=1, type_="mol", WithGrad_=True)

    # Spawn a manager for the TensorFlow instances 
    manager = tm.TFMolManage("", tset, False, "fc_sqdiff_BP_Direct_EandG_SymFunction")

    # Train the final model
    manager.Train(1)

TestTraining()

