=====================
Tutorials
=====================

These tutorials assume basic familiarity with terminals and interactive python sessions. TensorMol can be used interactively (in a terminal or Jupyter Notbook). It is easily imported as a python module if the `/TensorMol` directory is located in your `$PYTHONPATH` for example:

.. code-block:: python

	> python
	>>> import TensorMol as tm

Alternatively you can author your own python scripts which import TensorMol when they are first executed. This is the convention taken by several of the test-XXX.py scripts provided with the package. This short tutorial walks through some of the tests in test_h2o.py, performing various optimizations and dynamics with water molecules. In whatever directory is the working directory of the python executing, TensorMol assumes the existence of the following folders:

.. code-block:: python

	/datasets/ (Training data)
	/networks/ (Pre-Trained and new Neural networks & Managers)
	/results/ (Results of calculations)
	/logs/ (Execution logs)

Once you have TensorMol installed we suggest opening an interactive python session, and copy-pasting some lines of test_h2o.py into an interactive python session to get started using TensorMol. There are very few allowed Global Variables in TensorMol the most important being the PARAMS dictionary which holds parameters, and a set of physical constants. Default parameters are populated when TensorMol is imported, and can be modified at run-time by altering PARAMS before instantiating objects. Without further ado, let's jump in with some annotated examples.

TensorMol provides a molecule class (Mol), and a class for a set of Molecules (MSet) which takes as it's argument arrays of atomic numbers and coordinates. For example a set containing one water molecule can be made like this:

.. code-block:: python

	a = MSet()
	a.mols.append(Mol(np.array([1,1,8]),np.array([[0.9,0.1,0.1],[1.,0.9,1.],[0.1,0.1,0.1]])))
	m = a.mols[0]

Sets of molecules can be read/written off disk, in .xyz format, or in a .pdb format which is nothing more than a python Pickle of the object. The advantage of the pickle object is that is preserves the Mol.properties dictionary which contains many derived properties of a molecule which aren't saved or loaded from the plaintext.xyz format

.. code-block:: python

	a = MSet("water6")
	a.ReadXYZ() # Reads ./datasets/water6.xyz
	a.Save() # makes ./datasets/water6.pdb
	a.Load() # Reads ./datasets/water6.pdb

Now that we have some molecules, it is time to do some chemistry. To do this we need to instantiate the object which produces energies and forces from a Mol. *TFManage* is the root class for objects which manage the TensorFlow instances which evaluate Energies, Forces, etc. in TensorMol, and also train instances. Under-the-hood *TFManage* owns *TFInstances* which are particular Neural-Network models. TFManagers have the typical *.Load()* *.Save()* *.Train()* routines that you would expect. In order to function, TFManage instances require two additional components to function, a *Digester*, which is an object which maps *Mol()* cartesian coordinates and atomic numbers onto appropriate neural network descriptors. They also require a *TensorMolData* object which specifies how batches of molecular information may be fed into the Manager. We can recall a *TFManager* off of disk by constructing one and invoking its name.

.. code-block:: python

	TreatedAtoms = a.AtomTypes() # Makes the list of element types we'll use.
	# Create a descriptor which generates AN1 embeddings, and also provides the energy and dipole of a molecule.
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")
	# Create a data provider
	tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	# Recall the pre-trained manager off disk
	manager=TFMolManage("Mol_H2O_wb97xd_1to21_with_prontonated_ANI1_Sym_Direct_fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout_act_sigmoid100", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)

This particular sort of manager can predict energies, charges and dipoles of molecules. Once the manager has been instantiated we are ready to use it to do chemistry.
