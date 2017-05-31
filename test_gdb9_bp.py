from TensorMol import *

if (1):
        a = MSet("morphine")   # Make a set of molecules
        a.ReadXYZ("morphine")  # Include your .xyz file in the folder "datasets". The second line of the xyz file should begin with  "Comment: "
        a.Make_Graphs()        # Define all the bonds in the molecule
        manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1_release" , None, False)  # Load the pretrained network, which should be included in the folder "networks"
        manager.Eval_Bond_BP(a, True)  # Evaluting the energies of bonds in the molecule.

