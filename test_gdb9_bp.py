from TensorMol import *
import cProfile

# John's tests
if (1):
	if (1):
                a = MSet("morphine")
                a.ReadXYZ("morphine")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1_release" , None, False)
                manager.Eval_Bond_BP(a, True)

