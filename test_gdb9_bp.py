from TensorMol import *
import cProfile

# John's tests
if (1):
	if (0):
                a = MSet("morphine")
                a.ReadXYZ("morphine")
                a.Make_Graphs()
                a.Save()
                a.Load()
                manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1_release" , None, False)
                manager.Eval_Bond_BP(a, True)

        if (1):
		manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1_release" , None, False)
	        print "use dummy data to preload neural network, it may cost about few seconds..\n"
                a = MSet("dummy")
                a.ReadXYZ("dummy")
                a.Make_Graphs()
                manager.Eval_Bond_BP(a, True)
                print "end of preload"		

		print "begin timing test:"
                a = MSet("nanotube")
                a.ReadXYZ("nanotube")
                a.Make_Graphs()
                #a.Save()
                #a.Load()
                manager.Eval_Bond_BP(a, True)

	if (0):
		manager= TFMolManage("Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_fc_sqdiff_BP_1_release" , None, False)

		print "use dummy data to preload neural network, it may cost about few seconds..\n"
	        a = MSet("dummy")
                a.ReadXYZ("dummy")
                a.Make_Graphs()
                manager.Eval_Bond_BP(a, True)
		print "end of preload" 

		print "begin timing test:"
                a = MSet("c80h162")
                a.ReadXYZ("c80h162")
                a.Make_Graphs()
                manager.Eval_Bond_BP(a, True)

		a = MSet("c40h82")
                a.ReadXYZ("c40h82")
                a.Make_Graphs()
                manager.Eval_Bond_BP(a, True)

                a = MSet("c20h42")
                a.ReadXYZ("c20h42")
                a.Make_Graphs()
                manager.Eval_Bond_BP(a, True)

                a = MSet("c10h22")
                a.ReadXYZ("c10h22")
                a.Make_Graphs()
                manager.Eval_Bond_BP(a, True)

                a = MSet("c5h12")
                a.ReadXYZ("c5h12")
                a.Make_Graphs()
                manager.Eval_Bond_BP(a, True)
