from TensorMol import *
import os,sys

a=MSet()
m=Mol()
m.atoms = np.array([6, 1, 1, 1, 1],dtype=np.uint8)
m.coords = np.array([[0, 0, 0], [-1, 0, 0], [0, -0.5, -1], [0, 1, 0], [1, 0, 0]],dtype=np.float64)
a.mols.append(m)

def GetChemSpiderNetwork(a, Solvation_=False):
    TreatedAtoms = np.array([1,6,7,8], dtype=np.uint8)
    PARAMS["tm_root"] = "/home/animal/Packages/TensorMol"
    PARAMS["tf_prec"] = "tf.float64"
    PARAMS["NeuronType"] = "sigmoid_with_param"
    PARAMS["sigmoid_alpha"] = 100.0
    PARAMS["HiddenLayers"] = [2000, 2000, 2000]
    PARAMS["EECutoff"] = 15.0
    PARAMS["EECutoffOn"] = 0
    PARAMS["Elu_Width"] = 4.6  # when elu is used EECutoffOn should always equal to 0
    PARAMS["EECutoffOff"] = 15.0
    PARAMS["AddEcc"] = True
    PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 0.7]
    d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="EnergyAndDipole")  # Initialize a digester that apply descriptor for the fragme
    tset = TensorMolData_BP_Direct_EE_WithEle(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
    PARAMS["DSFAlpha"] = 0.18*BOHRPERA

    manager=TFMolManage("chemspider12_nosolvation", tset,False,"fc_sqdiff_BP_Direct_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize_Dropout",False,False)
    return manager

manager = GetChemSpiderNetwork(a, False) # load chemspider network

def EnAndForce(x_, DoForce=True):
    mtmp = Mol(m.atoms,x_)
    Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = manager.EvalBPDirectEEUpdateSingle(mtmp, PARAMS["AN1_r_Rc"], PARAMS["AN1_a_Rc"], PARAMS["EECutoffOff"], True)
    energy = Etotal[0]
    force = gradient[0]
    if DoForce:
        return energy, force
    else:
        return energy


# Perform geometry optimization
PARAMS["OptMaxCycles"]= 2000
PARAMS["OptThresh"] =0.00002
Opt = GeomOptimizer(EnAndForce)
m=Opt.Opt(m)
