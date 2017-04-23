from TensorMol import *
import cProfile

def Write_LEDA(mol, folder="magna"):
	leda = open(folder+"/"+mol.name.split()[-1]+".gw","w+")
	leda.write(mol.name+"\nstring\nlong\n-2\n"+str(mol.NAtoms())+"\n")
	for i in range (0, mol.NAtoms()):
		leda.write("|{"+str(mol.atoms[i])+"_"+str(i)+"}|\n")
	leda.write(str(mol.NBonds())+"\n")
	for i in range (0, mol.NBonds()):
		leda.write(str(int(mol.bonds[i][2]))+" "+str(int(mol.bonds[i][3]))+" 0 "+"|{0}|\n")
	leda.close()
	return

def Wirte_Similarity(mol_1, mol_2, folder = "magna"):
	if mol_1.NAtoms() > mol_2.NAtoms():
		tmp_mol = mol_1
		mol_1 = mol_2
		mol_2 = tmp_mol
	mat_f = open(folder+"/"+mol_1.name.split()[-1]+"_"+mol_2.name.split()[-1]+".txt","w+")
	mol_1_mat=mol_1.atoms.reshape((mol_1.NAtoms(), 1))
	mol_1_mat=np.repeat(mol_1_mat, mol_2.NAtoms(),axis=1)
	mol_2_mat=mol_2.atoms.reshape((1, mol_2.NAtoms()))
	mol_2_mat=np.repeat(mol_2_mat, mol_1.NAtoms(),axis=0)
	simi_mat = (np.equal(mol_1_mat, mol_2_mat)).astype(int)
	np.savetxt(folder+"/"+mol_1.name.split()[-1]+"_"+mol_2.name.split()[-1]+".txt", simi_mat, fmt='%i', header=str(mol_1.NAtoms())+" "+str(mol_2.NAtoms()), comments="")
	return

if (1):
	a = MSet("MAGNA_test")
        a.ReadXYZ("MAGNA_test")
        a.Make_Graphs()
	for mol in a.mols:
		Write_LEDA(mol)
	for i in range (0, len(a.mols)):
		for j in range (i+1, len(a.mols)):	
			Wirte_Similarity(a.mols[i], a.mols[j])

