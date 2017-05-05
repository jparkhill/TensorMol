"""
 Calculate an embeeding for a molecule, such as coulomb matrix
 Todo: Should inherit from Digest.py (which needs cleanup)
"""

from Mol import *
from Util import *

class MolDigester:
	def __init__(self, eles_, name_="Coulomb", OType_="FragEnergy", SensRadius_=6):
		self.name = name_
		self.OType = OType_
		self.lshape = None  # output is just the energy
		self.eshape = None
		self.SensRadius = SensRadius_
		self.eles = eles_
		self.neles = len(eles_) # Consistent list of atoms in the order they are treated.
		self.ngrid = 5 #this is a shitty parameter if we go with anything other than RDF and should be replaced.
		self.nsym = self.neles+(self.neles+1)*self.neles  # channel of sym functions

	def make_sym_update(self, mol, Rc = 4.0, g1_para_mat = None, g2_para_mat = None): # para_mat should contains the parameters of Sym_Func, on the order of: G1:Rs, eta1 ; G2: zeta, eta2
		if g1_para_mat == None or g2_para_mat == None: # use the default setting
			ngrid = 5
		Rs = []; eta1 = []; zeta = []; eta2 = []
		for i in range (0, ngrid):
			zeta.append(1.5**i); eta1.append(0.008*(2**i)); eta2.append(0.002*(2**i)); Rs.append(i*Rc/float(ngrid))
		g1_para_mat = np.array(np.meshgrid(Rs, eta1)).T.reshape((-1,2))
		g2_para_mat = np.array(np.meshgrid(zeta, eta2)).T.reshape((-1,2))
		SYM_Ins = MolEmb.Make_Sym_Update(mol.coords,  mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), Rc, g1_para_mat,g2_para_mat,  -1) # -1 means do it for all atoms
		SYM_Ins_deri = np.zeros((SYM_Ins.shape[0], SYM_Ins.shape[1]))
		return SYM_Ins, SYM_Ins_deri

	def make_sym(self, mol):
		zeta=[]
		eta1=[]
		eta2=[]
		Rs=[]
		eles = list(set(list(mol.atoms)))
		SYM = []
		for i in range (0, self.ngrid):
			zeta.append(1.5**i)    # set some value for zeta, eta, Rs
			eta1.append(0.008*(2**i))
			eta2.append(0.002*(2**i))
			Rs.append(i*self.SensRadius/float(self.ngrid))
		for i in range (0, mol.NAtoms()):
			sym =  MolEmb.Make_Sym(mol.coords, (mol.coords[i]).reshape((1,-1)), mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), i, self.SensRadius, zeta, eta1, eta2, Rs)
			sym = np.asarray(sym[0], dtype=np.float32)
			sym = sym.reshape((self.nsym*sym.shape[1] *  sym.shape[2]))
			#print "sym", sym
			SYM.append(sym)
		SYM =  np.asarray(SYM)
		print "mol.atoms", mol.atoms, "SYM", SYM
		SYM_deri = np.zeros((SYM.shape[0], SYM.shape[1])) # debug, it will take some work to implement to derivative of sym func.
		return SYM, SYM_deri

	def make_cm_bp(self, mol):
		CM_BP = []
		ngrids = 10
		for i in range (0, mol.NAtoms()):
			cm_bp = MolEmb.Make_CM(mol.coords, (mol.coords[i]).reshape((1,-1)), mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), self.SensRadius, ngrids, i,  0.0 )
			cm_bp = np.asarray(cm_bp[0], dtype=np.float32)
			cm_bp.flatten()
			CM_BP.append(cm_bp)
		CM_BP = np.array(CM_BP)
		CM_BP = CM_BP.reshape((CM_BP.shape[0],-1))
		CM_BP_deri = np.zeros((CM_BP.shape[0], CM_BP.shape[1])) # debug, it will take some work to implement to derivative of coloumb_bp func.
		return 	CM_BP, CM_BP_deri

	def make_cm_bond_bp(self, mol):
		CM_Bond_BP = []
		ngrids = 10
		for i in range (0, mol.NBonds()):
			cm_bp_1 = MolEmb.Make_CM(mol.coords, mol.coords[int(mol.bonds[i][2])].reshape((1,-1)), mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), self.SensRadius, ngrids, int(mol.bonds[i][2]),  0.0 )
			cm_bp_2 = MolEmb.Make_CM(mol.coords, mol.coords[int(mol.bonds[i][3])].reshape((1,-1)), mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), self.SensRadius, ngrids, int(mol.bonds[i][3]),  0.0 )
			dist = mol.bonds[i][1]
			cm_bp_1  = cm_bp_1[0][0].reshape(-1)
			cm_bp_2  = cm_bp_2[0][0].reshape(-1)
			cm_bond_bp = np.asarray(list(cm_bp_1)+list(cm_bp_2)+[1.0/dist], dtype=np.float32)
			cm_bond_bp.flatten()
			CM_Bond_BP.append(cm_bond_bp)
		CM_Bond_BP = np.array(CM_Bond_BP)
		CM_Bond_BP_deri = np.zeros((CM_Bond_BP.shape[0], CM_Bond_BP.shape[1])) # debug, it will take some work to implement to derivative of coloumb_bp func.
		return  CM_Bond_BP, CM_Bond_BP_deri

	def make_ANI1_sym(self, mol, MakeGradients_ = False):  # ANI-1 default setting
		eles = list(set(list(mol.atoms)))
		ANI1_Ins = MolEmb.Make_ANI1_Sym(PARAMS, mol.coords,  mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), -1) # -1 means do it for all atoms
		if (MakeGradients_):
			ANI1_Ins_deri = MolEmb.Make_ANI1_Sym_deri(PARAMS, mol.coords,  mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), -1)
			return ANI1_Ins, ANI1_Ins_deri
		else:
			return ANI1_Ins, None

	def make_ANI1_sym_bond_bp(self, mol):  # ANI-1 default setting
		ANI1_Ins_bond_bp = []
		for i in range (0, mol.NBonds()):
			ANI1_Ins_1 = MolEmb.Make_ANI1_Sym(PARAMS, mol.coords,  mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), int(mol.bonds[i][2])) # -1 means do it for all atoms
			ANI1_Ins_2 = MolEmb.Make_ANI1_Sym(PARAMS, mol.coords,  mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), int(mol.bonds[i][3])) # -1 means do it for all atoms
			dist = mol.bonds[i][1]
			ANI1_Ins_1 = ANI1_Ins_1.reshape((-1))
			ANI1_Ins_2 = ANI1_Ins_2.reshape((-1))
			ANI1_Ins_bond_bp.append(np.asarray(list(ANI1_Ins_1)+list(ANI1_Ins_2)+[1.0/dist], dtype=np.float32))
		ANI1_Ins_bond_bp = np.asarray(ANI1_Ins_bond_bp)
		ANI1_Ins_bond_bp_deri = np.zeros((ANI1_Ins_bond_bp.shape[0], ANI1_Ins_bond_bp.shape[1]))
		return ANI1_Ins_bond_bp, ANI1_Ins_bond_bp_deri

	def make_ANI1_sym_center_bond_bp(self, mol):  # ANI-1 default setting
		ANI1_Ins_bond_bp = []
		for i in range (0, mol.NBonds()):
			center = (mol.coords[ int(mol.bonds[i][2])] + mol.coords[ int(mol.bonds[i][3])] ) /2.0
			tmpcoords = np.zeros((mol.NAtoms()+1, 3))
			tmpcoords[:-1] = mol.coords
			tmpcoords[-1] = center
			tmpatoms = np.zeros(mol.NAtoms()+1)
			tmpatoms[:-1] = mol.atoms
			tmpatoms[-1] = 0
			ANI1_Ins = MolEmb.Make_ANI1_Sym(PARAMS, tmpcoords, tmpatoms.astype(np.uint8), self.eles.astype(np.uint8), mol.NAtoms()) # -1 means do it for all atoms
			dist = mol.bonds[i][1]
			ANI1_Ins = ANI1_Ins.reshape((-1))
			ANI1_Ins_bond_bp.append(np.asarray(list(ANI1_Ins)+[1.0/dist], dtype=np.float32))
		ANI1_Ins_bond_bp = np.asarray(ANI1_Ins_bond_bp)
		ANI1_Ins_bond_bp_deri = np.zeros((ANI1_Ins_bond_bp.shape[0], ANI1_Ins_bond_bp.shape[1]))
		return ANI1_Ins_bond_bp, ANI1_Ins_bond_bp_deri

	def make_cm_bp(self, mol):
		CM_BP = []
		ngrids = 10
		for i in range (0, mol.NAtoms()):
			cm_bp = MolEmb.Make_CM(mol.coords, (mol.coords[i]).reshape((1,-1)), mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), self.SensRadius, ngrids, i,  0.0 )
			cm_bp = np.asarray(cm_bp[0], dtype=np.float32)
			cm_bp.flatten()
			CM_BP.append(cm_bp)
		CM_BP = np.array(CM_BP)
		CM_BP = CM_BP.reshape((CM_BP.shape[0],-1))
		CM_BP_deri = np.zeros((CM_BP.shape[0], CM_BP.shape[1])) # debug, it will take some work to implement to derivative of coloumb_bp func.
		return 	CM_BP, CM_BP_deri

	def make_connectedbond_bond_bp(self, mol):
		ConnectedBond_Bond_BP = []
		for i in range (0, mol.NBonds()):
			atom1 = int(mol.bonds[i][2])
			atom2 = int(mol.bonds[i][3])
			#print "bond :", i, "index:", atom1, atom2, "ele type:", mol.atoms[atom1], mol.atoms[atom2]
			connected_bond1 = [[] for j in range (0, self.neles)]
			for node in mol.atom_nodes[atom1].connected_nodes:
				atom_index = node.node_index
				if atom_index != atom2:
					dist = mol.DistMatrix[atom1][atom_index]
					connected_bond1[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
			connected_bond1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond1 ]
			connected_bond1 = [item for sublist in connected_bond1 for item in sublist]

			connected_bond2 = [[] for j in range (0, self.neles)]
			for node in mol.atom_nodes[atom2].connected_nodes:
				atom_index = node.node_index
				if atom_index != atom1:
					dist = mol.DistMatrix[atom2][atom_index]
					connected_bond2[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
			connected_bond2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond2 ]
			connected_bond2 = [item for sublist in connected_bond2 for item in sublist]

			dist = mol.bonds[i][1]
			connectedbond_bond_bp = np.asarray(connected_bond1+ connected_bond2 +[1.0/dist], dtype=np.float32)
			#print "connectedbond_bond_bp", connectedbond_bond_bp
			ConnectedBond_Bond_BP.append(connectedbond_bond_bp)
		ConnectedBond_Bond_BP = np.array(ConnectedBond_Bond_BP)
		ConnectedBond_Bond_BP_deri = np.zeros((ConnectedBond_Bond_BP.shape[0], ConnectedBond_Bond_BP.shape[1]))
		return  ConnectedBond_Bond_BP, ConnectedBond_Bond_BP_deri

	def make_connectedbond_angle_bond_bp(self, mol):
		self.ngrid = 3
		ConnectedBond_Angle_Bond_BP = []
		for i in range (0, mol.NBonds()):
			atom1 = int(mol.bonds[i][2])
			atom2 = int(mol.bonds[i][3])
			bond_dist =  mol.bonds[i][1]
			connected_bond1 = [[] for j in range (0, self.neles)]
			connected_angle1 = [[] for j in range (0, self.neles)]
			for node in mol.atom_nodes[atom1].connected_nodes:
				atom_index = node.node_index
				if atom_index != atom2:
					dist = mol.DistMatrix[atom1][atom_index]
					dist2 = mol.DistMatrix[atom2][atom_index]
					connected_bond1[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
					connected_angle1[list(self.eles).index(mol.atoms[atom_index])].append((bond_dist**2+dist**2-dist2**2)/(2*bond_dist*dist))
			#print "before sorting 1:",connected_bond1, connected_angle1
			for j  in range (0, len(connected_bond1)):
				connected_angle1[j] = [x for (y, x) in sorted(zip(connected_bond1[j], connected_angle1[j]))]
				connected_bond1[j].sort()
				connected_bond1[j].reverse()
				connected_angle1[j].reverse()
			connected_bond1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond1 ]
			connected_bond1 = [item for sublist in connected_bond1 for item in sublist]
			connected_angle1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_angle1 ]
			connected_angle1 = [item for sublist in connected_angle1 for item in sublist]

			connected_bond2 = [[] for j in range (0, self.neles)]
			connected_angle2 = [[] for j in range (0, self.neles)]
			for node in mol.atom_nodes[atom2].connected_nodes:
				atom_index = node.node_index
				if atom_index != atom1:
					dist = mol.DistMatrix[atom2][atom_index]
					dist2 = mol.DistMatrix[atom1][atom_index]
					connected_bond2[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
					connected_angle2[list(self.eles).index(mol.atoms[atom_index])].append((bond_dist**2+dist**2-dist2**2)/(2*bond_dist*dist))

			for j  in range (0, len(connected_bond2)):
				connected_angle2[j] = [x for (y, x) in sorted(zip(connected_bond2[j], connected_angle2[j]))]
				connected_bond2[j].sort()
				connected_bond2[j].reverse()
				connected_angle2[j].reverse()
			connected_bond2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond2 ]
			connected_bond2 = [item for sublist in connected_bond2 for item in sublist]
			connected_angle2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_angle2 ]
			connected_angle2 = [item for sublist in connected_angle2 for item in sublist]
			connectedbond_angle_bond_bp = np.asarray(connected_bond1 + connected_bond2 + connected_angle1 + connected_angle2 + [1.0/bond_dist], dtype=np.float32)
			ConnectedBond_Angle_Bond_BP.append(connectedbond_angle_bond_bp)
		ConnectedBond_Angle_Bond_BP = np.array(ConnectedBond_Angle_Bond_BP)
		ConnectedBond_Angle_Bond_BP_deri = np.zeros((ConnectedBond_Angle_Bond_BP.shape[0], ConnectedBond_Angle_Bond_BP.shape[1]))
		return  ConnectedBond_Angle_Bond_BP, ConnectedBond_Angle_Bond_BP_deri

	def make_gauinv(self, mol):
		""" This is a totally inefficient way of doing this
			MolEmb should loop atoms. """
		GauInv = []
		for i in range (0, mol.NAtoms()):
			gauinv = MolEmb.Make_Inv(mol.coords, (mol.coords[i]).reshape((1,-1)), mol.atoms.astype(np.uint8),self.SensRadius, i)
			gauinv = np.asarray(gauinv[0], dtype = np.float32 )
			gauinv = gauinv.reshape(-1)
			GauInv.append(gauinv)
		GauInv = np.asarray(GauInv)
		GauInv_deri = np.zeros((GauInv.shape[0], GauInv.shape[1]))
		return GauInv, GauInv_deri

	def make_cm(self, mol_):
		natoms  = mol_.NAtoms()
		CM=np.zeros((natoms, natoms))
		deri_CM = np.zeros((natoms, natoms, 6))
		xyz = (mol_.coords).copy()
		#ele = (mol_.atoms).copy()
		ele = np.zeros(natoms)  # for eebug
		ele.fill(1.0)   # for debug, set ele value to 1
		code = """
		double dist = 0.0;
		double deri[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		for (int j=0; j < natoms; j++) {
			for (int k=0; k<natoms; k++) {
				if (k==j) {
					dist=1.0;
					deri[0] =0.0;
					deri[1] =0.0;
					deri[2] =0.0;
					deri[3] =0.0;
					deri[4] =0.0;
					deri[5] =0.0;
				}
				else {
					dist=sqrt(pow(xyz[j*3+0]-xyz[k*3+0],2) + pow(xyz[j*3+1]-xyz[k*3+1],2) + pow(xyz[j*3+2]-xyz[k*3+2],2));
					deri[0] = -(xyz[j*3+0]-xyz[k*3+0])/(dist*dist*dist);
					deri[1] = -(xyz[j*3+1]-xyz[k*3+1])/(dist*dist*dist);
					deri[2] = -(xyz[j*3+2]-xyz[k*3+2])/(dist*dist*dist);
					deri[3] = -deri[0];
					deri[4] = -deri[1];
					deri[5] = -deri[2];
				}
				CM[natoms*j+k]= ele[j]*ele[k]/dist;
				deri_CM[natoms*j*6+k*6+0]=ele[j]*ele[k]*deri[0];
				deri_CM[natoms*j*6+k*6+1]=ele[j]*ele[k]*deri[1];
				deri_CM[natoms*j*6+k*6+2]=ele[j]*ele[k]*deri[2];
				deri_CM[natoms*j*6+k*6+3]=ele[j]*ele[k]*deri[3];
				deri_CM[natoms*j*6+k*6+4]=ele[j]*ele[k]*deri[4];
				deri_CM[natoms*j*6+k*6+5]=ele[j]*ele[k]*deri[5];
			}

		}

		"""
		res = inline(code, ['CM','natoms','xyz','ele', 'deri_CM'],headers=['<math.h>','<iostream>'], compiler='gcc')
		return CM, deri_CM

	def GetUpTri(self, CM):
		CM = CM[np.triu_indices(CM.shape[0], 1)].copy() ##for debug purpose, ignore the diagnoal element
		#index = np.array([2,3,4,6,7,8,9,10,11]) # for debug, ignore the AA, BB block, only remain AB
		#CM = CM[index].copy()  # for debug
		return  CM  #for debug purpose, ignore the diagnoal element

	def EvalDigest(self, mol_):
		return self.Emb(mol_,False, True)

	def Emb(self, mol_, MakeOutputs=True, MakeGradients=False):
		"""
		Generates various molecular embeddings.
		If the embedding has BP on the end it comes out atomwise and includes all atoms in the molecule.
		Args:
			mol_: a Molecule to be digested
			MakeOutputs: generates outputs according to self.OType.
			MakeGradients: generates outputs according to self.OType.
		Returns:
			Output embeddings, and possibly labels and gradients.
		Todo:
			Hook up the gradients.
		"""
		Ins=None
		Grads=None
		Outs=None
		if (self.name == "Coulomb"):
			CM, deri_CM = self.make_cm(mol_)
			Ins = self.GetUpTri(CM)
		elif(self.name == "Coulomb_BP"):
			Ins, deri_CM_BP =  self.make_cm_bp(mol_)
			Ins = Ins.reshape([Ins.shape[0],-1])
		elif(self.name == "Coulomb_Bond_BP"):
		    Ins, Grads =  self.make_cm_bond_bp(mol_)
		    Ins = Ins.reshape([Ins.shape[0],-1])
		elif(self.name == "SymFunc"):
			Ins, SYM_deri = self.make_sym(mol_)
		elif(self.name == "SymFunc_Update"):
			Ins, Grads = self.make_sym_update(mol_)
		elif(self.name == "GauInv_BP"):
			Ins =  MolEmb.Make_Inv(PARAMS, mol_.coords, mol_.atoms, -1)
		elif(self.name == "GauSH_BP"):
		    # Here I have to add options to octahedrally average SH
			Ins =  MolEmb.Make_SH(PARAMS, mol_.coords, mol_.atoms, -1)
		elif(self.name == "ConnectedBond_Bond_BP"):
			Ins, Grads =  self.make_connectedbond_bond_bp(mol_)
			Ins = Ins.reshape([Ins.shape[0],-1])
		elif(self.name == "ConnectedBond_Angle_Bond_BP"):
			Ins, Grads = self.make_connectedbond_angle_bond_bp(mol_)
		elif(self.name == "ANI1_Sym"):
			Ins, Grads = self.make_ANI1_sym(mol_, MakeGradients_ = MakeGradients)
		elif(self.name == "ANI1_Sym_Bond_BP"):
			Ins, Grads = self.make_ANI1_sym_bond_bp(mol_)
		elif(self.name == "ANI1_Sym_Center_Bond_BP"):
			Ins, Grads = self.make_ANI1_sym_center_bond_bp(mol_)
		else:
			raise Exception("Unknown MolDigester Type.", self.name)
		if (self.eshape == None):
			self.eshape=Ins.shape[1:] # The first dimension is atoms. eshape is per-atom.
		if (MakeOutputs):
			if (self.OType == "Energy"):
				Outs = np.array([mol_.properties["energy"]])
			elif (self.OType == "AtomizationEnergy"):
			    Outs = np.array([mol_.properties["atomization"]])
			elif (self.OType == "EleEmbAtEn"):
				if (PARAMS["EEOrder"]==2):
					AE = mol_.properties["atomization"]
					if (PARAMS["EEVdw"]==True):
						AE -= mol_.properties["Vdw"]
					Outs = np.zeros(5) # AtEnergy, monopole, 3-dipole.
					Outs[0] = AE
					Outs[1] = 0.0
					Outs[1:] = mol_.properties["dipole"]
				else:
					raise Exception("Code higher orders... ")
			elif (self.OType == "Atomization_novdw"):
			    Outs = np.array([mol_.properties["atomization"] - mol_.properties["vdw"]])
			elif (self.OType == "FragEnergy"):
				Outs = np.array([mol_.properties["frag_mbe_energy"]])
			elif (self.OType == "GoEnergy"):
				Outs = np.array([mol_.GoEnergy()])
			else:
				raise Exception("Unknown Output Type... "+self.OType)

			if (self.lshape == None):
				self.lshape=Outs.shape
			if (MakeGradients):
				return Ins, Grads, Outs
			else:
				return Ins, Outs
		else:
			if (MakeGradients):
                                return Ins, Grads
			else:
				return Ins

	def TrainDigest(self, mol_):
		"""
		Returns list of inputs and outputs for a molecule.
		Uses self.Emb() uses Mol to get the Desired output type (Energy,Force,Probability etc.)
		Args:
			mol_: a molecule to be digested
		"""
		return self.Emb(mol_,True,False)

	def Print(self):
		print "Digest name: ", self.name
