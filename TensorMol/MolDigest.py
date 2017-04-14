"""
 Calculate an embeeding for a molecule, such as coulomb matrix
 Todo: Should inherit from Digest.py (which needs cleanup)
"""

from Mol import *
from Util import *

class MolDigester:
	def __init__(self, eles_, name_="Coulomb", OType_="FragEnergy", SensRadius_=5):
		self.name = name_
		self.OType = OType_
		self.lshape = None  # output is just the energy
		self.eshape = None
		self.SensRadius = SensRadius_
		self.eles = eles_
		self.neles = len(eles_) # Consistent list of atoms in the order they are treated.
		self.ngrid = 3 #this is a shitty parameter if we go with anything other than RDF and should be replaced.
		self.nsym = self.neles+(self.neles+1)*self.neles  # channel of sym functions

	def AssignNormalization(self,mn,sn):
		self.MeanNorm=mn
		self.StdNorm=sn
		return

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

	def make_ANI1_sym(self, mol, r_Rc = 4.6, a_Rc = 3.1, eta = 4.00, zeta = 8.00, num_r_Rs = 32, num_a_Rs = 8, num_a_As =8):  # ANI-1 default setting
		eles = list(set(list(mol.atoms)))
		r_Rs = [ r_Rc*i/num_r_Rs for i in range (0, num_r_Rs)]
		a_Rs = [ a_Rc*i/num_a_Rs for i in range (0, num_a_Rs)]
		a_As = [ 2.0*math.pi*i/num_a_As for i in range (0, num_a_As)]
		ANI1_Ins = MolEmb.Make_ANI1_Sym(mol.coords,  mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), r_Rc, a_Rc, r_Rs, a_Rs, a_As, eta, zeta, , -1) # -1 means do it for all atoms
		ANI1_Ins_deri = np.zeros((ANI1_Ins.shape[0], ANI1_Ins.shape[1]))
		return ANI1_Ins, ANI1_Ins_deri


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
			#print list(cm_bp_1), list(cm_bp_2), dist
                        cm_bond_bp = np.asarray(list(cm_bp_1)+list(cm_bp_2)+[1.0/dist], dtype=np.float32)
			#print  "cm_bond_bp:", cm_bond_bp
                        cm_bond_bp.flatten()
                        CM_Bond_BP.append(cm_bond_bp)
                CM_Bond_BP = np.array(CM_Bond_BP)
                CM_Bond_BP_deri = np.zeros((CM_Bond_BP.shape[0], CM_Bond_BP.shape[1])) # debug, it will take some work to implement to derivative of coloumb_bp func.
                return  CM_Bond_BP, CM_Bond_BP_deri

	def make_rdf_bond_bp(self, mol):
		RDF_Bond_BP = []
		ngrids = 50
		width = 0.1 
		for i in range (0, mol.NBonds()):
			rdf_bp_1 = MolEmb.Make_RDF(mol.coords, mol.coords[int(mol.bonds[i][2])].reshape((1,-1)), mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), self.SensRadius, ngrids, int(mol.bonds[i][2]),  width)
			rdf_bp_2 = MolEmb.Make_RDF(mol.coords, mol.coords[int(mol.bonds[i][3])].reshape((1,-1)), mol.atoms.astype(np.uint8), self.eles.astype(np.uint8), self.SensRadius, ngrids, int(mol.bonds[i][3]),  width)
			rdf_bp_1  = rdf_bp_1[0][0].reshape(-1)
                        rdf_bp_2  = rdf_bp_2[0][0].reshape(-1)
			#print "rdf_bp_1:", rdf_bp_1, "atom type:", mol.atoms[int(mol.bonds[i][2])]
			dist = mol.bonds[i][1]
			rdf_bond_bp = np.asarray(list(rdf_bp_1)+list(rdf_bp_2)+[1.0/dist], dtype=np.float32)
			rdf_bond_bp.flatten()
                        RDF_Bond_BP.append(rdf_bond_bp)
                RDF_Bond_BP = np.array(RDF_Bond_BP)
                RDF_Bond_BP_deri = np.zeros((RDF_Bond_BP.shape[0], RDF_Bond_BP.shape[1])) # debug, it will take some work to implement to derivative of coloumb_bp func.
                return  RDF_Bond_BP, RDF_Bond_BP_deri

        def make_dist_bond_bp(self, mol):
                Dist_Bond_BP = []
                for i in range (0, mol.NBonds()):
                        dist = mol.bonds[i][1]
                        #print list(cm_bp_1), list(cm_bp_2), dist
                        dist_bond_bp = np.asarray([1.0/dist], dtype=np.float32)
                        #print  "cm_bond_bp:", cm_bond_bp
                        Dist_Bond_BP.append(dist_bond_bp)
                Dist_Bond_BP = np.array(Dist_Bond_BP)
                Dist_Bond_BP_deri = np.zeros((Dist_Bond_BP.shape[0], Dist_Bond_BP.shape[1])) # debug, it will take some work to implement to derivative of coloumb_bp func.
                return  Dist_Bond_BP, Dist_Bond_BP_deri

	def make_connectedbond_cm_bond_bp(self, mol):
		CM_Bond_BP, CM_Bond_BP_deri = self.make_cm_bond_bp(mol)
		ConnectedBond_Bond_BP, ConnectedBond_Bond_BP_deri = self.make_connectedbond_bond_bp(mol)
		#print "ConnectedBond_Bond_BP.shape[1]", ConnectedBond_Bond_BP.shape, "CM_Bond_BP.shape[1]", CM_Bond_BP.shape
		ConnectedBond_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
		ConnectedBond_CM_Bond_BP[:,:ConnectedBond_Bond_BP.shape[1]] = ConnectedBond_Bond_BP
		ConnectedBond_CM_Bond_BP[:,ConnectedBond_Bond_BP.shape[1]:] = CM_Bond_BP
		deri_ConnectedBond_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
		return ConnectedBond_CM_Bond_BP, deri_ConnectedBond_CM_Bond_BP

        def make_connectedbond_cm_bp(self, mol):
                CM_BP, CM_BP_deri = self.make_cm_bp(mol)
                ConnectedBond_BP, ConnectedBond_deri = self.make_connectedbond_bp(mol)
		#print "ConnectedBond_BP.shape", ConnectedBond_BP.shape, "CM_BP.shape:", CM_BP.shape
                ConnectedBond_CM_BP = np.zeros((mol.NAtoms(), ConnectedBond_BP.shape[1]+CM_BP.shape[1]))
                ConnectedBond_CM_BP[:,:ConnectedBond_BP.shape[1]] = ConnectedBond_BP
                ConnectedBond_CM_BP[:,ConnectedBond_BP.shape[1]:] = CM_BP
                deri_ConnectedBond_CM_BP = np.zeros((mol.NAtoms(), ConnectedBond_BP.shape[1]+CM_BP.shape[1]))
                return ConnectedBond_CM_BP, deri_ConnectedBond_CM_BP

        def make_connectedbond_angle_cm_bond_bp(self, mol):
		CM_Bond_BP, CM_Bond_BP_deri = self.make_cm_bond_bp(mol)
                ConnectedBond_Angle_Bond_BP, ConnectedBond_Angle_Bond_BP_deri = self.make_connectedbond_angle_bond_bp(mol)
                ConnectedBond_Angle_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
                ConnectedBond_Angle_CM_Bond_BP[:,:ConnectedBond_Angle_Bond_BP.shape[1]] = ConnectedBond_Angle_Bond_BP
                ConnectedBond_Angle_CM_Bond_BP[:,ConnectedBond_Angle_Bond_BP.shape[1]:] = CM_Bond_BP
                deri_ConnectedBond_Angle_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
                return ConnectedBond_Angle_CM_Bond_BP, deri_ConnectedBond_Angle_CM_Bond_BP

	def make_connectedbond_angle_rdf_bond_bp(self, mol):
                RDF_Bond_BP, RDF_Bond_BP_deri = self.make_rdf_bond_bp(mol)
                ConnectedBond_Angle_Bond_BP, ConnectedBond_Angle_Bond_BP_deri = self.make_connectedbond_angle_bond_bp(mol)
                ConnectedBond_Angle_RDF_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Bond_BP.shape[1]+ RDF_Bond_BP.shape[1]))
                ConnectedBond_Angle_RDF_Bond_BP[:,:ConnectedBond_Angle_Bond_BP.shape[1]] = ConnectedBond_Angle_Bond_BP
                ConnectedBond_Angle_RDF_Bond_BP[:,ConnectedBond_Angle_Bond_BP.shape[1]:] = RDF_Bond_BP
                deri_ConnectedBond_Angle_RDF_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Bond_BP.shape[1]+RDF_Bond_BP.shape[1]))
                return ConnectedBond_Angle_RDF_Bond_BP, deri_ConnectedBond_Angle_RDF_Bond_BP



        def make_connectedbond_angle_2_cm_bond_bp(self, mol):
                CM_Bond_BP, CM_Bond_BP_deri = self.make_cm_bond_bp(mol)
                ConnectedBond_Angle_Bond_BP, ConnectedBond_Angle_Bond_BP_deri = self.make_connectedbond_angle_2_bond_bp(mol)
		#print "ConnectedBond_Angle_Bond_BP:",ConnectedBond_Angle_Bond_BP
                ConnectedBond_Angle_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
                ConnectedBond_Angle_CM_Bond_BP[:,:ConnectedBond_Angle_Bond_BP.shape[1]] = ConnectedBond_Angle_Bond_BP
                ConnectedBond_Angle_CM_Bond_BP[:,ConnectedBond_Angle_Bond_BP.shape[1]:] = CM_Bond_BP
                deri_ConnectedBond_Angle_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
                return ConnectedBond_Angle_CM_Bond_BP, deri_ConnectedBond_Angle_CM_Bond_BP

        def make_connectedbond_angle_dihed_cm_bond_bp(self, mol):
                CM_Bond_BP, CM_Bond_BP_deri = self.make_cm_bond_bp(mol)
                ConnectedBond_Angle_Dihed_Bond_BP, ConnectedBond_Angle_Dihed_Bond_BP_deri = self.make_connectedbond_angle_dihed_bond_bp(mol)
                ConnectedBond_Angle_Dihed_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Dihed_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
                ConnectedBond_Angle_Dihed_CM_Bond_BP[:,:ConnectedBond_Angle_Dihed_Bond_BP.shape[1]] = ConnectedBond_Angle_Dihed_Bond_BP
                ConnectedBond_Angle_Dihed_CM_Bond_BP[:,ConnectedBond_Angle_Dihed_Bond_BP.shape[1]:] = CM_Bond_BP
                deri_ConnectedBond_Angle_Dihed_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Dihed_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
                return ConnectedBond_Angle_Dihed_CM_Bond_BP, deri_ConnectedBond_Angle_Dihed_CM_Bond_BP

        def make_connectedbond_angle_dihed_2_cm_bond_bp(self, mol):
                CM_Bond_BP, CM_Bond_BP_deri = self.make_cm_bond_bp(mol)
                ConnectedBond_Angle_Dihed_Bond_BP, ConnectedBond_Angle_Dihed_Bond_BP_deri = self.make_connectedbond_angle_dihed_2_bond_bp(mol)
                ConnectedBond_Angle_Dihed_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Dihed_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
                ConnectedBond_Angle_Dihed_CM_Bond_BP[:,:ConnectedBond_Angle_Dihed_Bond_BP.shape[1]] = ConnectedBond_Angle_Dihed_Bond_BP
                ConnectedBond_Angle_Dihed_CM_Bond_BP[:,ConnectedBond_Angle_Dihed_Bond_BP.shape[1]:] = CM_Bond_BP
                deri_ConnectedBond_Angle_Dihed_CM_Bond_BP = np.zeros((mol.NBonds(), ConnectedBond_Angle_Dihed_Bond_BP.shape[1]+CM_Bond_BP.shape[1]))
                return ConnectedBond_Angle_Dihed_CM_Bond_BP, deri_ConnectedBond_Angle_Dihed_CM_Bond_BP


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

	
        def make_connectedbond_bp(self, mol):
                ConnectedBond_BP = []
		self.ngrid = 4
                for i in range (0, mol.NAtoms()):
                        #atom = int(mol.atoms[i])
                        #print "bond :", i, "index:", atom1, atom2, "ele type:", mol.atoms[atom1], mol.atoms[atom2] 
                        connected_atom = [[] for j in range (0, self.neles)]
                        for node in mol.atom_nodes[i].connected_nodes:
                                atom_index = node.node_index
                                dist = mol.DistMatrix[i][atom_index]
                                connected_atom[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
                        connected_atom = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_atom ]
                        connected_atom = [item for sublist in connected_atom for item in sublist]

                        connectedbond_bp = np.asarray(connected_atom, dtype=np.float32)
                        ConnectedBond_BP.append(connectedbond_bp)
                ConnectedBond_BP = np.array(ConnectedBond_BP)
                ConnectedBond_BP_deri = np.zeros((ConnectedBond_BP.shape[0], ConnectedBond_BP.shape[1]))
		#print "ConnectedBond_BP", ConnectedBond_BP
                return  ConnectedBond_BP, ConnectedBond_BP_deri

        def make_connectedbond_angle_bond_bp(self, mol):
		self.ngrid = 3
                ConnectedBond_Angle_Bond_BP = []
                for i in range (0, mol.NBonds()):
                        atom1 = int(mol.bonds[i][2])
                        atom2 = int(mol.bonds[i][3])
			bond_dist =  mol.bonds[i][1]
                        #print "bond :", i, "index:", atom1, atom2, "ele type:", mol.atoms[atom1], mol.atoms[atom2] 
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
			#print "after sorting 1:", connected_bond1, connected_angle1
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
			#print "before sorting 2:",connected_bond2, connected_angle2
			for j  in range (0, len(connected_bond2)):
                                connected_angle2[j] = [x for (y, x) in sorted(zip(connected_bond2[j], connected_angle2[j]))]
                                connected_bond2[j].sort()
				connected_bond2[j].reverse()
                                connected_angle2[j].reverse()
			#print "after sorting 2:",connected_bond2, connected_angle2
                        connected_bond2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond2 ]
                        connected_bond2 = [item for sublist in connected_bond2 for item in sublist]
			connected_angle2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_angle2 ]
                        connected_angle2 = [item for sublist in connected_angle2 for item in sublist]
				

                        connectedbond_angle_bond_bp = np.asarray(connected_bond1 + connected_bond2 + connected_angle1 + connected_angle2 + [1.0/bond_dist], dtype=np.float32)
                        #print "connectedbond_bond_bp", connectedbond_bond_bp
                        ConnectedBond_Angle_Bond_BP.append(connectedbond_angle_bond_bp)
                ConnectedBond_Angle_Bond_BP = np.array(ConnectedBond_Angle_Bond_BP)
                ConnectedBond_Angle_Bond_BP_deri = np.zeros((ConnectedBond_Angle_Bond_BP.shape[0], ConnectedBond_Angle_Bond_BP.shape[1]))
                return  ConnectedBond_Angle_Bond_BP, ConnectedBond_Angle_Bond_BP_deri


        def make_connectedbond_angle_dihed_bond_bp(self, mol):
                self.ngrid = 3
		dihed_max = 9
		ConnectedBond_Angle_Dihed_Bond_BP = []
		#print "mol name", mol.name, "\n\n\n"
                for i in range (0, mol.NBonds()):
                        atom1 = int(mol.bonds[i][2])
                        atom2 = int(mol.bonds[i][3])
                        bond_dist =  mol.bonds[i][1]
                        #print "bond :", i, "index:", atom1, atom2, "ele type:", mol.atoms[atom1], mol.atoms[atom2] 
                        connected_bond1 = [[] for j in range (0, self.neles)]
                        connected_angle1 = [[] for j in range (0, self.neles)]
			connected_dihed1 = [[] for j in range (0, len(dihed_pair.keys()))]
			connected_dihed1_dist1 = [[] for j in range (0, len(dihed_pair.keys()))]
			connected_dihed1_dist2 = [[] for j in range (0, len(dihed_pair.keys()))]
			connected_atom1 = [[] for j in range (0, self.neles)]
                        for node in mol.atom_nodes[atom1].connected_nodes:
                                atom_index = node.node_index
                                if atom_index != atom2:
                                        dist = mol.DistMatrix[atom1][atom_index]
                                        dist2 = mol.DistMatrix[atom2][atom_index]
                                        connected_bond1[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
                                        connected_angle1[list(self.eles).index(mol.atoms[atom_index])].append((bond_dist**2+dist**2-dist2**2)/(2*bond_dist*dist))
					connected_atom1[list(self.eles).index(mol.atoms[atom_index])].append(atom_index)
                        #print "before sorting 1:",connected_bond1, connected_angle1
                        for j  in range (0, len(connected_bond1)):
                                connected_angle1[j] = [x for (y, x) in sorted(zip(connected_bond1[j], connected_angle1[j]))]
				connected_atom1[j] = [x for (y, x) in sorted(zip(connected_bond1[j], connected_atom1[j]))]
                                connected_bond1[j].sort()
                                connected_bond1[j].reverse()
				connected_atom1[j].reverse()
                                connected_angle1[j].reverse()
			#print "connected_atom1:", connected_atom1
			for j in range (0, len(connected_atom1)):
				for k in range (0, len(connected_atom1[j])):
					atom_index = connected_atom1[j][k]
					atom_ele =  mol.atoms[atom_index]
					tmp_dist1 = [[] for m in range (0, self.neles)]
					tmp_dist2 = [[] for m in range (0, self.neles)]
					tmp_dihed = [[] for m in range (0, self.neles)]
					#print "atom_index:", atom_index
					for node in mol.atom_nodes[atom_index].connected_nodes:
						dihed_atom_index = node.node_index
						if dihed_atom_index != atom1:
							dihed_atom_ele_index = list(self.eles).index(mol.atoms[dihed_atom_index])
							dist1 = mol.DistMatrix[atom_index][dihed_atom_index]
							dist2 = mol.DistMatrix[atom_index][atom1]
							tmp_dist1[dihed_atom_ele_index].append(1.0/dist1)
							tmp_dist2[dihed_atom_ele_index].append(1.0/dist2)
							import warnings
							with warnings.catch_warnings():
                                                                warnings.filterwarnings('error')
                                                                try:
                                                                        theta = Dihed_4Points(mol.coords[atom2], mol.coords[atom1], mol.coords[atom_index], mol.coords[dihed_atom_index])
                                                                except Warning as e:
									print "coplane..."
									theta = 0.0
							tmp_dihed[dihed_atom_ele_index].append(math.cos(theta))
					for m in range (0, len(tmp_dist1)):
						tmp_dihed[m] = [x for (y, x) in sorted(zip(tmp_dist1[m], tmp_dihed[m]))]
						tmp_dist1[m].sort()
						tmp_dist1[m].reverse()
						tmp_dihed[m].reverse()
					tmp_dist1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in tmp_dist1]
					tmp_dihed = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in tmp_dihed]
					tmp_dist2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in tmp_dist2]
					for m in range (0, self.neles):
						try:
							pair = int(self.eles[m]*1000 + atom_ele)  # hacky way to find the index
							pair_index = dihed_pair[pair] 
							connected_dihed1[pair_index] += tmp_dihed[m]
							connected_dihed1_dist1[pair_index] += tmp_dist1[m]
							connected_dihed1_dist2[pair_index] += tmp_dist2[m]
						except:
							pass
                        #print "after sorting 1:", connected_bond1, connected_angle1
                        connected_bond1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond1 ]
                        connected_bond1 = [item for sublist in connected_bond1 for item in sublist]
                        connected_angle1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_angle1 ]
                        connected_angle1 = [item for sublist in connected_angle1 for item in sublist]
			#print "atom1:", atom1, "atom2:", atom2,"connected_dihed1:", connected_dihed1, " connected_dihed1_dist1:", connected_dihed1_dist1, "connected_dihed1_dist2", connected_dihed1_dist2
			connected_dihed1 = [b[0:dihed_max] + [0]*(dihed_max-len(b)) for b in connected_dihed1 ]
			connected_dihed1 = [item for sublist in connected_dihed1 for item in sublist]
			connected_dihed1_dist1 = [b[0:dihed_max] + [0]*(dihed_max-len(b)) for b in connected_dihed1_dist1 ]
                        connected_dihed1_dist1 = [item for sublist in connected_dihed1_dist1 for item in sublist]
			connected_dihed1_dist2 = [b[0:dihed_max] + [0]*(dihed_max-len(b)) for b in connected_dihed1_dist2 ]
                        connected_dihed1_dist2 = [item for sublist in connected_dihed1_dist2 for item in sublist]
			#print "connected_dihed1:", connected_dihed1, " connected_dihed1_dist1:", connected_dihed1_dist1, "connected_dihed1_dist2", connected_dihed1_dist2

			connected_bond2 = [[] for j in range (0, self.neles)]
                        connected_angle2 = [[] for j in range (0, self.neles)]
                        connected_dihed2 = [[] for j in range (0, len(dihed_pair.keys()))]
                        connected_dihed2_dist1 = [[] for j in range (0, len(dihed_pair.keys()))]
                        connected_dihed2_dist2 = [[] for j in range (0, len(dihed_pair.keys()))]
                        connected_atom2 = [[] for j in range (0, self.neles)]
                        for node in mol.atom_nodes[atom2].connected_nodes:
                                atom_index = node.node_index
                                if atom_index != atom1:
                                        dist = mol.DistMatrix[atom2][atom_index]
                                        dist2 = mol.DistMatrix[atom1][atom_index]
                                        connected_bond2[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
                                        connected_angle2[list(self.eles).index(mol.atoms[atom_index])].append((bond_dist**2+dist**2-dist2**2)/(2*bond_dist*dist))
                                        connected_atom2[list(self.eles).index(mol.atoms[atom_index])].append(atom_index)
                        #print "before sorting 1:",connected_bond1, connected_angle1
                        for j  in range (0, len(connected_bond2)):
                                connected_angle2[j] = [x for (y, x) in sorted(zip(connected_bond2[j], connected_angle2[j]))]
                                connected_atom2[j] = [x for (y, x) in sorted(zip(connected_bond2[j], connected_atom2[j]))]
                                connected_bond2[j].sort()
                                connected_bond2[j].reverse()
                                connected_atom2[j].reverse()
                                connected_angle2[j].reverse()
                        #print "connected_atom2:", connected_atom2
                        for j in range (0, len(connected_atom2)):
                                for k in range (0, len(connected_atom2[j])):
                                        atom_index = connected_atom2[j][k]
                                        atom_ele =  mol.atoms[atom_index]
                                        tmp_dist1 = [[] for m in range (0, self.neles)]
                                        tmp_dist2 = [[] for m in range (0, self.neles)]
                                        tmp_dihed = [[] for m in range (0, self.neles)]
                                        #print "atom_index:", atom_index
                                        for node in mol.atom_nodes[atom_index].connected_nodes:
                                                dihed_atom_index = node.node_index
                                                if dihed_atom_index != atom2:
                                                        dihed_atom_ele_index = list(self.eles).index(mol.atoms[dihed_atom_index])
                                                        dist1 = mol.DistMatrix[atom_index][dihed_atom_index]
                                                        dist2 = mol.DistMatrix[atom_index][atom2]
                                                        tmp_dist1[dihed_atom_ele_index].append(1.0/dist1)
                                                        tmp_dist2[dihed_atom_ele_index].append(1.0/dist2)
							import warnings
							with warnings.catch_warnings():
								warnings.filterwarnings('error')
								try:
									theta = Dihed_4Points(mol.coords[atom1], mol.coords[atom2], mol.coords[atom_index], mol.coords[dihed_atom_index])
								except Warning as e:
									print "coplane..."
									theta = 0.0
                                                        tmp_dihed[dihed_atom_ele_index].append(math.cos(theta))
                                        for m in range (0, len(tmp_dist1)):
                                                tmp_dihed[m] = [x for (y, x) in sorted(zip(tmp_dist1[m], tmp_dihed[m]))]
						tmp_dist1[m].sort()
                                                tmp_dist1[m].reverse()
                                                tmp_dihed[m].reverse()
                                        tmp_dist1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in tmp_dist1]
                                        tmp_dihed = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in tmp_dihed]
                                        tmp_dist2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in tmp_dist2]
                                        for m in range (0, self.neles):
                                                try:
                                                        pair = int(self.eles[m]*1000 + atom_ele)  # hacky way to find the index
                                                        pair_index = dihed_pair[pair]
                                                        connected_dihed2[pair_index] += tmp_dihed[m]
                                                        connected_dihed2_dist1[pair_index] += tmp_dist1[m]
                                                        connected_dihed2_dist2[pair_index] += tmp_dist2[m]
                                                except:
                                                        pass
                        #print "after sorting 1:", connected_bond1, connected_angle1
                        connected_bond2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond2 ]
                        connected_bond2 = [item for sublist in connected_bond2 for item in sublist]
                        connected_angle2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_angle2 ]
                        connected_angle2 = [item for sublist in connected_angle2 for item in sublist]
                        #print "atom1:", atom1, "atom2:", atom2,"connected_dihed2:", connected_dihed2, " connected_dihed2_dist1:", connected_dihed2_dist1, "connected_dihed2_dist2", connected_dihed2_dist2, connected_bond2, connected_angle2
                        connected_dihed2 = [b[0:dihed_max] + [0]*(dihed_max-len(b)) for b in connected_dihed2 ]
                        connected_dihed2 = [item for sublist in connected_dihed2 for item in sublist]
                        connected_dihed2_dist1 = [b[0:dihed_max] + [0]*(dihed_max-len(b)) for b in connected_dihed2_dist1 ]
                        connected_dihed2_dist1 = [item for sublist in connected_dihed2_dist1 for item in sublist]
                        connected_dihed2_dist2 = [b[0:dihed_max] + [0]*(dihed_max-len(b)) for b in connected_dihed2_dist2 ]
                        connected_dihed2_dist2 = [item for sublist in connected_dihed2_dist2 for item in sublist]

			connectedbond_angle_dihed_bond_bp = np.asarray(connected_bond1 + connected_bond2 + connected_angle1 + connected_angle2 + connected_dihed1 + connected_dihed1_dist1 + connected_dihed1_dist2 + connected_dihed2 + connected_dihed2_dist1 + connected_dihed2_dist2 + [1.0/bond_dist], dtype=np.float32)
                        #print "connectedbond_bond_bp", connectedbond_bond_bp
                        ConnectedBond_Angle_Dihed_Bond_BP.append(connectedbond_angle_dihed_bond_bp)
                ConnectedBond_Angle_Dihed_Bond_BP = np.array(ConnectedBond_Angle_Dihed_Bond_BP)
                ConnectedBond_Angle_Dihed_Bond_BP_deri = np.zeros((ConnectedBond_Angle_Dihed_Bond_BP.shape[0], ConnectedBond_Angle_Dihed_Bond_BP.shape[1]))
                return  ConnectedBond_Angle_Dihed_Bond_BP, ConnectedBond_Angle_Dihed_Bond_BP_deri


        def make_connectedbond_angle_2_bond_bp(self, mol):
		self.ngrid = 3
                ConnectedBond_Angle_Bond_BP = []
                for i in range (0, mol.NBonds()):
                        atom1 = int(mol.bonds[i][2])
                        atom2 = int(mol.bonds[i][3])
			bond_dist =  mol.bonds[i][1]
                        #print "bond :", i, "index:", atom1, atom2, "ele type:", mol.atoms[atom1], mol.atoms[atom2] 
                        connected_bond1 = []
			connected_angle1 = []
			connected_ele1 = []
                        for node in mol.atom_nodes[atom1].connected_nodes:
                                atom_index = node.node_index
                                if atom_index != atom2:
                                        dist = mol.DistMatrix[atom1][atom_index]
					dist2 = mol.DistMatrix[atom2][atom_index]
                                        connected_bond1.append(1.0/dist)
					connected_angle1.append((bond_dist**2+dist**2-dist2**2)/(2*bond_dist*dist))
					connected_ele1.append(mol.atoms[atom_index])
			#print "before sorting 1:",connected_bond1, connected_angle1
			connected_bond1 =[ x for (y, x) in sorted(zip(connected_ele1, connected_bond1))]
			connected_angle1 =[ x for (y, x) in sorted(zip(connected_ele1, connected_angle1))]
			connected_ele1.sort()
			#print "after sorting 1:", connected_bond1, connected_angle1
                        connected_bond1 = connected_bond1[0:self.ngrid] + [0]*(self.ngrid-len(connected_bond1))
			connected_ele1 = connected_ele1[0:self.ngrid] + [0]*(self.ngrid-len(connected_ele1))
			connected_angle1 = connected_angle1[0:self.ngrid] + [0]*(self.ngrid-len(connected_angle1))	

			connected_bond2 = []
                        connected_angle2 = []
                        connected_ele2 = []
                        for node in mol.atom_nodes[atom2].connected_nodes:
                                atom_index = node.node_index
                                if atom_index != atom1:
                                        dist = mol.DistMatrix[atom2][atom_index]
                                        dist2 = mol.DistMatrix[atom1][atom_index]
                                        connected_bond2.append(1.0/dist)
                                        connected_angle2.append((bond_dist**2+dist**2-dist2**2)/(2*bond_dist*dist))
                                        connected_ele2.append(mol.atoms[atom_index])
                        #print "before sorting 1:",connected_bond1, connected_angle1
                        for j  in range (0, len(connected_bond1)):
                                connected_bond2 = [x for (y, x) in sorted(zip(connected_ele1, connected_bond2))]
                                connected_angle2 =[ x for (y, x) in sorted(zip(connected_ele1, connected_angle2))]
                                connected_ele2.sort()
                        #print "after sorting 1:", connected_bond1, connected_angle1
                        connected_bond2 = connected_bond2[0:self.ngrid] + [0]*(self.ngrid-len(connected_bond2))
                        connected_ele2 = connected_ele2[0:self.ngrid] + [0]*(self.ngrid-len(connected_ele2))
                        connected_angle2 = connected_angle2[0:self.ngrid] + [0]*(self.ngrid-len(connected_angle2))	

                        connectedbond_angle_bond_bp = np.asarray(connected_ele1 + connected_bond1 + connected_angle1 + connected_ele2 + connected_bond2 + connected_angle2 + [1.0/bond_dist], dtype=np.float32)
                        #print "connectedbond_bond_bp", connectedbond_bond_bp
                        ConnectedBond_Angle_Bond_BP.append(connectedbond_angle_bond_bp)
			#print "atom1:", atom1, "atom2:", atom2, "connectedbond_angle_bond_bp:", connectedbond_angle_bond_bp
                ConnectedBond_Angle_Bond_BP = np.array(ConnectedBond_Angle_Bond_BP)
		#print "ConnectedBond_Angle_Bond_BP :", ConnectedBond_Angle_Bond_BP 
                ConnectedBond_Angle_Bond_BP_deri = np.zeros((ConnectedBond_Angle_Bond_BP.shape[0], ConnectedBond_Angle_Bond_BP.shape[1]))
                return  ConnectedBond_Angle_Bond_BP, ConnectedBond_Angle_Bond_BP_deri


        def make_connectedbond_angle_dihed_2_bond_bp(self, mol):
		dihed_max = 9
		ConnectedBond_Angle_Dihed_Bond_BP = []
		#print "mol name", mol.name, "\n\n\n"
                for i in range (0, mol.NBonds()):
                        atom1 = int(mol.bonds[i][2])
                        atom2 = int(mol.bonds[i][3])
                        bond_dist =  mol.bonds[i][1]
                        #print "bond :", i, "index:", atom1, atom2, "ele type:", mol.atoms[atom1], mol.atoms[atom2] 
                        connected_bond1 = [[] for j in range (0, self.neles)]
                        connected_angle1 = [[] for j in range (0, self.neles)]
			connected_dihed1 = []
			connected_dihed1_dist1 = []
			connected_dihed1_dist2 = []
			connected_dihed1_pairindex = []
			connected_atom1 = [[] for j in range (0, self.neles)]
                        for node in mol.atom_nodes[atom1].connected_nodes:
                                atom_index = node.node_index
                                if atom_index != atom2:
                                        dist = mol.DistMatrix[atom1][atom_index]
                                        dist2 = mol.DistMatrix[atom2][atom_index]
                                        connected_bond1[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
                                        connected_angle1[list(self.eles).index(mol.atoms[atom_index])].append((bond_dist**2+dist**2-dist2**2)/(2*bond_dist*dist))
					connected_atom1[list(self.eles).index(mol.atoms[atom_index])].append(atom_index)
                        #print "before sorting 1:",connected_bond1, connected_angle1
                        for j  in range (0, len(connected_bond1)):
                                connected_angle1[j] = [x for (y, x) in sorted(zip(connected_bond1[j], connected_angle1[j]))]
				connected_atom1[j] = [x for (y, x) in sorted(zip(connected_bond1[j], connected_atom1[j]))]
                                connected_bond1[j].sort()
                                connected_bond1[j].reverse()
				connected_atom1[j].reverse()
                                connected_angle1[j].reverse()
			#print "connected_atom1:", connected_atom1
			for j in range (0, len(connected_atom1)):
				for k in range (0, len(connected_atom1[j])):
					atom_index = connected_atom1[j][k]
					atom_ele =  mol.atoms[atom_index]
					tmp_dist1 = [[] for m in range (0, self.neles)]
					tmp_dist2 = [[] for m in range (0, self.neles)]
					tmp_dihed = [[] for m in range (0, self.neles)]
					#print "atom_index:", atom_index
					for node in mol.atom_nodes[atom_index].connected_nodes:
						dihed_atom_index = node.node_index
						if dihed_atom_index != atom1:
							dihed_atom_ele_index = list(self.eles).index(mol.atoms[dihed_atom_index])
							dist1 = mol.DistMatrix[atom_index][dihed_atom_index]
							dist2 = mol.DistMatrix[atom_index][atom1]
							tmp_dist1[dihed_atom_ele_index].append(1.0/dist1)
							tmp_dist2[dihed_atom_ele_index].append(1.0/dist2)
							import warnings
							with warnings.catch_warnings():
                                                                warnings.filterwarnings('error')
                                                                try:
                                                                        theta = Dihed_4Points(mol.coords[atom2], mol.coords[atom1], mol.coords[atom_index], mol.coords[dihed_atom_index])
                                                                except Warning as e:
									print "coplane..."
									theta = 0.0
							tmp_dihed[dihed_atom_ele_index].append(math.cos(theta))
					for m in range (0, len(tmp_dist1)):
						tmp_dihed[m] = [x for (y, x) in sorted(zip(tmp_dist1[m], tmp_dihed[m]))]
						tmp_dist1[m].sort()
						tmp_dist1[m].reverse()
						tmp_dihed[m].reverse()
					for m in range (0, self.neles):
						try:
							pair = int(self.eles[m]*1000 + atom_ele)  # hacky way to find the index
							pair_index = dihed_pair[pair] 
							connected_dihed1 += tmp_dihed[m]
							connected_dihed1_dist1 += tmp_dist1[m]
							connected_dihed1_dist2 += tmp_dist2[m]
							connected_dihed1_pairindex += [pair_index for i in range (0, len(tmp_dist2[m]))]
						except:
							pass
                        #print "after sorting 1:", connected_bond1, connected_angle1
                        connected_bond1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond1 ]
                        connected_bond1 = [item for sublist in connected_bond1 for item in sublist]
                        connected_angle1 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_angle1 ]
                        connected_angle1 = [item for sublist in connected_angle1 for item in sublist]
			#print "atom1:", atom1, "atom2:", atom2,"connected_dihed1:", connected_dihed1, " connected_dihed1_dist1:", connected_dihed1_dist1, "connected_dihed1_dist2", connected_dihed1_dist2
			connected_dihed1 = connected_dihed1[0:dihed_max] + [0]*(dihed_max-len(connected_dihed1)) 
			connected_dihed1_dist1 = connected_dihed1_dist1[0:dihed_max] + [0]*(dihed_max-len(connected_dihed1_dist1)) 
			connected_dihed1_dist2 = connected_dihed1_dist2[0:dihed_max] + [0]*(dihed_max-len(connected_dihed1_dist2)) 
			connected_dihed1_pairindex = connected_dihed1_pairindex[0:dihed_max] + [0]*(dihed_max-len(connected_dihed1_pairindex))

			connected_dihed1 = [x for (y, x) in sorted(zip(connected_dihed1_pairindex, connected_dihed1))]
			connected_dihed1_dist1 = [x for (y, x) in sorted(zip(connected_dihed1_pairindex, connected_dihed1_dist1))] 
			connected_dihed1_dist2 = [x for (y, x) in sorted(zip(connected_dihed1_pairindex, connected_dihed1_dist2))]
			connected_dihed1_pairindex.sort()
			#print "atom1:", atom1, "atom2:", atom2, "connected_dihed1:", connected_dihed1, " connected_dihed1_dist1:", connected_dihed1_dist1, "connected_dihed1_dist2", connected_dihed1_dist2, "connected_dihed1_pairindex:",connected_dihed1_pairindex 

			connected_bond2 = [[] for j in range (0, self.neles)]
                        connected_angle2 = [[] for j in range (0, self.neles)]
                        connected_dihed2 = []
                        connected_dihed2_dist1 = []
                        connected_dihed2_dist2 = []
			connected_dihed2_pairindex = []
                        connected_atom2 = [[] for j in range (0, self.neles)]
                        for node in mol.atom_nodes[atom2].connected_nodes:
                                atom_index = node.node_index
                                if atom_index != atom1:
                                        dist = mol.DistMatrix[atom2][atom_index]
                                        dist2 = mol.DistMatrix[atom1][atom_index]
                                        connected_bond2[list(self.eles).index(mol.atoms[atom_index])].append(1.0/dist)
                                        connected_angle2[list(self.eles).index(mol.atoms[atom_index])].append((bond_dist**2+dist**2-dist2**2)/(2*bond_dist*dist))
                                        connected_atom2[list(self.eles).index(mol.atoms[atom_index])].append(atom_index)
                        #print "before sorting 1:",connected_bond1, connected_angle1
                        for j  in range (0, len(connected_bond2)):
                                connected_angle2[j] = [x for (y, x) in sorted(zip(connected_bond2[j], connected_angle2[j]))]
                                connected_atom2[j] = [x for (y, x) in sorted(zip(connected_bond2[j], connected_atom2[j]))]
                                connected_bond2[j].sort()
                                connected_bond2[j].reverse()
                                connected_atom2[j].reverse()
                                connected_angle2[j].reverse()
                        #print "connected_atom2:", connected_atom2
                        for j in range (0, len(connected_atom2)):
                                for k in range (0, len(connected_atom2[j])):
                                        atom_index = connected_atom2[j][k]
                                        atom_ele =  mol.atoms[atom_index]
                                        tmp_dist1 = [[] for m in range (0, self.neles)]
                                        tmp_dist2 = [[] for m in range (0, self.neles)]
                                        tmp_dihed = [[] for m in range (0, self.neles)]
                                        #print "atom_index:", atom_index
                                        for node in mol.atom_nodes[atom_index].connected_nodes:
                                                dihed_atom_index = node.node_index
                                                if dihed_atom_index != atom2:
                                                        dihed_atom_ele_index = list(self.eles).index(mol.atoms[dihed_atom_index])
                                                        dist1 = mol.DistMatrix[atom_index][dihed_atom_index]
                                                        dist2 = mol.DistMatrix[atom_index][atom2]
                                                        tmp_dist1[dihed_atom_ele_index].append(1.0/dist1)
                                                        tmp_dist2[dihed_atom_ele_index].append(1.0/dist2)
							import warnings
							with warnings.catch_warnings():
								warnings.filterwarnings('error')
								try:
									theta = Dihed_4Points(mol.coords[atom1], mol.coords[atom2], mol.coords[atom_index], mol.coords[dihed_atom_index])
								except Warning as e:
									print "coplane..."
									theta = 0.0
                                                        tmp_dihed[dihed_atom_ele_index].append(math.cos(theta))
                                        for m in range (0, len(tmp_dist1)):
                                                tmp_dihed[m] = [x for (y, x) in sorted(zip(tmp_dist1[m], tmp_dihed[m]))]
						tmp_dist1[m].sort()
                                                tmp_dist1[m].reverse()
                                                tmp_dihed[m].reverse()
                                        for m in range (0, self.neles):
                                                try:
                                                        pair = int(self.eles[m]*1000 + atom_ele)  # hacky way to find the index
                                                        pair_index = dihed_pair[pair]
                                                        connected_dihed2  += tmp_dihed[m]
                                                        connected_dihed2_dist1  += tmp_dist1[m]
                                                        connected_dihed2_dist2  += tmp_dist2[m]
							#print "pair:", [pair_index for i in range (0, len(tmp_dist2[m]))], pair, "self.eles[m]:",self.eles[m] , "atom_ele:", atom_ele
							connected_dihed2_pairindex += [pair_index for i in range (0, len(tmp_dist2[m]))]
                                                except:
                                                        pass
                        #print "after sorting 1:", connected_bond1, connected_angle1
                        connected_bond2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_bond2 ]
                        connected_bond2 = [item for sublist in connected_bond2 for item in sublist]
                        connected_angle2 = [b[0:self.ngrid] + [0]*(self.ngrid-len(b)) for b in connected_angle2 ]
                        connected_angle2 = [item for sublist in connected_angle2 for item in sublist]
                        #print "atom1:", atom1, "atom2:", atom2,"connected_dihed2:", connected_dihed2, " connected_dihed2_dist1:", connected_dihed2_dist1, "connected_dihed2_dist2", connected_dihed2_dist2, "connected_dihed2_pairindex:",connected_dihed2_pairindex
                        connected_dihed2 = connected_dihed2[0:dihed_max] + [0]*(dihed_max-len(connected_dihed2)) 
                        connected_dihed2_dist1 = connected_dihed2_dist1[0:dihed_max] + [0]*(dihed_max-len(connected_dihed2_dist1))
                        connected_dihed2_dist2 = connected_dihed2_dist2[0:dihed_max] + [0]*(dihed_max-len(connected_dihed2_dist2))
			connected_dihed2_pairindex = connected_dihed2_pairindex[0:dihed_max] + [0]*(dihed_max-len(connected_dihed2_pairindex))

			connected_dihed2 = [x for (y, x) in sorted(zip(connected_dihed2_pairindex, connected_dihed2))]      
                        connected_dihed2_dist1 = [x for (y, x) in sorted(zip(connected_dihed2_pairindex, connected_dihed2_dist1))]
                        connected_dihed2_dist2 = [x for (y, x) in sorted(zip(connected_dihed2_pairindex, connected_dihed2_dist2))]
                        connected_dihed2_pairindex.sort()

			connectedbond_angle_dihed_bond_bp = np.asarray(connected_bond1 + connected_bond2 + connected_angle1 + connected_angle2 + connected_dihed1 + connected_dihed1_dist1 + connected_dihed1_dist2 + connected_dihed1_pairindex + connected_dihed2 + connected_dihed2_dist1 + connected_dihed2_dist2 + connected_dihed2_pairindex + [1.0/bond_dist], dtype=np.float32)
                        #print "connectedbond_bond_bp", connectedbond_bond_bp
                        ConnectedBond_Angle_Dihed_Bond_BP.append(connectedbond_angle_dihed_bond_bp)
                ConnectedBond_Angle_Dihed_Bond_BP = np.array(ConnectedBond_Angle_Dihed_Bond_BP)
                ConnectedBond_Angle_Dihed_Bond_BP_deri = np.zeros((ConnectedBond_Angle_Dihed_Bond_BP.shape[0], ConnectedBond_Angle_Dihed_Bond_BP.shape[1]))
                return  ConnectedBond_Angle_Dihed_Bond_BP, ConnectedBond_Angle_Dihed_Bond_BP_deri

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
		return self.Emb(mol_,False)

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
                        Ins, deri_CM_Bond_BP =  self.make_cm_bond_bp(mol_)
                        Ins = Ins.reshape([Ins.shape[0],-1])
		elif(self.name == "RDF_Bond_BP"):
                        Ins, deri_CM_Bond_BP =  self.make_rdf_bond_bp(mol_)
                        Ins = Ins.reshape([Ins.shape[0],-1])
		elif(self.name == "Dist_Bond_BP"):
                        Ins, deri_Dist_Bond_BP =  self.make_dist_bond_bp(mol_)
                        Ins = Ins.reshape([Ins.shape[0],-1])
		elif(self.name == "SymFunc"):
			Ins, SYM_deri = self.make_sym(mol_)
		elif(self.name == "GauInv_BP"):
			Ins =  MolEmb.Make_Inv(mol_.coords, mol_.coords, mol_.atoms ,  self.SensRadius,-1)
		elif(self.name == "ConnectedBond_Bond_BP"):
			Ins, deri_Dist_Bond_BP =  self.make_connectedbond_bond_bp(mol_)
                        Ins = Ins.reshape([Ins.shape[0],-1])
		elif(self.name == "ConnectedBond_CM_Bond_BP"):
			Ins, deri_BP = self.make_connectedbond_cm_bond_bp(mol_)
		elif(self.name == "ConnectedBond_CM_BP"):
                        Ins, deri_BP = self.make_connectedbond_cm_bp(mol_)
		elif(self.name == "ConnectedBond_Angle_Bond_BP"):
			Ins, deri_BP = self.make_connectedbond_angle_bond_bp(mol_)
		elif(self.name == "ConnectedBond_Angle_2_Bond_BP"):
                        Ins, deri_BP = self.make_connectedbond_angle_2_bond_bp(mol_)
		elif(self.name == "ConnectedBond_Angle_2_CM_Bond_BP"):
                        Ins, deri_BP = self.make_connectedbond_angle_2_cm_bond_bp(mol_)
		elif(self.name == "ConnectedBond_Angle_Dihed_Bond_BP"):
			Ins, deri_BP = self.make_connectedbond_angle_dihed_bond_bp(mol_)
		elif(self.name == "ConnectedBond_Angle_CM_Bond_BP"):
                        Ins, deri_BP = self.make_connectedbond_angle_cm_bond_bp(mol_)
		elif(self.name == "ConnectedBond_Angle_RDF_Bond_BP"):
                        Ins, deri_BP = self.make_connectedbond_angle_rdf_bond_bp(mol_)
		elif(self.name == "ConnectedBond_Angle_Dihed_CM_Bond_BP"):
                        Ins, deri_BP = self.make_connectedbond_angle_dihed_cm_bond_bp(mol_)
		elif(self.name == "ConnectedBond_Angle_Dihed_2_CM_Bond_BP"):
                        Ins, deri_BP = self.make_connectedbond_angle_dihed_2_cm_bond_bp(mol_)
		elif(self.name == "ANI1_Sym"):
			Ins, deri_BP = self.make_ANI1_sym(mol_)
		else:
			raise Exception("Unknown MolDigester Type.", self.name)

		if (self.eshape == None):
			self.eshape=Ins.shape[1:] # The first dimension is atoms. eshape is per-atom.
		if (MakeOutputs):
			if (self.OType == "Energy"):
				Outs = np.array([mol_.energy])
			elif (self.OType == "FragEnergy"):
				Outs = np.array([mol_.frag_mbe_energy])
			elif (self.OType == "GoEnergy"):
				Outs = np.array([mol_.GoEnergy()])
			elif (self.OType == "Atomization"):
				Outs = np.array([mol_.atomization])
			elif (self.OType == "Atomization_novdw"):
                                Outs = np.array([mol_.atomization - mol_.vdw])
			else:
				raise Exception("Unknown Output Type... "+self.OType)
			if (self.lshape == None):
				self.lshape=Outs.shape
			if (MakeGradients):
				return Ins, Grads, Outs
			else:
				return Ins, Outs
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

	def unscld(self,a):
	    return (a*self.StdNorm+self.MeanNorm)

	def EvaluateTestOutputs(self, desired, predicted):
		print desired.shape
		print predicted.shape
		print "Evaluating, ", len(desired), " predictions... "
		if (self.OType=="GoEnergy" or self.OType == "Energy"):
			predicted=predicted.flatten()
			desired=desired.flatten()
			print "Mean Norm and Std", self.MeanNorm, self.StdNorm
			print "NCases: ", len(desired)
			print "Mean Energy ", self.StdNorm*np.average(desired)+self.MeanNorm
			print "Mean Predicted Energy ", self.StdNorm*np.average(predicted)+self.MeanNorm
			for i in range(100):
				print "Desired: ",i,self.unscld(desired[i])," Predicted ",self.unscld(predicted[i])
			print "MAE ", np.average(self.StdNorm*np.abs(desired-predicted)+self.MeanNorm)
			print "std ", self.StdNorm*np.std(desired-predicted)+self.MeanNorm
		elif (self.OType=="Atomization"):
			pass
		else:
			raise Exception("Unknown Digester Output Type.")
		return

	def Print(self):
		print "Digest name: ", self.name
