#
# These work Moleculewise the versions without the mol prefix work atomwise.
# but otherwise the behavior of these is the same as TFManage etc.
#
from TFManage import *
from TensorMolData import *
from TFMolInstance import *
import numpy as np
import gc

class TFMolManage(TFManage):
	"""
		A manager of tensorflow instances which perform molecule-wise predictions 
		including Many Body and Behler-Parinello
	"""
	def __init__(self, Name_="", TData_=None, Train_=False, NetType_="fc_sqdiff", RandomTData_=True):
		"""
			Args: 
				Name_: If not blank, will try to load a network with that name using Prepare()
				TData_: A TensorMolData instance to provide and process data. 
				Train_: Whether to train the instances raised. 
				NetType_: Choices of Various network architectures. 
				RandomTData_: Modifes the preparation of training batches.
		"""
		TFManage.__init__(self, Name_, TData_, False, NetType_, RandomTData_)
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.TData.order)
		if (Name_!=""):
			self.name = Name_
			self.Prepare()
			return
		self.TrainedAtoms=[] # In order of the elements in TData
		self.TrainedNetworks=[] # In order of the elements in TData
		self.Instances=None # In order of the elements in TData
		if (Train_):
			self.Train()
			return
		return

	def Train(self, maxstep=3000):
		"""
		Instantiates and trains a Molecular network.
		
		Args:
			maxstep: The number of training steps.
		"""
		if (self.TData.dig.eshape==None):
			raise Exception("Must Have Digester")
		# It's up the TensorData to provide the batches and input output shapes.
		if (self.NetType == "fc_classify"):
			self.Instances = MolInstance_fc_classify(self.TData, None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = MolInstance_fc_sqdiff(self.TData, None)
		elif (self.NetType == "fc_sqdiff_BP"):
			self.Instances = MolInstance_fc_sqdiff_BP(self.TData)
		else:
			raise Exception("Unknown Network Type!")
		self.Instances.train(maxstep) # Just for the sake of debugging.
		nm = self.Instances.name
		# Here we should print some summary of the pupil's progress as well, maybe.
		if self.TrainedNetworks.count(nm)==0:
			self.TrainedNetworks.append(nm)
		self.Save()
		gc.collect()
		return

	def Eval(self, test_input):
		return self.Instances.evaluate(test_input)   


        def Eval_BP(self, mol_set, total_energy = False):
                nmols = len(mol_set.mols)
                natoms = mol_set.NAtoms()
                cases = np.zeros(tuple([natoms]+list(self.TData.dig.eshape)))
                dummy_outputs = np.zeros((nmols))
                meta = np.zeros((natoms, 4), dtype = np.int)
                casep = 0
                mols_done = 0
                for mol in mol_set.mols:
                        ins = self.TData.dig.EvalDigest(mol)
                        nat = mol.NAtoms()
                        cases[casep:casep+nat] = ins
                        for i in range (casep, casep+nat):
                                meta[i, 0] = mols_done
                                meta[i, 1] = mol.atoms[i - casep]
                                meta[i, 2] = casep
                                meta[i, 3] = casep + nat
                        casep += nat
                        mols_done += 1
                sto = np.zeros(len(self.TData.eles),dtype = np.int32)
                offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
                inputs = []
                matrices = []
                outputpointer = 0
                for i in range (0, natoms):
                        sto[self.TData.eles.index(meta[i, 1])] += 1
                currentmol = 0
                for e in range (len(self.TData.eles)):
                        inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
                        matrices.append(np.zeros((sto[e], nmols)))
                for i in range (0, natoms):
                        if currentmol != meta[i, 0]:
                                outputpointer += 1
                                currentmol = meta[i, 0]
                        e = meta[i, 1]
                        ei = self.TData.eles.index(e)
                        inputs[ei][offsets[ei], :] = cases[i]
                        matrices[ei][offsets[ei], outputpointer] = 1.0
                        offsets[ei] += 1
                mol_out, atom_out, gradient = self.Instances.evaluate([inputs, matrices, dummy_outputs])

                pointers = [0 for ele in self.TData.eles]
                diff = 0
                for i in range (0, nmols):
                        mol = mol_set.mols[i]
                        print "for mol :", mol.name," energy:", mol.energy, "  gradient:", gradient
                        print "total atomization energy: %.10f"%mol_out[0][i]
                        #diff += abs(mol.energy - mol_out[0][i])
                        if total_energy:
                                total = mol_out[0][i]
                                for j in range (0, mol.NAtoms()):
                                        total += ele_U[mol.atoms[j]]
                                print "total electronic energy:", total
                        for j in range (0, mol.atoms.shape[0]):
                                atom_type = mol.atoms[j]
                                atom_index = self.TData.eles.index(atom_type)
                                print "atom: ", mol.atoms[j], " energy:", atom_out[atom_index][0][pointers[atom_index]]
                                pointers[atom_index] += 1
                #print "mol out:", mol_out, " atom_out", atom_out
                #return diff / nmols
                return



	def Eval_Bond_BP(self, mol_set, total_energy = False):
		nmols = len(mol_set.mols)
		nbonds = mol_set.NBonds()
		cases = np.zeros(tuple([nbonds]+list(self.TData.dig.eshape)))
		dummy_outputs = np.zeros((nmols))
		meta = np.zeros((nbonds, 4), dtype = np.int)	
		casep = 0
		mols_done = 0
		for mol in mol_set.mols:
			ins = self.TData.dig.EvalDigest(mol)
			nbo = mol.NBonds()
			cases[casep:casep+nbo] = ins
			for i in range (casep, casep+nbo):
				meta[i, 0] = mols_done
				meta[i, 1] = mol.bonds[i - casep,0]
				meta[i, 2] = casep
				meta[i, 3] = casep + nbo
			casep += nbo
			mols_done += 1
		sto = np.zeros(len(self.TData.eles),dtype = np.int32)
		offsets = np.zeros(len(self.TData.eles),dtype = np.int32)
		inputs = []
		matrices = []
		outputpointer = 0
		for i in range (0, nbonds):
			sto[self.TData.eles.index(meta[i, 1])] += 1
		currentmol = 0
		for e in range (len(self.TData.eles)):
			inputs.append(np.zeros((sto[e], np.prod(self.TData.dig.eshape))))
			matrices.append(np.zeros((sto[e], nmols)))
		for i in range (0, nbonds):
			if currentmol != meta[i, 0]:
				outputpointer += 1
				currentmol = meta[i, 0]
			e = meta[i, 1]
			ei = self.TData.eles.index(e)
			inputs[ei][offsets[ei], :] = cases[i]
			matrices[ei][offsets[ei], outputpointer] = 1.0
			offsets[ei] += 1
		#print "[inputs, matrices, dummy_outputs]", [inputs, matrices, dummy_outputs]
		mol_out, atom_out = self.Instances.evaluate([inputs, matrices, dummy_outputs])

		pointers = [0 for ele in self.TData.eles]
		diff = 0
		for i in range (0, nmols):
			mol = mol_set.mols[i]
			print "for mol :", mol.name," energy:", mol.energy
			print "total atomization energy:", mol_out[0][i]
			#diff += abs(mol.energy - mol_out[0][i])
			if total_energy:
				total = mol_out[0][i]
				for j in range (0, mol.NAtoms()):
					total += ele_U[mol.atoms[j]]
				print "total electronic energy:", total
			for j in range (0, mol.bonds.shape[0]):
				bond_type = mol.bonds[j, 0]
				bond_index = self.TData.eles.index(bond_type)
				print "bond: ", mol.bonds[j], " energy:", atom_out[bond_index][0][pointers[bond_index]] 
				pointers[bond_index] += 1		
		#print "mol out:", mol_out, " atom_out", atom_out
		#return	diff / nmols
		return 

	def Eval_Mol(self, mol):
		total_case = len(mol.mbe_frags[self.TData.order])
		if total_case == 0:
			return 0.0
		natom = mol.mbe_frags[self.TData.order][0].NAtoms()
		cases = np.zeros((total_case, self.TData.dig.eshape))
		cases_deri = np.zeros((total_case, natom, natom, 6)) # x1,y1,z1,x2,y2,z2
		casep = 0 
		for frag in mol.mbe_frags[self.TData.order]:
			ins, embed_deri =  self.TData.dig.EvalDigest(frag)
			cases[casep:casep+1] += ins
			cases_deri[casep:casep+1]=embed_deri
			casep += 1
		print "evaluating order:", self.TData.order
		nn, nn_deri=self.Eval(cases)
		mean, std = self.TData.Get_Mean_Std()
		nn = nn*std+mean
		nn_deri = nn_deri*std
		#print "nn:",nn, "nn_deri:",nn_deri, "cm_deri:", cases_deri, "cases:",cases, "coord:", mol.coords
		mol.Set_Frag_Force_with_Order(cases_deri, nn_deri, self.TData.order)
		return nn.sum()

	def Prepare(self):
		self.Load()
		self.Instances= None # In order of the elements in TData
		if (self.NetType == "fc_classify"):
			self.Instances = MolInstance_fc_classify(None,  self.TrainedNetworks[0], None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances = MolInstance_fc_sqdiff(None, self.TrainedNetworks[0], None)
		elif (self.NetType == "fc_sqdiff_BP"):
			self.Instances = MolInstance_fc_sqdiff_BP(None,self.TrainedNetworks[0])
		else:
			raise Exception("Unknown Network Type!")	
		# Raise TF instances for each atom which have already been trained.
		return

# This has to be totally re-written to be more like the
# testing in TFInstance.

	def Continue_Training(self, maxsteps):   # test a pretrained network
                self.Instances.TData = self.TData
                self.Instances.TData.LoadDataToScratch()
		#self.Instances.chk_file = "./networks/Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_1_None/Mol_gdb9_energy_1_6_7_8_cleaned_ConnectedBond_Angle_Bond_BP_1_None-chk-182" 
                self.Instances.Prepare()
                self.Instances.continue_training(maxsteps)
		self.Save()
                return


	def Test(self):   # test a pretrained network
		self.Instances.TData = self.TData
		self.Instances.TData.LoadDataToScratch()
		self.Instances.Prepare()
		self.Instances.test_after_training(-1)
		self.Instances.TData.Init_TraceBack()
		return


        def Test_MBE(self, save_file="mbe_test.dat"):
                ti, to = self.TData.LoadData( True)
                NTest = int(self.TData.TestRatio * ti.shape[0])
                ti= ti[ti.shape[0]-NTest:]
                to = to[to.shape[0]-NTest:]
                acc_nn = np.zeros((to.shape[0],2))
                acc=self.TData.ApplyNormalize(to)
                nn, gradient=self.Eval(ti)
                acc_nn[:,0]=acc.reshape(acc.shape[0])
                acc_nn[:,1]=nn.reshape(nn.shape[0])
                mean, std = self.TData.Get_Mean_Std()
                acc_nn = acc_nn*std+mean
                np.savetxt(save_file,acc_nn)
                np.savetxt("dist_2b.dat", ti[:,1])
                return

