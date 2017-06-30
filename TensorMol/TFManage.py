"""
 Either trains, tests, evaluates or provides an interface for optimization.
"""
from TensorData import *
from TFInstance import *
import numpy as np
import gc

class TFManage:
	"""
		A manager of tensorflow instances which perform atom-wise predictions
		and parent of the molecular instance mangager.
	"""
	def __init__(self, Name_="", TData_=None, Train_=True, NetType_="fc_sqdiff", RandomTData_=True, Trainable_ = True):
		"""
			Args:
				Name_: If not blank, will try to load a network with that name using Prepare()
				TData_: A TensorData instance to provide and process data.
				Train_: Whether to train the instances raised.
				NetType_: Choices of Various network architectures.
				RandomTData_: Modifes the preparation of training batches.
				ntrain_: Number of steps to train an element.
		"""
		self.path = "./networks/"
		self.TData = TData_
		self.Trainable = Trainable_
		self.NetType = NetType_
		self.n_train = PARAMS["max_steps"]
		if (Name_ != ""):
			# This will unpickle and instantiate TData...
			self.name = Name_
			self.Prepare()
			return
		if (RandomTData_==False):
			self.TData.Randomize=False
		# All done if you're doing molecular calculations
		print self.TData.AvailableElements
		print self.TData.AvailableDataFiles
		print self.TData.SamplesPerElement
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		print "--- TF will be fed by ---", self.TData.name
		self.TrainedAtoms=[] # In order of the elements in TData
		self.TrainedNetworks=[] # In order of the elements in TData
		self.Instances=[None for i in range(MAX_ATOMIC_NUMBER)] # In order of the elements in TData
		if (Train_):
			self.Train()
			return
		return

	def Print(self):
		print "-- TensorMol, Tensorflow Manager Status--"
		return

	def Train(self):
		print "Will train a NNetwork for each element in: ", self.TData.name
		for i in range(len(self.TData.AvailableElements)):
			self.TrainElement(self.TData.AvailableElements[i])
		return

	def Save(self):
		print "Saving TFManager:",self.path+self.name+".tfm"
		self.TData.CleanScratch()
		f=open(self.path+self.name+".tfm","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

	def Load(self):
		print "Unpickling TFManager..."
		f = open(self.path+self.name+".tfm","rb")
		import TensorMol.PickleTM
		tmp = TensorMol.PickleTM.UnPickleTM(f)
		self.__dict__.update(tmp)
		f.close()
		print "TFManager Metadata Loaded, Reviving Networks."
		self.Print()
		return

	def TrainElement(self, ele):
		print "Training Element:", ele
		if (self.TData.dig.eshape==None):
			raise Exception("Must Have Digester")
		# It's up the TensorData to provide the batches and input output shapes.
		if (self.NetType == "fc_classify" or PARAMS["Classify"]):
			self.Instances[ele] = Instance_fc_classify(self.TData, ele, None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances[ele] = Instance_fc_sqdiff(self.TData, ele, None)
		elif (self.NetType == "del_fc_sqdiff"):
			self.Instances[ele] = Instance_fc_sqdiff(self.TData, ele, None)
		elif (self.NetType == "3conv_sqdiff"):
			self.Instances[ele] = Instance_3dconv_sqdiff(self.TData, ele, None)
		elif (self.NetType == "KRR_sqdiff"):
			self.Instances[ele] = Instance_KRR(self.TData, ele, None)
		elif (self.NetType == "fc_sqdiff_queue"):
			self.Instances[ele] = Queue_Instance(self.TData, ele, None)
		else:
			raise Exception("Unknown Network Type!")
		self.Instances[ele].train(self.n_train) # Just for the sake of debugging.
		nm = self.Instances[ele].name
		if self.TrainedNetworks.count(nm)==0:
			self.TrainedNetworks.append(nm)
		if self.TrainedAtoms.count(ele)==0:
			self.TrainedAtoms.append(ele)
		self.Save()
		gc.collect()
		return

	def EvalElement(self, ele, test_input):
		return self.Instances[ele].evaluate(test_input)    # for debugging

	def Prepare(self):
		self.Load()
		self.Instances=[None for i in range(MAX_ATOMIC_NUMBER)] # In order of the elements in TData
		for i  in range (0, len(self.TrainedAtoms)):
			if (self.NetType == "fc_classify"):
				self.Instances[self.TrainedAtoms[i]] = Instance_fc_classify(None, self.TrainedAtoms[i], self.TrainedNetworks[i])
			elif (self.NetType == "fc_sqdiff"):
				self.Instances[self.TrainedAtoms[i]] = Instance_fc_sqdiff(None, self.TrainedAtoms[i], self.TrainedNetworks[i])
			elif (self.NetType == "3conv_sqdiff"):
				self.Instances[self.TrainedAtoms[i]] = Instance_3dconv_sqdiff(None, self.TrainedAtoms[i], self.TrainedNetworks[i])
			else:
				raise Exception("Unknown Network Type!")
		# Raise TF instances for each atom which have already been trained.
		return

	def SampleAtomGrid(self, mol, atom, maxstep, ngrid):
	    # use TF instances for each atom.
	    return  self.TData.dig.UniformDigest(mol,atom,maxstep,ngrid)

	def SmoothPOneAtom(self, mol_, atom_):
		"""
			Uses smooth fitting of Go probability to pick the next point as predicted by the network
			This should eventually be made faster by doing all atoms of an element type at once.
		"""
		inputs = self.TData.dig.Emb(mol_, atom_, [mol_.coords[atom_]],False)
		output = self.Instances[mol_.atoms[atom_]].evaluate(inputs)[0,0]
		if(not np.all(np.isfinite(output))):
			print output
			raise Exception("BadTFOutput")
		p = mol_.UseGoProb(atom_, output)
		return p

	def EvalRotAvForceOld(self, mol, RotAv=10, Debug=False):
		"""
		Goes without saying we should do this in batches for each element,
		if it actually improves accuracy. And improve rotational sampling.
		But for the time being I'm doing this sloppily.
		"""
		if(self.TData.dig.name != "GauSH"):
			raise Exception("Don't average this...")
		p = np.zeros((mol.NAtoms(),3))
		pi = np.zeros((3,RotAv,mol.NAtoms(),3))
		for atom in range(mol.NAtoms()):
			inputs = np.zeros((3*RotAv,PARAMS["SH_NRAD"]*(PARAMS["SH_LMAX"]+1)*(PARAMS["SH_LMAX"]+1)))
			for ax in range(3):
				axis = [0,0,0]
				axis[ax] = 1
				for i, theta in enumerate(np.linspace(-Pi, Pi, RotAv)):
					mol_t = Mol(mol.atoms, mol.coords)
					mol_t.Rotate(axis, theta, mol.coords[atom])
					inputs[ax*RotAv+i] = self.TData.dig.Emb(mol_t, atom, mol_t.coords[atom],False)
			outs = self.Instances[mol_t.atoms[atom]].evaluate(inputs)
			for ax in range(3):
				axis = [0,0,0]
				axis[ax] = 1
				for i, theta in enumerate(np.linspace(-Pi, Pi, RotAv)):
					pi[ax,i,atom] = np.dot(RotationMatrix(axis, -1.0*theta),outs[0,ax*RotAv+i].T).reshape(3)
		p[atom] += pi[ax,i,atom]
		if (Debug):
			print "Checking Rotations... "
			for atom in range(mol.NAtoms()):
				print "Atom ", atom, " mean: ", np.mean(pi[:,:,atom],axis=(0,1)), " std ",np.std(pi[:,:,atom],axis=(0,1))
				for ax in range(3):
					for i, theta in enumerate(np.linspace(-Pi, Pi, RotAv)):
						print atom,ax,theta,":",pi[ax,i,atom]
		return p/(3.0*RotAv)

	def EvalRotAvForce(self, mol, RotAv=10, Debug=False):
		"""
		Rewritten for optimal performance with rotational averaging and atom ordering.
		"""
		if(self.TData.dig.name != "GauSH"):
		    raise Exception("Don't average this...")
		p = np.zeros((mol.NAtoms(),3)) # Forces to output
		eles = mol.AtomTypes().tolist()
		transfs = np.zeros((3*RotAv,3,3))
		itransfs = np.zeros((3*RotAv,3,3))
		ind = 0
		for ax in range(3):
			axis = [0,0,0]
			axis[ax] = 1
			for i, theta in enumerate(np.linspace(-Pi, Pi, RotAv)):
				transfs[ind] = RotationMatrix(axis, theta)
				itransfs[ind] = RotationMatrix(axis, -1.0*theta)
				ind = ind+1
		for ele in eles:
			eats = [i for i in range(mol.NAtoms()) if mol.atoms[i] == ele]
			na = len(eats)
			inputs = np.zeros((na*3*RotAv,PARAMS["SH_NRAD"]*(PARAMS["SH_LMAX"]+1)*(PARAMS["SH_LMAX"]+1)))
			# Create an index matrix to contract these down and increment the right places.
			for i,ei in enumerate(eats):
				inputs[i*3*RotAv:(i+1)*3*RotAv] = self.TData.dig.Emb(mol, ei, mol.coords[ei], False, False, transfs)
			#print inputs.shape, na*3*RotAv , na, 3*RotAv, 3
			outs = self.Instances[ele].evaluate(inputs)
			#print inputs.shape, outs.shape, na, 3*RotAv, 3
			outs = outs.reshape(na,3*RotAv,3)
			ou = np.einsum("txy,aty->ax",itransfs,outs)/(3.*RotAv)
			for i,ei in enumerate(eats):
				p[ei] = ou[i].copy()
		return p

	def EvalOctAvForce(self, mol, Debug=False):
		"""
		Goes without saying we should do this in batches for each element,
		if it actually improves accuracy. And improve rotational sampling.
		But for the time being I'm doing this sloppily.
		"""
		if(self.TData.dig.name != "GauSH"):
		    raise Exception("Don't average this...")
		ops = OctahedralOperations()
		invops = map(np.linalg.inv,ops)
		pi = np.zeros((mol.NAtoms(),len(ops),3))
		p = np.zeros((mol.NAtoms(),3))
		for atom in range(mol.NAtoms()):
			inputs = np.zeros((len(ops),PARAMS["SH_NRAD"]*(PARAMS["SH_LMAX"]+1)*(PARAMS["SH_LMAX"]+1)))
			for i in range(len(ops)):
				op = ops[i]
				mol_t = Mol(mol.atoms, mol.coords)
				mol_t.Transform(op, mol.coords[atom])
				inputs[i] = self.TData.dig.Emb(mol_t, atom, mol_t.coords[atom],False)
				if (Debug):
					print inputs[i]
			outs = self.Instances[mol_t.atoms[atom]].evaluate(inputs)[0]
			for i in range(len(ops)):
				pi[atom,i] = np.dot(invops[i],outs[i].T).reshape(3)
				p[atom] += np.sum(pi[atom,i], axis=0)
		if (Debug):
			print "Checking Rotations... "
			for atom in range(mol.NAtoms()):
				print "Atom ", atom, " mean: ", np.mean(pi[atom,:],axis=0), " std ",np.std(pi[atom,:],axis=0)
				for i in range(len(ops)):
					print atom, i, pi[atom,i]
		return p/(len(ops))

	def evaluate(self, mol, atom):
		inputs = self.TData.dig.Emb(mol, atom, mol.coords[atom],False)
		if (self.Instances[mol.atoms[atom]].tformer.innorm != None):
			inputs = self.Instances[mol.atoms[atom]].tformer.NormalizeIns(inputs, train=False)
		outs = self.Instances[mol.atoms[atom]].evaluate(inputs)
		if (self.Instances[mol.atoms[atom]].tformer.outnorm != None):
			outs = self.Instances[mol.atoms[atom]].tformer.UnNormalizeOuts(outs)
		return outs[0]

	def EvalOneAtom(self, mol, atom, maxstep = 0.2, ngrid = 50):
		xyz, inputs = self.SampleAtomGrid( mol, atom, maxstep, ngrid)
		p = self.Instances[mol.atoms[atom]].evaluate(inputs)
		if (np.sum(p**2)**0.5 != 0):
			p = p/(np.sum(p**2))**0.5
		else:
			p.fill(1.0)
		#Check finite-ness or throw
		if(not np.all(np.isfinite(p))):
			print p
			raise Exception("BadTFOutput")
		return xyz, p

	def EvalAllAtoms(self, mol, maxstep = 1.5, ngrid = 50):
		XYZ=[]
		P=[]
		for i in range (0, mol.atoms.shape[0]):
			print ("Evaluating atom: ", mol.atoms[i])
			xyz, p = self.EvalOneAtom(mol, i, maxstep, ngrid)
			XYZ.append(xyz)
			P.append(p)
		XYZ = np.asarray(XYZ)
		P = np.asarray(P)
		return XYZ, P

	def EvalOneAtomMB(self, mol, atom, maxstep=0.2, ngrid=50):
		# This version samples all the other atoms as well and supposes the prob of move is the product of all.
		satoms=mol.AtomsWithin(self.TData.dig.SensRadius,mol.coords[atom])
		ele = mol.atoms[atom]
		xyz, inputs = self.SampleAtomGrid( mol, atom, maxstep, ngrid)
		plist = []
		p = self.Instances[ele].evaluate(inputs.reshape((inputs.shape[0], -1)))
		print "p: ",p
		if (np.sum(p**2)**0.5 != 0):
			p = p/(np.sum(p**2))**0.5
		else:
			p.fill(1.0)
		for i in satoms:
			if (i == atom ):
		#			if (i == atom or mol.atoms[i] == 1):  # we ignore the hyrodgen here
				continue
			ele = mol.atoms[i]
			# The conditional digest moves the coordinates of catom to all points in the grid, and then evaluates
			# the 'put-back probability' of atom i to it's place
			#print xyz.shape
			MB_inputs = self.TData.dig.ConditionDigest(mol, atom, i, xyz)
			tmp_p = self.Instances[ele].evaluate(MB_inputs.reshape((MB_inputs.shape[0], -1)))
			print "tmp_p",tmp_p
			if (np.sum(tmp_p**2)**0.5 != 0):
				tmp_p = tmp_p/(np.sum(tmp_p**2))**0.5  #just normlized it...
			else:
				tmp_p.fill(1.0)
		#			tmp_p = np.log10(p)  # take the log..
			p *=  tmp_p  # small p is what we want
		p = np.absolute(p)
		return xyz, p

	def EvalAllAtomsMB(self, mol, maxstep = 0.2, ngrid = 50):
		XYZ=[]
		P=[]
		for i in range (0, mol.atoms.shape[0]):
				print ("Evaluating atom: ", mol.atoms[i])
				xyz, p = self.EvalOneAtomMB(mol, i, maxstep, ngrid)
				XYZ.append(xyz)
				P.append(p)
		XYZ = np.asarray(XYZ)
		P = np.asarray(P)
		return XYZ, P

	def EvalMol(self, mol):
		P=1.0
		for i in range (0, mol.atoms.shape[0]):
			inputs = (self.TData.dig.Emb(i, mol.atoms, mol.coords,  ((mol.coords)[i]).reshape((1,3))))[0]
			inputs = np.array(inputs)
			inputs = inputs.reshape((1,-1))
			p = float(self.Instances[mol.atoms[i]].evaluate(inputs))
			print ("p:", p)
			P  *= p
		return P

	def EvalMol_v2(self, mol):
		logP=0.0
		for i in range (1, mol.NAtoms()):
			tmp_mol = Mol(mol.atoms[0:i+1], mol.coords[0:i+1])
			inputs = (self.TData.dig.Emb(i, tmp_mol.atoms, tmp_mol.coords,  ((tmp_mol.coords)[i]).reshape((1,3))))[0]
			inputs = np.array(inputs)
			inputs = inputs.reshape((1,-1))
			p = float(self.Instances[tmp_mol.atoms[i]].evaluate(inputs))
			print ("i:",i, "p:", p, "logp:", math.log10(p))
			logP += math.log10(p)
		print ("logP:", logP)
		return logP
