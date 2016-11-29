#
# Either trains, tests, evaluates or provides an interface for optimization.
#
from TensorData import *
from TFInstance import *
import numpy as np
import gc


class TFManage:
	def __init__(self, Name_="", TData_=None, Train_=True, NetType_="fc_classify", RandomTData_=True):  #Test_TData_ is some other randon independent test data
	        self.path = "./networks/"	
		if (Name_ != ""):
			# This will unpickle and instantiate TData...
			self.name = Name_
			self.Prepare()
			return
		self.TData = TData_
		if (RandomTData_==False):
			self.TData.Randomize=False
		self.NetType = NetType_
		print self.TData.AvailableElements
		print self.TData.AvailableDataFiles
		print self.TData.SamplesPerElement
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		print "--- TF will be fed by ---",self.TData.name

		self.TrainedAtoms=[] # In order of the elements in TData
		self.TrainedNetworks=[] # In order of the elements in TData
		self.Instances=[None for i in range(MAX_ATOMIC_NUMBER)] # In order of the elements in TData
		if (Train_):
			self.TrainAllAtoms()
			return
		return

	def Print(self):
		print "-- TensorMol, Tensorflow Manager Status--"
		return
	
	def TrainAllAtoms(self):
		print "Will train a NNetwork for each element in: ", self.TData.name
		for i in range(len(self.TData.AvailableElements)):
			self.TrainElement(self.TData.AvailableElements[i])
		self.Save()
		return

	def Save(self):
		print "Saving TFManager."
		self.TData.CleanScratch()
		f=open(self.path+self.name+".tfm","wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
		return

	def Load(self):
		print "Unpickling TFManager..."
		f = open(self.path+self.name+".tfm","rb")
		tmp=pickle.load(f)
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
		if (self.NetType == "fc_classify"):
			self.Instances[ele] = Instance_fc_classify(self.TData, ele, None)
		elif (self.NetType == "fc_sqdiff"):
			self.Instances[ele] = Instance_fc_sqdiff(self.TData, ele, None)
		elif (self.NetType == "3conv_sqdiff"):
			self.Instances[ele] = Instance_3dconv_sqdiff(self.TData, ele, None)
		else:
			raise Exception("Unknown Network Type!")
		#self.Instances[ele].train_prepare()
		#for step in range (0, 10):
			#self.Instances[ele].train_step(step)
		#tself.Instances[ele].test(step)
		self.Instances[ele].train(1500) # Just for the sake of debugging.
		nm = self.Instances[ele].name
		# Here we should print some summary of the pupil's progress as well, maybe.
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
		''' 
			Uses smooth fitting of Go probability to pick the next point as predicted by the network
			This should eventually be made faster by doing all atoms of an element type at once.
		'''
		inputs = self.TData.dig.Emb(mol_, atom_, [mol_.coords[atom_]],False)
		output = self.Instances[mol_.atoms[atom_]].evaluate(inputs)[0,0]
		if(not np.all(np.isfinite(output))):
			print output
			raise Exception("BadTFOutput")
		p = mol_.UseGoProb(atom_, output)
		return p

	def evaluate(self, mol, atom):
		input = self.TData.dig.Emb(mol, atom, mol.coords[atom])
		p = self.Instances[mol.atoms[atom]].evaluate(input)
		return p[0]

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
