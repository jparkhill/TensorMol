#
# A molecule set is not a training set.
#
from Mol import *
from MolGraph import *
from Util import *
from MolFrag import *
import numpy as np
import os,sys,re,copy,time
import cPickle as pickle

class MSet:
	""" A molecular database which
		provides structures """
	def __init__(self, name_ ="gdb9", path_="./datasets/", center_=True):
		self.mols=[]
		self.path=path_
		self.name=name_
		self.suffix=".pdb" #Pickle Database? Poor choice.
		self.center=center_

	def Save(self, filename=None):
		for mol in self.mols:
			mol.Clean()
		if filename == None:
			filename = self.name
		LOGGER.info("Saving set to: %s ", self.path+filename+self.suffix)
		#print "Saving set to: ", self.path+self.name+self.suffix
		f=open(self.path+filename+self.suffix,"wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

	def Load(self, filename=None):
		if filename == None:
			filename = self.name
		f = open(self.path+filename+self.suffix,"rb")
		tmp=pickle.load(f)
		self.__dict__.update(tmp)
		f.close()
		LOGGER.info("Loaded, "+str(len(self.mols))+" molecules "+str(self.NAtoms())+" Atoms total "+str(self.AtomTypes())+" Types ")
		return

	def DistortAlongNormals(self, npts=8, random=True, disp=.2):
		'''
		Create a distorted copy of a set

		Args:
			npts: the number of points to sample along the normal mode coordinate.
			random: whether to randomize the order of the new set.
			disp: the maximum displacement of atoms along the mode

		Returns:
			A set containing distorted versions of the original set.
		'''
		print "Making distorted clone of:", self.name
		s = MSet(self.name+"_NEQ")
		ord = range(len(self.mols))
		if(random):
			np.random.seed(int(time.time()))
			ord=np.random.permutation(len(self.mols))
		for j in ord:
			newcoords = self.mols[j].ScanNormalModes(npts,disp)
			for i in range(newcoords.shape[0]): # Loop modes
				for k in range(newcoords.shape[1]): # loop points
					s.mols.append(Mol(self.mols[j].atoms,newcoords[i,k,:,:]))
					s.mols[-1].DistMatrix = self.mols[j].DistMatrix
		return s

	def RotatedClone(self, NRots=3):
		"""
		Rotate every molecule NRots Times.
		We should toss some reflections in the mix too...
		"""
		print "Making Rotated clone of:", self.name
		s = MSet(self.name)
		ord = range(len(self.mols))
		if(random):
			np.random.seed(int(time.time()))
			ord=np.random.permutation(len(self.mols))
		for j in ord:
			for i in range (0, NRots):
				s.mols.append(copy.deepcopy(self.mols[j]))
				s.mols[-1].coords -= s.mols[-1].Center()
				s.mols[-1].RotateRandomUniform()
		return s

	def DistortedClone(self, NDistorts=1, random=True):
			''' Create a distorted copy of a set'''
			print "Making distorted clone of:", self.name
			s = MSet(self.name+"_NEQ")
			ord = range(len(self.mols))
			if(random):
				np.random.seed(int(time.time()))
				ord=np.random.permutation(len(self.mols))
			for j in ord:
				for i in range (0, NDistorts):
					s.mols.append(copy.deepcopy(self.mols[j]))
					s.mols[-1].Distort()
			return s

	def TransformedClone(self, transfs):
		''' make a linearly transformed copy of a set. '''
		LOGGER.info("Making Transformed clone of:"+self.name)
		s = MSet(self.name)
		ord = range(len(self.mols))
		for j in ord:
			for k in range(len(transfs)):
				s.mols.append(copy.deepcopy(self.mols[j]))
				s.mols[-1].Transform(transfs[k])
		return s

	def CenterSet(self):
		"""
		Translates every Mol such that the center is at 0.
		"""
		ord = range(len(self.mols))
		for j in ord:
			self.mols[j].coords -= self.mols[j].Center()

	def SplitTest(self):
		"""
		Split the set into a train and test set based on PARAMS["test_ratio"].
		"""
		b=MSet(self.name+"_train")
		c=MSet(self.name+"_test")
		mols = random.sample(range(len(self.mols)), int(len(self.mols)*PARAMS["TestRatio"]))
		for i in xrange(len(self.mols)):
			if i in mols:
				c.mols.append(self.mols[i])
			else:
				b.mols.append(self.mols[i])
		b.Save()
		c.Save()
		return b, c

	def NAtoms(self):
		nat=0
		for m in self.mols:
			nat += m.NAtoms()
		return nat

	def MaxNAtoms(self):
		return np.max([m.NAtoms() for m in self.mols])

	def AtomTypes(self):
		types = np.array([],dtype=np.uint8)
		for m in self.mols:
			types = np.union1d(types,m.AtomTypes())
		return types

	def BondTypes(self):
		return np.asarray([x for x in itertools.product(self.AtomTypes().tolist(), repeat=2)])

	def ReadXYZUnpacked(self, path="/Users/johnparkhill/gdb9/", has_energy=False, has_force=False, has_charge=False, has_mmff94=False):
		"""
		Reads XYZs in distinct files in one directory as a molset
		Args:
			path: the directory which contains the .xyz files to be read
			has_energy: switch to turn on reading the energy from the comment line as formatted from the md_dataset on quantum-machine.org
			has_force: switch to turn on reading the force from the comment line as formatted from the md_dataset on quantum-machine.org
		"""
		from os import listdir
		from os.path import isfile, join
		#onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		for file in onlyfiles:
			if ( file[-4:]!='.xyz' ):
					continue
			self.mols.append(Mol())
			self.mols[-1].ReadGDB9(path+file, file)
			self.mols[-1].properties["set_name"] = self.name
			if has_force:
				self.mols[-1].ForceFromXYZ(path+file)
			if has_energy:
				self.mols[-1].EnergyFromXYZ(path+file)
			if has_charge:
				self.mols[-1].ChargeFromXYZ(path+file)
			if has_mmff94:
				self.mols[-1].MMFF94FromXYZ(path+file)
		if (self.center):
			self.CenterSet()
		return

	def ReadXYZ(self,filename = None, xyz_type = 'mol'):
		""" Reads XYZs concatenated into a single file separated by \n\n as a molset """
		if filename == None:
			filename = self.name
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].count('Comment:')>0):
				line0=line-1
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(Mol())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]))
				self.mols[-1].name = str(txts[line0+1])
				self.mols[-1].properties["set_name"] = self.name
		if (self.center):
			self.CenterSet()
		LOGGER.debug("Read "+str(len(self.mols))+" molecules from XYZ")
		return

	def AppendFromDirectory(self, apath_):
		"""
		Append all xyz files in apath_ to this set.
		"""
		for file in os.listdir(apath_):
			if file.endswith(".xyz"):
				m = Mol()
				m.properties = {"from_file":file}
				f = open(file,'r')
				fs = f.read()
				m.FromXYZString(fs)
				self.mols.append(m)
		return

	def WriteXYZ(self,filename=None):
		if filename == None:
			filename = self.name
		for mol in self.mols:
			mol.WriteXYZfile(self.path,filename)
		LOGGER.info('Wrote %s ', filename)
		return

	def pop(self, ntopop):
		for i in range(ntopop):
			self.mols.pop()
		return

	def OnlyWithElements(self, allowed_eles):
		mols=[]
		for mol in self.mols:
			if set(list(mol.atoms)).issubset(allowed_eles):
				mols.append(mol)
		for i in allowed_eles:
			self.name += "_"+str(i)
		self.mols=mols
		return

	def AppendSet(self, b):
		if (self.name == None):
			self.name = self.name + b.name
		self.mols = self.mols+b.mols
		return

	def rms(self,other_):
		if (len(self.mols) != len(other_.mols)):
			raise Exception("Bad Comparison")
		rmss = [self.mols[i].rms_inv(other_.mols[i]) for i in range(len(self.mols))]
		return rmss

	def Statistics(self):
		""" Return some energy information about the samples we have... """
		print "Set Statistics----"
		print "Nmol: ", len(self.mols)
		sampfrac = 0.1;
		np.random.seed(int(time.time()))
		ord=np.random.permutation(int(len(self.mols)*sampfrac))
		if len(ord)==0:
			return
		ens = np.zeros(len(ord))
		rmsd = np.zeros(len(ord))
		n=0
		for j in ord:
			if ("energy" in self.mols[j].properties.keys()):
				ens[n] = self.mols[j].properties["energy"]
			else :
				ens[n] = self.mols[j].GoEnergy(self.mols[j].coords.flatten())
				tmp = MolEmb.Make_DistMat(self.mols[j].coords) - self.mols[j].DistMatrix
				rmsd[n] = np.sum(tmp*tmp)/len(self.mols[j].coords)
				n=n+1
		print "Mean and Std. Energy", np.average(ens), np.std(ens)
		print "Energy Histogram", np.histogram(ens, 100)
		print "RMSD Histogram", np.histogram(rmsd, 100)
		return

	def Clean_GDB9(self):
		s = MSet(self.name+"_cleaned")
		s.path = self.path
		for mol in self.mols:
			if float('inf') in mol.Bonds_Between:
				print "disconnected atoms in mol.. discard"
			elif -1 in mol.bond_type or 0 in mol.bond_type:
				print "allowed bond type in mol... discard"
			else:
				s.mols.append(mol)
		return s

	def Calculate_vdw(self):
		for mol in self.mols:
			mol.Calculate_vdw()
			print "atomization:", mol.atomization, " vdw:", mol.vdw
		return

	def WriteSmiles(self):
		for mol in self.mols:
			mol.WriteSmiles()
		return

	def MakeBonds(self):
		self.NBonds = 0
		for m in self.mols:
			self.NBonds += m.MakeBonds()
		self.BondTypes = np.unique(np.concatenate([m.bondtypes for m in self.mols],axis=0),axis=0)




class FragableMSet(MSet):
	def __init__(self, name_ ="NaClH2O", path_="./datasets/"):
		MSet.__init__(self, name_, path_)
		return

	def ReadGDB9Unpacked(self, path="/Users/johnparkhill/gdb9/"):
		""" Reads the GDB9 dataset as a pickled list of molecules"""
		from os import listdir
		from os.path import isfile, join
		#onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		for file in onlyfiles:
			if ( file[-4:]!='.xyz' ):
				continue
			self.mols.append(FragableCluster())
			self.mols[-1].ReadGDB9(path+file, file)
			self.mols[-1].properties["set_name"] = self.name
			self
		return

	def ReadXYZ(self,filename, xyz_type = 'mol'):
		""" Reads XYZs concatenated into a single separated by \n\n file as a molset """
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].count('Comment:')>0):
				line0=line-1
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(FragableCluster())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]))
				self.mols[-1].name = str(line)
				self.mols[-1].properties["set_name"] = self.name
		return

	def MBE(self,  atom_group=1, cutoff=10, center_atom=0):
		for mol in self.mols:
			mol.MBE(atom_group, cutoff, center_atom)
		return

	def PySCF_Energy(self):
		for mol in self.mols:
			mol.properties["energy"] = PySCFMP2Energy(mol)
		return

	def Generate_All_MBE_term(self,  atom_group=1, cutoff=10, center_atom=0, max_case = 1000000):
		for mol in self.mols:
			mol.Generate_All_MBE_term(atom_group, cutoff, center_atom, max_case)
		return

	def Generate_All_MBE_term_General(self, frag_list=[], cutoff=10, center_atom=0):
		for mol in self.mols:
			mol.Generate_All_MBE_term_General(frag_list, cutoff, center_atom)
		return

	def Calculate_All_Frag_Energy(self, method="pyscf"):
		for mol in self.mols:
			mol.Calculate_All_Frag_Energy(method)
			# 	mol.Set_MBE_Energy()
		return

	def Calculate_All_Frag_Energy_General(self, method="pyscf"):
		for mol in self.mols:
			#print mol.properties
			#print "Mol set_name", mol.properties["set_name"]
			mol.Calculate_All_Frag_Energy_General(method)
			#        mol.Set_MBE_Energy()
		return

	def Get_All_Qchem_Frag_Energy(self):
		for mol in self.mols:
			mol.Get_All_Qchem_Frag_Energy()
		return

	def Get_All_Qchem_Frag_Energy_General(self):
		for mol in self.mols:
			mol.Get_All_Qchem_Frag_Energy_General()
		return

	def Generate_All_Pairs(self, pair_list=[]):
		for mol in self.mols:
			mol.Generate_All_Pairs(pair_list)
		return

	def Get_Permute_Frags(self, indis=[0]):
		for mol in self.mols:
			mol.Get_Permute_Frags(indis)
		return

	def Set_Qchem_Data_Path(self):
		for mol in self.mols:
			mol.Set_Qchem_Data_Path()
		return



class FragableMSetBF(FragableMSet):
	def __init__(self, name_ ="NaClH2O", path_="./datasets/"):
		MSet.__init__(self, name_, path_)
		return

	def ReadXYZ(self,filename, xyz_type = 'mol'):
		""" Reads XYZs concatenated into a single separated by \n\n file as a molset """
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].count('Comment:')>0):
				line0=line-1
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(FragableClusterBF())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]))
				self.mols[-1].name = str(line)
				self.mols[-1].properties["set_name"] = self.name
		return

	def Generate_All_MBE_term_General(self, frag_list=[]):
		for mol in self.mols:
			mol.Generate_All_MBE_term_General(frag_list)
		return




class GraphSet(MSet):
	def __init__(self, name_ ="gdb9", path_="./datasets/", center_=True):
		MSet.__init__(self, name_, path_, center_)
		self.graphs=[]
		self.path=path_
		self.name=name_
		self.suffix=".graph" #Pickle Database? Poor choice.

	def BondTypes(self):
		types = np.array([],dtype=np.uint8)
		for m in self.mols:
			types = np.union1d(types,m.BondTypes())
		return types

	def NBonds(self):
		nbonds=0
		for m in self.mols:
			nbonds += m.NBonds()
		return nbonds

	def MakeGraphs(self):
		graphs = map(MolGraph, self.mols)
		return graphs

	def ReadXYZ(self,filename = None, xyz_type = 'mol', eqforce=False):
		""" Reads XYZs concatenated into a single file separated by \n\n as a molset """
		if filename == None:
			filename = self.name
		f = open(self.path+filename+".xyz","r")
		txts = f.readlines()
		for line in range(len(txts)):
			if (txts[line].count('Comment:')>0):
				line0=line-1
				nlines=int(txts[line0])
				if xyz_type == 'mol':
					self.mols.append(MolGraph())
				elif xyz_type == 'frag_of_mol':
					self.mols.append(Frag_of_Mol())
				else:
					raise Exception("Unknown Type!")
				self.mols[-1].FromXYZString(''.join(txts[line0:line0+nlines+2]))
				self.mols[-1].name = str(txts[line0+1])
				self.mols[-1].properties["set_name"] = self.name
		if (self.center):
			self.CenterSet()
		if (eqforce):
			self.EQ_forces()
		LOGGER.debug("Read "+str(len(self.mols))+" molecules from XYZ")
		return

	def Save(self):
		print "Saving set to: ", self.path+self.name+self.suffix
		f=open(self.path+self.name+self.suffix,"wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
		return

	def Load(self):
		f = open(self.path+self.name+self.suffix,"rb")
		tmp=pickle.load(f)
		self.__dict__.update(tmp)
		f.close()
		print "Loaded, ", len(self.mols), " molecules "
		print self.NAtoms(), " Atoms total"
		print self.AtomTypes(), " Types "
		return
