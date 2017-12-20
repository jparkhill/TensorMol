from __future__ import absolute_import
from __future__ import print_function
from .Mol import *
from ..Util import *
import os,sys,re
import numpy as np
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle
from ..Math import LinearOperations
if (HAS_EMB):
	import MolEmb

class Digester:
	"""
	An Embedding gives some chemical description of a molecular
	Environment around a point. This one is for networks that will embed properties of atoms.
	please refer to /C_API/setup.py

	note: Molecule embeddings and Behler-Parrinello are in DigestMol.
	"""
	def __init__(self, eles_, name_="GauSH", OType_="Disp"):
		"""

		Args:
			eles_ : a list of elements in the Tensordata that I'll digest
			name_: type of digester to reduce molecules to NN inputs.
			OType_: property of the molecule which will be learned (energy, force, etc)
		"""

		 # In Atomic units at 300K
		# These are the key variables which determine the type of digestion.
		self.name = name_ # Embedding type.
		self.eshape=None  #shape of an embedded case
		self.lshape=None  #shape of the labels of an embedded case.
		self.OType = OType_ # Output Type: HardP, SmoothP, StoP, Disp, Force, Energy etc. See Emb() for options.

		self.NTrainSamples=1 # Samples per atom. Should be made a parameter.
		if (self.OType == "SmoothP" or self.OType == "Disp"):
			self.NTrainSamples=1 #Smoothprobability only needs one sample because it fits the go-probability and pgaussians-center.

		self.eles = np.array(eles_)
		self.eles.sort() # Consistent list of atoms in the order they are treated.
		self.neles = len(eles_) # Consistent list of atoms in the order they are treated.
		self.nsym = self.neles+(self.neles+1)*self.neles  # channel of sym functions
		self.npgaussian = self.neles # channel of PGaussian
		# Instead self.emb should know it's return shape or it should be testable.

		self.SamplingType = PARAMS["dig_SamplingType"]
		self.TrainSampDistance=2.0 #how far in Angs to sample on average.
		self.ngrid = PARAMS["dig_ngrid"] #this is a shitty parameter if we go with anything other than RDF and should be replaced.
		self.BlurRadius = PARAMS["BlurRadius"] # Stdev of gaussian used as prob of atom
		self.SensRadius=6.0 # Distance which is used for input.
		self.embtime=0.0
		self.outtime=0.0
		self.Print()
		return

	def Print(self):
		LOGGER.info("-------------------- ")
		LOGGER.info("Digester Information ")
		LOGGER.info("self.name: "+self.name)
		LOGGER.info("self.OType: "+self.OType)
		LOGGER.debug("self.NTrainSamples: "+str(self.NTrainSamples))
		LOGGER.debug("self.TrainSampDistance: "+str(self.TrainSampDistance))
		LOGGER.debug("self.OType: "+self.OType)
		LOGGER.info("-------------------- ")
		return

	def MakeSamples_v2(self,point):    # with sampling function f(x)=M/(x+1)^2+N; f(0)=maxdisp,f(maxdisp)=0; when maxdisp =5.0, 38 % lie in (0, 0.1)
		disps = samplingfunc_v2(self.TrainSampDistance * np.random.random(self.NTrainSamples), self.TrainSampDistance)
		theta  = np.random.random(self.NTrainSamples)* math.pi
		phi = np.random.random(self.NTrainSamples)* math.pi * 2
		grids  = np.zeros((self.NTrainSamples,3),dtype=np.float64)
		grids[:,0] = disps*np.cos(theta)
		grids[:,1] = disps*np.sin(theta)*np.cos(phi)
		grids[:,2] = disps*np.sin(theta)*np.sin(phi)
		return grids + point

#
#  Embedding functions, called by batch digests. Use outside of Digester() is discouraged.
#  Instead call a batch digest routine.
#

	def Emb(self, mol_, at_, xyz_, MakeOutputs=True, MakeGradients=False, Transforms=None):
		"""
		Generates various molecular embeddings.

		Args:
			mol_: a Molecule to be digested
			at_: an atom to be digested or moved. if at_ < 0 it usually returns arrays for each atom in the molecule
			xyz_: makes inputs with at_ moved to these positions.
			MakeOutputs: generates outputs according to self.OType.
			MakeGradients: Generate nuclear derivatives of inputs.
			Transforms: Generate inputs for all the linear transformations appended.

		Returns:
			Output embeddings, and possibly labels and gradients.
			if at_ < 0 the first dimension loops over atoms in mol_
		"""
		#start = time.time()
		if (self.name=="CZ"):
			if (at_ > 0):
				raise Exception("CoordZ embedding is done moleculewise always")
			Ins = np.zeros((mol_.NAtoms,4))
			Ins[:,0] = mol_.atoms
			Ins[:,1:] = mol_.coords
		elif (self.name=="Coulomb"):
			Ins= MolEmb.Make_CM(mol_.coords, xyz_, mol_.atoms , self.eles ,  self.SensRadius, self.ngrid, at_, 0.0)
		elif (self.name=="GauSH"):
			if (Transforms == None):
				Ins =  MolEmb.Make_SH(PARAMS, mol_.coords, mol_.atoms, at_)
			else:
				Ins =  MolEmb.Make_SH_Transf(PARAMS, mol_.coords, mol_.atoms, at_, Transforms)
		elif (self.name=="GauInv"):
			Ins= MolEmb.Make_Inv(PARAMS, mol_.coords, mol_.atoms, at_)
		elif (self.name=="RDF"):
			Ins= MolEmb.Make_RDF(mol_.coords, xyz_, mol_.atoms , self.eles ,  self.SensRadius, self.ngrid, at_, 0.0)
		elif (self.name=="SensoryBasis"):
			Ins= mol_.OverlapEmbeddings(mol_.coords, xyz_, mol_.atoms , self.eles ,  self.SensRadius, self.ngrid, at_, 0.0)
		elif (self.name=="SymFunc"):
			Ins= self.make_sym(mol_.coords, xyz_, mol_.atoms , self.eles ,  self.SensRadius, self.ngrid, at_, 0.0)
		elif(self.name == "ANI1_Sym"):
			Ins = MolEmb.Make_ANI1_Sym(PARAMS, mol_.coords,  mol_.atoms, self.eles, at_)
		else:
			raise Exception("Unknown Embedding Function.")
		#self.embtime += (time.time() - start)
		#start = time.time()
		Outs=None
		if (MakeOutputs):
			if (self.OType=="HardP"):
				Outs = self.HardCut(xyz_-coords_[at_])
			elif (self.OType=="SmoothP"):
				Outs = mol_.FitGoProb(at_)
			elif (self.OType=="Disp"):
				Outs = mol_.GoDisp(at_)
			elif (self.OType=="GoForce"):
				Outs = mol_.GoForce(at_)
			elif (self.OType=="GoForceSphere"):
				Outs = mol_.GoForce(at_, 1) # See if the network is better at doing spherical=>spherical
			elif (self.OType=="Force"):
				if ( "forces" in mol_.properties):
					if (at_<0):
						Outs = mol_.properties['forces']
						#print "Outs", Outs
					else:
						Outs = mol_.properties['forces'][at_].reshape((1,3))
				else:
					raise Exception("Mol Is missing force. ")
			elif (self.OType=="Del_Force"):
				if ( "forces" in mol_.properties):
					if ( "mmff94forces" in mol_.properties):
						if (at_<0):
							Outs = mol_.properties['forces']
							Ins = np.append(Ins, mol_.properties["mmff94forces"],axis=1)
						else:
							Outs = mol_.properties['forces'][at_].reshape((1,3))
							Ins = np.append(Ins, mol_.properties['mmff94forces'][at_].reshape((1,3)), axis=1)
					else:
						raise Exception("Mol Is missing MMFF94 force. ")
				else:
					raise Exception("Mol Is missing force. ")
			elif (self.OType=="ForceSphere"):
				if ( "sphere_forces" in mol_.properties):
					if (at_<0):
						Outs = mol_.properties['sphere_forces']
						#print "Outs", Outs
					else:
						Outs = mol_.properties['sphere_forces'][at_].reshape((1,3))
				else:
					raise Exception("Mol Is missing spherical force. ")
			elif (self.OType=="ForceMag"):
				if ( "forces" in mol_.properties):
					if (at_<0):
						Outs = np.array(np.linalg.norm(mol_.properties['forces'], axis=1))
					else:
						Outs = np.array(np.linalg.norm(mol_.properties['forces'][at_])).reshape((1,1))
				else:
					raise Exception("Mol Is missing force. ")
			elif (self.OType=="StoP"):
				ens_ = mol_.EnergiesOfAtomMoves(xyz_,at_)
				if (ens_==None):
					raise Exception("Empty energies...")
				print(ens_.min(), ens_.max())
				Es=ens_-ens_.min()
				Boltz=np.exp(-1.0*Es/PARAMS["KAYBEETEE"])
				rnds = np.random.rand(len(xyz_))
				Outs = np.array([1 if rnds[i]<Boltz[i] else 0 for i in range(len(ens_))])
			elif (self.OType=="Energy"):
				if ("energy" in mol_.properties):
					ens_ = mol_.properties["energy"]
				else:
					raise Exception("Empty energies...")
			elif (self.OType=="AtomizationEnergy"):
				if ("atomization" in mol_.properties):
					ens_ = mol_.properties["atomization"]
				else:
					raise Exception("Empty energies...")
			elif (self.OType=="CalcEnergy"):
				ens_ = mol_.EnergiesOfAtomMoves(xyz_,at_)
				if (ens_==None):
					raise Exception("Empty energies...")
				E0=np.min(ens_)
				Es=ens_-E0
				Outs = Es
			else:
				raise Exception("Unknown Digester Output Type.")
			#self.outtime += (time.time() - start)
			#print "Embtime: ", self.embtime, " OutTime: ", self.outtime
			return Ins,Outs
		else:
			return Ins

#
#  Various types of Batch Digests.
#

	def TrainDigestMolwise(self, mol_, MakeOutputs_=True):
		"""
		Returns list of inputs and outputs for a molecule.
		Uses self.Emb() uses Mol to get the Desired output type (Energy,Force,Probability etc.)
		This version works mol-wise to try to speed up and avoid calling C++ so much...

		Args:
			mol_: a molecule to be digested
			eles_: A list of elements coming from Tensordata to order the output.

		Returns:
			Two lists: containing inputs and outputs in order of eles_
		"""
		return self.Emb(mol_,-1,mol_.coords[0], MakeOutputs_) # will deal with getting energies if it's needed.

	def TrainDigest(self, mol_, ele_, MakeDebug=False):
		"""
		Returns list of inputs and outputs for a molecule.
		Uses self.Emb() uses Mol to get the Desired output type (Energy,Force,Probability etc.)

		Args:
			mol_: a molecule to be digested
			ele_: an element for which training data will be made.
			MakeDebug: if MakeDebug is True, it also returns a list with debug information to trace possible errors in digestion.
		"""
		if (self.eshape==None or self.lshape==None):
			tinps, touts = self.Emb(mol_,0,np.array([[0.0,0.0,0.0]]))
			self.eshape = list(tinps[0].shape)
			self.lshape = list(touts[0].shape)
			LOGGER.debug("Assigned Digester shapes: "+str(self.eshape)+str(self.lshape))
		ncase = mol_.NumOfAtomsE(ele_)*self.NTrainSamples
		ins = np.zeros(shape=tuple([ncase]+list(self.eshape)),dtype=np.float64)
		outs = np.zeros(shape=tuple([ncase]+list(self.lshape)),dtype=np.float64)
		dbg=[]
		casep=0
		for i in range(len(mol_.atoms)):
			if (mol_.atoms[i]==ele_):
				if (self.OType == "SmoothP" or self.OType == "Disp" or self.OType == "Force"):
					inputs, outputs = self.Emb(mol_,i,mol_.coords[i]) # will deal with getting energies if it's needed.
				elif(self.SamplingType=="Smooth"): #If Smooth is now a property of the Digester: OType SmoothP
					samps=PointsNear(mol_.coords[i], self.NTrainSamples, self.TrainSampDistance)
					inputs, outputs = self.Emb(mol_,i,samps) # will deal with getting energies if it's needed.
				else:
					samps=self.MakeSamples_v2(mol_.coords[i])
					inputs, outputs = self.Emb(mol_,i,samps)
				# Here we should write a short routine to debug/print the inputs and outputs.
				#				print "Smooth",outputs
				#print i, mol_.atoms, mol_.coords,mol_.coords[i],"Samples:",samps,"inputs ", inputs, "Outputs",outputs, "Distances",np.array(map(np.linalg.norm,samps-mol_.coords[i]))

				ins[casep:casep+self.NTrainSamples] = np.array(inputs)
				outs[casep:casep+self.NTrainSamples] = outputs
				casep += self.NTrainSamples
		if (MakeDebug):
			return ins,outs,dbg
		else:
			return ins,outs

	def UniformDigest(self, mol_, at_, mxstep, num):
		""" Returns list of inputs sampled on a uniform cubic grid around at """
		ncase = num*num*num
		samps=MakeUniform(mol_.coords[at_],mxstep,num)
		if (self.name=="SymFunc"):
			inputs = self.Emb(self, mol_, at_, samps, None, False) #(self.EmbF())(mol_.coords, samps, mol_.atoms, self.eles ,  self.SensRadius, self.ngrid, at_, 0.0)
			inputs = np.asarray(inputs)
		else:
			inputs = self.Emb(self, mol_, at_, samps, None, False)
			inputs = np.assrray(inputs[0])
		return samps, inputs

	def emb_vary_coords(self, coords, xyz, atoms, eles, Radius, ngrid, vary_at, tar_at):
		return  MolEmb.Make_CM_vary_coords(coords, xyz, atoms, eles, Radius, ngrid, vary_at, tar_at)

	def make_sym(self, coords_, xyz_, ats_,  eles , SensRadius, ngrid, at_, dummy):    #coords_, xyz_, ats_, self.eles ,  self.SensRadius, self.ngrid, at_, 0.0
		zeta=[]
		eta1=[]
		eta2=[]
		Rs=[]
		for i in range (0, ngrid):
			zeta.append(1.5**i)    # set some value for zeta, eta, Rs
			eta1.append(0.008*(2**i))
			eta2.append(0.002*(2**i))
			Rs.append(i*SensRadius/float(ngrid))
		SYM =  MolEmb.Make_Sym(coords_, xyz_, ats_, eles, at_, SensRadius, zeta, eta1, eta2, Rs)
		SYM = np.asarray(SYM[0], dtype=np.float64)
		SYM = SYM.reshape((SYM.shape[0]/self.nsym, self.nsym,  SYM.shape[1] *  SYM.shape[2]))
		return SYM
