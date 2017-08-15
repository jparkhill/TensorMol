"""
Very similar to neighbors, but does a many body expansion
Requires a list of atom fragments, prepared by the user or MSet::ToFragments
TODO: extend to triples.
"""

from __future__ import absolute_import
import numpy as np
from MolEmb import Make_NListNaive, Make_NListLinear, Make_DistMat

class MBNeighbors:
	"""
	The purpose of this class is to provide:
	self.sings, singz : the monomer part of the set batch, and it's atomic numbers.
	self.pairs, self.trips (likewise)
	self.singC : the coefficient of each in the many-body expansion
	"""
	def __init__(self,x_,z_,frags_):
		"""
		Initialize a Many-Body Neighbor list which
		can generate in linear time.
		terms in a many body expansion up to three.

		Args:
			x_ : coordinates of all the atoms.
			z_ : atomic numbers of all the atoms.
			frags_: list of lists containing these atoms.
		"""
		# Coordinates to pass to networks. (unique)
		self.sings = None
		self.pairs = None
		self.trips = None
		# Atomic numbers to pass to networks. (unique)
		self.singz = None
		self.pairz = None
		self.tripz = None
		# indices in terms of monomers.
		self.pairi = None
		self.tripi = None
		# Coefficients of the expansion.
		self.singC = None
		self.pairC = None
		self.tripC = None
		self.x = x_.copy()
		self.z = z_.copy()
		self.frags = frags_
		self.nt = self.x.shape[0]
		self.nf = len(frags_)
		self.ntrip = 0
		self.npair = 0
		self.maxnatom = 3*max(map(len,frags_))
		self.sings = np.zeros((self.nf,self.maxnatom,3))
		self.singz = np.zeros((self.nf,self.maxnatom), dtype=np.uint8)
		for i in range(self.nf):
			self.sings[i,:len(self.frags[i]),:] = self.x[self.frags[i]].copy()
			self.singz[i,:len(self.frags[i])] = self.z[self.frags[i]].copy()

	def Update(self,x_, R2=10.0, R3=5.0):
		"""
		Update the lists of bodies and their coefficients.

		Args:
			x: A new position vector
			R2: pair cutoff
			R3: triples cutoff
		"""
		if (R2<R3):
			raise Exception("R3<R2 Assumption Violated.")
		self.x = x_.copy()
		for i in range(self.nf):
			self.sings[i,:len(self.frags[i]),:] = self.x[self.frags[i]].copy()
			self.singz[i,:len(self.frags[i])] = self.z[self.frags[i]].copy()
		centers = np.zeros((self.nf,3))
		for i in range(self.nf):
			centers[i] = np.average(self.x[self.frags[i]],axis=0)
		TwoBodyPairs=None
		ThreeBodyPairs=None
		#print "Number of Frags",self.nf
		if (self.nf < 500):
			TwoBodyPairs = Make_NListNaive(centers,R2,self.nf,int(False))
			ThreeBodyPairs = Make_NListNaive(centers,R3,self.nf,int(False))
		else:
			TwoBodyPairs = Make_NListLinear(centers,R2,self.nf,int(False))
			ThreeBodyPairs = Make_NListLinear(centers,R3,self.nf,int(False))
		self.pairi = set()
		self.tripi = set()
		#print TwoBodyPairs
		#print ThreeBodyPairs
		for i in range(self.nf):
			nnt = len(ThreeBodyPairs[i])
			nnp = len(TwoBodyPairs[i])
			if (nnp>0):
				for j in range(nnp):
					if (j != None):
						self.pairi.add(tuple(sorted([i,TwoBodyPairs[i][j]])))
			if (nnt>=2):
				for j in range(nnt):
					if (j != None):
						for k in range(j+1,nnt):
							if (k != None):
								if ThreeBodyPairs[i][k] in ThreeBodyPairs[ThreeBodyPairs[i][j]]:
									self.tripi.add(tuple(sorted([i,ThreeBodyPairs[i][j],ThreeBodyPairs[i][k]])))
		DistMatrix = Make_DistMat(self.x)
		self.ntrip = len(self.tripi)
		self.npair = len(self.pairi)
		#print "num pairs", self.npair
		#print "num trips", self.ntrip
		self.pairi = map(list,self.pairi)
		self.tripi = map(list,self.tripi)
		#print "Pairs", self.pairi
		#print "Trips", self.tripi
		# Now generate the coeffs of each order and sum them.
		self.singC = np.ones(self.nf)
		self.pairC = np.ones(self.npair)
		self.tripC = np.ones(self.ntrip)
		self.singI = self.frags
		self.pairI = []
		self.tripI = []
		self.pairs = np.zeros((self.npair,self.maxnatom,3))
		self.trips = np.zeros((self.ntrip,self.maxnatom,3))
		self.pairz = np.zeros((self.npair,self.maxnatom), dtype=np.uint8)
		self.tripz = np.zeros((self.ntrip,self.maxnatom), dtype=np.uint8)
		for trip_index, trip in enumerate(self.tripi):
			i,j,k = trip[0],trip[1],trip[2]
			self.tripI.append([self.frags[i]+self.frags[j]+self.frags[k]])
			ni,nj,nk = len(self.frags[i]),len(self.frags[j]),len(self.frags[k])
			self.trips[trip_index,:ni,:] = self.x[self.frags[i]].copy()
			self.trips[trip_index,ni:(ni+nj),:] = self.x[self.frags[j]].copy()
			self.trips[trip_index,(ni+nj):(ni+nj+nk),:] = self.x[self.frags[k]].copy()
			self.tripz[trip_index,:ni] = self.z[self.frags[i]].copy()
			self.tripz[trip_index,ni:(ni+nj)] = self.z[self.frags[j]].copy()
			self.tripz[trip_index,(ni+nj):(ni+nj+nk)] = self.z[self.frags[k]].copy()
			#@softcut =
			self.pairC[self.pairi.index(sorted([i,j]))] -= 1
			self.pairC[self.pairi.index(sorted([j,k]))] -= 1
			self.pairC[self.pairi.index(sorted([k,i]))] -= 1
			self.singC[i] += 1
			self.singC[j] += 1
			self.singC[k] += 1
		for pair_index, pair in enumerate(self.pairi):
			i,j = pair[0],pair[1]
			self.pairI.append([self.frags[i]+self.frags[j]])
			ni,nj = len(self.frags[i]),len(self.frags[j])
			self.pairs[pair_index,:ni,:] = self.x[self.frags[i]].copy()
			self.pairs[pair_index,ni:(ni+nj),:] = self.x[self.frags[j]].copy()
			self.pairz[pair_index,:ni] = self.z[self.frags[i]].copy()
			self.pairz[pair_index,ni:(ni+nj)] = self.z[self.frags[j]].copy()
			self.singC[i] -= 1
			self.singC[j] -= 1
		return
