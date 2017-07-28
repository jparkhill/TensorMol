"""
Very similar to neighbors, but does a many body expansion
Requires a list of atom fragments, prepared by the user or MSet::ToFragments
"""

from Neighbors import *

class MBNeighbors:
	def __init__(x_,frags_):
		"""
		Initialize a Many-Body Neighbor list which
		can generate terms in a many body expansion up to three.
		"""
		self.sings = None
		self.pairs = None #
		self.triples = None
		self.singsC = None # Coefficients of the expansion.
		self.pairsC = None
		self.triplesC = None

	def Update(x_, frags_, R2=10.0, R3=5.0):
		"""
		Update the lists of bodies and their coefficients.
		"""
		centers = None
		return
