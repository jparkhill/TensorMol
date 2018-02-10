"""
	This is a crazy type of simulation
	which finds minimum energy paths between
	multiple geometries using repeated NEB calculations.
	This could be used for example to make better training data.
	for reactive species.
"""

from __future__ import absolute_import
from __future__ import print_function
from .Opt import *
from .Neb import *
import random
import time

class LocalReactions:
	def __init__(self,f_, g0_, Nstat_ = 20):
		"""
		In this case the bumps are pretty severe
		and the metadynamics is designed to sample reactions rapidly.

		Args:
			f_: an energy, force routine (energy Hartree, Force Kcal/ang.)
			g0_: initial molecule.
		"""
		self.NStat = Nstat_
		LOGGER.info("Finding reactions between %i local minima... ", self.NStat)
		MOpt = MetaOptimizer(f_,g0_,StopAfter_=self.NStat)
		mins = MOpt.MetaOpt()
		LOGGER.info("---------------------------------")
		LOGGER.info("--------- Nudged Bands ----------")
		LOGGER.info("---------------------------------")
		nbead = 15
		self.path = np.zeros(shape = ((self.NStat-1)*nbead,g0_.NAtoms(),3))
		for i in range(self.NStat-1):
			g0 = Mol(g0_.atoms,mins[i])
			g1 = Mol(g0_.atoms,mins[i+1])
			neb = NudgedElasticBand(f_,g0,g1,name_ = "Neb"+str(i),thresh_=0.004,nbeads_=nbead)
			self.path[i*nbead:(i+1)*nbead] = neb.Opt()
		# finally write the whole trajectory.
		for i in range(self.path.shape[0]):
			m = Mol(g0_.atoms, self.path[i])
			m.WriteXYZfile("./results/","WebPath")
		return
