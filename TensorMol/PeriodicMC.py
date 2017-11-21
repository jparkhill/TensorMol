"""
Periodic Monte Carlo.
"""

from __future__ import absolute_import
from __future__ import print_function
from .Sets import *
from .TFManage import *
from .Neighbors import *
from .Electrostatics import *
from .QuasiNewtonTools import *
from .Statistics import *
from .Periodic import *
from .SimpleMD import *
from .PeriodicMD import *

class PeriodicMonteCarlo(PeriodicVelocityVerlet):
	def __init__(self, Force_, name_ ="PdicMC"):
		"""

		Args:
			Force_: A PERIODIC energy, force CLASS.
			PMol_: initial molecule.
		Returns:
			Nothing.
		"""
		PeriodicVelocityVerlet.__init__(self, Force_,name_)
		e0, f0 = self.PForce(self.PForce.mol0.coords)
		self.eold = e0
		self.Estat = OnlineEstimator(e0)
		self.RDFold = self.PForce.RDF(self.PForce.mol0.coords)
		self.RDFstat = OnlineEstimator(self.RDFold)
		self.Xstat = OnlineEstimator(self.x)
		self.PACCstat = OnlineEstimator(1.0)
		self.kbt = KAYBEETEE*(PARAMS["MDTemp"]/300.0) # Hartrees.
		self.Eav = None
		self.dE2 = None
		self.Xav = None
		self.dX2 = None
		self.Pacc = 0.0
	def RandomVectorField(self,x_):
		"""
		This is a random move which is locally continuous
		so that nearby atoms are pulled together to improve
		acceptance probability. This is done by interpolating
		randomly placed vectors of random orientations to the points
		where the atoms are. (9 such vectors.)

		To accomplish rotations, there is also a solenoidal field
		which acts as a cross product.

		V(x_j) = \sum_(pts, i) v_i * exp(-dij^2)
		"""
		mxx = np.max(x_)
		mnx = np.min(x_)
		npts = 4
		pts = np.random.uniform(mxx-mnx,size=(npts,3))+mnx
		rmagn = np.random.normal(scale = 0.07, size=(npts,1))
		theta = np.random.uniform(3.1415, size=(npts,1))
		phi = np.random.uniform(2.0*3.1415, size=(npts,1))
		rx = np.sin(theta)*np.cos(phi)
		ry = np.sin(theta)*np.sin(phi)
		rz = np.cos(theta)
		magn = np.concatenate([rmagn*rx,rmagn*ry,rmagn*rz],axis=1)
		jd = np.concatenate([pts,x_])
		DMat = MolEmb.Make_DistMat_ForReal(jd,npts)
		sigma = np.random.uniform(2.2)+0.02
		expd = (1.0/np.sqrt(6.2831*sigma*sigma))*np.exp(-1.0*DMat[:,npts:]*DMat[:,npts:]/(2*sigma*sigma))
		tore = np.einsum('jk,ji->ik', magn, expd)
		# do the solenoidal piece.
		for i in range(npts):
			if (np.random.random() < 0.6):
				vs = x_ - pts[i]
				for j in range(x_.shape[0]):
					vs[j] /= np.linalg.norm(vs[j])
				soil = np.cross(vs,4.0*magn[i])*(expd[i,:,np.newaxis])
				#print(soil)
				tore += soil
		return tore
	def MetropolisHastings(self,x_):
		"""
		Perform the Metropolis step.
		"""
		dx = self.RandomVectorField(x_)
		dx += np.random.uniform(size=x_.shape)*0.0005
		edx , tmp = self.PForce(x_+dx,DoForce=False)
		PMove = min(1.0,np.exp(-(edx - self.eold)/self.kbt))
		if (np.random.random()<PMove):
			self.x = self.PForce.lattice.ModuloLattice(x_ + dx)
			self.eold = edx
			self.RDFold = self.PForce.RDF(self.x)
			self.Pacc,t = self.PACCstat(1.0)
			print("accept")
		else:
			self.Pacc,t = self.PACCstat(0.0)
			print('reject',PMove,(edx - self.eold),self.kbt)
		self.Eav, self.dE2 = self.Estat(self.eold)
		self.Xav, self.dX2 = self.Xstat(self.x)
		return
	def Prop(self):
		"""
		Propagate Monte Carlo.
		MEEETROOOPPOLLIISSS
		"""
		step = 0
		self.md_log = np.zeros((self.maxstep, 7)) # time Dipoles Energy
		while(step < self.maxstep):
			self.t = step
			t = time.time()
			self.MetropolisHastings(self.x)
			rdf, rdf2 = self.RDFstat(self.RDFold)
			self.md_log[step,0] = self.t
			self.md_log[step,5] = self.EPot
			if (step%3==0 and PARAMS["MDLogTrajectory"]):
				self.WriteTrajectory()
			if (step%500==0):
				np.savetxt("./results/"+"MDLog"+self.name+".txt",self.md_log)
				np.savetxt("./results/"+"MCRDF"+self.name+".txt",rdf)
				np.savetxt("./results/"+"MCRDF2"+self.name+".txt",rdf2)
			step+=1
			LOGGER.info("Step: %i <E>(kJ/mol): %.5f sqrt(<dE2>): %.5f sqrt(<dX2>): %.5f Paccept %.5f Rho(g/cm**3): %.5f ", step, self.Eav, np.sqrt(self.dE2), np.sqrt(np.linalg.norm(self.dX2)), self.Pacc,self.Density())
			print(("per step cost:", time.time() -t ))
		return
