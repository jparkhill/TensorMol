"""
Changes that need to be made:
"""

from Sets import *
from TFManage import *
from DIIS import *
from LinearOperations import *
import random
import time

class Optimizer:
	def __init__(self,tfm_):
		"""
		Geometry optimizations based on NN-PES's etc.
		Args:
			tfm_: a TFManage or TFMolManage instance to use as a molecular model.
		"""
		self.thresh = PARAMS["OptThresh"]
		self.maxstep = PARAMS["OptMaxStep"]
		self.fscale = PARAMS["OptStepSize"]
		self.momentum = PARAMS["OptMomentum"]
		self.momentum_decay = PARAMS["OptMomentumDecay"]
		self.max_opt_step = PARAMS["OptMaxCycles"]
		self.step = self.maxstep
		self.ngrid = 10 # Begin with 500 pts sampled 0.2A in each direction.
		self.probtype = 0 # 0 = one atom probability, 1 = product of all probabilities for each sample.
		self.tfm = tfm_
		if (self.tfm!=None):
			self.OType = self.tfm.TData.dig.OType
			print "Optimizer will use ",self.OType, " outputs from tensorflow to optimize."
		return

	def CenterOfMass(self, xyz, probs):
		#Check for garbage...
		isn=np.all(np.isfinite(probs))
		if (not isn):
			print "Infinite Probability predicted."
			print probs
			raise Exception("NanTFOutput")
		nprobs=probs/np.sum(probs)
		pxyz=(xyz.T*nprobs).T
		return np.sum(pxyz,axis=0)

	def LargestP(self, xyz, probs):
		return xyz[np.argmax(probs)]

	def SmallestP(self, xyz, probs):
		return xyz[np.argmin(probs)]

	def GaussianConv(self, xyz, probs, width = 0.05):
		return best_xyz

	def ChooseBest(self, xyz, probs, method = "COM"):
		''' This routine actually belongs in the digester, because the digester knows the relationship between outputs and desired coordinates. '''
		if (method == "COM"):
			return self.CenterOfMass(xyz, probs)
		elif (method == "largest"):
			return self.LargestP(xyz, probs)
		elif (method == "smallest"):
			return self.SmallestP(xyz, probs)
		elif (method == "gconv"):
			return self.GaussianConv(xyz, probs)
		else:
			raise Exception("Unknown Method")

	def RemoveAverageTorque(self,velocs,coords):
		p0=np.average(coords,axis=0)
		ccoords = coords-p0
		Ls=np.cross(ccoords,velocs)
		n_L=np.sum(Ls,axis=0)
		print "Net Angular Momentum:",n_L
		return velocs

	def OptGoForce(self,m):
		"""
		Simple test of the Go-Force
		Args:
			m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print "Orig Coords", m.coords
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		while(err>self.thresh and step < self.max_opt_step):
			for i in range(m.NAtoms()):
				veloc[i] = -0.1*m.GoForce(i)
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			# Remove translation.
			c_veloc = c_veloc - np.average(c_veloc,axis=0)
			prev_m = Mol(m.atoms, m.coords)
			m.coords = m.coords + c_veloc
			old_veloc = self.momentum_decay*c_veloc
			err = m.rms(prev_m)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", "OptLog")
			step+=1
			print "Step:", step, " RMS Error: ", err, " Coords: ", m.coords
		return

	def OptLJForce(self,m):
		"""
		Simple test of the LJ-Force
		Args:
			m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print "Orig Coords", m.coords
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		self.momentum = 0.0
		while(err>self.thresh and step < self.max_opt_step):
			HessD = np.clip(m.NumericLJHessDiag(), 1e-3, 1e16)
			# for i in range(m.NAtoms()):
			# 	veloc[i] = -0.0001*m.LJForce(i)
			# 	#print veloc[i]
			veloc = -0.01*m.LJForce()#/HessD
			#print "Coords", m.coords
			# print "Hess", HessD
			# print m.NumericLJHessDiag()
			# print "velco", veloc
			#print m.properties["LJE"]nergy(m.coords)
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			# Remove translation.
			c_veloc = c_veloc - np.average(c_veloc,axis=0)
			prev_m = Mol(m.atoms, m.coords)
			m.coords = m.coords + c_veloc
			old_veloc = self.momentum_decay*c_veloc
			err = m.rms(prev_m)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile(PARAMS["results_dir"], "OptLog")
			step+=1
			print "Step:", step, " RMS Error: ", err, " Coords: ", m.coords
		return

	def OptTFGoForce(self,m,Debug=True):
		"""
		Optimize using force output of an atomwise network.
		Args:
			m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print "Orig Coords", m.coords
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		while(err>self.thresh and step < self.max_opt_step):
			if (PARAMS["RotAvOutputs"]):
				veloc = -0.001*self.tfm.EvalRotAvForce(m, RotAv=10)
			elif (PARAMS["OctahedralAveraging"]):
				veloc = -0.001*self.tfm.EvalOctAvForce(m)
			else:
				for i in range(m.NAtoms()):
					veloc[i] = -0.001*self.tfm.evaluate(m,i)
			if (Debug):
				for i in range(m.NAtoms()):
					print "Real & TF ",m.atoms[i], ":" , veloc[i], "::", -1.0*m.GoForce(i)
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			# Remove translation.
			c_veloc = c_veloc - np.average(c_veloc,axis=0)
			prev_m = Mol(m.atoms, m.coords)
			m.coords = m.coords + c_veloc
			old_veloc = self.momentum_decay*c_veloc
			err = m.rms(prev_m)
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./datasets/", "OptLog")
			step+=1
			print "Step:", step, " RMS Error: ", err, " Coords: ", m.coords
		return

	def OptTFRealForce(self,m, filename="OptLog",Debug=False):
		"""
		Optimize using force output of an atomwise network.
		now also averages over rotations...
		Args:
			m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		diis = DIIS()
		print "Orig Coords", m.coords
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		while((rmsdisp>self.thresh or rmsgrad>self.thresh)  and step < self.max_opt_step):
			if (PARAMS["RotAvOutputs"]):
				veloc = self.tfm.EvalRotAvForce(m, RotAv=PARAMS["RotAvOutputs"], Debug=False)
			elif (PARAMS["OctahedralAveraging"]):
				veloc = self.tfm.EvalOctAvForce(m, Debug=True)
			else:
				for i in range(m.NAtoms()):
					veloc[i] = self.tfm.evaluate(m,i)
			if (Debug):
				for i in range(m.NAtoms()):
					print "TF veloc: ",m.atoms[i], ":" , veloc[i]
			veloc = veloc - np.average(veloc,axis=0)
			#Remove translation.
			prev_m = Mol(m.atoms, m.coords)
			#ForceFunction = lambda x: self.tfm.EvalRotAvForce(Mol(m.atoms,x), RotAv=1, Debug=False)
			#DHess = DiagHess(ForceFunction,m.coords,veloc)
			if (rmsgrad > 0.06):
				m.coords = diis.NextStep(m.coords,veloc)
			else:
				c_veloc = (1.0-self.momentum)*self.fscale*veloc+self.momentum*old_veloc
				old_veloc = self.momentum_decay*c_veloc
				m.coords = m.coords + c_veloc
			rmsgrad = np.sum(np.linalg.norm(veloc,axis=1))/veloc.shape[0]
			maxgrad = np.amax(np.linalg.norm(veloc,axis=1))
			rmsdisp = np.sum(np.linalg.norm((prev_m.coords-m.coords),axis=1))/m.coords.shape[0]
			maxdisp = np.amax(np.linalg.norm((prev_m.coords - m.coords), axis=1))
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			step+=1
			LOGGER.info("Step: %i RMS Disp: %.5f Max Disp: %.5f RMS Gradient: %.5f  Max Gradient: %.5f ", step, rmsdisp, maxdisp, rmsgrad, maxgrad)
		return prev_m

	def OptANI1(self,m, filename="OptLog",Debug=False):
		"""
		Optimize using force output of an atomwise network.
		now also averages over rotations...
		Args:
		        m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		diis = DIIS()
		print "Orig Coords", m.coords
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		veloc=np.zeros(m.coords.shape)
		old_veloc=np.zeros(m.coords.shape)
		while( step < self.max_opt_step):
			prev_m = Mol(m.atoms, m.coords)
			energy, veloc = self.tfm.Eval_BPForce(m,total_energy=True)
			veloc = RemoveInvariantForce(m.coords, veloc, m.atoms)
			rmsgrad = np.sum(np.linalg.norm(veloc,axis=1))/veloc.shape[0]
			veloc *= self.fscale
			if (rmsgrad > 30.0):
				m.coords = diis.NextStep(m.coords,veloc)
			else:
				c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
				old_veloc = self.momentum_decay*c_veloc
				m.coords = m.coords + c_veloc
			print "step: ", step ," energy: ", energy, " rmsgrad ", rmsgrad
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			step+=1
		return prev_m

	def OptTFRealForceLBFGS(self,m, filename="OptLog",Debug=False):
		"""
		Optimize using force output of an atomwise network.
		now also averages over rotations...
		Args:
			m: A distorted molecule to optimize
		"""
		# Sweeps one at a time
		rmsdisp = 10.0
		maxdisp = 10.0
		rmsgrad = 10.0
		maxgrad = 10.0
		step=0
		mol_hist = []
		prev_m = Mol(m.atoms, m.coords)
		print "Orig Coords", m.coords
		#print "Initial force", self.tfm.evaluate(m, i), "Real Force", m.properties["forces"][i]
		veloc=np.zeros(m.coords.shape)
		self.m_max = PARAMS["OptMaxBFGS"]
		R_Hist = np.zeros(([self.m_max]+list(m.coords.shape)))
		F_Hist = np.zeros(([self.m_max]+list(m.coords.shape)))
		while(rmsdisp>self.thresh and step < self.max_opt_step):
			if (PARAMS["RotAvOutputs"]):
				veloc = self.tfm.EvalRotAvForce(m, RotAv=PARAMS["RotAvOutputs"], Debug=False)
			elif (PARAMS["OctahedralAveraging"]):
				veloc = self.tfm.EvalOctAvForce(m, Debug=True)
			else:
				for i in range(m.NAtoms()):
					veloc[i] = self.tfm.evaluate(m,i)
			if (Debug):
				for i in range(m.NAtoms()):
					print "TF veloc: ",m.atoms[i], ":" , veloc[i]
			veloc = veloc - np.average(veloc,axis=0)
			if step < self.m_max:
				R_Hist[step] = m.coords.copy()
				F_Hist[step] = veloc.copy()
			else:
				R_Hist = np.roll(R_Hist,-1,axis=0)
				F_Hist = np.roll(R_Hist,-1,axis=0)
				R_Hist[-1] = m.coords.copy()
				F_Hist[-1] = veloc.copy()
			# Quasi Newton L-BFGS global step.
			q = veloc.copy()
			for i in range(min(self.m_max,step)-1, 0, -1):
				s = R_Hist[i] - R_Hist[i-1]
				y = F_Hist[i] - F_Hist[i-1]
				rho = 1.0/np.einsum("ia,ia",y,s)#y.dot(s)
				a = rho * np.einsum("ia,ia",s,q)#s.dot(q)
				#print "a ",a
				q -= a*y
			if step < 1:
				H=1.0
			else:
				num = min(self.m_max-1,step)
				v1 = (R_Hist[num] - R_Hist[num-1])
				v2 = (F_Hist[num] - F_Hist[num-1])
				H = (np.einsum("ia,ia",v1,v2))/(np.einsum("ia,ia",v2,v2))
				#print "H:", H
			z = H*q
			for i in range (1,min(self.m_max,step)):
				s = (R_Hist[i] - R_Hist[i-1])
				y = (F_Hist[i] - F_Hist[i-1])
				rho = 1.0/np.einsum("ia,ia",y,s)#y.dot(s)
				a=rho*np.einsum("ia,ia",s,q)#s.dot(q)
				beta = rho*np.einsum("ia,ia",y,z)#(force_his[i] - force_his[i-1]).dot(z)
				#print "a-b: ", (a-beta)
				z += s*(a-beta)
			c_veloc = self.fscale*z
			prev_m = Mol(m.atoms, m.coords)
			m.coords = m.coords + c_veloc
			rmsgrad = np.sum(np.linalg.norm(veloc,axis=1))/veloc.shape[0]
			maxgrad = np.amax(np.linalg.norm(veloc,axis=1))
			rmsdisp = np.sum(np.linalg.norm((prev_m.coords-m.coords),axis=1))/m.coords.shape[0]
			maxdisp = np.amax(np.linalg.norm((prev_m.coords - m.coords), axis=1))
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results/", filename)
			step+=1
			LOGGER.info("Step: %i RMS Disp: %.5f Max Disp: %.5f RMS Gradient: %.5f  Max Gradient: %.5f ", step, rmsdisp, maxdisp, rmsgrad, maxgrad)
		return prev_m

	def OptProb(self,m):
		''' This version tests if the Go-opt converges when atoms are moved to
			the points of mean-probability. CF POfAtomMoves() '''
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		new_m = Mol()
		new_m = m
 		print "Orig Coords", new_m.coords
		old_veloc=np.zeros(new_m.coords.shape)
		while(err>self.thresh and step < self.max_opt_step):
			coords=np.array(new_m.coords,copy=True)
			tmp_m = Mol(new_m.atoms, coords)
			newcoord = self.tfm.SmoothPOneAtom(new_m, i)
			veloc = new_m.GoMeanProbForce() # Tries to update atom positions to average of Go-Probability.
			# Remove average torque
			#veloc = self.RemoveAverageTorque(new_m.coords,veloc)
			#TODO remove rotation.
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			# Remove translation.
			c_veloc = c_veloc - np.average(c_veloc,axis=0)

			new_m.coords = tmp_m.coords + c_veloc
			old_veloc = self.momentum_decay*c_veloc

			err = new_m.rms(tmp_m)
			mol_hist.append(tmp_m)
			tmp_m.WriteXYZfile("./datasets/", "OptLog")
			if (0): # Prints density along the optimization.
				xs,ps = self.tfm.EvalAllAtoms(tmp_m)
				grids = tmp_m.MolDots()
				grids = tmp_m.AddPointstoMolDots(grids, xs, ps)
				GridstoRaw(grids,250,str(step))
			step+=1
			Energy = new_m.EnergyAfterAtomMove(new_m.coords[0],0)
                        print "Step:", step, " RMS Error: ", err, "Energy: ", Energy#" Coords: ", new_m.coords
			#print "Step:", step, " RMS Error: ", err, " Coords: ", new_m.coords
		return

	def Opt(self,m):
		# Sweeps one at a time
		if (self.OType=="SmoothP"):
			return self.OptSmoothP(m)
		if (self.OType=="Force"):
			return self.OptForce(m)
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		new_m = Mol(m.atoms, m.coords)
		print "Orig Coords", new_m.coords
		old_veloc=np.zeros(new_m.coords.shape)
		while(err>self.thresh and step < self.max_opt_step):
			coords=np.array(new_m.coords,copy=True)
			tmp_m = Mol(new_m.atoms, coords)
			for i in range(m.NAtoms()):
				if (self.probtype==0):
					xyz,probs = self.tfm.EvalOneAtom(new_m, i, self.step, self.ngrid)
				else:
					xyz,probs = self.tfm.EvalOneAtomMB(new_m, i, self.step,self.ngrid)
				#print "xyz:", xyz
				#print "probs:", probs
				new_m.coords[i] = self.ChooseBest(xyz, probs)

				#Update with momentum.
			veloc = new_m.coords - tmp_m.coords
			# Remove average torque
			#veloc = self.RemoveAverageTorque(new_m.coords,veloc)
			#TODO remove rotation.
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			# Remove translation.
			c_veloc = c_veloc - np.average(c_veloc,axis=0)
			new_m.coords = tmp_m.coords + c_veloc
			old_veloc = self.momentum_decay*c_veloc
			err = new_m.rms(tmp_m)
			mol_hist.append(tmp_m)
			tmp_m.WriteXYZfile("./datasets/", "OptLog")
			if (0): # Prints density along the optimization.
				xs,ps = self.tfm.EvalAllAtoms(tmp_m)
				grids = tmp_m.MolDots()
				grids = tmp_m.AddPointstoMolDots(grids, xs, ps)
				GridstoRaw(grids,250,str(step))
			step+=1
			print "Step:", step, " RMS Error: ", err, " Coords: ", new_m.coords
		return

	def OptSmoothP(self,m):
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		new_m = copy.deepcopy(m)
		print "Orig Coords", new_m.coords
		old_veloc=np.zeros(new_m.coords.shape)
		while(err>self.thresh and step < self.max_opt_step):
			coords=np.array(new_m.coords,copy=True)
			tmp_m = Mol(new_m.atoms, coords)
			for i in range(m.NAtoms()):
				newcoord = self.tfm.SmoothPOneAtom(new_m, i)
				new_m.coords[i] = newcoord
			veloc = new_m.coords - tmp_m.coords
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			c_veloc = c_veloc - np.average(c_veloc,axis=0)

			new_m.coords = tmp_m.coords + c_veloc
			old_veloc = self.momentum_decay*c_veloc

			err = new_m.rms(tmp_m)
			mol_hist.append(tmp_m)
			tmp_m.WriteXYZfile("./datasets/", "OptLog")
			if (0): # Prints density along the optimization.
				xs,ps = self.tfm.EvalAllAtoms(tmp_m)
				grids = tmp_m.MolDots()
				grids = tmp_m.AddPointstoMolDots(grids, xs, ps)
				GridstoRaw(grids,250,str(step))
			step+=1
			print "Step:", step, " RMS Error: ", err, " Coords: ", new_m.coords
		return

	def GoOptProb(self,m):
		''' This version tests if the Go-opt converges when atoms are moved to
			the points of mean-probability. CF POfAtomMoves() '''
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		new_m = Mol()
		new_m = m
 		print "Orig Coords", new_m.coords
		old_veloc=np.zeros(new_m.coords.shape)
		while(err>self.thresh and step < self.max_opt_step):
			coords=np.array(new_m.coords,copy=True)
			tmp_m = Mol(new_m.atoms, coords)

			veloc = new_m.GoMeanProbForce() # Tries to update atom positions to average of Go-Probability.
			# Remove average torque
			#veloc = self.RemoveAverageTorque(new_m.coords,veloc)
			#TODO remove rotation.
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			# Remove translation.
			c_veloc = c_veloc - np.average(c_veloc,axis=0)

			new_m.coords = tmp_m.coords + c_veloc
			old_veloc = self.momentum_decay*c_veloc

			err = new_m.rms(tmp_m)
			mol_hist.append(tmp_m)
			tmp_m.WriteXYZfile("./datasets/", "OptLog")
			if (0): # Prints density along the optimization.
				xs,ps = self.tfm.EvalAllAtoms(tmp_m)
				grids = tmp_m.MolDots()
				grids = tmp_m.AddPointstoMolDots(grids, xs, ps)
				GridstoRaw(grids,250,str(step))
			step+=1
			Energy = new_m.EnergyAfterAtomMove(new_m.coords[0],0)
                        print "Step:", step, " RMS Error: ", err, "Energy: ", Energy#" Coords: ", new_m.coords
			#print "Step:", step, " RMS Error: ", err, " Coords: ", new_m.coords
		return

	def GoOpt(self,m):
		''' Tests if the Force of the Go-potential yields the equilibrium structure, which it totally does. '''
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		new_m = Mol()
		new_m = m
 		print "Orig Coords", new_m.coords
		old_veloc=np.zeros(new_m.coords.shape)
		while(err>self.thresh and step < self.max_opt_step):
			coords=np.array(new_m.coords,copy=True)
			tmp_m = Mol(new_m.atoms, coords)

			veloc = new_m.SoftCutGoForce()
			# Remove average torque
			#veloc = self.RemoveAverageTorque(new_m.coords,veloc)
			#TODO remove rotation.
			c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
			# Remove translation.
			c_veloc = c_veloc - np.average(c_veloc,axis=0)

			new_m.coords = tmp_m.coords + c_veloc
			old_veloc = self.momentum_decay*c_veloc

			err = new_m.rms(tmp_m)
			mol_hist.append(tmp_m)
			tmp_m.WriteXYZfile("./datasets/", "OptLog")
			if (0): # Prints density along the optimization.
				xs,ps = self.tfm.EvalAllAtoms(tmp_m)
				grids = tmp_m.MolDots()
				grids = tmp_m.AddPointstoMolDots(grids, xs, ps)
				GridstoRaw(grids,250,str(step))
			step+=1
			Energy = new_m.GoPotential()
                        print "Step:", step, " RMS Error: ", err, "Energy: ", Energy#" Coords: ", new_m.coords
			#print "Step:", step, " RMS Error: ", err, " Coords: ", new_m.coords
		return

	def GoOpt_ScanForce(self,m):
                # Sweeps one at a time
                err=10.0
                lasterr=10.0
                step=0
                mol_hist = []
                new_m = Mol()
                new_m = m
                print "Orig Coords", new_m.coords
                old_veloc=np.zeros(new_m.coords.shape)
                while(err>self.thresh and step < self.max_opt_step):
                        coords=np.array(new_m.coords,copy=True)
                        tmp_m = Mol(new_m.atoms, coords)

                        new_coords = new_m.GoForce_Scan(self.maxstep, self.ngrid)
			veloc = new_coords - tmp_m.coords
                        # Remove average torque
                        #veloc = self.RemoveAverageTorque(new_m.coords,veloc)
                        #TODO remove rotation.
                        c_veloc = (1.0-self.momentum)*veloc+self.momentum*old_veloc
                        # Remove translation.
                        c_veloc = c_veloc - np.average(c_veloc,axis=0)

                        new_m.coords = tmp_m.coords + c_veloc
                        old_veloc = self.momentum_decay*c_veloc

                        err = new_m.rms(tmp_m)
                        mol_hist.append(tmp_m)
                        tmp_m.WriteXYZfile("./datasets/", "OptLog")
                        if (0): # Prints density along the optimization.
                                xs,ps = self.tfm.EvalAllAtoms(tmp_m)
                                grids = tmp_m.MolDots()
                                grids = tmp_m.AddPointstoMolDots(grids, xs, ps)
                                GridstoRaw(grids,250,str(step))
                        step+=1
			Energy = new_m.GoPotential()
                        print "Step:", step, " RMS Error: ", err, "Energy: ", Energy#" Coords: ", new_m.coords
                return

	def Interpolate_OptForce(self,m1,m2):
		# Interpolates between lattices of two stoichiometric molecules from the center outwards
		err=10.0
		lasterr=10.0
		step=0
		mol_hist = []
		prev_m = Mol(m1.atoms, m1.coords)
		#print "Orig Coords", m1.coords
		#Needs to confirm the two molecules are oriented the same here too
		if (m1.Center()-m2.Center()).all() != 0:
			m2.coords += m1.Center() - m2.Center()
		center_dist = np.array(np.linalg.norm(m2.coords - m2.Center(), axis=1))
		veloc=np.zeros(m1.coords.shape)
		#old_veloc=np.zeros(m1.coords.shape)
		m1.BuildDistanceMatrix()
		m1eq = m1.DistMatrix
		m2.BuildDistanceMatrix()
		while(err>self.thresh and step < self.max_opt_step):
			for i in range(m1.NAtoms()):
				seq = (3*(1-2*np.linalg.norm(m2.coords[i]-m2.Center())/np.amax(center_dist)))
				m1.DistMatrix = m2.DistMatrix
				m2force = m1.GoForceLocal(i)
				m1.DistMatrix = m1eq
				#linear interpolation
				#veloc[i] = -0.1*(m1.GoForceLocal(i)*(1-np.linalg.norm(m2.coords[i]-m2.Center())/np.amax(center_dist)) + m2force*(np.linalg.norm(m2.coords[i]-m2.Center())/np.amax(center_dist)))
				#tanh interpolation
				veloc[i] = -0.01*(m1.GoForceLocal(i)*(math.tanh(seq)+1)/2 + m2force*(math.tanh(-seq)+1)/2)
				#print (math.tanh(seq)+1)/2, (math.tanh(-seq)+1)/2, (math.tanh(seq)+1)/2 + (math.tanh(-seq)+1)/2
			c_veloc = veloc - np.average(veloc,axis=0)
			prev_m = Mol(m1.atoms, m1.coords)
			m1.coords = m1.coords + c_veloc
			err = m1.rms(prev_m)
			#err = np.sqrt(np.mean(np.square(c_veloc)))
			mol_hist.append(prev_m)
			prev_m.WriteXYZfile("./results", "OptLog")
			step+=1
			print "Step:", step, " RMS Error: ", err, " Coords: ", m1.coords
		return
