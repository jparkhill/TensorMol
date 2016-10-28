#
# Optimization algorithms
#

from Sets import *
from TFManage import *
import random

class Optimizer:
	def __init__(self,tfm_):
		self.thresh = 0.01
		self.maxstep = 0.1
		self.momentum = 0.9
		self.momentum_decay = 0.8
		self.max_opt_step = 100
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

	def Opt(self,m):
		# Sweeps one at a time
		if (self.OType=="SmoothP"):
			return self.OptSmoothP(m)
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
#				print "xyz:", xyz
#				print "probs:", probs
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
#			print "Step:", step, " RMS Error: ", err, " Coords: ", new_m.coords
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
#			print "Step:", step, " RMS Error: ", err, " Coords: ", new_m.coords
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
