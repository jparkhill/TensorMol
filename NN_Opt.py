from NN_MBE import *

class NN_Optimizer:
        def __init__(self,nn_mbe_):
                self.energy_thresh = 1e-7
		self.force_thresh = 0.001
                self.step_size = 0.1
                self.momentum = 0.9
                self.momentum_decay = 0.7
                self.max_opt_step = 1000
                self.nn_mbe = nn_mbe_
                return

	def NN_Opt(self, m):
		coords_hist = []
		step = 0
		energy_err = 100
		force_err = 100
		self.nn_mbe.NN_Energy(m)
		while(energy_err>self.energy_thresh and step < self.max_opt_step ):
			coords = np.array(m.coords,copy=True)
			force = np.array(m.mbe_deri,copy=True)
			if step==0:
				old_force =force
			force = (1-self.momentum)*force + self.momentum*old_force
			old_force =force
			energy = m.nn_energy
			coords_hist.append(coords)
			step += 1
			m.coords = m.coords - self.step_size*force
			m.Reset_Frags()
			self.nn_mbe.NN_Energy(m)
			energy_err = abs(m.nn_energy - energy)
			print "old_energy:", energy, "new_energy:", m.nn_energy, "energy_err:", energy_err
                        print "step:", step, "coords:", m.coords	
			m.WriteXYZfile("./datasets/", "OptLog")	
