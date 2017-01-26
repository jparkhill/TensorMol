"""
Changes that need to be made: 
 - LBFGS needs to be modularized (Solvers.py)?
 - The filename needs to be changed to Opt_MBE
 - This needs to be a child of the ordinary optimizer class.
"""

from NN_MBE import *

class MBE_Optimizer:
	def __init__(self,nn_mbe_):
		self.energy_thresh = 1e-9
		self.force_thresh = 0.001
		self.step_size = 0.1
		self.momentum = 0.0
		self.momentum_decay = 0.7
		self.max_opt_step = 500
		self.nn_mbe = nn_mbe_
		self.m_max = 7
		return

	def MBE_Opt(self, m):
		coords_hist = []
		step = 0
		energy_err = 100
		force_err = 100
		energy_his = []
		self.nn_mbe.NN_Energy(m)
		force = np.array(m.properties["mbe_deri"],copy=True)
		while( ( force_err >self.force_thresh or energy_err > self.energy_thresh) and  step < self.max_opt_step ):
			coords = np.array(m.coords,copy=True)
			force = np.array(m.properties["mbe_deri"],copy=True)
			if step==0:
				old_force =force
			force = (1-self.momentum)*force + self.momentum*old_force
			old_force =force
			energy = m.nn_energy
			energy_his.append(energy)
			coords_hist.append(coords)
			step += 1
			m.coords = m.coords - self.step_size*force
			m.Reset_Frags()
			self.nn_mbe.NN_Energy(m)
			energy_err = abs(m.nn_energy - energy)
			force_err = (np.absolute(force)).max()
			print "\n\n"
			print "step:", step
			print "old_energy:", energy, "new_energy:", m.nn_energy, "energy_err:", energy_err
			if (step%10 == 0):
				m.WriteXYZfile("./datasets/", "OptLog")
		np.savetxt("gd_opt_no_momentum.dat", np.asarray(energy_his))
                return


	def MBE_LBFGS_Opt(self, m):
                step = 0
                energy_err = 100
                force_err = 100
                self.nn_mbe.NN_Energy(m)
                force = np.array(m.properties["mbe_deri"],copy=True)
		force_his = []
		coord_his = []
		energy_his = []
                while( ( force_err >self.force_thresh or energy_err > self.energy_thresh) and  step < self.max_opt_step ):


                        coords = np.array(m.coords,copy=True)
                        force = np.array(m.properties["mbe_deri"],copy=True)

			if step < self.m_max:
				force_his.append(force.reshape(force.shape[0]*force.shape[1]))
				coord_his.append(coords.reshape(coords.shape[0]*coords.shape[1]))
			else:
				force_his.pop(0)
				force_his.append(force.reshape(force.shape[0]*force.shape[1]))
				coord_his.pop(0)
				coord_his.append(coords.reshape(coords.shape[0]*coords.shape[1]))

			print "force:", force
			q = (force.reshape(force.shape[0]*force.shape[1])).copy()
                        for i in range (len(force_his)-1, 0, -1):
                                s=coord_his[i] - coord_his[i-1]
				y=force_his[i] - force_his[i-1]
				rho = 1/y.dot(s)
				a=rho*s.dot(q)
				q = q - a*y

			if step == 0:
				H= 1.0
			else:
				H = ((coord_his[-1] - coord_his[-2]).dot(force_his[-1] - force_his[-2]))/((force_his[-1] - force_his[-2]).dot(force_his[-1] - force_his[-2]))

			z = H*q
			for i in range (1, len(force_his)):
				s=coord_his[i] - coord_his[i-1]
                                y=force_his[i] - force_his[i-1]
                                rho = 1/y.dot(s)
                                a=rho*s.dot(q)
				beta = rho*(force_his[i] - force_his[i-1]).dot(z)
				z = z + s*(a -beta)
			z = z.reshape((m.NAtoms(), -1))
			print "z: ",z

                        energy = m.nn_energy
			energy_his.append(energy)
                        m.coords = m.coords - self.step_size*z
                        m.Reset_Frags()
                        self.nn_mbe.NN_Energy(m)
                        energy_err = abs(m.nn_energy - energy)
                        force_err = (np.absolute(force)).max()
			step += 1
                        print "\n\n"
                        print "step:", step
                        print "old_energy:", energy, "new_energy:", m.nn_energy, "energy_err:", energy_err
                        if (step%10 == 0):
                                m.WriteXYZfile("./datasets/", "OptLog")
		np.savetxt("lbfgs_opt.dat", np.asarray(energy_his))
		return 
	
