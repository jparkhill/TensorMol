"""
Generates artificial data for H_3, learns a potential for it, tests it in optimizations and whatnot.
"""
from TensorMol import *
class PorterKarplus(ForceHolder):
	def __init__(self,natom_=3):
		"""
		simple porter karplus for three atoms for a training example.
		"""
		ForceHolder.__init__(self,natom_)
		self.lat_pl = None # boundary vectors of the cell.
		#self.a = sqrt(self.k/(2.0*self.de))
		self.Prepare()
	def PorterKarplus(self,x_pl):
		x1 = x_pl[0] - x_pl[1]
		x2 = x_pl[2] - x_pl[1]
		x12 = x_pl[0] - x_pl[2]
		r1 = tf.norm(x1)
		r2 = tf.norm(x2)
		r12 = tf.norm(x12)
		v1 = 0.7*tf.pow(1.-tf.exp(-(r1-0.7)),2.0)
		v2 = 0.7*tf.pow(1.-tf.exp(-(r2-0.7)),2.0)
		v3 = 0.7*tf.pow(1.-tf.exp(-((r12)-0.7)),2.0)
		return v1+v2+v3
	def Prepare(self):
			self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.natom,3]))
			self.Energy = self.PorterKarplus(self.x_pl)
			self.Force =tf.gradients(-1.0*self.PorterKarplus(self.x_pl),self.x_pl)
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
	def __call__(self,x_):
		"""
		Args:
			x_: the coordinates on which to evaluate the force.
			lat_: the lattice boundary vectors.
		Returns:
			the Energy and Force (Eh and Eh/ang.) associated with the quadratic walls.
		"""
#		print("lat",lat_)
#		print("Xinlat",self.sess.run([XInLat(self.x_pl,self.lat_pl)], feed_dict = {self.x_pl:x_, self.lat_pl:lat_}))
		e,f = self.sess.run([self.Energy,self.Force], feed_dict = {self.x_pl:x_})
		#print("Min max and lat", np.min(x_), np.max(x_), lat_, e ,f)
		return e, f[0]

def GenerateData():
	"""
	Generate random configurations in a reasonable range.
	and calculate their energies and forces.
	"""
	nsamp = 10000
	crds = np.random.uniform(4.0,size = (nsamp,3,3))
	st = MSet()
	PK = PorterKarplus()
	for s in range(nsamp):
		st.mols.append(Mol(np.array([1.,1.,1.]),crds[s]))
		en,f = PK(crds[s])
		st.mols[-1].properties["energy"] = en
		st.mols[-1].properties["force"] = f
		st.mols[-1].properties["gradients"] = -1.0*f
		st.mols[-1].properties["dipole"] = np.array([0.,0.,0.])
		st.mols[-1].CalculateAtomization()
	return st

def TestTraining_John():
	PARAMS["train_dipole"] = False
	tset = GenerateData()
	net = BehlerParinelloDirectGauSH(tset)
	net.train()
	return

def TestTraining():
	a = GenerateData()
	TreatedAtoms = a.AtomTypes()
	PARAMS["NetNameSuffix"] = "training_sample"
	PARAMS["learning_rate"] = 0.00001
	PARAMS["momentum"] = 0.95
	PARAMS["max_steps"] = 15 # Train for 5 epochs in total
	PARAMS["batch_size"] =  100
	PARAMS["test_freq"] = 5 # Test for every epoch
	PARAMS["tf_prec"] = "tf.float64" # double precsion
	PARAMS["EnergyScalar"] = 1.0
	PARAMS["GradScalar"] = 1.0/20.0
	PARAMS["NeuronType"] = "sigmoid_with_param" # choose activation function
	PARAMS["sigmoid_alpha"] = 100.0  # activation params
	PARAMS["KeepProb"] = [1.0, 1.0, 1.0, 1.0] # each layer's keep probability for dropout
	d = MolDigester(TreatedAtoms, name_="ANI1_Sym_Direct", OType_="AtomizationEnergy")
	tset = TensorMolData_BP_Direct_EandG_Release(a, d, order_=1, num_indis_=1, type_="mol",  WithGrad_ = True)
	manager=TFMolManage("",tset,False,"fc_sqdiff_BP_Direct_EandG_SymFunction")
	PARAMS['Profiling']=0
	manager.Train(1)


def TestOpt():
	return

def TestMD():
	return

TestTraining_John()
