"""
	Tensorflow holders for simple force field ingredients
	That cannot be trained.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TensorMol.TFInstance import *
from TensorMol.TensorMolData import *
from TensorMol.TFMolInstance import *
from TensorMol.ElectrostaticsTF import *
from TensorMol.Neighbors import *
from tensorflow.python.client import timeline
import threading

def XInLat(x_, lat_):
	"""
	quadratic wall forces which go off into infinity at the sides of a lattice cell. Within the cell they increase linearly with a slope of 10Eh / 0.1 Angstrom. beginning at zero at a skin-depth of self.skinD. This can be used to crush a
	simulation.

	Args:
		x_: points to evaluate this force on.
		lat_: The Lattice.
	"""
	latmet = TFMatrixPower(tf.matmul(lat_,tf.transpose(lat_)),-0.5)
	xinlat = tf.matmul(x_,latmet) # The points in the lattice coordinates.
	return xinlat
	xinlat2 = xinlat*xinlat
	xinlatm12 = (xinlat-1.0)*(xinlat-1.0)
	lower_walls = tf.where(xinlat<0.0, xinlat2, tf.zeros_like(x_))
	upper_walls = tf.where(xinlat>1.0, xinlatm12, tf.zeros_like(x_))
	return x_

def WallEn(x_, lat_):
	"""
	quadratic wall forces which go off into infinity at the sides of a lattice cell. Within the cell they increase linearly with a slope of 10Eh / 0.1 Angstrom. beginning at zero at a skin-depth of self.skinD. This can be used to crush a
	simulation.

	Args:
		x_: points to evaluate this force on.
		lat_: The Lattice.
	"""
	latmet = TFMatrixPower(tf.matmul(lat_,tf.transpose(lat_)),-0.5)
	xinlat = tf.matmul(x_,latmet) # The points in the lattice coordinates.
	xinlat2 = xinlat*xinlat
	xinlatm12 = (xinlat-1.0)*(xinlat-1.0)
	lower_walls = tf.where(xinlat<0.0, xinlat2, tf.zeros_like(x_))
	upper_walls = tf.where(xinlat>1.0, xinlatm12, tf.zeros_like(x_))
	return tf.reduce_sum(lower_walls+upper_walls)

class ForceHolder:
	def __init__(self,natom_):
		"""
		Base Force holder for an aperiodic force.

		args:
			natom_: number of atoms the force can be evaluated on.
		"""
		self.natom = natom_
		self.sess = None
		self.x_pl = None
		return
	def Prepare(self):
		with tf.Graph().as_default():
			self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.natom,3]))
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

class BoxHolder(ForceHolder):
	def __init__(self,natom_):
		"""
		Holds quadratic wall forces which go off into infinity at the sides of a lattice cell. Within the cell they increase linearly with a slope of 10Eh / 0.1 Angstrom.
		beginning at zero at a skin-depth of self.skinD. This can be used to crush a
		simulation.
		"""
		ForceHolder.__init__(self,natom_)
		self.lat_pl = None # boundary vectors of the cell.
		self.slope = None
		self.Prepare()
	def Prepare(self):
			self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.natom,3]))
			self.lat_pl=tf.placeholder(tf.float64, shape=tuple([3,3]))
			self.slope=tf.Variable(10.0,dtype=tf.float64)
			self.SkinD=tf.Variable(0.1,dtype=tf.float64)
			self.Energy = self.slope*WallEn(self.x_pl,self.lat_pl)
			self.Force = tf.gradients(WallEn(self.x_pl,self.lat_pl),self.x_pl)
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
	def __call__(self,x_,lat_):
		"""
		Args:
			x_: the coordinates on which to evaluate the force.
			lat_: the lattice boundary vectors.
		Returns:
			the Energy and Force (Eh and Eh/ang.) associated with the quadratic walls.
		"""
#		print("lat",lat_)
#		print("Xinlat",self.sess.run([XInLat(self.x_pl,self.lat_pl)], feed_dict = {self.x_pl:x_, self.lat_pl:lat_}))
		e,f = self.sess.run([self.Energy,self.Force], feed_dict = {self.x_pl:x_, self.lat_pl:lat_})
		print("Min max and lat", np.min(x_), np.max(x_), lat_, e ,f)
		return e, f[0]

class BumpHolder(ForceHolder):
	def __init__(self,natom_,maxbump_,bowlk_=0.0):
		"""
		Holds a bump-function graph to allow for rapid
		metadynamics. Can also hold an attractive bump which draws
		atoms towards 0,0,0
		Args:
			m: a molecule.
		"""
		ForceHolder.__init__(self, natom_)
		self.maxbump = maxbump_
		self.nb_pl = None
		self.h = None
		self.w = None
		self.BowlK = bowlk_
		self.Prepare()
		return
	def Prepare(self):
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(tf.float64, shape=tuple([self.maxbump,self.natom,3]))
			self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.natom,3]))
			self.nb_pl=tf.placeholder(tf.int32)
			self.h = tf.Variable(0.5,dtype = tf.float64)
			self.w = tf.Variable(1.0,dtype = tf.float64)
			self.BowlKv = tf.Variable(self.BowlK,dtype = tf.float64)
			init = tf.global_variables_initializer()
			self.BE = BumpEnergy(self.h, self.w, self.xyzs_pl, self.x_pl, self.nb_pl)
			self.BF = tf.gradients(BumpEnergy(self.h, self.w, self.xyzs_pl, self.x_pl, self.nb_pl), self.x_pl)
			self.BowlE = BowlEnergy(self.BowlKv, self.x_pl)
			self.BowlF = tf.gradients(BowlEnergy(self.BowlKv, self.x_pl), self.x_pl)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return
	def Bump(self, BumpCoords, x_, NBump_):
		"""
		Returns the Bump energy force.
		"""
		if (self.BowlK == 0.0):
			return self.sess.run([self.BE,self.BF], feed_dict = {self.xyzs_pl:BumpCoords, self.x_pl:x_, self.nb_pl:NBump_})
		else:
			e,f,we,wf = self.sess.run([self.BE,self.BF,self.BowlE,self.BowlF], feed_dict = {self.xyzs_pl:BumpCoords, self.x_pl:x_, self.nb_pl:NBump_})
			return (e+we), ([f[0]+wf[0]])
	def Bowl(self, x_):
		"""
		Returns the Bowl force.
		which is a linear attraction to 0.0.0
		"""
		return self.sess.run([self.BowlE,self.BowlF], feed_dict = {self.x_pl:x_})

class MolInstance_DirectForce(MolInstance_fc_sqdiff_BP):
	"""
	An instance which can evaluate and optimize some model force field.
	The force routines are in ElectrostaticsTF.py
	The force routines can take some parameters described here.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True, ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "LJForce"
		self.TData = TData_
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		self.MaxNAtoms = TData_.MaxNAtoms
		self.batch_size_output = 4096
		self.PreparedFor = 0
		self.inp_pl=None
		self.nzp_pl=None
		self.x_pl=None
		self.z_pl=None
		self.nzp2_pl=None
		self.frce_pl=None
		self.sess = None
		self.ForceType = ForceType_
		self.forces = None
		self.energies = None
		self.forcesLinear = None
		self.energiesLinear = None
		self.forceLinear = None
		self.energyLinear = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None
		self.LJe = None
		self.LJr = None
		self.Deq = None
		self.dbg1 = None
		self.dbg2 = None
		self.NL = None
		self.max_checkpoints = 5000
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.

	def loss_op(self, output, labels):
		"""
		The loss operation of this model is complicated
		Because you have to construct the electrostatic energy moleculewise,
		and the mulitpoles.

		Emats and Qmats are constructed to accerate this process...
		"""
		output = tf.Print(output,[output],"Comp'd",1000,1000)
		labels = tf.Print(labels,[labels],"Desired",1000,1000)
		diff  = tf.subtract(output, labels)
		#tf.Print(diff, [diff], message="This is diff: ",first_n=10000000,summarize=100000000)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def LJFrc(self, inp_pl):
		"""
		Compute forces for a batch of molecules
		with the current LJe, and LJr.

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
		"""
		# separate out the Z from the XYZ.
		inp_shp = tf.shape(inp_pl)
		nmol = inp_shp[0]
		maxnatom = inp_shp[1]
		XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int64)
		#self.LJe = tf.Print(self.LJe,[self.LJe],"LJe",1000,1000)
		#self.LJr = tf.Print(self.LJr,[self.LJr],"LJr",1000,1000)
		LJe2 = self.LJe*self.LJe
		LJr2 = self.LJr*self.LJr
		#LJe2 = tf.Print(LJe2,[LJe2],"LJe2",1000,1000)
		#LJr2 = tf.Print(LJr2,[LJr2],"LJr2",1000,1000)
		Ens = LJEnergies(XYZs, Zs, LJe2, LJr2)
		#Ens = tf.Print(Ens,[Ens],"Energies",5000,5000)
		frcs = -1.0*(tf.gradients(Ens, XYZs)[0])
		return Ens, frcs

	def LJFrcLinear(self, z_pl, x_pl, nzp2_pl):
		"""
		Compute forces
		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
			nzp_pl: placeholder for the NMol X 3 tensor of nonzero pairs.
		"""
		LJe2 = 0.116*tf.ones([8,8],dtype=tf.float64)
		LJr2 = tf.ones([8,8],dtype=tf.float64)
		x_plshp = tf.shape(x_pl)
		xpl = tf.reshape(x_pl,[1,x_plshp[0],x_plshp[1]])
		Ens = LJEnergyLinear(xpl, z_pl, LJe2, LJr2, nzp2_pl)
		frcs = -1.0*(tf.gradients(Ens, xpl)[0])
		return Ens, frcs

	def LJFrcsLinear(self, inp_pl, nzp_pl):
		"""
		Compute forces for a batch of molecules
		with the current LJe, and LJr with linear scaling.

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
			nzp_pl: placeholder for the NMol X 3 tensor of nonzero pairs.
		"""
		# separate out the Z from the XYZ.
		inp_shp = tf.shape(inp_pl)
		nmol = inp_shp[0]
		maxnatom = inp_shp[1]
		XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int64)
		LJe2 = 0.116*tf.ones([8,8],dtype=tf.float64)
		LJr2 = tf.ones([8,8],dtype=tf.float64)
		Ens = LJEnergiesLinear(XYZs, Zs, LJe2, LJr2, nzp_pl)
		frcs = -1.0*(tf.gradients(Ens, XYZs)[0])
		return Ens, frcs

	def HarmFrc(self, inp_pl):
		"""
		Compute Harmonic Forces with equilibrium distance matrix
		Deqs, and force constant matrix, Keqs

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
		"""
		# separate out the Z from the XYZ.
		inp_shp = tf.shape(inp_pl)
		nmol = inp_shp[0]
		maxnatom = inp_shp[1]
		XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int64)
		ZZeroTensor = tf.cast(tf.where(tf.equal(Zs,0),tf.ones_like(Zs),tf.zeros_like(Zs)),tf.float64)
		# Construct a atomic number masks.
		Zshp = tf.shape(Zs)
		Zzij1 = tf.tile(ZZeroTensor,[1,1,Zshp[1]]) # mol X atom X atom.
		Zzij2 = tf.transpose(Zzij1,perm=[0,2,1]) # mol X atom X atom.
		Deqs = tf.ones((nmol,maxnatom,maxnatom))
		Keqs = 0.001*tf.ones((nmol,maxnatom,maxnatom))
		K = HarmKernels(XYZs, Deqs, Keqs)
		K = tf.where(tf.equal(Zzij1,1.0),tf.zeros_like(K),K)
		K = tf.where(tf.equal(Zzij2,1.0),tf.zeros_like(K),K)
		Ens = tf.reduce_sum(K,[1,2])
		frcs = -1.0*(tf.gradients(Ens, XYZs)[0])
		#frcs = tf.Print(frcs,[frcs],"Forces",1000,1000)
		return Ens, frcs

	def EvalForce(self,m):
		Ins = self.TData.dig.Emb(m,False,False)
		Ins = Ins.reshape(tuple([1]+list(Ins.shape)))
		feeddict = {self.inp_pl:Ins}
		En,Frc = self.sess.run([self.energies, self.forces],feed_dict=feeddict)
		return En, JOULEPERHARTREE*Frc[0] # Returns energies and forces.
	def CallLinearLJForce(self,z,x,NZ):
		if (z.shape[0] != self.PreparedFor):
			self.MaxNAtoms = z.shape[0]
			self.Prepare()
		feeddict = {self.z_pl:z.astype(np.int64).reshape(z.shape[0],1),self.x_pl:x, self.nzp2_pl:NZ}
		En,Frc = self.sess.run([self.energyLinear, self.forceLinear],feed_dict=feeddict)
		return En, JOULEPERHARTREE*Frc[0] # Returns energies and forces.
	def EvalForceLinear(self,m):
		Ins = self.TData.dig.Emb(m,False,False)
		if (Ins.shape[0] != self.PreparedFor):
			self.MaxNAtoms = Ins.shape[0]
			self.Prepare()
		Ins = Ins.reshape(tuple([1]+list(Ins.shape))) # mol X 4
		if (self.NL==None):
			self.NL = NeighborListSet(Ins[:,:,1:],np.array([m.NAtoms()]))
		self.NL.Update(Ins[:,:,1:],7.0)
		feeddict = {self.inp_pl:Ins, self.nzp_pl:self.NL.pairs}
		En,Frc = self.sess.run([self.energiesLinear, self.forcesLinear],feed_dict=feeddict)
		return En, JOULEPERHARTREE*Frc[0] # Returns energies and forces.
	def print_training(self, step, loss, Ncase, duration, Train=True):
		print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		return
	def Prepare(self):
		self.TrainPrepare()
		return
	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = len(self.TData.set.mols)
		start_time = time.time()
		train_loss =  0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_train/self.batch_size_output)):
			#print ("ministep: ", ministep, " Ncase_train:", Ncase_train, " self.batch_size", self.batch_size)
			batch_data = self.TData.RawBatch()
			if (not np.all(np.isfinite(batch_data[0]))):
				print("Bad Batch...0 ")
			if (not np.all(np.isfinite(batch_data[1]))):
				print("Bad Batch...1 ")
			feeddict={i:d for i,d in zip([self.inp_pl,self.frce_pl],[batch_data[0],batch_data[1]])}
			dump_2, total_loss_value, loss_value = self.sess.run([self.train_op, self.total_loss, self.loss], feed_dict=feeddict)
			train_loss = train_loss + loss_value
			duration = time.time() - start_time
			num_of_mols += self.batch_size_output
			#print ("atom_outputs:", atom_outputs, " mol outputs:", mol_output)
			#print ("atom_outputs shape:", atom_outputs[0].shape, " mol outputs", mol_output.shape)
		#print("train diff:", (mol_output[0]-batch_data[2])[:actual_mols], np.sum(np.square((mol_output[0]-batch_data[2])[:actual_mols])))
		#print ("train_loss:", train_loss, " Ncase_train:", Ncase_train, train_loss/num_of_mols)
		#print ("diff:", mol_output - batch_data[2], " shape:", mol_output.shape)
		self.print_training(step, train_loss, num_of_mols, duration)
		return
	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		self.PreparedFor = self.MaxNAtoms
		with tf.Graph().as_default():
			self.inp_pl=tf.placeholder(tf.float64, shape=tuple([None,self.MaxNAtoms,4]))
			self.nzp_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.z_pl=tf.placeholder(tf.int64, shape=tuple([self.MaxNAtoms,1]))
			self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.MaxNAtoms,3]))
			self.nzp2_pl=tf.placeholder(tf.int64, shape=tuple([None,2]))
			self.frce_pl = tf.placeholder(tf.float64, shape=tuple([None,self.MaxNAtoms,3])) # Forces.
			if (self.ForceType=="LJ"):
				self.LJe = tf.Variable(0.40*tf.ones([8,8],dtype=tf.float64),trainable=True,dtype=tf.float64)
				self.LJr = tf.Variable(1.1*tf.ones([8,8],dtype=tf.float64),trainable=True,dtype=tf.float64)
				# These are squared later to keep them positive.
				self.energies, self.forces = self.LJFrc(self.inp_pl)
				self.energyLinear, self.forceLinear = self.LJFrcLinear(self.z_pl,self.x_pl,self.nzp2_pl)
				self.energiesLinear, self.forcesLinear = self.LJFrcsLinear(self.inp_pl,self.nzp_pl)
				self.total_loss, self.loss = self.loss_op(self.forces, self.frce_pl)
				self.train_op = self.training(self.total_loss, PARAMS["learning_rate"], PARAMS["momentum"])
				self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			elif (self.ForceType=="Harm"):
				self.energies, self.forces = self.HarmFrc(self.inp_pl)
			else:
				raise Exception("Unknown Kernel")
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def train(self, mxsteps=10000):
		self.TrainPrepare()
		LOGGER.info("MolInstance_LJForce.train()")
		test_freq = PARAMS["test_freq"]
		mini_test_loss = float('inf') # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
		self.SaveAndClose()
		return
