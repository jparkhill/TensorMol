"""
	These instances work directly on raw coordinate, atomic number data.
	They either generate their own descriptor or physical model.
	They are also simplified relative to the usual MolInstance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TensorMol.TFInstance import *
from TensorMol.TensorMolData import *
from TensorMol.TFMolInstance import *
from TensorMol.ElectrostaticsTF import *
from TensorMol.RawEmbeddings import *
from tensorflow.python.client import timeline

class BumpHolder:
	def __init__(self,natom_,maxbump_):
		"""
		Holds a bump-function graph to allow for rapid
		metadynamics.

		Args:
			m: a molecule.
		"""
		self.natom = natom_
		self.maxbump = maxbump_
		self.sess = None
		self.Prepare()
		self.xyzs_pl = None
		self.x_pl = None
		self.nb_pl = None
		self.h = None
		self.w = None
		self.Prepare()
		return
	def Prepare(self):
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(tf.float64, shape=tuple([self.maxbump,self.natom,3]))
			self.x_pl=tf.placeholder(tf.float64, shape=tuple([self.natom,3]))
			self.nb_pl=tf.placeholder(tf.int32)
			self.h = tf.Variable(0.6,dtype = tf.float64)
			self.w = tf.Variable(1.0,dtype = tf.float64)
			init = tf.global_variables_initializer()
			self.BE = BumpEnergy(self.h, self.w, self.xyzs_pl, self.x_pl, self.nb_pl)
			self.BF = tf.gradients(BumpEnergy(self.h, self.w, self.xyzs_pl, self.x_pl, self.nb_pl), self.x_pl)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return
	def Bump(self, BumpCoords, x_, NBump_):
		"""Returns the Bump force.
		"""
		return self.sess.run([self.BE,self.BF], feed_dict = {self.xyzs_pl:BumpCoords, self.x_pl:x_, self.nb_pl:NBump_})

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
		self.inp_pl=None
		self.frce_pl=None
		self.sess = None
		self.ForceType = ForceType_
		self.forces = None
		self.energies = None
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
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.inp_pl=tf.placeholder(tf.float32, shape=tuple([None,self.MaxNAtoms,4]))
			self.frce_pl = tf.placeholder(tf.float32, shape=tuple([None,self.MaxNAtoms,3])) # Forces.
			if (self.ForceType=="LJ"):
				self.LJe = tf.Variable(0.316*tf.ones([8,8]),trainable=True)
				self.LJr = tf.Variable(tf.ones([8,8]),trainable=True)
				# These are squared later to keep them positive.
				self.energies, self.forces = self.LJFrc(self.inp_pl)
				self.total_loss, self.loss = self.loss_op(self.forces, self.frce_pl)
				self.train_op = self.training(self.total_loss, PARAMS["learning_rate"], PARAMS["momentum"])
				self.saver = tf.train.Saver()
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
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
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
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
		ZZeroTensor = tf.cast(tf.where(tf.equal(Zs,0),tf.ones_like(Zs),tf.zeros_like(Zs)),tf.float32)
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

	def print_training(self, step, loss, Ncase, duration, Train=True):
		print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		return

	def Prepare(self):
		self.train_prepare()
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

	def train(self, mxsteps=10000):
		self.train_prepare()
		LOGGER.info("MolInstance_LJForce.train()")
		test_freq = PARAMS["test_freq"]
		mini_test_loss = float('inf') # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
		self.SaveAndClose()
		return

class MolInstance_DirectForce_tmp(MolInstance_fc_sqdiff_BP):
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
		self.NetType = "LJE"
		self.TData = TData_
		self.MaxNAtoms = TData_.MaxNAtoms
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.batch_size_output = 10000
		self.inp_pl=None
		self.frce_pl=None
		self.sess = None
		self.ForceType = ForceType_
		self.forces = None
		self.energies = None
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
		self.batch_data = self.TData.RawBatch(nmol=30000)
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
		        continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.LJe = tf.placeholder(tf.float32, shape=(),name="Ee_pl")
			self.LJr = tf.placeholder(tf.float32, shape=(),name="Re_pl")
			# self.Ee_pl = tf.constant(0.316, dtype=tf.float32)
			# self.Re_pl = tf.constant(1.0, dtype=tf.float32)
			self.inp_shp = tf.shape(self.batch_data[0])
			self.nmol = self.inp_shp[0]
			self.maxnatom = self.inp_shp[1]
			self.XYZs = tf.to_float(tf.slice(self.batch_data[0],[0,0,1],[-1,-1,-1]))
			self.REns = tf.convert_to_tensor(self.batch_data[1][:,0,0],dtype=tf.float32)
			self.Zs = tf.cast(tf.reshape(tf.slice(self.batch_data[0],[0,0,0],[-1,-1,1]),[self.nmol,self.maxnatom,1]),tf.int32)
			self.Ens = LJEnergies(self.XYZs, self.Zs, self.LJe, self.LJr)
			self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.Ens, self.REns)))
			# params = (XYZs, Zs, REns)
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.sess.run(init)

	def LJER(self,inp_pl,E_pl,R_pl):
		"""
		with the current LJe, and LJr.

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
		"""
		# separate out the Z from the XYZ.
		inp_shp = tf.shape(inp_pl)
		nmol = inp_shp[0]
		maxnatom = inp_shp[1]
		XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
		#self.LJe = tf.Print(self.LJe,[self.LJe],"LJe",1000,1000)
		#self.LJr = tf.Print(self.LJr,[self.LJr],"LJr",1000,1000)
		LJe2 = E_pl*E_pl
		LJr2 = R_pl*R_pl
		#LJe2 = tf.Print(LJe2,[LJe2],"LJe2",1000,1000)
		#LJr2 = tf.Print(LJr2,[LJr2],"LJr2",1000,1000)
		Ens = LJEnergies(XYZs, Zs, LJe2, LJr2)
		#Ens = tf.Print(Ens,[Ens],"Energies",5000,5000)
		return Ens

	def LJFrc(self, params):
		"""
		Compute forces for a batch of molecules
		with the current LJe, and LJr.

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
		"""
		print(params)
		feeddict = {self.LJe:params[0], self.LJr:params[1]}
		result = self.sess.run(self.mae,feed_dict=feeddict)
		return result

class MolInstance_DirectBP(MolInstance_fc_sqdiff_BP):
	"""
	An Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True, ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP"
		self.TData = TData_
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		self.MaxNAtoms = TData_.MaxNAtoms
		self.batch_size_output = 4096
		self.inp_pl=None
		self.frce_pl=None
		self.sess = None
		self.ForceType = ForceType_
		self.forces = None
		self.energies = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		#with tf.Graph().as_default():

		return

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
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
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
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
		ZZeroTensor = tf.cast(tf.where(tf.equal(Zs,0),tf.ones_like(Zs),tf.zeros_like(Zs)),self.tf_prec)
		# Construct a atomic number masks.
		Zshp = tf.shape(Zs)
		Zzij1 = tf.tile(ZZeroTensor,[1,1,Zshp[1]]) # mol X atom X atom.
		Zzij2 = tf.transpose(Zzij1,perm=[0,2,1]) # mol X atom X atom.
		Deqs = tf.ones((nmol,maxnatom,maxnatom),dtype=self.tf_prec)
		Keqs = 0.001*tf.ones((nmol,maxnatom,maxnatom),dtype=self.tf_prec)
		K = HarmKernels(XYZs, Deqs, Keqs)
		K = tf.where(tf.equal(Zzij1,1.0),tf.zeros_like(K,dtype=self.tf_prec),K)
		K = tf.where(tf.equal(Zzij2,1.0),tf.zeros_like(K,dtype=self.tf_prec),K)
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

	def print_training(self, step, loss, Ncase, duration, Train=True):
		print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		return

	def Prepare(self):
		self.train_prepare()
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

	def train(self, mxsteps=10000):
		self.train_prepare()
		LOGGER.info("MolInstance_LJForce.train()")
		test_freq = PARAMS["test_freq"]
		mini_test_loss = float('inf') # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
		self.SaveAndClose()
		return

class MolInstance_DirectBP_NoGrad(MolInstance_fc_sqdiff_BP):
	"""
	An Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	Do not use gradient in training
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_noGrad"
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		#if (Name_ != None):
		#	return
		self.SFPa = None
		self.SFPr = None
		self.Ra_cut = None
                self.Rr_cut = None
		self.MaxNAtoms = self.TData.MaxNAtoms
                self.eles = self.TData.eles
                self.n_eles = len(self.eles)
                self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
                self.eles_pairs = []
                for i in range (len(self.eles)):
                        for j in range(i, len(self.eles)):
                                self.eles_pairs.append([self.eles[i], self.eles[j]])
                self.eles_pairs_np = np.asarray(self.eles_pairs)
		self.SetANI1Param()
		self.batch_size = PARAMS["batch_size"] 
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		if (self.Trainable):
                        self.TData.LoadDataToScratch(self.tformer)
		self.xyzs_pl = None
		self.Zs_pl = None
		self.label_pl = None
		self.sess = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None

	def SetANI1Param(self, prec=np.float64):
		self.Ra_cut = PARAMS["AN1_a_Rc"]
		self.Rr_cut = PARAMS["AN1_r_Rc"]
		zetas = np.array([[PARAMS["AN1_zeta"]]], dtype = prec)
		etas = np.array([[PARAMS["AN1_eta"]]], dtype = prec)
		AN1_num_a_As = PARAMS["AN1_num_a_As"]
		AN1_num_a_Rs = PARAMS["AN1_num_a_Rs"]
		thetas = np.array([ 2.0*Pi*i/AN1_num_a_As for i in range (0, AN1_num_a_As)], dtype = prec)
		rs =  np.array([ self.Ra_cut*i/AN1_num_a_Rs for i in range (0, AN1_num_a_Rs)], dtype = prec)
		# Create a parameter tensor. 4 x nzeta X neta X ntheta X nr
		p1 = np.tile(np.reshape(zetas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(etas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
		p3 = np.tile(np.reshape(thetas,[1,1,AN1_num_a_As,1,1]),[1,1,1,AN1_num_a_Rs,1])
		p4 = np.tile(np.reshape(rs,[1,1,1,AN1_num_a_Rs,1]),[1,1,AN1_num_a_As,1,1])
		SFPa = np.concatenate([p1,p2,p3,p4],axis=4)
		self.SFPa = np.transpose(SFPa, [4,0,1,2,3])
		etas_R = np.array([[PARAMS["AN1_eta"]]], dtype = prec)
		AN1_num_r_Rs = PARAMS["AN1_num_r_Rs"]
		rs_R =  np.array([ self.Rr_cut*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)], dtype = prec)
		# Create a parameter tensor. 2 x  neta X nr
		p1_R = np.tile(np.reshape(etas_R,[1,1,1]),[1,AN1_num_r_Rs,1])
		p2_R = np.tile(np.reshape(rs_R,[1,AN1_num_r_Rs,1]),[1,1,1])
		SFPr = np.concatenate([p1_R,p2_R],axis=2)
		self.SFPr = np.transpose(SFPr, [2,0,1])
		self.inshape = int(len(self.eles)*AN1_num_r_Rs + len(self.eles_pairs)*AN1_num_a_Rs*AN1_num_a_As)

		p1 = np.tile(np.reshape(thetas,[AN1_num_a_As,1,1]),[1,AN1_num_a_Rs,1])
                p2 = np.tile(np.reshape(rs,[1,AN1_num_a_Rs,1]),[AN1_num_a_As,1,1])
                SFPa2 = np.concatenate([p1,p2],axis=2)
                self.SFPa2 = np.transpose(SFPa2, [2,0,1])
		p1_new = np.reshape(rs_R,[AN1_num_r_Rs,1])
                self.SFPr2 = np.transpose(p1_new, [1,0])
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]

		print ("self.inshape:", self.inshape)

	def Clean(self):
		Instance.Clean(self)
		self.xyzs_pl=None
		self.check = None
		self.Zs_pl=None
		self.label_pl=None
		self.atom_outputs = None
		self.Scatter_Sym = None
		self.Sym_Index = None
		self.options = None
		self.run_metadata = None
		return


	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int32)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int32)
			SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update2(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2, self.Rr_cut, Elep, self.SFPa2,self.zeta, self.eta, self.Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Rr_cut_tf = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
                        #self.Ra_cut_tf = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
                        #self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut_tf, Elep, self.SFPa, self.Ra_cut_tf)
			#tf.verify_tensor_all_finite(self.Scatter_Sym[0], "Nan in output!!! 0 ")
			#tf.verify_tensor_all_finite(self.Scatter_Sym[1], "Nan in output!!! 1")
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.output, self.xyzs_pl)
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
			#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			#self.run_metadata = tf.RunMetadata()
		return

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def inference(self, inp, indexs):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		# convert the index matrix from bool to float
		branches=[]
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3

		output = tf.zeros([self.batch_size], dtype=self.tf_prec)
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		print("Norms:", nrm1,nrm2,nrm3)
		LOGGER.info("Layer initial Norms: %f %f %f", nrm1,nrm2,nrm3)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp[e]
			shp_in = tf.shape(inputs)
			index = tf.cast(indexs[e], tf.int64)
			if (PARAMS["check_level"]>2):
				tf.Print(tf.to_float(shp_in), [tf.to_float(shp_in)], message="Element "+str(e)+"input shape ",first_n=10000000,summarize=100000000)
				index_shape = tf.shape(index)
				tf.Print(tf.to_float(index_shape), [tf.to_float(index_shape)], message="Element "+str(e)+"index shape ",first_n=10000000,summarize=100000000)
			if (PARAMS["check_level"]>3):
				tf.Print(tf.to_float(inputs), [tf.to_float(inputs)], message="This is input shape ",first_n=10000000,summarize=100000000)
			with tf.name_scope(str(self.eles[e])+'_hidden_1'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev=nrm1, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden1_units], dtype=self.tf_prec), name='biases')
				branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_hidden_2'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev=nrm2, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden2_units], dtype=self.tf_prec), name='biases')
				branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_hidden_3'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev=nrm3, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden3_units], dtype=self.tf_prec), name='biases')
				branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
				#tf.Print(branches[-1], [branches[-1]], message="This is layer 2: ",first_n=10000000,summarize=100000000)
			with tf.name_scope(str(self.eles[e])+'_regression_linear'):
				shp = tf.shape(inputs)
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden3_units, 1], var_stddev=nrm4, var_wd=None)
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
				branches[-1].append(tf.matmul(branches[-1][-1], weights) + biases)
				shp_out = tf.shape(branches[-1][-1])
				cut = tf.slice(branches[-1][-1],[0,0],[shp_out[0],1])
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				rshpflat = tf.reshape(cut,[shp_out[0]])
				range_index = tf.range(tf.cast(shp_out[0], tf.int64), dtype=tf.int64)
				sparse_index =tf.stack([index, range_index], axis=1)
				sp_atomoutputs = tf.SparseTensor(sparse_index, rshpflat, dense_shape=[tf.cast(self.batch_size, tf.int64), tf.cast(shp_out[0], tf.int64)])
				mol_tmp = tf.sparse_reduce_sum(sp_atomoutputs, axis=1)
				output = tf.add(output, mol_tmp)
			tf.verify_tensor_all_finite(output,"Nan in output!!!")
			#tf.Print(output, [output], message="This is output: ",first_n=10000000,summarize=100000000)
		return output, atom_outputs

	def fill_feed_dict(self, batch_data):
		"""
		Fill the tensorflow feed dictionary.

		Args:
			batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
			and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

		Returns:
			Filled feed dictionary.
		"""
		# Don't eat shit.
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.label_pl], [batch_data[0]]+[batch_data[1]]+[batch_data[2]])}
		return feed_dict

	def print_training(self, step, loss, Ncase, duration, Train=True):
		print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		return

	def Prepare(self):
		self.train_prepare()
		return

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
		        step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep:", ministep)
			batch_data = self.TData.GetTrainBatch(self.batch_size)
			actual_mols  = self.batch_size
			t = time.time()
			print ("batch_size:", self.batch_size)
			dump_, dump_2, total_loss_value, loss_value, mol_output, atom_outputs, gradient = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output,  self.atom_outputs, self.gradient], feed_dict=self.fill_feed_dict(batch_data))
			#dump_, dump_2, total_loss_value, loss_value, mol_output, atom_outputs, gradient = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output,  self.atom_outputs, self.gradient], feed_dict=self.fill_feed_dict(batch_data), options=self.options, run_metadata=self.run_metadata)
			print ("time:", time.time() - t)
			#print ("gradient:", gradient[0][:4])
			
			#print ("gradient:", np.sum(gradient[0]))
			#print ("gradient:", np.sum(np.isinf(gradient[0])))
			#print ("gradient:", np.where(np.isinf(gradient[0]) == True))
			train_loss = train_loss + loss_value
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
                        #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        #with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
                        #       f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
		self.print_training(step, train_loss, num_of_mols, duration)
		return

	def test(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		test_loss =  0.0
		start_time = time.time()
		Ncase_test = self.TData.NTest
		num_of_mols = 0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data=self.TData.GetTestBatch(self.batch_size)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = self.batch_size
			preds, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols
		duration = time.time() - start_time
		print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		return test_loss, feed_dict

	def print_training(self, step, loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f", step, duration, (float(loss)/(Ncase)))
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f", step, duration, (float(loss)/(Ncase)))
		return

	def evaluate(self, batch_data, IfGrad=True):   #this need to be modified
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
		if not self.sess:
			print ("loading the session..")
			self.Eval_Prepare()
		feed_dict=self.fill_feed_dict(batch_data)
		#mol_output, total_loss_value, loss_value, atom_outputs, gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
		#for i in range (0, batch_data[0][-1][-1].shape[0]):
		#        print("i:", i)
		#        import copy
		#        new_batch_data=copy.deepcopy(batch_data)
		#        #new_batch_data = list(batch_data)
		#        new_batch_data[0][-1][-1][i] += 0.01
		#        feed_dict=self.fill_feed_dict(new_batch_data)
		#       new_mol_output, total_loss_value, loss_value, new_atom_outputs, new_gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
		#        print ("new_charge_gradient: ", gradient[-1][-1][i],  new_gradient[-1][-1][i], " numerical: ", (new_atom_outputs[-1][-1][-1]- atom_outputs[-1][-1][-1])/0.01)
		if (IfGrad):
			mol_output, total_loss_value, loss_value, atom_outputs, gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
			#print ("atom_outputs:", atom_outputs)
			return mol_output, atom_outputs, gradient
		else:
			mol_output, total_loss_value, loss_value, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs],  feed_dict=feed_dict)
			return mol_output, atom_outputs

	def Eval_Prepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int32)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int32)
			SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			elf.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Rr_cut_tf = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			#self.Ra_cut_tf = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut_tf, Elep, self.SFPa, self.Ra_cut_tf)
			#tf.verify_tensor_all_finite(self.Scatter_Sym[0], "Nan in output!!! 0 ")
			#tf.verify_tensor_all_finite(self.Scatter_Sym[1], "Nan in output!!! 1")
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.output, self.xyzs_pl)
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			#self.run_metadata = tf.RunMetadata()
		return

	def continue_training(self, mxsteps):
		self.Eval_Prepare()
		test_loss , feed_dict = self.test(-1)
		test_freq = 1
		mini_test_loss = test_loss
		for step in  range (0, mxsteps+1):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss, feed_dict = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					self.save_chk(step, feed_dict)
		self.SaveAndClose()
		return
