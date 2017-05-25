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

class MolInstance_LJForce(MolInstance_fc_sqdiff_BP):
	"""
	Optimizes a LJ force where pairs of atoms have specific
	Epsilon and Re parameters.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "LJForce"
		MolInstance_fc_sqdiff_BP.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "LJForce"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		self.MaxNAtoms = TData_.MaxNAtoms
		self.batch_size_output = 4096
		self.LJe = None
		self.LJr = None
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.inp_pl=tf.placeholder(tf.float32, shape=tuple([None,self.inshape]))
			self.mats_pl=tf.placeholder(tf.float32, shape=tuple([self.batch_size_output,self.MaxNAtoms]))
			self.frce_pl = tf.placeholder(tf.float32, shape=tuple([None,3])) # Forces.
			self.LJe = self._variable_with_weight_decay(var_name='LJe', var_shape=[MAX_ATOMIC_NUMBER,MAX_ATOMIC_NUMBER], var_stddev=nrm1, var_wd=0.001)
			self.LJr = self._variable_with_weight_decay(var_name='LJr', var_shape=[MAX_ATOMIC_NUMBER,MAX_ATOMIC_NUMBER], var_stddev=nrm1, var_wd=0.001)
			# These are squared later to keep them positive.
			self.output = self.LJFrc(self.inp_pl, self.mats_pl)
			self.total_loss, self.loss = self.loss_op(self.output, self.frce_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
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
		diff  = tf.subtract(output, labels)
		#tf.Print(diff, [diff], message="This is diff: ",first_n=10000000,summarize=100000000)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def LJFrc(self, inp_pl, keys_pl):
		"""
		Compute forces for a batch of molecules
		with the current LJe, and LJr.

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
			keys_pl: placeholder for the NMol X MaxNatom molecule keys.
		"""
		# separate out the Z from the XYZ.
		inp_shp = tf.shape(inp_pl)
		keys_shp = tf.shape(keys_pl)
		natom = inp_shp[0]
		maxnatom = inp_shp[1]
		nmol = keys_shp[0]
		XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
		Zs = tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[natom,maxnatom,1])
		frcs = LJForce(XYZ, Zs, keys_pl, self.LJe, self.LJr)
		return frcs

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
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep: ", ministep, " Ncase_train:", Ncase_train, " self.batch_size", self.batch_size)
			batch_data = self.TData.RawBatch()
			feeddict={i:d for i,d in zip([self.inp_pl,self.mats_pl,self.label_pl],[batch_data[0],batch_data[1],batch_data[2]])}
			dump_2, total_loss_value, loss_value = self.sess.run([self.train_op, self.total_loss, self.loss, self.output], feed_dict=self.fill_feed_dict(batch_data))
			train_loss = train_loss + loss_value
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#print ("atom_outputs:", atom_outputs, " mol outputs:", mol_output)
			#print ("atom_outputs shape:", atom_outputs[0].shape, " mol outputs", mol_output.shape)
		#print("train diff:", (mol_output[0]-batch_data[2])[:actual_mols], np.sum(np.square((mol_output[0]-batch_data[2])[:actual_mols])))
		#print ("train_loss:", train_loss, " Ncase_train:", Ncase_train, train_loss/num_of_mols)
		#print ("diff:", mol_output - batch_data[2], " shape:", mol_output.shape)
		self.print_training(step, train_loss, num_of_mols, duration)
		return

	def train(self, mxsteps=10000):
		LOGGER.info("MolInstance_LJForce.train()")
		test_freq = PARAMS["test_freq"]
		mini_test_loss = float('inf') # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
		self.SaveAndClose()
		return
