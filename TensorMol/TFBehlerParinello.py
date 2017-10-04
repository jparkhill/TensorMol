"""
Behler-Parinello network classes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import itertools

from TensorMol.TensorMolData import *
from TensorMol.RawEmbeddings import *
from TensorMol.Neighbors import *
from tensorflow.python.client import timeline

class BehlerParinelloDirect:
	"""
	Behler-Parinello network using embedding from RawEmbeddings.py
	"""
	def __init__(self, tensor_data, embedding_type, name = None):
		"""
		Args:
			tensor_data (TensorMol.TensorMolData object): a class which holds the training data
			name (str): a name used to recall this network

		Notes:
			if name != None, attempts to load a previously saved network, otherwise assumes a new network
		"""
		#Network and training parameters
		self.tf_precision = eval(PARAMS["tf_prec"])
		self.hidden_layers = PARAMS["HiddenLayers"]
		self.learning_rate = PARAMS["learning_rate"]
		self.weight_decay = PARAMS["weight_decay"]
		self.momentum = PARAMS["momentum"]
		self.max_steps = PARAMS["max_steps"]
		self.batch_size = PARAMS["batch_size"]
		self.max_checkpoints = PARAMS["max_checkpoints"]
		self.train_energy_gradients = PARAMS["train_energy_gradients"]
		self.profiling = PARAMS["Profiling"]
		self.activation_function_type = PARAMS["NeuronType"]
		self.assign_activation()
		self.embedding_type = embedding_type


		#Reloads a previous network if name variable is not None
		if name !=  None:
			self.name = name
			self.load_network()
			LOGGER.info("Reloaded network from %s", self.network_directory)
			return

		#Data parameters
		self.tensor_data = tensor_data
		self.elements = self.tensor_data.molecule_set.Atom_Types().sort()
		self.element_pairs = np.array([[self.elements[i], self.elements[j]] for i in range(len(self.elements)) for j in range(i, len(self.elements))])
		self.network_type = "BehlerParinelloDirect"
		self.name = self.NetType+"_"+self.tensor_data.molecule_set_name+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.network_directory = './networks/'+self.name

		LOGGER.info("self.learning_rate: %d", self.learning_rate)
		LOGGER.info("self.batch_size: %d", self.batch_size)
		LOGGER.info("self.max_steps: %d", self.max_steps)
		return

		# The tensorflow objects go up here.
		# self.inshape = None
		# self.outshape = None
		# self.sess = None
		# self.loss = None
		# self.output = None
		# self.train_op = None
		# self.total_loss = None
		# self.embeds_placeholder = None
		# self.labels_placeholder = None
		# self.saver = None
		# self.gradient =None
		# self.summary_op =None
		# self.summary_writer=None
		self.Trainable = Trainable_
		self.embedding_shape =  self.TData.dig.eshape  # use the flatted version
		self.output_shape = self.TData.dig.lshape    # use the flatted version
		return

	def assign_activation(self):
		LOGGER.debug("Assigning Activation Function: %s", PARAMS["NeuronType"])
		try:
			if self.activation_function_type == "relu":
				self.activation_function = tf.nn.relu
			elif self.activation_function_type == "elu":
				self.activation_function = tf.nn.elu
			elif self.activation_function_type == "selu":
				self.activation_function = self.selu
			elif self.activation_function_type == "softplus":
				self.activation_function = tf.nn.softplus
			elif self.activation_function_type == "tanh":
				self.activation_function = tf.tanh
			elif self.activation_function_type == "sigmoid":
				self.activation_function = tf.sigmoid
			else:
				print ("unknown activation function, set to relu")
				self.activation_function = tf.nn.relu
		except Exception as Ex:
			print(Ex)
			print ("activation function not assigned, set to relu")
			self.activation_function = tf.nn.relu
		return

	def save_checkpoint(self, step):
		checkpoint_file = os.path.join(self.network_directory,self.name+'-checkpoint-'+str(step))
		LOGGER.info("Saving checkpoint file %s", checkpoint_file)
		self.saver.save(self.sess, checkpoint_file)
		return

	def save_network(self):
		print("Saving TFInstance")
		if (self.tensor_data != None):
			self.tensor_data.CleanScratch()
		self.Clean()
		f = open(self.path+self.name+".tfn","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

	def load_network(self):
		LOGGER.info("Loading TFInstance")
		f = open(self.path+self.name+".tfn","rb")
		import TensorMol.PickleTM
		network_member_variables = TensorMol.PickleTM.UnPickleTM(f)
		self.Clean()
		self.__dict__.update(network_member_variables)
		f.close()
		checkpoint_files = [x for x in os.listdir(self.network_directory) if (x.count('checkpoint')>0 and x.count('meta')==0)]
		if (len(checkpoint_files)>0):
			self.latest_checkpoint_file = checkpoint_files[0]
		else:
			LOGGER.error("Network not found in directory: %s", self.network_directory)
			LOGGER.error("Network directory contents: %s", str(os.listdir(self.network_directory)))
		return

	def variable_summaries(self, var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def variable_with_weight_decay(self, shape, stddev, weight_decay, name = None):
		"""
		Creates random tensorflow variable from a truncated normal distribution with weight decay

		Args:
			name: name of the variable
			shape: list of ints
			stddev: standard deviation of a truncated Gaussian
			wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.

		Returns:
			Variable Tensor

		Notes:
			Note that the Variable is initialized with a truncated normal distribution.
			A weight decay is added only if one is specified.
		"""
		variable = tf.Variable(tf.truncated_normal(shape, stddev = stddev, dtype = self.tf_prec), name = name)
		if weight_decay is not None:
			weightdecay = tf.multiply(tf.nn.l2_loss(variable), weight_decay, name='weight_loss')
			tf.add_to_collection('losses', weightdecay)
		return variable

	def compute_normalization_constants(self):
		batch_data = self.tensor_data.get_train_batch(self.batch_size)
		self.tensor_data.train_scratch_pointer = 0
		xyzs, Zs, rad_p_ele, ang_t_elep, mil_jk = tf.Variable(batch_data[0], dtype=self.tf_prec), \
				tf.Variable(batch_data[1], dtype=tf.int32), tf.Variable(batch_data[5], dtype=tf.int32), \
				tf.Variable(batch_data[6], dtype=tf.int32), tf.Variable(batch_data[7], dtype=tf.int32)
		Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int32)
		Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int32)
		SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
		SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
		Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
		Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
		zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
		eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
		element_factors = tf.Variable(np.array([2.20, 2.55, 3.04, 3.44]), trainable=True, dtype=tf.float64)
		element_pair_factors = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], trainable=True, dtype=tf.float64)
		Scatter_Sym, Sym_Index = TFSymSet_Linear_channel(xyzs, Zs, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, rad_p_ele, ang_t_elep, mil_jk, element_factors, element_pair_factors)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			embed, _ = sess.run([Scatter_Sym, Sym_Index])
		self.inmean, self.instd = np.mean(np.concatenate(embed), axis=0), np.std(np.concatenate(embed), axis=0)
		self.outmean, self.outstd = np.mean(batch_data[2]), np.std(batch_data[2])
		self.gradmean, self.gradstd = np.mean(batch_data[3]), np.std(batch_data[3])
		return

	def set_symmetry_function_params(self, prec=np.float64):
		self.radial_grid_cutoff = PARAMS["AN1_r_Rc"]
		self.angular_grid_cutoff = PARAMS["AN1_a_Rc"]
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]

		#Define radial grid parameters
		num_radial_grid_points = PARAMS["AN1_num_r_Rs"]
		radial_grid = self.radial_grid_cutoff * np.linspace(0, (num_radial_grid_points - 1.0) / num_radial_grid_points, num_radial_grid_points)
		self.SFPr2 = np.transpose(np.reshape(radial_grid,[num_radial_grid_points,1]), [1,0])

		#Define angular grid parameters
		num_radial_angular_grid_points = PARAMS["AN1_num_a_As"]
		num_angular_grid_points = PARAMS["AN1_num_a_Rs"]
		thetas = 2.0 * np.pi * np.linspace(0, (num_angular_grid_points - 1.0) / num_angular_grid_points, num_angular_grid_points)
		rs = self.angular_grid_cutoff * np.linspace(0, (num_radial_angular_grid_points - 1.0) / num_radial_angular_grid_points, num_radial_angular_grid_points)
		p1 = np.tile(np.reshape(thetas,[num_angular_grid_points,1,1]),[1,num_radial_angular_grid_points,1])
		p2 = np.tile(np.reshape(rs,[1,num_radial_angular_grid_points,1]),[num_angular_grid_points,1,1])
		self.SFPa2 = np.transpose(np.concatenate([p1,p2],axis=2), [2,0,1])
		return

	def Clean(self):
		MolInstance.Clean()
		self.Radp_pl = None
		self.Angt_pl = None
		return

	def fill_feed_dict(self, batch_data):
		"""
		Fill the tensorflow feed dictionary.

		Args:
			batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
			and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

		Returns:
			Filled feed dictionary.
		"""
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.label_pl] + [self.grads_pl] + [self.n_atoms] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.mil_jk_pl], batch_data)}
		return feed_dict

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
		output = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		atom_outputs = []
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp[e]
			shp_in = tf.shape(inputs)
			index = tf.cast(indexs[e], tf.int64)
			for i in range(len(self.HiddenLayers)):
				if i == 0:
					with tf.name_scope(str(self.eles[e])+'_hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.inshape, self.hidden_layers[i]], stddev=1.0/(10+math.sqrt(float(self.inshape))), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
				else:
					with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]], stddev=1.0/(10+math.sqrt(float(self.hidden_layers[i-1]))), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_regression_linear'):
				shp = tf.shape(inputs)
				weights = self._variable_with_weight_decay(shape=[self.HiddenLayers[-1], 1], stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
				branches[-1].append(tf.matmul(branches[-1][-1], weights) + biases)
				shp_out = tf.shape(branches[-1][-1])
				cut = tf.slice(branches[-1][-1],[0,0],[shp_out[0],1])
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				rshpflat = tf.reshape(cut,[shp_out[0]])
				atom_indice = tf.slice(index, [0,1], [shp_out[0],1])
				ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
				output = tf.add(output, ToAdd)
			tf.verify_tensor_all_finite(output,"Nan in output!!!")
		return tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size]), atom_outputs

	def loss_op(self, output, nn_grads, labels, grads, n_atoms):
		energy_diff  = tf.subtract(output, labels)
		energy_loss = tf.nn.l2_loss(energy_diff)
		grads_diff = tf.subtract(nn_grads, grads)
		nonzero_grads_diff = tf.gather_nd(grads_diff, tf.where(tf.not_equal(grads_diff, 0)))
		grads_loss = tf.nn.l2_loss(nonzero_grads_diff) / tf.reduce_sum(n_atoms) * self.batch_size
		#loss = tf.multiply(grads_loss, energy_loss)
		# loss = tf.add(energy_loss, tf.multiply(grads_loss, self.GradScalar))
		loss = energy_loss + grads_loss
		#loss = tf.identity(energy_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		train_grads_loss = 0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data = self.TData.GetTrainBatch(self.batch_size)
			actual_mols  = self.batch_size
			t = time.time()
			_, _, total_loss_value, loss_value, energy_loss, grads_loss, mol_output, atom_outputs = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.output, self.atom_outputs], feed_dict=self.fill_feed_dict(batch_data))
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
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
		test_energy_loss = 0.0
		test_grads_loss = 0.0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data=self.TData.GetTestBatch(self.batch_size)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = self.batch_size
			preds, total_loss_value, loss_value, energy_loss, grads_loss, mol_output, atom_outputs, element_factors, element_pair_factors = self.sess.run([self.output, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.output, self.atom_outputs, self.element_factors, self.element_pair_factors],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
		duration = time.time() - start_time
		print( "testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, num_of_mols, duration)
		LOGGER.info("Element factors: %s", element_factors)
 		LOGGER.info("Element pair factors: %s", element_pair_factors)
		return test_loss

	def train(self, mxsteps, continue_training= False):
		self.compute_normalization_constants()
		self.TrainPrepare(continue_training)
		test_freq = PARAMS["test_freq"]
		mini_test_loss = 100000000 # some big numbers
		for step in range(1, mxsteps+1):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_chk(step)
		self.SaveAndClose()
		return

	def print_training(self, step, loss, energy_loss, grads_loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f  energy_loss: %.10f  grad_loss: %.10f", step, duration, float(loss)/(Ncase), float(energy_loss)/(Ncase), float(grads_loss)/(Ncase))
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f energy_loss: %.10f  grad_loss: %.10f", step, duration, float(loss)/(Ncase), float(energy_loss)/(Ncase), float(grads_loss)/(Ncase))
		return

	def evaluate(self, batch_data):
		"""
		Evaluate the energy, atom energies, and IfGrad = True the gradients
		of this Direct Behler-Parinello graph.

		Args:
			batch_data: a list containing
			XYZ,Z,radial pairs, angular triples (all set format Mol X MaxNAtoms... )
		"""
		# Check sanity of input
		xf = batch_data[0].copy()
		zf = batch_data[1].copy()
		MustPrepare = not self.sess
		if (batch_data[0].shape[1] > self.MaxNAtoms or self.batch_size > batch_data[0].shape[0]):
			print("Natoms Match?", batch_data[0].shape[1] , self.MaxNAtoms)
			print("BatchSizes Match?", self.batch_size , batch_data[0].shape[0])
			self.batch_size = batch_data[0].shape[0]
			self.MaxNAtoms = batch_data[0].shape[1]
			MustPrepare = True
			# Create tensors with the right shape, and sub-fill them.
		elif (batch_data[0].shape[1] != self.MaxNAtoms or self.batch_size != batch_data[0].shape[0]):
			xf = np.zeros((self.batch_size,self.MaxNAtoms,3))
			zf = np.zeros((self.batch_size,self.MaxNAtoms))
			xf[:batch_data[0].shape[0],:batch_data[0].shape[1],:] = batch_data[0]
			zf[:batch_data[1].shape[0],:batch_data[1].shape[1]] = batch_data[1]
		LOGGER.debug("Batch_Size: %i", self.batch_size)
		if MustPrepare:
			print ("loading the session..")
			self.EvalPrepare()
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Radp_pl]+[self.Angt_pl], [xf]+[zf]+[batch_data[2]]+[batch_data[3]])}
		mol_output, atom_outputs, gradient = self.sess.run([self.output, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
		return mol_output, atom_outputs, gradient

	def EvalPrepare(self):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.
		"""
		with tf.Graph().as_default():
			self.SetANI1Param()
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_Ele_pl=tf.placeholder(tf.int32, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int32, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int32, shape=tuple([None,4]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int32)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int32)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			element_factors = tf.Variable(np.array([2.20, 2.55, 3.04, 3.44]), trainable=False, dtype=tf.float64)
			element_pair_factors = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], trainable=False, dtype=tf.float64)
			self.Scatter_Sym, self.Sym_Index = TFSymSet_Linear_channel(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl, mil_jkt, element_factors, element_pair_factors )
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.gradient = tf.gradients(self.output, self.xyzs_pl)
			self.summary_op = tf.summary.merge_all()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		return

	def TrainPrepare(self,  continue_training =False):
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
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_Ele_pl=tf.placeholder(tf.int32, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int32, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int32, shape=tuple([None,4]))
			self.n_atoms = tf.placeholder(tf.float64, shape=tuple([self.batch_size]))
			inmean = tf.constant(self.inmean, dtype=self.tf_prec)
			instd = tf.constant(self.instd, dtype=self.tf_prec)
			outmean = tf.constant(self.outmean, dtype=self.tf_prec)
			outstd = tf.constant(self.outstd, dtype=self.tf_prec)
			gradmean = tf.constant(self.gradmean, dtype=self.tf_prec)
			gradstd = tf.constant(self.gradstd, dtype=self.tf_prec)
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int32)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int32)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.element_factors = tf.Variable(np.array([2.20, 2.55, 3.04, 3.44]), trainable=True, dtype=tf.float64)
			self.element_pair_factors = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], trainable=True, dtype=tf.float64)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.Scatter_Sym, self.Sym_Index = TFSymSet_Linear_channel(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl, self.element_factors, self.element_pair_factors)
			self.norm_embedding_list = []
			for embedding in self.Scatter_Sym:
				self.norm_embedding_list.append((embedding - inmean) / instd)
			self.norm_output, self.atom_outputs = self.inference(self.norm_embedding_list, self.Sym_Index)
			self.output = (self.norm_output * outstd) - outmean
			self.check = tf.add_check_numerics_ops()
			self.norm_gradient = tf.gradients(self.output, self.xyzs_pl)
			self.gradient = (self.norm_gradient * gradstd) - gradmean
			self.total_loss, self.loss, self.energy_loss, self.grads_loss = self.loss_op(self.output, self.gradient, self.label_pl, self.grads_pl, self.n_atoms)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return
