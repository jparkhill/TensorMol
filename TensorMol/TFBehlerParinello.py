"""
Behler-Parinello network classes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import itertools
import random

from TensorMol.TensorMolData import *
from TensorMol.RawEmbeddings import *
from TensorMol.Neighbors import *
from tensorflow.python.client import timeline

class BehlerParinelloDirectSymFunc:
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
		TensorMol.RawEmbeddings.data_precision = self.tf_precision
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
		self.elements = self.tensor_data.elements
		self.max_num_atoms = self.tensor_data.max_num_atoms
		self.network_type = "BehlerParinelloDirect"
		self.name = self.network_type+"_"+self.tensor_data.molecule_set_name
		self.network_directory = './networks/'+self.name+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")

		if self.embedding_type == "symmetry_functions":
			self.set_symmetry_function_params()

		LOGGER.info("self.learning_rate: %d", self.learning_rate)
		LOGGER.info("self.batch_size: %d", self.batch_size)
		LOGGER.info("self.max_steps: %d", self.max_steps)
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
			elif self.activation_function_type == "sigmoid_with_param":
				self.activation_function = sigmoid_with_param
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
			self.tensor_data.clean_scratch()
		self.clean()
		f = open(self.network_directory+".tfn","wb")
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
		variable = tf.Variable(tf.truncated_normal(shape, stddev = stddev, dtype = self.tf_precision), name = name)
		if weight_decay is not None:
			weightdecay = tf.multiply(tf.nn.l2_loss(variable), weight_decay, name='weight_loss')
			tf.add_to_collection('losses', weightdecay)
		return variable

	def compute_normalization(self):
		elements = tf.constant(self.elements, dtype = tf.int32)
		element_pairs = tf.constant(self.element_pairs, dtype = tf.int32)
		radial_rs = tf.constant(self.radial_rs, dtype = self.tf_precision)
		angular_rs = tf.constant(self.angular_rs, dtype = self.tf_precision)
		theta_s = tf.constant(self.theta_s, dtype = self.tf_precision)
		radial_cutoff = tf.constant(self.radial_cutoff, dtype = self.tf_precision)
		angular_cutoff = tf.constant(self.angular_cutoff, dtype = self.tf_precision)
		zeta = tf.constant(self.zeta, dtype = self.tf_precision)
		eta = tf.constant(self.eta, dtype = self.tf_precision)
		xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
		Zs_pl = tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.max_num_atoms]))
		embeddings, molecule_indices = tf_symmetry_functions(xyzs_pl, Zs_pl, elements, element_pairs, radial_cutoff,
										angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)
		embeddings_list = [[], [], [], []]
		labels_list = []
		gradients_list = []

		self.embeddings_max = []
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		for ministep in range (0, max(2, int(0.1 * self.tensor_data.num_train_cases/self.batch_size))):
			batch_data = self.tensor_data.get_train_batch(self.batch_size)
			num_atoms = batch_data[4]
			labels_list.append(batch_data[2])
			for molecule in range(self.batch_size):
				gradients_list.append(batch_data[3][molecule,:num_atoms[molecule]])
			embedding, molecule_index = sess.run([embeddings, molecule_indices], feed_dict = {xyzs_pl:batch_data[0], Zs_pl:batch_data[1]})
			for element in range(len(self.elements)):
				embeddings_list[element].append(embedding[element])
		sess.close()
		for element in range(len(self.elements)):
			self.embeddings_max.append(np.amax(np.concatenate(embeddings_list[element])))
		labels = np.concatenate(labels_list)
		self.labels_mean = np.mean(labels)
		self.labels_stddev = np.std(labels)
		gradients = np.concatenate(gradients_list)
		self.gradients_mean = np.mean(gradients)
		self.gradients_stddev = np.std(gradients)
		self.tensor_data.train_scratch_pointer = 0

		#Set the embedding and label shape
		self.embedding_shape = embedding[0].shape[1]
		self.label_shape = labels[0].shape
		return

	def set_symmetry_function_params(self):
		self.element_pairs = np.array([[self.elements[i], self.elements[j]] for i in range(len(self.elements)) for j in range(i, len(self.elements))])
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]

		#Define radial grid parameters
		num_radial_rs = PARAMS["AN1_num_r_Rs"]
		self.radial_cutoff = PARAMS["AN1_r_Rc"]
		self.radial_rs = self.radial_cutoff * np.linspace(0, (num_radial_rs - 1.0) / num_radial_rs, num_radial_rs)

		#Define angular grid parameters
		num_angular_rs = PARAMS["AN1_num_a_Rs"]
		num_angular_theta_s = PARAMS["AN1_num_a_As"]
		self.angular_cutoff = PARAMS["AN1_a_Rc"]
		self.theta_s = 2.0 * np.pi * np.linspace(0, (num_angular_theta_s - 1.0) / num_angular_theta_s, num_angular_theta_s)
		self.angular_rs = self.angular_cutoff * np.linspace(0, (num_angular_rs - 1.0) / num_angular_rs, num_angular_rs)
		return

	def clean(self):
		if (self.sess != None):
			self.sess.close()
		self.sess = None
		self.total_loss = None
		self.train_op = None
		self.saver = None
		self.summary_writer = None
		self.summary_op = None
		self.activation_function = None
		self.options = None
		self.run_metadata = None
		self.xyzs_pl = None
		self.Zs_pl = None
		self.labels_pl = None
		self.num_atoms_pl = None
		self.gradients_pl = None
		self.energy_loss = None
		self.gradients_loss = None
		self.output = None
		self.gradients = None
		return

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			#Define the placeholders to be fed in for each batch
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
			self.Zs_pl = tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.max_num_atoms]))
			self.labels_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size]))
			self.gradients_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=([self.batch_size]))

			#Define the constants/Variables for the symmetry function basis
			elements = tf.constant(self.elements, dtype = tf.int32)
			element_pairs = tf.constant(self.element_pairs, dtype = tf.int32)
			radial_rs = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_rs = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			theta_s = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.constant(self.radial_cutoff, dtype = self.tf_precision)
			angular_cutoff = tf.constant(self.angular_cutoff, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)

			#Define normalization constants
			embeddings_max = tf.constant(self.embeddings_max, dtype = self.tf_precision)
			labels_mean = tf.constant(self.labels_mean, dtype = self.tf_precision)
			labels_stddev = tf.constant(self.labels_stddev, dtype = self.tf_precision)
			gradients_mean = tf.constant(self.gradients_mean, dtype = self.tf_precision)
			gradients_stddev = tf.constant(self.gradients_stddev, dtype = self.tf_precision)
			num_atoms_batch = tf.reduce_sum(self.num_atoms_pl)

			#Define the graph for computing the embedding, feeding through the network, and evaluating the loss
			element_embeddings, mol_indices = tf_symmetry_functions(self.xyzs_pl, self.Zs_pl, elements,
					element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)
			for element in range(len(self.elements)):
				element_embeddings[element] /= embeddings_max[element]
			self.normalized_output = self.inference(element_embeddings, mol_indices)
			self.output = (self.normalized_output * self.labels_stddev) + self.labels_mean
			self.normalized_gradients = tf.gradients(self.output, self.xyzs_pl)[0]
			self.gradients = (self.normalized_gradients * self.gradients_stddev) + self.gradients_mean
			non_padded_atoms = tf.where(tf.not_equal(self.Zs_pl, 0))
			self.non_padded_gradients = tf.gather_nd(self.gradients, non_padded_atoms)
			self.non_padded_gradients_label = tf.gather_nd(self.gradients_pl, non_padded_atoms)
			if self.train_energy_gradients:
				self.total_loss, self.energy_loss, self.gradient_loss = self.loss_op(self.output,
						self.labels_pl, self.non_padded_gradients, self.non_padded_gradients_label, num_atoms_batch)
			else:
				self.total_loss, self.energy_loss = self.loss_op(self.output, self.labels_pl)
			self.train_op = self.optimizer(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
			self.sess.run(init)
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
		feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl, self.labels_pl, self.gradients_pl, self.num_atoms_pl], batch_data)}
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
		branches=[]
		output = tf.zeros([self.batch_size, self.max_num_atoms], dtype=self.tf_precision)
		for e in range(len(self.elements)):
			branches.append([])
			inputs = inp[e]
			index = indexs[e]
			for i in range(len(self.hidden_layers)):
				if i == 0:
					with tf.name_scope(str(self.elements[e])+'_hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.embedding_shape, self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.embedding_shape)), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
				else:
					with tf.name_scope(str(self.elements[e])+'_hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.elements[e])+'_regression_linear'):
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
						stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
				branches[-1].append(tf.squeeze(tf.matmul(branches[-1][-1], weights) + biases))
				output += tf.scatter_nd(index, branches[-1][-1], [self.batch_size, self.max_num_atoms])
			tf.verify_tensor_all_finite(output,"Nan in output!!!")
		return tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size])

	def optimizer(self, loss, learning_rate, momentum):
		"""
		Sets up the training Ops.
		Creates a summarizer to track the loss over time in TensorBoard.
		Creates an optimizer and applies the gradients to all trainable variables.
		The Op returned by this function is what must be passed to the
		`sess.run()` call to cause the model to train.

		Args:
			loss: Loss tensor, from loss().
			learning_rate: the learning rate to use for gradient descent.

		Returns:
			train_op: the tensorflow operation to call for training.
		"""
		tf.summary.scalar(loss.op.name, loss)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def loss_op(self, output, labels, gradients = None, gradient_labels = None, num_atoms = None):
		energy_loss = tf.nn.l2_loss(tf.subtract(output, labels))
		tf.add_to_collection('losses', energy_loss)
		if self.train_energy_gradients:
			gradients_loss = self.batch_size * tf.nn.l2_loss(tf.subtract(gradients, gradient_labels)) / tf.cast(num_atoms, self.tf_precision)
			tf.add_to_collection('losses', gradients_loss)
			return tf.add_n(tf.get_collection('losses'), name='total_loss'), energy_loss, gradients_loss
		else:
			return tf.add_n(tf.get_collection('losses'), name='total_loss'), energy_loss

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.tensor_data.num_train_cases
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		train_gradient_loss = 0.0
		num_mols = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data = self.tensor_data.get_train_batch(self.batch_size)
			if self.train_energy_gradients:
				_, total_loss_value, energy_loss, gradient_loss, mol_output, n_atoms, gradients, gradient_labels = self.sess.run([self.train_op,
						self.total_loss, self.energy_loss, self.gradient_loss, self.output, self.num_atoms_pl, self.gradients,
						self.gradients_pl], feed_dict=self.fill_feed_dict(batch_data))
				train_gradient_loss += gradient_loss
			else:
				_, total_loss_value, energy_loss, mol_output = self.sess.run([self.train_op, self.total_loss,
							self.energy_loss, self.output], feed_dict=self.fill_feed_dict(batch_data))
			train_loss += total_loss_value
			train_energy_loss += energy_loss
			num_mols += self.batch_size
		duration = time.time() - start_time
		if self.train_energy_gradients:
			self.print_training(step, train_loss, train_energy_loss, num_mols, duration, train_gradient_loss)
		else:
			self.print_training(step, train_loss, train_energy_loss, num_mols, duration)
		return

	def test_step(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		test_loss =  0.0
		start_time = time.time()
		Ncase_test = self.tensor_data.num_test_cases
		num_mols = 0
		test_energy_loss = 0.0
		test_gradient_loss = 0.0
		test_epoch_energy_labels, test_epoch_energy_outputs = [], []
		test_epoch_force_labels, test_epoch_force_outputs = [], []
		num_atoms_epoch = []
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.tensor_data.get_test_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_energy_gradients:
				output, labels, gradients, gradient_labels, total_loss_value, energy_loss, gradient_loss, num_atoms = self.sess.run([self.output,
						self.labels_pl, self.gradients, self.gradients_pl, self.total_loss, self.energy_loss, self.gradient_loss,
						self.num_atoms_pl],  feed_dict=feed_dict)
				test_gradient_loss += gradient_loss
			else:
				output, labels, gradients, gradient_labels, total_loss_value, energy_loss, num_atoms = self.sess.run([self.output,
						self.labels_pl, self.gradients, self.gradients_pl, self.total_loss, self.energy_loss,
						self.num_atoms_pl],  feed_dict=feed_dict)
			test_loss += total_loss_value
			num_mols += self.batch_size
			test_energy_loss += energy_loss
			test_epoch_energy_labels.append(labels)
			test_epoch_energy_outputs.append(output)
			num_atoms_epoch.append(num_atoms)
			for molecule in range(self.batch_size):
				test_epoch_force_labels.append(-1.0 * gradient_labels[molecule,:num_atoms[molecule]])
				test_epoch_force_outputs.append(-1.0 * gradients[molecule,:num_atoms[molecule]])
		test_epoch_energy_labels = np.concatenate(test_epoch_energy_labels)
		test_epoch_energy_outputs = np.concatenate(test_epoch_energy_outputs)
		test_epoch_energy_errors = test_epoch_energy_labels - test_epoch_energy_outputs
		test_epoch_force_labels = np.concatenate(test_epoch_force_labels)
		test_epoch_force_outputs = np.concatenate(test_epoch_force_outputs)
		test_epoch_force_errors = test_epoch_force_labels - test_epoch_force_outputs
		num_atoms_epoch = np.sum(np.concatenate(num_atoms_epoch))
		duration = time.time() - start_time
		for i in [random.randint(0, self.batch_size - 1) for _ in xrange(20)]:
			LOGGER.info("Energy label: %.8f  Energy output: %.8f", test_epoch_energy_labels[i], test_epoch_energy_outputs[i])
		for i in [random.randint(0, num_atoms_epoch - 1) for _ in xrange(20)]:
			LOGGER.info("Forces label: %s  Forces output: %s", test_epoch_force_labels[i], test_epoch_force_outputs[i])
		LOGGER.info("MAE  Energy: %11.8f    Forces: %11.8f", np.mean(np.abs(test_epoch_energy_errors)), np.mean(np.abs(test_epoch_force_errors)))
		LOGGER.info("MSE  Energy: %11.8f    Forces: %11.8f", np.mean(test_epoch_energy_errors), np.mean(test_epoch_force_errors))
		LOGGER.info("RMSE Energy: %11.8f    Forces: %11.8f", np.sqrt(np.mean(np.square(test_epoch_energy_errors))), np.sqrt(np.mean(np.square(test_epoch_force_errors))))
		if self.train_energy_gradients:
			self.print_testing(step, test_loss, test_energy_loss, num_mols, duration, test_gradient_loss)
		else:
			self.print_testing(step, test_loss, test_energy_loss, num_mols, duration)
		return test_loss

	def train(self):
		self.tensor_data.load_data_to_scratch()
		self.compute_normalization()
		self.train_prepare()
		test_freq = PARAMS["test_freq"]
		mini_test_loss = 100000000 # some big numbers
		for step in range(1, self.max_steps+1):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test_step(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_checkpoint(step)
		self.save_network()
		return

	def print_training(self, step, loss, energy_loss, num_mols, duration, gradient_loss=None):
		if self.train_energy_gradients:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
						step, duration, loss / num_mols, energy_loss / num_mols, gradient_loss / num_mols)
		else:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f energy loss: %.10f",
						step, duration, loss / num_mols, energy_loss / num_mols)
		return

	def print_testing(self, step, loss, energy_loss, num_mols, duration, gradient_loss=None):
		if self.train_energy_gradients:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
						step, duration, loss / num_mols, energy_loss / num_mols, gradient_loss / num_mols)
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f energy loss: %.10f",
						step, duration, loss / num_mols, energy_loss / num_mols)
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
		if (batch_data[0].shape[1] > self.max_num_atoms or self.batch_size > batch_data[0].shape[0]):
			print("Natoms Match?", batch_data[0].shape[1] , self.max_num_atoms)
			print("BatchSizes Match?", self.batch_size , batch_data[0].shape[0])
			self.batch_size = batch_data[0].shape[0]
			self.max_num_atoms = batch_data[0].shape[1]
			MustPrepare = True
			# Create tensors with the right shape, and sub-fill them.
		elif (batch_data[0].shape[1] != self.max_num_atoms or self.batch_size != batch_data[0].shape[0]):
			xf = np.zeros((self.batch_size,self.max_num_atoms,3))
			zf = np.zeros((self.batch_size,self.max_num_atoms))
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
			self.xyzs_pl=tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms,3]))
			self.Zs_pl=tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.max_num_atoms]))
			self.gradients_pl=tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms,3]))
			self.Radp_Ele_pl=tf.placeholder(tf.int32, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int32, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int32, shape=tuple([None,4]))
			Ele = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
			Elep = tf.Variable(self.element_pairs, trainable=False, dtype = tf.int32)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_precision)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_precision)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_precision)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			element_factors = tf.Variable(np.array([2.20, 2.55, 3.04, 3.44]), trainable=False, dtype=tf.float64)
			element_pair_factors = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], trainable=False, dtype=tf.float64)
			self.Scatter_Sym, self.Sym_Index = TFSymSet_Linear_channel(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl, mil_jkt, element_factors, element_pair_factors )
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.gradient = tf.gradients(self.output, self.xyzs_pl)
			self.summary_op = tf.summary.merge_all()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
		return

class BehlerParinelloDirectGauSH:
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
		TensorMol.RawEmbeddings.data_precision = self.tf_precision
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
		self.number_radial = PARAMS["SH_NRAD"]
		self.l_max = PARAMS["SH_LMAX"]
		self.gaussian_params = PARAMS["RBFS"][:self.number_radial]
		self.atomic_embed_factors = PARAMS["ANES"]
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
		self.elements = self.tensor_data.elements
		self.max_num_atoms = self.tensor_data.max_num_atoms
		self.network_type = "BehlerParinelloDirect"
		self.name = self.network_type+"_"+self.tensor_data.molecule_set_name
		self.network_directory = './networks/'+self.name+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")

		if self.embedding_type == "symmetry_functions":
			self.set_symmetry_function_params()

		LOGGER.info("self.learning_rate: %d", self.learning_rate)
		LOGGER.info("self.batch_size: %d", self.batch_size)
		LOGGER.info("self.max_steps: %d", self.max_steps)
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
			self.tensor_data.clean_scratch()
		self.clean()
		f = open(self.network_directory+".tfn","wb")
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
		variable = tf.Variable(tf.truncated_normal(shape, stddev = stddev, dtype = self.tf_precision), name = name)
		if weight_decay is not None:
			weightdecay = tf.multiply(tf.nn.l2_loss(variable), weight_decay, name='weight_loss')
			tf.add_to_collection('losses', weightdecay)
		return variable

	def compute_normalization(self):
		xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([None, self.max_num_atoms, 3]))
		Zs_pl = tf.placeholder(tf.int32, shape=tuple([None, self.max_num_atoms]))
		gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_precision)
		atomic_embed_factors = tf.Variable(self.atomic_embed_factors, trainable=False, dtype=self.tf_precision)

		elements = tf.constant(self.elements, dtype = tf.int32)
		rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
				np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
				tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision)], axis=-1, name="rotation_params")
		rotated_xyzs = tf_random_rotate(xyzs_pl, rotation_params)
		embeddings, molecule_indices = tf_gaussian_spherical_harmonics(rotated_xyzs, Zs_pl, elements,
				gaussian_params, atomic_embed_factors, self.l_max)

		embeddings_list = []
		for element in range(len(self.elements)):
			embeddings_list.append([])
		labels_list = []
		gradients_list = []
		self.embeddings_mean = []
		self.embeddings_stddev = []

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		for ministep in range (0, max(2, int(0.1 * self.tensor_data.num_train_cases/self.batch_size))):
			batch_data = self.tensor_data.get_train_batch(self.batch_size)
			num_atoms = batch_data[4]
			labels_list.append(batch_data[2])
			for molecule in range(self.batch_size):
				gradients_list.append(batch_data[3][molecule,:num_atoms[molecule]])
			embedding, molecule_index = sess.run([embeddings, molecule_indices], feed_dict = {xyzs_pl:batch_data[0], Zs_pl:batch_data[1]})
			for element in range(len(self.elements)):
				embeddings_list[element].append(embedding[element])
		sess.close()
		for element in range(len(self.elements)):
			self.embeddings_mean.append(np.mean(np.concatenate(embeddings_list[element])))
			self.embeddings_stddev.append(np.std(np.concatenate(embeddings_list[element])))
		labels = np.concatenate(labels_list)
		self.labels_mean = np.mean(labels)
		self.labels_stddev = np.std(labels)
		gradients = np.concatenate(gradients_list)
		self.gradients_mean = np.mean(gradients)
		self.gradients_stddev = np.std(gradients)
		self.tensor_data.train_scratch_pointer = 0

		#Set the embedding and label shape
		self.embedding_shape = embedding[0].shape[1]
		self.label_shape = labels[0].shape
		return

	def clean(self):
		if (self.sess != None):
			self.sess.close()
		self.sess = None
		self.total_loss = None
		self.train_op = None
		self.saver = None
		self.summary_writer = None
		self.summary_op = None
		self.activation_function = None
		self.options = None
		self.run_metadata = None
		self.xyzs_pl = None
		self.Zs_pl = None
		self.labels_pl = None
		self.num_atoms_pl = None
		self.gradients_pl = None
		self.energy_loss = None
		self.gradients_loss = None
		self.output = None
		self.gradients = None
		return

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			#Define the placeholders to be fed in for each batch
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([None, self.max_num_atoms, 3]))
			self.Zs_pl = tf.placeholder(tf.int32, shape=tuple([None, self.max_num_atoms]))
			self.labels_pl = tf.placeholder(self.tf_precision, shape=tuple([None]))
			self.gradients_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=([self.batch_size]))
			self.gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_precision)
			self.atomic_embed_factors = tf.Variable(self.atomic_embed_factors, trainable=False, dtype=self.tf_precision)

			elements = tf.constant(self.elements, dtype = tf.int32)
			embeddings_mean = tf.constant(self.embeddings_mean, dtype = self.tf_precision)
			embeddings_stddev = tf.constant(self.embeddings_stddev, dtype = self.tf_precision)
			labels_mean = tf.constant(self.labels_mean, dtype = self.tf_precision)
			labels_stddev = tf.constant(self.labels_stddev, dtype = self.tf_precision)
			# gradients_mean = tf.constant(self.gradients_mean, dtype = self.tf_precision)
			# gradients_stddev = tf.constant(self.gradients_stddev, dtype = self.tf_precision)
			rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
					np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
					tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision)], axis=-1, name="rotation_params")
			rotated_xyzs = tf_random_rotate(self.xyzs_pl, rotation_params)
			embeddings, molecule_indices = tf_gaussian_spherical_harmonics(rotated_xyzs, self.Zs_pl, elements,
					self.gaussian_params, self.atomic_embed_factors, self.l_max)
			for element in range(len(self.elements)):
				embeddings[element] -= embeddings_mean[element]
				embeddings[element] /= embeddings_stddev[element]
			norm_output = self.inference(embeddings, molecule_indices)
			self.output = (norm_output * labels_stddev) + labels_mean
			self.total_loss, self.loss = self.loss_op(self.output, self.labels_pl)

			# if self.train_energy_gradients:
			# 	self.total_loss, self.energy_loss, self.gradient_loss = self.loss_op(self.output,
			# 			self.labels_pl, self.non_padded_gradients, self.non_padded_gradients_label, num_atoms_batch)
			# else:
			self.total_loss, self.energy_loss = self.loss_op(self.output, self.labels_pl)
			self.train_op = self.optimizer(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
			self.sess.run(init)
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
		feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl, self.labels_pl, self.gradients_pl, self.num_atoms_pl], batch_data)}
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
		branches=[]
		output = tf.zeros([self.batch_size, self.max_num_atoms], dtype=self.tf_precision)
		for e in range(len(self.elements)):
			branches.append([])
			inputs = inp[e]
			index = indexs[e]
			for i in range(len(self.hidden_layers)):
				if i == 0:
					with tf.name_scope(str(self.elements[e])+'_hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.embedding_shape, self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.embedding_shape)), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
				else:
					with tf.name_scope(str(self.elements[e])+'_hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
								stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.elements[e])+'_regression_linear'):
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
						stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
				branches[-1].append(tf.squeeze(tf.matmul(branches[-1][-1], weights) + biases))
				output += tf.scatter_nd(index, branches[-1][-1], [self.batch_size, self.max_num_atoms])
			tf.verify_tensor_all_finite(output,"Nan in output!!!")
		return tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size])

	def optimizer(self, loss, learning_rate, momentum):
		"""
		Sets up the training Ops.
		Creates a summarizer to track the loss over time in TensorBoard.
		Creates an optimizer and applies the gradients to all trainable variables.
		The Op returned by this function is what must be passed to the
		`sess.run()` call to cause the model to train.

		Args:
			loss: Loss tensor, from loss().
			learning_rate: the learning rate to use for gradient descent.

		Returns:
			train_op: the tensorflow operation to call for training.
		"""
		tf.summary.scalar(loss.op.name, loss)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def loss_op(self, output, labels, gradients = None, gradient_labels = None, num_atoms = None):
		energy_loss = tf.nn.l2_loss(tf.subtract(output, labels))
		tf.add_to_collection('losses', energy_loss)
		if self.train_energy_gradients:
			gradients_loss = self.batch_size * tf.nn.l2_loss(tf.subtract(gradients, gradient_labels)) / tf.cast(num_atoms, self.tf_precision)
			tf.add_to_collection('losses', gradients_loss)
			return tf.add_n(tf.get_collection('losses'), name='total_loss'), energy_loss, gradients_loss
		else:
			return tf.add_n(tf.get_collection('losses'), name='total_loss'), energy_loss

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.tensor_data.num_train_cases
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		train_gradient_loss = 0.0
		num_mols = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data = self.tensor_data.get_train_batch(self.batch_size)
			if self.train_energy_gradients:
				_, total_loss_value, energy_loss, gradient_loss, mol_output, n_atoms, gradients, gradient_labels = self.sess.run([self.train_op,
						self.total_loss, self.energy_loss, self.gradient_loss, self.output, self.num_atoms_pl, self.gradients,
						self.gradients_pl], feed_dict=self.fill_feed_dict(batch_data))
				train_gradient_loss += gradient_loss
			else:
				_, total_loss_value, energy_loss, mol_output = self.sess.run([self.train_op, self.total_loss,
							self.energy_loss, self.output], feed_dict=self.fill_feed_dict(batch_data))
			train_loss += total_loss_value
			train_energy_loss += energy_loss
			num_mols += self.batch_size
		duration = time.time() - start_time
		if self.train_energy_gradients:
			self.print_training(step, train_loss, train_energy_loss, num_mols, duration, train_gradient_loss)
		else:
			self.print_training(step, train_loss, train_energy_loss, num_mols, duration)
		return

	def test_step(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		test_loss =  0.0
		start_time = time.time()
		Ncase_test = self.tensor_data.num_test_cases
		num_mols = 0
		test_energy_loss = 0.0
		test_gradient_loss = 0.0
		test_epoch_energy_labels, test_epoch_energy_outputs = [], []
		test_epoch_force_labels, test_epoch_force_outputs = [], []
		num_atoms_epoch = []
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.tensor_data.get_test_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			if self.train_energy_gradients:
				output, labels, gradients, gradient_labels, total_loss_value, energy_loss, gradient_loss, num_atoms = self.sess.run([self.output,
						self.labels_pl, self.gradients, self.gradients_pl, self.total_loss, self.energy_loss, self.gradient_loss,
						self.num_atoms_pl],  feed_dict=feed_dict)
				test_gradient_loss += gradient_loss
			else:
				output, labels, total_loss_value, energy_loss, num_atoms = self.sess.run([self.output,
						self.labels_pl, self.total_loss, self.energy_loss,
						self.num_atoms_pl],  feed_dict=feed_dict)
			test_loss += total_loss_value
			num_mols += self.batch_size
			test_energy_loss += energy_loss
			test_epoch_energy_labels.append(labels)
			test_epoch_energy_outputs.append(output)
			num_atoms_epoch.append(num_atoms)
			# for molecule in range(self.batch_size):
			# 	test_epoch_force_labels.append(-1.0 * gradient_labels[molecule,:num_atoms[molecule]])
			# 	test_epoch_force_outputs.append(-1.0 * gradients[molecule,:num_atoms[molecule]])
		test_epoch_energy_labels = np.concatenate(test_epoch_energy_labels)
		test_epoch_energy_outputs = np.concatenate(test_epoch_energy_outputs)
		test_epoch_energy_errors = test_epoch_energy_labels - test_epoch_energy_outputs
		# test_epoch_force_labels = np.concatenate(test_epoch_force_labels)
		# test_epoch_force_outputs = np.concatenate(test_epoch_force_outputs)
		# test_epoch_force_errors = test_epoch_force_labels - test_epoch_force_outputs
		num_atoms_epoch = np.sum(np.concatenate(num_atoms_epoch))
		duration = time.time() - start_time
		for i in [random.randint(0, self.batch_size - 1) for _ in xrange(20)]:
			LOGGER.info("Energy label: %.8f  Energy output: %.8f", test_epoch_energy_labels[i], test_epoch_energy_outputs[i])
		# for i in [random.randint(0, num_atoms_epoch - 1) for _ in xrange(20)]:
		# 	LOGGER.info("Forces label: %s  Forces output: %s", test_epoch_force_labels[i], test_epoch_force_outputs[i])
		# LOGGER.info("MAE  Energy: %11.8f    Forces: %11.8f", np.mean(np.abs(test_epoch_energy_errors)), np.mean(np.abs(test_epoch_force_errors)))
		# LOGGER.info("MSE  Energy: %11.8f    Forces: %11.8f", np.mean(test_epoch_energy_errors), np.mean(test_epoch_force_errors))
		# LOGGER.info("RMSE Energy: %11.8f    Forces: %11.8f", np.sqrt(np.mean(np.square(test_epoch_energy_errors))), np.sqrt(np.mean(np.square(test_epoch_force_errors))))
		LOGGER.info("MAE  Energy: %11.8f", np.mean(np.abs(test_epoch_energy_errors)))
		LOGGER.info("MSE  Energy: %11.8f", np.mean(test_epoch_energy_errors))
		LOGGER.info("RMSE Energy: %11.8f", np.sqrt(np.mean(np.square(test_epoch_energy_errors))))
		if self.train_energy_gradients:
			self.print_testing(step, test_loss, test_energy_loss, num_mols, duration, test_gradient_loss)
		else:
			self.print_testing(step, test_loss, test_energy_loss, num_mols, duration)
		return test_loss

	def train(self):
		self.tensor_data.load_data_to_scratch()
		self.compute_normalization()
		self.train_prepare()
		test_freq = PARAMS["test_freq"]
		mini_test_loss = 100000000 # some big numbers
		for step in range(1, self.max_steps+1):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test_step(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_checkpoint(step)
		self.save_network()
		return

	def print_training(self, step, loss, energy_loss, num_mols, duration, gradient_loss=None):
		if self.train_energy_gradients:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
						step, duration, loss / num_mols, energy_loss / num_mols, gradient_loss / num_mols)
		else:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f energy loss: %.10f",
						step, duration, loss / num_mols, energy_loss / num_mols)
		return

	def print_testing(self, step, loss, energy_loss, num_mols, duration, gradient_loss=None):
		if self.train_energy_gradients:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
						step, duration, loss / num_mols, energy_loss / num_mols, gradient_loss / num_mols)
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f energy loss: %.10f",
						step, duration, loss / num_mols, energy_loss / num_mols)
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
		if (batch_data[0].shape[1] > self.max_num_atoms or self.batch_size > batch_data[0].shape[0]):
			print("Natoms Match?", batch_data[0].shape[1] , self.max_num_atoms)
			print("BatchSizes Match?", self.batch_size , batch_data[0].shape[0])
			self.batch_size = batch_data[0].shape[0]
			self.max_num_atoms = batch_data[0].shape[1]
			MustPrepare = True
			# Create tensors with the right shape, and sub-fill them.
		elif (batch_data[0].shape[1] != self.max_num_atoms or self.batch_size != batch_data[0].shape[0]):
			xf = np.zeros((self.batch_size,self.max_num_atoms,3))
			zf = np.zeros((self.batch_size,self.max_num_atoms))
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
			self.xyzs_pl=tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms,3]))
			self.Zs_pl=tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.max_num_atoms]))
			self.gradients_pl=tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms,3]))
			self.Radp_Ele_pl=tf.placeholder(tf.int32, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int32, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int32, shape=tuple([None,4]))
			Ele = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
			Elep = tf.Variable(self.element_pairs, trainable=False, dtype = tf.int32)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_precision)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_precision)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_precision)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			element_factors = tf.Variable(np.array([2.20, 2.55, 3.04, 3.44]), trainable=False, dtype=tf.float64)
			element_pair_factors = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], trainable=False, dtype=tf.float64)
			self.Scatter_Sym, self.Sym_Index = TFSymSet_Linear_channel(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl, mil_jkt, element_factors, element_pair_factors )
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.gradient = tf.gradients(self.output, self.xyzs_pl)
			self.summary_op = tf.summary.merge_all()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
		return
