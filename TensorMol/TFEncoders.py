"""
Encoder network classes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Autoencoder(object):
	def __init__(self):
		self.tf_precision = eval(PARAMS["tf_prec"])
		self.hidden_layers = PARAMS["HiddenLayers"]
		self.learning_rate = PARAMS["learning_rate"]
		self.max_steps = PARAMS["max_steps"]
		self.batch_size = PARAMS["batch_size"]
		self.max_checkpoints = PARAMS["max_checkpoints"]
		self.activation_function_type = PARAMS["NeuronType"]

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

	def encoder(self, inputs):
		layers = []
		with tf.name_scope("encoder"):
			for i, num_neurons in enumerate(self.hidden_layers):
				if i == 0:
					with tf.name_scope('hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.input_shape, num_neurons],
								stddev=math.sqrt(2.0 / self.embedding_shape), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([num_neurons], dtype=self.tf_precision), name='biases')
						layers.append(self.activation_function(tf.matmul(inputs, weights) + biases))
				else:
					with tf.name_scope('hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], num_neurons],
								stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([num_neurons], dtype=self.tf_precision), name='biases')
						layers.append(self.activation_function(tf.matmul(layers[-1], weights) + biases))
			with tf.name_scope('regression_linear'):
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], self.output_shape],
						stddev=math.sqrt(2.0 / self.hidden_layers[-1]), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([self.output_shape], dtype=self.tf_precision), name='biases')
				output = self.activation_function(tf.matmul(layers[-1], weights) + biases)
		return output

	def decoder(self, input):
		layers = []
		with tf.name_scope("encoder"):
			for i, num_neurons in enumerate(reversed(self.hidden_layers)):
				if i == 0:
					with tf.name_scope('hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.input_shape, num_neurons],
								stddev=math.sqrt(2.0 / self.embedding_shape), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([num_neurons], dtype=self.tf_precision), name='biases')
						layers.append(self.activation_function(tf.matmul(inputs, weights) + biases))
				else:
					with tf.name_scope('hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], num_neurons],
								stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([num_neurons], dtype=self.tf_precision), name='biases')
						layers.append(self.activation_function(tf.matmul(layers[-1], weights) + biases))
			with tf.name_scope('regression_linear'):
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], self.output_shape],
						stddev=math.sqrt(2.0 / self.hidden_layers[-1]), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([self.output_shape], dtype=self.tf_precision), name='biases')
				output = self.activation_function(tf.matmul(layers[-1], weights) + biases)
		return output

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
		tf.add_to_collection('losses', tf.nn.l2_loss(tf.subtract(output, labels)))
		return tf.add_n(tf.get_collection('losses'), name='total_loss')

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			#Define the placeholders to be fed in for each batch
			self.input_pl = tf.placeholder(self.tf_precision, shape=tuple([None, self.max_num_atoms, 3]))

			self.encoded_output = self.encoder(self.input_pl)
			self.decoded_output = self.decoder(self.encoded_output)

			self.total_loss = self.loss_op(self.decoded_output, self.input_pl)
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
		feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl, self.labels_pl, self.gradients_pl, self.num_atoms_pl], batch_data)}
		return feed_dict

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		num_train_cases = self.tensor_data.num_train_cases
		train_loss = 0.0
		start_time = time.time()
		for ministep in range(0, int(num_train_cases / self.batch_size)):
			batch_data = self.tensor_data.get_train_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			_, batch_loss = self.sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)
			train_loss += batch_loss
		duration = time.time() - start_time
		self.print_training(step, train_loss, duration)
		return

	def test_step(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		num_test_cases = self.tensor_data.num_test_cases
		test_loss =  0.0
		start_time = time.time()
		for ministep in range(0, int(num_test_cases / self.batch_size)):
			batch_data = self.tensor_data.get_test_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			batch_loss = self.sess.run(self.total_loss, feed_dict=feed_dict)
			test_loss += batch_loss
		duration = time.time() - start_time
		self.print_testing(step, test_loss, duration)
		return test_loss

	def train(self):
		self.train_prepare()
		test_freq = PARAMS["test_freq"]
		mini_test_loss = 1e6
		for step in range(1, self.max_steps+1):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test_step(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_checkpoint(step)
		self.save_network()
		return

class RestrictedBoltzmann(object):
	def __init__(self):
		self.tf_precision = eval(PARAMS["tf_prec"])
		self.hidden_layers = PARAMS["HiddenLayers"]
		self.learning_rate = PARAMS["learning_rate"]
		self.max_steps = PARAMS["max_steps"]
		self.batch_size = PARAMS["batch_size"]
		self.max_checkpoints = PARAMS["max_checkpoints"]
		self.activation_function_type = "sigmoid"

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
				self.activation_function = self.sigmoid_with_param
			else:
				print ("unknown activation function, set to relu")
				self.activation_function = tf.nn.relu
		except Exception as Ex:
			print(Ex)
			print ("activation function not assigned, set to relu")
			self.activation_function = tf.nn.relu
		return

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

	def loss_op(self, output, labels, gradients = None, gradient_labels = None, num_atoms = None):
		tf.add_to_collection('losses', tf.nn.l2_loss(tf.subtract(output, labels)))
		return tf.add_n(tf.get_collection('losses'), name='total_loss')

	def sample_prob(self, probs):
		return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			#Define the placeholders to be fed in for each batch
			self.input_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.input_shape])
			self.weights_pl = tf.placeholder(self.tf_precision, shape=[self.input_shape, self.hidden_layers[0]])
			self.hidden_bias_pl = tf.placeholder(self.tf_precision, shape=[self.hidden_layers[0]])
			self.visible_bias_pl = tf.placeholder(self.tf_precision, shape=[self.input_shape])
			alpha = tf.constant(1.0, dtype=self.tf_precision)

			self.hidden0prob = self.activation_function(tf.matmul(self.input_pl, self.weights_pl) + self.hidden_bias_pl)
			self.hidden0 = self.sample_prob(self.hidden0prob)
			self.visible1 = self.activation_function(tf.matmul(self.hidden0, tf.transpose(self.weights_pl)) + self.visible_bias_pl)
			self.hidden1 = self.activation_function(tf.matmul(self.visible1, self.weights_pl) + self.hidden_bias_pl)

			self.w_positive_grads = tf.matmul(tf.transpose(self.input_pl), self.hidden0)
			self.w_negative_grads = tf.matmul(tf.transpose(self.visible1), self.hidden1)

			self.update_w = self.weights_pl + alpha * (self.w_positive_grads - self.w_negative_grads) / tf.cast(tf.shape(inputs)[0], self.tf_precision)
			self.update_vb = self.visible_bias_pl + alpha * tf.reduce_mean(self.input_pl - self.visible1, axis=0)
			self.update_hb = self.hidden_bias_pl + alpha * tf.reduce_mean(self.hidden0prob - self.hidden1, axis=0)

			self.h_sample = self.activation_function(tf.matmul(self.input_pl, self.weights_pl) + self.hidden_bias_pl)
			self.v_sample = self.activation_function(tf.matmul(self.h_sample, tf.transpose(self.weights_pl)) + self.visible_bias_pl)

			self.cost = tf.nn.l2_loss(self.inputs_pl - self.v_sample)

			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
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
		feed_dict={i: d for i, d in zip([self.input_pl, self.weights_pl, self.hidden_bias_pl, self.visible_bias_pl],
										[batch_data, self.weights, self.hidden_bias, self.visible_bias])}
		return feed_dict

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		batch_data = self.tensor_data.get_train_batch(self.batch_size)
		feed_dict = self.fill_feed_dict(batch_data)
		self.weights, self.hidden_bias, self.visible_bias = self.sess.run([self.update_w,
				self.update_hb, self.update_vb], feed_dict=feed_dict)
		feed_dict = self.fill_feed_dict(batch_data)
		cost = self.sess.run(self.cost, feed_dict=feed_dict)
		print(cost)
		return

	def test_step(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		num_test_cases = self.tensor_data.num_test_cases
		test_loss =  0.0
		start_time = time.time()
		for ministep in range(0, int(num_test_cases / self.batch_size)):
			batch_data = self.tensor_data.get_test_batch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			batch_loss = self.sess.run(self.total_loss, feed_dict=feed_dict)
			test_loss += batch_loss
		duration = time.time() - start_time
		self.print_testing(step, test_loss, duration)
		return test_loss

	def train(self):
		self.train_prepare()
		test_freq = PARAMS["test_freq"]
		mini_test_loss = 1e6
		for step in range(1, self.max_steps+1):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test_step(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_checkpoint(step)
		self.save_network()
		return
