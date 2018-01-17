"""
For the sake of modularity, all direct access to dig
needs to be phased out...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..Containers.TensorData import *
from ..ForceModifiers.Transformer import *
from ..TFDescriptors.RawSymFunc import *
from ..Util import *
import numpy as np
import math
import time
import os
import sys
import numbers
import random
if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import _pickle as pickle
import os.path
if (HAS_TF):
	import tensorflow as tf

class Instance:
	"""
	Manages a persistent training network instance
	"""
	def __init__(self, TData_, ele_ = 1 , Name_=None, NetType_=None):
		"""
		Args:
			TData_: a TensorData
			ele_: an element type for this instance.
			Name_ : a name for this instance, attempts to load from checkpoint.
		"""
		# The tensorflow objects go up here.
		self.inshape = None
		self.outshape = None
		self.sess = None
		self.loss = None
		self.output = None
		self.train_op = None
		self.total_loss = None
		self.embeds_placeholder = None
		self.labels_placeholder = None
		self.saver = None
		self.gradient =None
		self.summary_op =None
		self.summary_writer=None
		# The parameters below belong to tensorflow and its graph
		# all tensorflow variables cannot be pickled they are populated by Prepare
		self.PreparedFor=0
		try:
			self.tf_prec
		except:
			self.tf_prec = eval(PARAMS["tf_prec"])
		self.HiddenLayers = PARAMS["HiddenLayers"]
		self.hidden1 = PARAMS["hidden1"]
		self.hidden2 = PARAMS["hidden2"]
		self.hidden3 = PARAMS["hidden3"]
		self.learning_rate = PARAMS["learning_rate"]
		self.momentum = PARAMS["momentum"]
		self.max_steps = PARAMS["max_steps"]
		self.batch_size = PARAMS["batch_size"]
		self.max_checkpoints = PARAMS["max_checkpoints"]
		self.activation_function_type = PARAMS["NeuronType"]
		self.activation_function = None
		self.profiling = PARAMS["Profiling"]
		self.AssignActivation()
		self.path=PARAMS["networks_directory"]
		if (Name_ !=  None):
			self.name = Name_
			#self.QueryAvailable() # Should be a sanity check on the data files.
			self.Load() # Network still cannot be used until it is prepared.
			self.train_dir = PARAMS["networks_directory"]+self.name
			LOGGER.info("raised network: "+ self.train_dir)
			return
		self.element = ele_
		self.TData = TData_
		self.tformer = Transformer(PARAMS["InNormRoutine"], PARAMS["OutNormRoutine"], self.element, self.TData.dig.name, self.TData.dig.OType)
		if (not os.path.isdir(self.path)):
			os.mkdir(self.path)
		self.chk_file = ''

		LOGGER.info("self.learning_rate: "+str(self.learning_rate))
		LOGGER.info("self.batch_size: "+str(self.batch_size))
		LOGGER.info("self.max_steps: "+str(self.max_steps))

		self.NetType = "None"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = PARAMS["networks_directory"]+self.name
		# if (self.element != 0):
			# self.TData.LoadElementToScratch(self.element, self.tformer)
			# self.tformer.Print()
			# self.TData.PrintStatus()
			# self.inshape = self.TData.dig.eshape
			# self.outshape = self.TData.dig.lshape
		return

	def __del__(self):
		if (self.sess != None):
			self.sess.close()
		self.Clean()

	def AssignActivation(self):
		LOGGER.debug("Assigning Activation... %s", PARAMS["NeuronType"])
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
			elif self.activation_function_type == "gaussian":
				self.activation_function = guassian_act
			elif self.activation_function_type == "gaussian_rev_tozero":
				self.activation_function = guassian_rev_tozero
			elif self.activation_function_type == "gaussian_rev_tozero_tolinear":
				self.activation_function = guassian_rev_tozero_tolinear
			elif self.activation_function_type == "square_tozero_tolinear":
				self.activation_function = square_tozero_tolinear
			else:
				print ("unknown activation function, set to relu")
				self.activation_function = tf.nn.relu
		except Exception as Ex:
			print(Ex)
			print ("activation function not assigned, set to relu")
			self.activation_function = tf.nn.relu
		return

	def evaluate(self, eval_input):
		# Check sanity of input
		if (not np.all(np.isfinite(eval_input))):
			LOGGER.error("WTF, you trying to feed me, garbage?")
			raise Exception("bad digest.")
		if (self.PreparedFor < eval_input.shape[0]):
			self.Prepare(eval_input, eval_input.shape[0])
		return

	def Prepare(self, eval_input, Ncase=1250):
		"""
		Called if only evaluations are being done, by evaluate()
		"""
		self.Clean()
		self.AssignActivation()
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default():
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
			self.output = self.inference(self.embeds_placeholder)
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			metafiles = [x for x in os.listdir(self.train_dir) if (x.count('meta')>0)]
			if (len(metafiles)>0):
				most_recent_meta_file=metafiles[0]
				LOGGER.debug("Restoring training from Meta file: "+most_recent_meta_file)
				config = tf.ConfigProto(allow_soft_placement=True)
				self.sess = tf.Session(config=config)
				self.saver = tf.train.import_meta_graph(self.train_dir+'/'+most_recent_meta_file)
				self.saver.restore(self.sess, tf.train.latest_checkpoint(self.train_dir))
		self.PreparedFor = Ncase
		return

	def TrainPrepare(self,  continue_training =False):
		""" Builds the graphs by calling inference """
		with tf.Graph().as_default():
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.embeds_placeholder)
			self.total_loss, self.loss = self.loss_op(self.output, self.labels_placeholder)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.sess.run(init)
			try:
				metafiles = [x for x in os.listdir(self.train_dir) if (x.count('meta')>0)]
				if (len(metafiles)>0):
					most_recent_meta_file=metafiles[0]
					LOGGER.info("Restoring training from Metafile: "+most_recent_meta_file)
					#Set config to allow soft device placement for temporary fix to known issue with Tensorflow up to version 0.12 atleast - JEH
					config = tf.ConfigProto(allow_soft_placement=True)
					self.sess = tf.Session(config=config)
					self.saver = tf.train.import_meta_graph(self.train_dir+'/'+most_recent_meta_file)
					self.saver.restore(self.sess, tf.train.latest_checkpoint(self.train_dir))
			except Exception as Ex:
				LOGGER.error("Restore Failed")
				pass
			self.summary_writer =  tf.summary.FileWriter(self.train_dir, self.sess.graph)
			return

	def Clean(self):
		if (self.sess != None):
			self.sess.close()
		self.sess = None
		self.loss = None
		self.output = None
		self.total_loss = None
		self.train_op = None
		self.embeds_placeholder = None
		self.labels_placeholder = None
		self.saver = None
		self.gradient =None
		self.summary_writer = None
		self.PreparedFor = 0
		self.summary_op = None
		self.activation_function = None
		self.options = None
		self.run_metadata = None
		return

	def SaveAndClose(self):
		print("Saving TFInstance...")
		if (self.TData!=None):
			self.TData.CleanScratch()
		self.Clean()
		#print("Going to pickle...\n",[(attr,type(ins)) for attr,ins in self.__dict__.items()])
		f=open(self.path+self.name+".tfn","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
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

	def save_chk(self,  step):  # this can be included in the Instance
		checkpoint_file_mini = os.path.join(self.train_dir,self.name+'-chk-'+str(step))
		LOGGER.info("Saving Checkpoint file, "+checkpoint_file_mini)
		self.saver.save(self.sess, checkpoint_file_mini)
		return

	def FindLastCheckpoint(self):
		chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')>0)]
		if (len(chkfiles)==0):
			return False
		chknums = sorted([int(chkfile.replace(self.name+'-chk-','').replace('.meta','')) for chkfile in chkfiles])
		lastchkfile = os.path.join(self.train_dir,self.name+'-chk-'+str(chknums[-1]))
		print("Found Last Checkpoint file: ",lastchkfile)
		return lastchkfile

	#this isn't really the correct way to load()
	# only the local class members (not any TF objects should be unpickled.)
	def Load(self):
		LOGGER.info("Unpickling TFInstance...")
		from ..Containers.PickleTM import UnPickleTM as UnPickleTM
		tmp = UnPickleTM(self.path+self.name+".tfn")
		# All this shit should be deleteable after re-training.
		self.__dict__.update(tmp)
		chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
		# if (len(chkfiles)>0):
		# 	self.chk_file = chkfiles[0]
		# else:
		# 	LOGGER.error("Network not found... Traindir:"+self.train_dir)
		# 	LOGGER.error("Traindir contents: "+str(os.listdir(self.train_dir)))
		return

	def _variable_with_weight_decay(self, var_name, var_shape, var_stddev, var_wd):
		"""Helper to create an initialized Variable with weight decay.

		Note that the Variable is initialized with a truncated normal distribution.
		A weight decay is added only if one is specified.

		Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
		decay is not added for this Variable.

		Returns:
		Variable Tensor
		"""
		var = tf.Variable(tf.truncated_normal(var_shape, stddev=var_stddev, dtype=self.tf_prec), name=var_name)
		if var_wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), var_wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var


	def _get_weight_variable(self, name, shape):
		return tf.get_variable(name, shape, self.tf_prec, tf.truncated_normal_initializer(stddev=0.01))

	def _get_bias_variable(self, name, shape):
		return tf.get_variable(name, shape, self.tf_prec, tf.constant_initializer(0.01, dtype=self.tf_prec))


	def _get_variable_with_weight_decay(self, var_name, var_shape, var_stddev, var_wd):
		"""Helper to create an initialized Variable with weight decay for sharing weights.

		Note that the Variable is initialized with a truncated normal distribution.
		A weight decay is added only if one is specified.

		Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
		decay is not added for this Variable.

		Returns:
		Variable Tensor
		"""
		var = tf.get_variable(var_name, var_shape, self.tf_prec, tf.truncated_normal_initializer(stddev=var_stddev))
		if var_wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), var_wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def dropout_selu(self, x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, noise_shape=None, seed=None, name=None, training=False):
		"""Dropout to a value with rescaling."""
		def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
			keep_prob = 1.0 - rate
			x = tf.convert_to_tensor(x, name="x")
			if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
				raise ValueError("keep_prob must be a scalar tensor or a float in the "
								"range (0, 1], got %g" % keep_prob)
			keep_prob = tf.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
			keep_prob.get_shape().assert_is_compatible_with([])

			alpha = tf.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
			keep_prob.get_shape().assert_is_compatible_with([])

			if tf.contrib.util.constant_value(keep_prob) == 1:
				return x

			noise_shape = noise_shape if noise_shape is not None else tf.shape(x)
			random_tensor = keep_prob
			random_tensor += tf.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
			binary_tensor = tf.floor(random_tensor)
			ret = x * binary_tensor + alpha * (1-binary_tensor)

			a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

			b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
			ret = a * ret + b
			ret.set_shape(x.get_shape())
			return ret

		with tf.name_scope(name, "dropout", [x]) as name:
			# return dropout_selu_impl(x, rate, alpha, noise_shape, seed, name) if training else array_ops.identity(x)
			return tf.cond(training,
				lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
				lambda: tf.identity(x))

	def selu(self, x):
		with tf.name_scope('elu') as scope:
			alpha = 1.6732632423543772848170429916717
			scale = 1.0507009873554804934193349852946
			return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

	def placeholder_inputs(self, batch_size):
		raise("Populate placeholder_inputs")
		return

	def fill_feed_dict(self, batch_data, embeds_pl, labels_pl):
		"""Fills the feed_dict for training the given step.
		A feed_dict takes the form of:
		feed_dict = {
		<placeholder>: <tensor of values to be passed for placeholder>,
		....
		}
		Args:
		data_set: The set of images and labels, from input_data.read_data_sets()
		embeds_pl: The images placeholder, from placeholder_inputs().
		labels_pl: The labels placeholder, from placeholder_inputs().
		Returns:
		feed_dict: The feed dictionary mapping from placeholders to values.
		"""
		# Don't eat shit.
		if (not np.all(np.isfinite(batch_data[0]))):
			LOGGER.error("I was fed shit")
			raise Exception("DontEatShit")
		if (not np.all(np.isfinite(batch_data[1]))):
			LOGGER.error("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict = {embeds_pl: batch_data[0], labels_pl: batch_data[1],}
		return feed_dict

	def inference(self, inputs):
		"""Builds the network architecture. Number of hidden layers and nodes in each layer defined in TMParams "HiddenLayers".
		Args:
			inputs: input placeholder for training data from Digester.
		Returns:
			output: scalar or vector of OType from Digester.
		"""

		hiddens = []
		for i in range(len(self.HiddenLayers)):
			if i == 0:
				with tf.name_scope('hidden1'):
					weights = self._variable_with_weight_decay(var_name='weights',
									var_shape=([self.inshape, self.HiddenLayers[i]]),
									var_stddev= 1.0 / math.sqrt(float(self.inshape)), var_wd= 0.00)
					biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
					hiddens.append(self.activation_function(tf.matmul(inputs, weights) + biases))
					# tf.scalar_summary('min/' + weights.name, tf.reduce_min(weights))
					# tf.histogram_summary(weights.name, weights)
			else:
				with tf.name_scope('hidden'+str(i+1)):
					weights = self._variable_with_weight_decay(var_name='weights',
									var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]],
									var_stddev= 1.0 / math.sqrt(float(self.HiddenLayers[i-1])), var_wd= 0.00)
					biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec),name='biases')
					hiddens.append(self.activation_function(tf.matmul(hiddens[-1], weights) + biases))
		with tf.name_scope('regression_linear'):
			weights = self._variable_with_weight_decay(var_name='weights',
							var_shape=[self.HiddenLayers[-1], self.outshape],
							var_stddev= 1.0 / math.sqrt(float(self.HiddenLayers[-1])), var_wd= 0.00)
			biases = tf.Variable(tf.zeros(self.outshape, dtype=self.tf_prec), name='biases')
			output = tf.matmul(hiddens[-1], weights) + biases
		return output

	def loss_op(self, output, labels):
		"""
		Calculates the loss from the logits and the labels.
		Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size].
		Returns:
		loss: Loss tensor of type float.
		"""
		raise Exception("Base Loss.")
		return

	def training(self, loss, learning_rate, momentum):
		"""Sets up the training Ops.
		Creates a summarizer to track the loss over time in TensorBoard.
		Creates an optimizer and applies the gradients to all trainable variables.
		The Op returned by this function is what must be passed to the
		`sess.run()` call to cause the model to train.
		Args:
		loss: Loss tensor, from loss().
		learning_rate: The learning rate to use for gradient descent.
		Returns:
		train_op: The Op for training.
		"""
		tf.summary.scalar(loss.op.name, loss)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def train(self, mxsteps, continue_training= False):
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

	def train_step(self,step):
		raise Exception("Cannot Train base...")
		return

	def TrainPrepare(self,  continue_training =False):
		"""Train for a number of steps."""
		with tf.Graph().as_default():
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.embeds_placeholder)
			self.total_loss, self.loss = self.loss_op(self.output, self.labels_placeholder)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.sess.run(init)
			try: # I think this may be broken
				chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
				metafiles = [x for x in os.listdir(self.train_dir) if (x.count('meta')>0)]
				if (len(metafiles)>0):
					most_recent_meta_file=metafiles[0]
					print("Restoring training from Metafile: ",most_recent_meta_file)
					#Set config to allow soft device placement for temporary fix to known issue with Tensorflow up to version 0.12 atleast - JEH
					config = tf.ConfigProto(allow_soft_placement=True)
					self.sess = tf.Session(config=config)
					self.saver = tf.train.import_meta_graph(self.train_dir+'/'+most_recent_meta_file)
					self.saver.restore(self.sess, tf.train.latest_checkpoint(self.train_dir))
			except Exception as Ex:
				print("Restore Failed 2341325",Ex)
				pass
			self.summary_writer =  tf.summary.FileWriter(self.train_dir, self.sess.graph)
			return

	def test(self,step):
		raise Exception("Base Test")
		return

	def print_training(self, step, loss, Ncase, duration, Train=True):
		denom = max((int(Ncase/self.batch_size)),1)
		if Train:
			LOGGER.info("step: %7d  duration: %.5f train loss: %.10f", step, duration,(float(loss)/(denom*self.batch_size)))
		else:
			LOGGER.info("step: %7d  duration: %.5f test loss: %.10f", step, duration,(float(loss)/(denom*self.batch_size)))
		return

class Instance_fc_classify(Instance):
	def __init__(self, TData_, ele_ = 1 , Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		self.NetType = "fc_classify"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = PARAMS["networks_directory"]+self.name
		self.prob = None
		#		self.inshape = self.TData.scratch_inputs.shape[1]
		self.correct = None
		self.summary_op =None
		self.summary_writer=None

	def n_correct(self, output, labels):
		# For a classifier model, we can use the in_top_k Op.
		# It returns a bool tensor with shape [batch_size] that is true for
		# the examples where the label is in the top k (here k=1)
		# of all logits for that example.
		labels = tf.to_int64(labels)
		correct = tf.nn.in_top_k(output, labels, 1)
		# Return the number of true entries.
		return tf.reduce_sum(tf.cast(correct, tf.int32))

	def evaluate(self, eval_input):
		# Check sanity of input
		Instance.evaluate(self, eval_input)
		eval_input_ = eval_input
		if (self.PreparedFor>eval_input.shape[0]):
			eval_input_ =np.copy(eval_input)
			eval_input_.resize((self.PreparedFor,eval_input.shape[1]))
			# pad with zeros
		eval_labels = np.zeros(self.PreparedFor)  # dummy labels
		batch_data = [eval_input_, eval_labels]
		#embeds_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.embeds_placeholder,self.labels_placeholder)
		tmp = (np.array(self.sess.run([self.prob], feed_dict=feed_dict))[0,:eval_input.shape[0],1])
		if (not np.all(np.isfinite(tmp))):
			LOGGER.error("TFsession returned garbage")
			LOGGER.error("TFInputs: "+str(eval_input) ) #If it's still a problem here use tf.Print version of the graph.
			raise Exception("Garbage...")
		if (self.PreparedFor > eval_input.shape[0]):
			return tmp[:eval_input.shape[0]]
		return tmp

	def Prepare(self, eval_input, Ncase=1250):
		Instance.Prepare(self)
		LOGGER.info("Preparing a "+self.NetType+"Instance")
		self.prob = None
		self.correct = None
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default():
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
			self.output = self.inference(self.embeds_placeholder)
			self.correct = self.n_correct(self.output, self.labels_placeholder)
			self.prob = self.justpreds(self.output)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
			if (len(chkfiles)>0):
				most_recent_chk_file=chkfiles[0]
				LOGGER.info("Restoring training from Checkpoint: "+most_recent_chk_file)
				self.saver.restore(self.sess, self.train_dir+'/'+most_recent_chk_file)
		self.PreparedFor = Ncase
		return

	def Save(self):
		self.prob = None
		self.correct = None
		self.summary_op =None
		self.summary_writer=None
		Instance.Save(self)
		return

	def placeholder_inputs(self, batch_size):
		"""Generate placeholder variables to represent the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded data in the .run() loop, below.
		Args:
		batch_size: The batch size will be baked into both placeholders.
		Returns:
		embeds_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
		"""
		# Note that the shapes of the placeholders match the shapes of the full
		# image and label tensors, except the first dimension is now batch_size
		# rather than the full size of the train or test data sets.
		inputs_pl = tf.placeholder(self.tf_prec, shape=(batch_size,self.inshape)) # JAP : Careful about the shapes... should be flat for now.
		outputs_pl = tf.placeholder(self.tf_prec, shape=(batch_size))
		return inputs_pl, outputs_pl

	def justpreds(self, output):
		"""Calculates the loss from the logits and the labels.
		Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size].
		Returns:
		loss: Loss tensor of type float.
		"""
		prob = tf.nn.softmax(output)
		return prob

	def loss_op(self, output, labels):
		"""Calculates the loss from the logits and the labels.
		Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size].
		Returns:
		loss: Loss tensor of type float.
		"""
		prob = tf.nn.softmax(output)
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels, name='xentropy')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), cross_entropy_mean, prob

	def print_training(self, step, loss, total_correct, Ncase, duration):
		denom=max(int(Ncase/self.batch_size),1)
		print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/denom),"accu:  %.5f"%(float(total_correct)/(denom*self.batch_size)))
		return

	def train_step(self,step):
		Ncase_train = self.TData.NTrainCasesInScratch()
		start_time = time.time()
		train_loss =  0.0
		total_correct = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data=self.TData.GetTrainBatch(self.element,  self.batch_size) #advances the case pointer in TData...
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			_, total_loss_value, loss_value, prob_value, correct_num  = self.sess.run([self.train_op, self.total_loss, self.loss, self.prob, self.correct], feed_dict=feed_dict)
			train_loss = train_loss + loss_value
			total_correct = total_correct + correct_num
		duration = time.time() - start_time
		#self.print_training(step, train_loss, total_correct, Ncase_train, duration)
		self.print_training(step, train_loss, Ncase_train, duration)
		return

	def test(self, step):
		Ncase_test = self.TData.NTest
		test_loss =  0.0
		test_correct = 0.
		test_start_time = time.time()
		test_loss = None
		feed_dict = None
		for  ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data=self.TData.GetTestBatch(self.element,  self.batch_size, ministep)
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			loss_value, prob_value, test_correct_num = self.sess.run([ self.loss, self.prob, self.correct],  feed_dict=feed_dict)
			test_loss = test_loss + loss_value
			test_correct = test_correct + test_correct_num
			duration = time.time() - test_start_time
			LOGGER.info("testing...")
			self.print_training(step, test_loss, test_correct, Ncase_test, duration)
		return test_loss, feed_dict

class Instance_fc_sqdiff(Instance):
	def __init__(self, TData_, ele_ = 1 , Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		self.NetType = "fc_sqdiff"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = PARAMS["networks_directory"]+self.name

	def evaluate(self, eval_input):
		# Check sanity of input
		Instance.evaluate(self, eval_input)
		given_cases = eval_input.shape[0]
		#print("given_cases:", given_cases)
		eis = list(eval_input.shape)
		eval_input_ = eval_input.copy()
		if (self.PreparedFor > given_cases):
			eval_input_.resize(([self.PreparedFor]+eis[1:]))
			# pad with zeros
		eval_labels = np.zeros(tuple([self.PreparedFor]+list(self.outshape)))  # dummy labels
		batch_data = [eval_input_, eval_labels]
		#embeds_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.embeds_placeholder, self.labels_placeholder)
		tmp = np.array(self.sess.run([self.output], feed_dict=feed_dict))
		if (not np.all(np.isfinite(tmp))):
			LOGGER.error("TFsession returned garbage")
			LOGGER.error("TFInputs"+str(eval_input) ) #If it's still a problem here use tf.Print version of the graph.
		return tmp[0,:given_cases]

	def Save(self):
		self.summary_op =None
		self.summary_writer=None
		Instance.Save(self)
		return

	def placeholder_inputs(self, batch_size):
		"""Generate placeholder variables to represent the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded data in the .run() loop, below.
		Args:
		batch_size: The batch size will be baked into both placeholders.
		Returns:
		embeds_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
		"""
		# Note that the shapes of the placeholders match the shapes of the full
		# image and label tensors, except the first dimension is now batch_size
		# rather than the full size of the train or test data sets.
		inputs_pl = tf.placeholder(self.tf_prec, shape=tuple([batch_size]+list(self.inshape)))
		outputs_pl = tf.placeholder(self.tf_prec, shape=tuple([batch_size]+list(self.outshape)))
		return inputs_pl, outputs_pl

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def train_step(self,step):
		Ncase_train = self.TData.NTrainCasesInScratch()
		start_time = time.time()
		train_loss =  0.0
		total_correct = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data=self.TData.GetTrainBatch(self.element,  self.batch_size) #advances the case pointer in TData...
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			_, total_loss_value, loss_value = self.sess.run([self.train_op, self.total_loss, self.loss], feed_dict=feed_dict)
			train_loss = train_loss + loss_value
		duration = time.time() - start_time
		#self.print_training(step, train_loss, total_correct, Ncase_train, duration)
		self.print_training(step, train_loss, Ncase_train, duration)
		return

	def test(self, step):
		Ncase_test = self.TData.NTestCasesInScratch()
		test_loss =  0.0
		test_start_time = time.time()
		#for ministep in range (0, int(Ncase_test/self.batch_size)):
		batch_data=self.TData.GetTestBatch(self.element,  self.batch_size)#, ministep)
		feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
		preds, total_loss_value, loss_value  = self.sess.run([self.output, self.total_loss,  self.loss],  feed_dict=feed_dict)
		self.TData.EvaluateTestBatch(batch_data[1],preds, self.tformer)
		test_loss = test_loss + loss_value
		duration = time.time() - test_start_time
		print("testing...")
		self.print_training(step, test_loss,  Ncase_test, duration, Train=False)
		return test_loss, feed_dict

	def PrepareData(self, batch_data):
		if (batch_data[0].shape[0]==self.batch_size):
			batch_data=[batch_data[0], batch_data[1].reshape((batch_data[1].shape[0],1))]
		elif (batch_data[0].shape[0] < self.batch_size):
			batch_data=[batch_data[0], batch_data[1].reshape((batch_data[1].shape[0],1))]
			tmp_input = np.copy(batch_data[0])
			tmp_output = np.copy(batch_data[1])
			tmp_input.resize((self.batch_size,  batch_data[0].shape[1]))
			tmp_output.resize((self.batch_size,  batch_data[1].shape[1]))
			batch_data=[ tmp_input, tmp_output]
		return batch_data

class Instance_fc_sqdiff_GauSH_direct(Instance):
	def __init__(self, TData=None, elements=None, trainable=True, name=None):
		Instance.__init__(self, TData, elements, name)
		self.number_radial = PARAMS["SH_NRAD"]
		self.l_max = PARAMS["SH_LMAX"]
		self.gaussian_params = PARAMS["RBFS"][:self.number_radial]
		self.atomic_embed_factors = PARAMS["ANES"]
		self.MaxNAtoms = self.TData.MaxNAtoms
		self.inshape =  self.number_radial * (self.l_max + 1) ** 2
		self.outshape = 3
		TensorMol.TFDescriptors.RawSH.data_precision = self.tf_prec
		if name != None:
			self.path = PARAMS["networks_directory"]
			self.name = name
			self.Load()
			self.gaussian_params = PARAMS["RBFS"][:self.number_radial]
			self.atomic_embed_factors = PARAMS["ANES"]
			self.MaxNAtoms = self.TData.MaxNAtoms
			self.inshape =  self.number_radial * (self.l_max + 1) ** 2
			self.outshape = 3
			self.AssignActivation()

			return
		self.NetType = "fc_sqdiff_GauSH_direct"
		self.name = self.TData.name+"_"+self.NetType+"_"+str(self.element)+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.train_dir = PARAMS["networks_directory"]+self.name
		self.trainable = trainable
		self.orthogonalize = True
		if (self.trainable):
			self.TData.LoadDataToScratch(self.tformer)

	def compute_normalization_constants(self):
		batch_data = self.TData.GetTrainBatch(20 * self.batch_size)
		self.TData.ScratchPointer = 0
		xyzs, Zs, labels = tf.convert_to_tensor(batch_data[0], dtype=self.tf_prec), tf.convert_to_tensor(batch_data[1]), tf.convert_to_tensor(batch_data[2], dtype=self.tf_prec)
		num_mols = tf.shape(xyzs)[0]
		rotation_params = tf.stack([np.pi * tf.random_uniform([num_mols], maxval=2.0, dtype=self.tf_prec),
				np.pi * tf.random_uniform([num_mols], maxval=2.0, dtype=self.tf_prec),
				tf.random_uniform([num_mols], maxval=2.0, dtype=self.tf_prec)], axis=-1)
		rotated_xyzs, rotated_labels = tf_random_rotate(xyzs, rotation_params, labels)
		embedding, labels, _, _ = tf_gaussian_spherical_harmonics_element(rotated_xyzs, Zs, rotated_labels,
											self.element, tf.Variable(self.gaussian_params, dtype=self.tf_prec),
											tf.Variable(self.atomic_embed_factors, trainable=False, dtype=self.tf_prec),
											self.l_max, self.orthogonalize)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			embed, label = sess.run([embedding, labels])
		self.inmean, self.instd = np.mean(embed, axis=0), np.std(embed, axis=0)
		self.outmean, self.outstd = np.mean(label), np.std(label)
		return

	def TrainPrepare(self):
		""" Builds the graphs by calling inference """
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms, 3]))
			self.Zs_pl = tf.placeholder(tf.int32, shape=tuple([None, self.MaxNAtoms]))
			self.labels_pl = tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms, 3]))
			self.gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_prec)
			self.atomic_embed_factors = tf.Variable(self.atomic_embed_factors, trainable=False, dtype=self.tf_prec)
			element = tf.constant(self.element, dtype=tf.int32)
			inmean = tf.constant(self.inmean, dtype=self.tf_prec)
			instd = tf.constant(self.instd, dtype=self.tf_prec)
			outmean = tf.constant(self.outmean, dtype=self.tf_prec)
			outstd = tf.constant(self.outstd, dtype=self.tf_prec)
			rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_prec),
					np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_prec),
					tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_prec)], axis=-1, name="rotation_params")
			rotated_xyzs, rotated_labels = tf_random_rotate(self.xyzs_pl, rotation_params, self.labels_pl)
			self.embedding, self.labels, _, min_eigenval = tf_gaussian_spherical_harmonics_element(rotated_xyzs, self.Zs_pl, element,
					self.gaussian_params, self.atomic_embed_factors, self.l_max, rotated_labels, orthogonalize=self.orthogonalize)
			self.norm_embedding = (self.embedding - inmean) / instd
			self.norm_labels = (self.labels - outmean) / outstd
			self.norm_output = self.inference(self.norm_embedding)
			self.output = (self.norm_output * outstd) + outmean
			self.n_atoms_batch = tf.shape(self.output)[0]
			self.total_loss, self.loss = self.loss_op(self.norm_output, self.norm_labels)
			barrier_function = -1000.0 * tf.log(tf.concat([self.gaussian_params + 0.9, tf.expand_dims(6.5 - self.gaussian_params[:,0], axis=-1), tf.expand_dims(1.75 - self.gaussian_params[:,1], axis=-1)], axis=1))
			truncated_barrier_function = tf.reduce_sum(tf.where(tf.greater(barrier_function, 0.0), barrier_function, tf.zeros_like(barrier_function)))
			gaussian_overlap_constraint = tf.square(0.001 / min_eigenval)
			loss_and_constraint = self.total_loss + truncated_barrier_function + gaussian_overlap_constraint
			self.train_op = self.training(loss_and_constraint, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.sess.run(init)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			if self.profiling:
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
			return

	def inference(self, inputs):
		"""
		Builds a Behler-Parinello graph

		Args:
			inputs: a list of (num_of atom type X flattened input shape) matrix of input cases.
		Returns:
			The BP graph output
		"""
		layers=[]
		for i in range(len(self.HiddenLayers)):
			if i == 0:
				with tf.name_scope('hidden1'):
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]],
																var_stddev = math.sqrt(2.0 / float(self.inshape)), var_wd=0.00)
					biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
					layers.append(self.activation_function(tf.matmul(inputs, weights) + biases))
			else:
				with tf.name_scope('hidden'+str(i+1)):
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1],self.HiddenLayers[i]],
																var_stddev = math.sqrt(2.0 / float(self.HiddenLayers[i-1])), var_wd=0.00)
					biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
					layers.append(self.activation_function(tf.matmul(layers[-1], weights) + biases))
		with tf.name_scope('regression_linear'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], self.outshape],
														var_stddev = math.sqrt(2.0 / float(self.HiddenLayers[-1])), var_wd=None)
			biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
			outputs = tf.matmul(layers[-1], weights) + biases
		tf.verify_tensor_all_finite(outputs,"Nan in output!!!")
		return outputs

	def evaluate(self, eval_input):
		# Check sanity of input
		Instance.evaluate(self, eval_input)
		given_cases = eval_input.shape[0]
		#print("given_cases:", given_cases)
		eis = list(eval_input.shape)
		eval_input_ = eval_input.copy()
		if (self.PreparedFor > given_cases):
			eval_input_.resize(([self.PreparedFor]+eis[1:]))
			# pad with zeros
		eval_labels = np.zeros(tuple([self.PreparedFor]+list(self.outshape)))  # dummy labels
		batch_data = [eval_input_, eval_labels]
		#embeds_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.embeds_placeholder, self.labels_placeholder)
		tmp = np.array(self.sess.run([self.output], feed_dict=feed_dict))
		if (not np.all(np.isfinite(tmp))):
			LOGGER.error("TFsession returned garbage")
			LOGGER.error("TFInputs"+str(eval_input) ) #If it's still a problem here use tf.Print version of the graph.
		return tmp[0,:given_cases]

	def Clean(self):
		if (self.sess != None):
			self.sess.close()
		self.sess = None
		self.loss = None
		self.output = None
		self.total_loss = None
		self.train_op = None
		self.saver = None
		self.gradient = None
		self.summary_writer = None
		self.PreparedFor = 0
		self.summary_op = None
		self.activation_function = None
		self.atomic_embed_factors = None
		self.gaussian_params = None
		self.labels = None
		self.norm_output = None
		self.norm_labels = None
		self.norm_embedding = None
		self.labels_pl = None
		self.Zs_pl = None
		self.xyzs_pl = None
		self.output = None
		self.embedding = None
		self.n_atoms_batch = None
		self.tf_prec = None
		return

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss



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
		if (not np.all(np.isfinite(batch_data[2]))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl] + [self.Zs_pl] + [self.labels_pl], [batch_data[0]] + [batch_data[1]] + [batch_data[2]])}
		return feed_dict

	def train(self, mxsteps, continue_training= False):
		self.compute_normalization_constants()
		self.TrainPrepare()
		test_freq = PARAMS["test_freq"]
		mini_test_loss = 1.e16
		for step in range(1, mxsteps+1):
			self.train_step(step)
			if step%test_freq==0:
				test_loss = self.test(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_chk(step)
		self.summary_writer.close()
		self.SaveAndClose()
		return

	def train_step(self,step):
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss, n_atoms_epoch = 0.0, 0.0
		for ministep in xrange(0, int(Ncase_train/self.batch_size)):
			batch_data = self.TData.GetTrainBatch(self.batch_size) #advances the case pointer in TData...
			feed_dict = self.fill_feed_dict(batch_data)
			if self.profiling:
				_, total_loss_value, loss_value, n_atoms_batch = self.sess.run([self.train_op, self.total_loss, self.loss, self.n_atoms_batch], feed_dict=feed_dict, options=self.options, run_metadata=self.run_metadata)
				fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
				chrome_trace = fetched_timeline.generate_chrome_trace_format()
				with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
					f.write(chrome_trace)
			else:
				_, total_loss_value, loss_value, n_atoms_batch = self.sess.run([self.train_op, self.total_loss, self.loss, self.n_atoms_batch], feed_dict=feed_dict)
			train_loss += total_loss_value
			n_atoms_epoch += n_atoms_batch
		duration = time.time() - start_time
		self.print_training(step, train_loss, n_atoms_epoch, duration)
		return

	def test(self, step):
		print("testing...")
		Ncase_test = self.TData.NTest
		test_loss, n_atoms_epoch = 0.0, 0.0
		test_start_time = time.time()
		mean_test_error, std_dev_test_error = 0.0, 0.0
		test_epoch_labels, test_epoch_outputs = [], []
		for ministep in xrange(0, int(Ncase_test/self.batch_size)):
			batch_data=self.TData.GetTestBatch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			output, labels, total_loss_value, loss_value, n_atoms_batch, gaussian_params, atomic_embed_factors = self.sess.run([self.output, self.labels, self.total_loss, self.loss, self.n_atoms_batch, self.gaussian_params, self.atomic_embed_factors],  feed_dict=feed_dict)
			test_loss += total_loss_value
			n_atoms_epoch += n_atoms_batch
			test_epoch_labels.append(labels)
			test_epoch_outputs.append(output)
		test_epoch_labels = np.concatenate(test_epoch_labels)
		test_epoch_outputs = np.concatenate(test_epoch_outputs)
		test_epoch_errors = test_epoch_labels - test_epoch_outputs
		duration = time.time() - test_start_time
		for i in [random.randint(0, self.batch_size) for j in xrange(20)]:
			LOGGER.info("Label: %s  Output: %s", test_epoch_labels[i], test_epoch_outputs[i])
		LOGGER.info("MAE: %f", np.mean(np.abs(test_epoch_errors)))
		LOGGER.info("MSE: %f", np.mean(test_epoch_errors))
		LOGGER.info("RMSE: %f", np.sqrt(np.mean(np.square(test_epoch_errors))))
		LOGGER.info("Std. Dev.: %f", np.std(test_epoch_errors))
		LOGGER.info("Gaussian paramaters: %s", gaussian_params)
		LOGGER.info("Atomic embedding factors: %s", atomic_embed_factors)
		self.print_testing(step, test_loss, n_atoms_epoch, duration)
		return test_loss

	def print_training(self, step, loss, n_cases, duration):
		LOGGER.info("step: %7d  duration: %.5f train loss: %.10f", step, duration,(loss / float(n_cases)))
		return

	def print_testing(self, step, loss, n_cases, duration):
		LOGGER.info("step: %7d  duration: %.5f test loss: %.10f", step, duration,(loss / float(n_cases)))
		return

	def PrepareData(self, batch_data):
		if (batch_data[0].shape[0]==self.batch_size):
			batch_data=[batch_data[0], batch_data[1].reshape((batch_data[1].shape[0],1))]
		elif (batch_data[0].shape[0] < self.batch_size):
			batch_data=[batch_data[0], batch_data[1].reshape((batch_data[1].shape[0],1))]
			tmp_input = np.copy(batch_data[0])
			tmp_output = np.copy(batch_data[1])
			tmp_input.resize((self.batch_size,  batch_data[0].shape[1]))
			tmp_output.resize((self.batch_size,  batch_data[1].shape[1]))
			batch_data=[ tmp_input, tmp_output]
		return batch_data

	def evaluate_prepare(self):
		""" Builds the graphs by calling inference """
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_prec, shape=tuple([None, 465, 3]))
			self.Zs_pl = tf.placeholder(tf.int32, shape=tuple([None, 465]))
			self.labels_pl = tf.placeholder(self.tf_prec, shape=tuple([None, 465, 3]))
			self.gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_prec)
			self.atomic_embed_factors = tf.Variable(self.atomic_embed_factors, trainable=False, dtype=self.tf_prec)
			element = tf.constant(self.element, dtype=tf.int32)
			inmean = tf.constant(self.inmean, dtype=self.tf_prec)
			instd = tf.constant(self.instd, dtype=self.tf_prec)
			outmean = tf.constant(self.outmean, dtype=self.tf_prec)
			outstd = tf.constant(self.outstd, dtype=self.tf_prec)
			# rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_prec),
			# 		np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_prec),
			# 		tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_prec)], axis=-1, name="rotation_params")
			# rotated_xyzs, rotated_labels = tf_random_rotate(self.xyzs_pl, rotation_params, self.labels_pl)
			self.embedding, self.atom_indices, min_eigenval = tf_gaussian_spherical_harmonics_element(self.xyzs_pl, self.Zs_pl, element,
					self.gaussian_params, self.atomic_embed_factors, self.l_max, orthogonalize=self.orthogonalize)
			self.norm_embedding = (self.embedding - inmean) / instd
			# self.norm_labels = (self.labels - outmean) / outstd
			self.norm_output = self.inference(self.norm_embedding)
			self.output = (self.norm_output * outstd) + outmean
			self.n_atoms_batch = tf.shape(self.output)[0]
			# self.total_loss, self.loss = self.loss_op(self.norm_output, self.norm_labels)
			# barrier_function = -1000.0 * tf.log(tf.concat([self.gaussian_params + 0.9, tf.expand_dims(6.5 - self.gaussian_params[:,0], axis=-1), tf.expand_dims(1.75 - self.gaussian_params[:,1], axis=-1)], axis=1))
			# truncated_barrier_function = tf.reduce_sum(tf.where(tf.greater(barrier_function, 0.0), barrier_function, tf.zeros_like(barrier_function)))
			# gaussian_overlap_constraint = tf.square(0.001 / min_eigenval)
			# loss_and_constraint = self.total_loss + truncated_barrier_function + gaussian_overlap_constraint
			# self.train_op = self.training(loss_and_constraint, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			# init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
			# self.sess.run(init)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			# if self.profiling:
			# 	self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			# 	self.run_metadata = tf.RunMetadata()
			return

	def evaluate_fill_feed_dict(self, xyzs, Zs):
		"""
		Fill the tensorflow feed dictionary.

		Args:
			batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
			and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

		Returns:
			Filled feed dictionary.
		"""
		feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl], [xyzs, Zs])}
		return feed_dict

	def evaluate(self, xyzs, Zs):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		if not self.sess:
			print("loading the session..")
			self.chk_file = self.FindLastCheckpoint()
			self.evaluate_prepare()
		new_xyzs = np.zeros((1, 465,3))
		new_xyzs[0,:np.shape(xyzs)[0]] = xyzs
		new_Zs = np.zeros((1, 465), dtype=np.int32)
		new_Zs[0,:np.shape(Zs)[0]] = Zs
		feed_dict=self.evaluate_fill_feed_dict(new_xyzs, new_Zs)
		forces, atom_indices = self.sess.run([self.output, self.atom_indices], feed_dict=feed_dict)
		return -forces, atom_indices

class FCGauSHDirectRotationInvariant(Instance_fc_sqdiff_GauSH_direct):
	def __init__(self, TData_, elements_ , Trainable_ = True, Name_ = None):
		Instance.__init__(self, TData_, elements_, Name_)
		self.NetType = "fc_sqdiff_GauSH_direct"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = PARAMS["networks_directory"]+self.name
		self.number_radial = PARAMS["SH_NRAD"]
		self.l_max = PARAMS["SH_LMAX"]
		self.gaussian_params = PARAMS["RBFS"][:self.number_radial]
		self.atomic_embed_factors = PARAMS["ANES"]
		TensorMol.TFDescriptors.RawSH.data_precision = self.tf_prec
		self.MaxNAtoms = self.TData.MaxNAtoms
		self.inshape =  self.number_radial * (self.l_max + 1) ** 2
		self.outshape = 3
		self.Trainable = Trainable_
		self.orthogonalize = True
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)

	def compute_normalization_constants(self):
		batch_data = self.TData.GetTrainBatch(20 * self.batch_size)
		self.TData.ScratchPointer = 0
		xyzs, Zs, labels = tf.convert_to_tensor(batch_data[0], dtype=self.tf_prec), tf.convert_to_tensor(batch_data[1]), tf.convert_to_tensor(batch_data[2], dtype=self.tf_prec)
		num_mols = tf.shape(xyzs)[0]
		rotation_params = tf.stack([np.pi * tf.random_uniform([num_mols], maxval=2.0, dtype=self.tf_prec),
				np.pi * tf.random_uniform([num_mols], maxval=2.0, dtype=self.tf_prec),
				tf.random_uniform([num_mols], maxval=2.0, dtype=self.tf_prec)], axis=-1)
		rotated_xyzs, rotated_labels = tf_random_rotate(xyzs, rotation_params, labels)
		embedding, labels, _, _ = tf_gaussian_spherical_harmonics_element(rotated_xyzs, Zs, rotated_labels,
											self.element, tf.Variable(self.gaussian_params, dtype=self.tf_prec),
											tf.Variable(self.atomic_embed_factors, trainable=False, dtype=self.tf_prec),
											self.l_max, self.orthogonalize)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			embed, label = sess.run([embedding, labels])
		self.inmean, self.instd = np.mean(embed, axis=0), np.std(embed, axis=0)
		self.outmean, self.outstd = np.mean(label), np.std(label)
		return

	def TrainPrepare(self):
		""" Builds the graphs by calling inference """
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms, 3]), name="xyzs_pl")
			self.Zs_pl = tf.placeholder(tf.int32, shape=tuple([None, self.MaxNAtoms]), name="Zs_pl")
			self.labels_pl = tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms, 3]), name="labels_pl")
			self.gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_prec)
			self.atomic_embed_factors = tf.Variable(self.atomic_embed_factors, trainable=False, dtype=self.tf_prec)
			num_mols = tf.shape(self.xyzs_pl)[0]
			element = tf.constant(self.element, dtype=tf.int32)
			inmean = tf.constant(self.inmean, dtype=self.tf_prec)
			instd = tf.constant(self.instd, dtype=self.tf_prec)
			outmean = tf.constant(self.outmean, dtype=self.tf_prec)
			outstd = tf.constant(self.outstd, dtype=self.tf_prec)
			with tf.name_scope("Rotation"):
				rotation_params = tf.stack([np.pi * tf.random_uniform([num_mols], minval=0.1, maxval=1.9, dtype=self.tf_prec),
						np.pi * tf.random_uniform([num_mols], minval=0.1, maxval=1.9, dtype=self.tf_prec),
						tf.random_uniform([num_mols], minval=0.1, maxval=1.9, dtype=self.tf_prec)], axis=-1, name="rotation_params")
				rotated_xyzs, rotation_matrix = tf_random_rotate(self.xyzs_pl, rotation_params, return_matrix=True)
			with tf.name_scope("Embedding_Normalization"):
				self.embedding, self.labels, mol_atom_indices, min_eigenval = tf_gaussian_spherical_harmonics_element(rotated_xyzs,
						self.Zs_pl, self.labels_pl, element, self.gaussian_params, self.atomic_embed_factors, self.l_max, orthogonalize=self.orthogonalize)
				self.norm_embedding = (self.embedding - inmean) / instd
				self.norm_labels = (self.labels - outmean) / outstd
			self.norm_output = self.inference(self.norm_embedding)
			with tf.name_scope("Inverse_Rotation"):
				inverse_rotation_matrix = tf.matrix_inverse(tf.gather(rotation_matrix, mol_atom_indices[:,0]))
				element_xyzs, rotated_element_xyzs = tf.gather_nd(self.xyzs_pl, mol_atom_indices), tf.gather_nd(rotated_xyzs, mol_atom_indices)
				unrotated_norm_output = tf.squeeze(tf.einsum("lij,lkj->lki", inverse_rotation_matrix,
						tf.expand_dims(rotated_element_xyzs + self.norm_output, axis=1))) - element_xyzs
			self.output = (unrotated_norm_output * outstd) + outmean
			self.n_atoms_batch = tf.cast(tf.shape(self.output)[0], self.tf_prec)
			self.total_loss, self.loss = self.loss_op(unrotated_norm_output, self.norm_labels)
			self.total_loss /= self.n_atoms_batch
			# barrier_function = -1000.0 * tf.log(tf.concat([self.gaussian_params + 0.9, tf.expand_dims(6.5 - self.gaussian_params[:,0], axis=-1), tf.expand_dims(1.75 - self.gaussian_params[:,1], axis=-1)], axis=1))
			# truncated_barrier_function = tf.reduce_sum(tf.where(tf.greater(barrier_function, 0.0), barrier_function, tf.zeros_like(barrier_function)))
			# gaussian_overlap_constraint = tf.square(0.001 / min_eigenval)
			self.rotation_constraint = tf.reduce_sum(tf.square(tf.clip_by_value(tf.gradients(self.output, rotation_params), -1, 1))) / self.n_atoms_batch
			loss_and_constraint = self.total_loss + self.rotation_constraint
			self.train_op = self.training(loss_and_constraint, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.sess.run(init)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			if self.profiling:
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
			return

	# def training(self, loss, learning_rate, momentum):
	# 	"""Sets up the training Ops.
	# 	Creates a summarizer to track the loss over time in TensorBoard.
	# 	Creates an optimizer and applies the gradients to all trainable variables.
	# 	The Op returned by this function is what must be passed to the
	# 	`sess.run()` call to cause the model to train.
	# 	Args:
	# 	loss: Loss tensor, from loss().
	# 	learning_rate: The learning rate to use for gradient descent.
	# 	Returns:
	# 	train_op: The Op for training.
	# 	"""
	# 	tf.summary.scalar(loss.op.name, loss)
	# 	optimizer = tf.train.AdamOptimizer(learning_rate)
	# 	grads_and_vars = optimizer.compute_gradients(loss)
	# 	capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
	# 	global_step = tf.Variable(0, name='global_step', trainable=False)
	# 	train_op = optimizer.apply_gradients(capped_grads_and_vars)
	# 	return train_op

	# def training_op(self, loss, learning_rate, momentum):
	# 	"""Sets up the training Ops.
	# 	Creates a summarizer to track the loss over time in TensorBoard.
	# 	Creates an optimizer and applies the gradients to all trainable variables.
	# 	The Op returned by this function is what must be passed to the
	# 	`sess.run()` call to cause the model to train.
	# 	Args:
	# 	loss: Loss tensor, from loss().
	# 	learning_rate: The learning rate to use for gradient descent.
	# 	Returns:
	# 	train_op: The Op for training.
	# 	"""
	# 	tf.summary.scalar(loss.op.name, loss)
	# 	optimizer = tf.train.AdamOptimizer(learning_rate)
	# 	global_step = tf.Variable(0, name='global_step', trainable=False)
	# 	gradients_vars = optimizer.compute_gradients(loss)
	# 	for gradient, var in gradients_vars:
	#
	# 	train_op = optimizer.minimize(loss, global_step=global_step)
	# 	return train_op

	def Clean(self):
		if (self.sess != None):
			self.sess.close()
		self.sess = None
		self.loss = None
		self.output = None
		self.total_loss = None
		self.train_op = None
		self.saver = None
		self.gradient = None
		self.summary_writer = None
		self.PreparedFor = 0
		self.summary_op = None
		self.activation_function = None
		self.atomic_embed_factors = None
		self.gaussian_params = None
		self.labels = None
		self.norm_output = None
		self.norm_labels = None
		self.norm_embedding = None
		self.labels_pl = None
		self.Zs_pl = None
		self.xyzs_pl = None
		self.output = None
		self.embedding = None
		self.n_atoms_batch = None
		self.rotation_constraint = None
		return

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def train_step(self,step):
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss, n_atoms_epoch = 0.0, 0.0
		rotation_loss = 0.0
		ministeps = int(Ncase_train/self.batch_size)
		for ministep in xrange(ministeps):
			batch_data = self.TData.GetTrainBatch(self.batch_size) #advances the case pointer in TData...
			feed_dict = self.fill_feed_dict(batch_data)
			if self.profiling:
				_, total_loss_value, loss_value, n_atoms_batch = self.sess.run([self.train_op, self.total_loss, self.loss,
						self.n_atoms_batch], feed_dict=feed_dict, options=self.options, run_metadata=self.run_metadata)
				fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
				chrome_trace = fetched_timeline.generate_chrome_trace_format()
				with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
					f.write(chrome_trace)
			else:
				_, total_loss_value, loss_value, n_atoms_batch, rotation_constraint = self.sess.run([self.train_op, self.total_loss,
						self.loss, self.n_atoms_batch, self.rotation_constraint], feed_dict=feed_dict)
			train_loss += total_loss_value
			n_atoms_epoch += n_atoms_batch
			rotation_loss += rotation_constraint
		train_loss /= ministeps
		rotation_loss /= ministeps
		duration = time.time() - start_time
		self.print_training(step, train_loss, n_atoms_epoch, duration, rotation_loss)
		return

	def test(self, step):
		print("testing...")
		Ncase_test = self.TData.NTest
		test_loss, n_atoms_epoch = 0.0, 0.0
		test_start_time = time.time()
		mean_test_error, std_dev_test_error = 0.0, 0.0
		test_epoch_labels, test_epoch_outputs = [], []
		rotation_loss = 0.0
		ministeps = int(Ncase_test/self.batch_size)
		for ministep in xrange(ministeps):
			batch_data=self.TData.GetTestBatch(self.batch_size)
			feed_dict = self.fill_feed_dict(batch_data)
			output, labels, total_loss_value, loss_value, n_atoms_batch, rotation_constraint, gaussian_params, atomic_embed_factors = self.sess.run([self.output,
					self.labels, self.total_loss, self.loss, self.n_atoms_batch, self.rotation_constraint, self.gaussian_params, self.atomic_embed_factors],  feed_dict=feed_dict)
			test_loss += total_loss_value
			n_atoms_epoch += n_atoms_batch
			rotation_loss += rotation_constraint
			test_epoch_labels.append(labels)
			test_epoch_outputs.append(output)
		test_loss /= ministeps
		rotation_loss /= ministeps
		test_epoch_labels = np.concatenate(test_epoch_labels)
		test_epoch_outputs = np.concatenate(test_epoch_outputs)
		test_epoch_errors = test_epoch_labels - test_epoch_outputs
		duration = time.time() - test_start_time
		for i in [random.randint(0, self.batch_size) for j in xrange(20)]:
			LOGGER.info("Label: %s  Output: %s", test_epoch_labels[i], test_epoch_outputs[i])
		LOGGER.info("MAE: %f", np.mean(np.abs(test_epoch_errors)))
		LOGGER.info("MSE: %f", np.mean(test_epoch_errors))
		LOGGER.info("RMSE: %f", np.sqrt(np.mean(np.square(test_epoch_errors))))
		LOGGER.info("Std. Dev.: %f", np.std(test_epoch_errors))
		# LOGGER.info("Gaussian paramaters: %s", gaussian_params)
		# LOGGER.info("Atomic embedding factors: %s", atomic_embed_factors)
		self.print_testing(step, test_loss, n_atoms_epoch, duration, rotation_loss)
		return test_loss

	def print_training(self, step, loss, n_cases, duration, rotation_loss):
		LOGGER.info("step: %7d  duration: %.5f train loss: %.10f rotation loss: %0.10f", step, duration, loss, rotation_loss)
		return

	def print_testing(self, step, loss, n_cases, duration, rotation_loss):
		LOGGER.info("step: %7d  duration: %.5f test loss: %.10f rotation loss: %0.10f", step, duration, loss, rotation_loss)
		return

class Instance_del_fc_sqdiff(Instance_fc_sqdiff):
	def __init__(self, TData_, ele_=1, Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		self.NetType = "del_fc_sqdiff"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = PARAMS["networks_directory"]+self.name

	def inference(self, inputs, bleep, bloop, blop):
		"""Build the MNIST model up to where it may be used for inference.
		Args:
		images: Images placeholder, from inputs().
		hidden1_units: Size of the first hidden layer.
		hidden2_units: Size of the second hidden layer.
		Returns:
		softmax_linear: Output tensor with the computed logits.
		"""
		hidden1_units = PARAMS["hidden1"]
		hidden2_units = PARAMS["hidden2"]
		hidden3_units = PARAMS["hidden3"]
		LOGGER.debug("hidden1_units: "+str(hidden1_units))
		LOGGER.debug("hidden2_units: "+str(hidden2_units))
		LOGGER.debug("hidden3_units: "+str(hidden3_units))
		# Hidden 1
		with tf.name_scope('hidden1'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=list(self.inshape)+[hidden1_units], var_stddev= 0.4 / math.sqrt(float(self.inshape[0])), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden1_units], dtype=self.tf_prec), name='biases')
			hidden1 = tf.nn.relu(tf.matmul(inputs[:-3], weights) + biases)
			#tf.summary.scalar('min/' + weights.name, tf.reduce_min(weights))
			#tf.summary.histogram(weights.name, weights)
		# Hidden 2
		with tf.name_scope('hidden2'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev= 0.4 / math.sqrt(float(hidden1_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden2_units], dtype=self.tf_prec),name='biases')
			hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

		# Hidden 3
		with tf.name_scope('hidden3'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev= 0.4 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden3_units], dtype=self.tf_prec),name='biases')
			hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
		#Delta Layer
		with tf.name_scope('delta_layer'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden3_units]+ list(2*self.outshape), var_stddev= 0.4 / math.sqrt(float(hidden3_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros(self.outshape, dtype=self.tf_prec), name='biases')
			delta = tf.matmul(hidden3, weights) + biases
		# Linear
		with tf.name_scope('regression_linear'):
			delta_out = tf.multiply(tf.slice(delta,[self.outshape],[self.outshape]),inputs[-3:])
			output = tf.add(tf.slice(delta,[0],[self.outshape]),delta_out)
		return output

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

class Instance_conv2d_sqdiff(Instance):
	def __init__(self, TData_, ele_ = 1 , Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		self.NetType = "conv2d_sqdiff"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = PARAMS["networks_directory"]+self.name

	def placeholder_inputs(self, batch_size):
		"""Generate placeholder variables to represent the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded data in the .run() loop, below.
		Args:
		batch_size: The batch size will be baked into both placeholders.
		Returns:
		embeds_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
		"""
		# Note that the shapes of the placeholders match the shapes of the full
		# image and label tensors, except the first dimension is now batch_size
		# rather than the full size of the train or test data sets.
		inputs_pl = tf.placeholder(self.tf_prec, shape=tuple([batch_size,self.inshape]))
		outputs_pl = tf.placeholder(self.tf_prec, shape=tuple([batch_size, self.outshape]))
		return inputs_pl, outputs_pl

	def _weight_variable(self, name, shape):
		return tf.get_variable(name, shape, self.tf_prec, tf.truncated_normal_initializer(stddev=0.01))

	def _bias_variable(self, name, shape):
		return tf.get_variable(name, shape, self.tf_prec, tf.constant_initializer(0.01, dtype=self.tf_prec))

	def conv2d(self, x, W, b, strides=1):
		"""
		2D Convolution wrapper with bias and relu activation
		"""
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return tf.nn.relu(x)

	def inference(self, input):
		FC_SIZE = 512
		with tf.variable_scope('conv1') as scope:
			in_filters = 1
			out_filters = 8
			kernel = self._weight_variable('weights', [2, 2, 2, in_filters, out_filters])
			conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME') # third arg. is the strides case,xstride,ystride,zstride,channel stride
			biases = self._bias_variable('biases', [out_filters])
			bias = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(bias, name=scope.name)
			prev_layer = conv1
			in_filters = out_filters

		# pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
		#norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')
		#prev_layer = norm1

		with tf.variable_scope('conv2') as scope:
			out_filters = 16
			kernel = self._weight_variable('weights', [2, 2, 2, in_filters, out_filters])
			conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
			biases = self._bias_variable('biases', [out_filters])
			bias = tf.nn.bias_add(conv, biases)
			conv2 = tf.nn.relu(bias, name=scope.name)
			prev_layer = conv2
			in_filters = out_filters

		# normalize prev_layer here
		# prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

		with tf.variable_scope('local1') as scope:
			dim = np.prod(prev_layer.get_shape().as_list()[1:])
			prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
			weights = self._weight_variable('weights', [dim, FC_SIZE])
			biases = self._bias_variable('biases', [FC_SIZE])
			local1 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
			prev_layer = local1

		with tf.variable_scope('local2') as scope:
			dim = np.prod(prev_layer.get_shape().as_list()[1:])
			prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
			weights = self._weight_variable('weights', [dim, FC_SIZE])
			biases = self._bias_variable('biases', [FC_SIZE])
			local2 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
			prev_layer = local2

		with tf.variable_scope('regression_linear') as scope:
			dim = np.prod(prev_layer.get_shape().as_list()[1:])
			weights = self._weight_variable('weights', [dim]+list(self.outshape))
			biases = self._bias_variable('biases', self.outshape)
			output = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)
		return output

	def evaluate(self, eval_input):
		# Check sanity of input
		Instance.evaluate(self, eval_input)
		eval_input_ = eval_input
		if (self.PreparedFor>eval_input.shape[0]):
			eval_input_ =np.copy(eval_input)
			eval_input_.resize(([self.PreparedFor]+self.inshape))
		# pad with zeros
		eval_labels = np.zeros(tuple([self.PreparedFor]+list(self.outshape)))  # dummy labels
		batch_data = self.PrepareData([eval_input_, eval_labels])
		#embeds_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.embeds_placeholder, self.labels_placeholder)
		tmp = np.array(self.sess.run([self.output], feed_dict=feed_dict))
		if (not np.all(np.isfinite(tmp))):
			LOGGER.error("TFsession returned garbage")
			LOGGER.error("TFInputs"+str(eval_input)) #If it's still a problem here use tf.Print version of the graph.
		if (self.PreparedFor>eval_input.shape[0]):
			return tmp[:eval_input.shape[0]]
		return tmp

	def Save(self):
		self.summary_op =None
		self.summary_writer=None
		Instance.Save(self)
		return

	def loss_op(self, output, labels):
		diff  = tf.slice(tf.sub(output, labels),[0,self.outshape[0]-3],[-1,-1])
		# this only compares direct displacement predictions.
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def train_step(self,step):
		Ncase_train = self.TData.NTrainCasesInScratch()
		start_time = time.time()
		train_loss =  0.0
		total_correct = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data=self.PrepareData(self.TData.GetTrainBatch(self.element,  self.batch_size)) #advances the case pointer in TData...
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			_, total_loss_value, loss_value = self.sess.run([self.train_op, self.total_loss, self.loss], feed_dict=feed_dict)
			train_loss = train_loss + loss_value
		duration = time.time() - start_time
		#self.print_training(step, train_loss, total_correct, Ncase_train, duration)
		self.print_training(step, train_loss, Ncase_train, duration)
		return

	def test(self, step):
		Ncase_test = self.TData.NTestCasesInScratch()
		test_loss =  0.0
		test_start_time = time.time()
		#for ministep in range (0, int(Ncase_test/self.batch_size)):
		batch_data=self.PrepareData(self.TData.GetTestBatch(self.element,  self.batch_size))#, ministep)
		feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
		preds, total_loss_value, loss_value  = self.sess.run([self.output, self.total_loss,  self.loss],  feed_dict=feed_dict)
		self.TData.EvaluateTestBatch(batch_data[1],preds, self.tformer)
		test_loss = test_loss + loss_value
		duration = time.time() - test_start_time
		LOGGER.info("testing...")
		self.print_training(step, test_loss,  Ncase_test, duration)
		return test_loss, feed_dict

	def PrepareData(self, batch_data):

		#for i in range(self.batch_size):
		#	ds=GRIDS.Rasterize(batch_data[0][i])
		#	GridstoRaw(ds, GRIDS.NPts, "Inp"+str(i))

		if (batch_data[0].shape[0]==self.batch_size):
			batch_data=[batch_data[0].reshape(batch_data[0].shape[0],GRIDS.NGau,GRIDS.NGau,GRIDS.NGau,1), batch_data[1]]
		elif (batch_data[0].shape[0] < self.batch_size):
			LOGGER.info("Resizing... ")
			batch_data=[batch_data[0].resize(self.batch_size,GRIDS.NGau,GRIDS.NGau,GRIDS.NGau,1), batch_data[1].resize((self.batch_size,  batch_data[1].shape[1]))]
#			batch_data=[batch_data[0], batch_data[1].reshape((batch_data[1].shape[0],1))]
#			tmp_input = np.copy(batch_data[0])
#			tmp_output = np.copy(batch_data[1])
#			tmp_input.resize((self.batch_size,  batch_data[0].shape[1]))
#			tmp_output.resize((self.batch_size,  batch_data[1].shape[1]))
#			batch_data=[ tmp_input, tmp_output]
		return batch_data

class Instance_3dconv_sqdiff(Instance):
	''' Let's see if a 3d-convolutional network improves the learning rate on the Gaussian grids. '''
	def __init__(self, TData_, ele_ = 1 , Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		self.NetType = "3conv_sqdiff"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = PARAMS["networks_directory"]+self.name

	def placeholder_inputs(self, batch_size):
		"""Generate placeholder variables to represent the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded data in the .run() loop, below.
		Args:
		batch_size: The batch size will be baked into both placeholders.
		Returns:
		embeds_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
		"""
		# Note that the shapes of the placeholders match the shapes of the full
		# image and label tensors, except the first dimension is now batch_size
		# rather than the full size of the train or test data sets.
		if (self.inshape[0]!=GRIDS.NGau3):
			print("Bad inputs... ", self.inshape)
			raise Exception("Nonsquare")
		inputs_pl = tf.placeholder(self.tf_prec, shape=tuple([batch_size,GRIDS.NGau,GRIDS.NGau,GRIDS.NGau,1]))
		outputs_pl = tf.placeholder(self.tf_prec, shape=tuple([batch_size]+list(self.outshape)))
		return inputs_pl, outputs_pl

	def _weight_variable(self, name, shape):
		return tf.get_variable(name, shape, self.tf_prec, tf.truncated_normal_initializer(stddev=0.01))

	def _bias_variable(self, name, shape):
		return tf.get_variable(name, shape, self.tf_prec, tf.constant_initializer(0.01, dtype=self.tf_prec))

	def inference(self, input):
		FC_SIZE = 512
		with tf.variable_scope('conv1') as scope:
			in_filters = 1
			out_filters = 8
			kernel = self._weight_variable('weights', [2, 2, 2, in_filters, out_filters])
			conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME') # third arg. is the strides case,xstride,ystride,zstride,channel stride
			biases = self._bias_variable('biases', [out_filters])
			bias = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(bias, name=scope.name)
			prev_layer = conv1
			in_filters = out_filters

		# pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
		#norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')
		#prev_layer = norm1

		with tf.variable_scope('conv2') as scope:
			out_filters = 16
			kernel = self._weight_variable('weights', [2, 2, 2, in_filters, out_filters])
			conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
			biases = self._bias_variable('biases', [out_filters])
			bias = tf.nn.bias_add(conv, biases)
			conv2 = tf.nn.relu(bias, name=scope.name)
			prev_layer = conv2
			in_filters = out_filters

		# normalize prev_layer here
		# prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

		with tf.variable_scope('local1') as scope:
			dim = np.prod(prev_layer.get_shape().as_list()[1:])
			prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
			weights = self._weight_variable('weights', [dim, FC_SIZE])
			biases = self._bias_variable('biases', [FC_SIZE])
			local1 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
			prev_layer = local1

		with tf.variable_scope('local2') as scope:
			dim = np.prod(prev_layer.get_shape().as_list()[1:])
			prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
			weights = self._weight_variable('weights', [dim, FC_SIZE])
			biases = self._bias_variable('biases', [FC_SIZE])
			local2 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
			prev_layer = local2

		with tf.variable_scope('regression_linear') as scope:
			dim = np.prod(prev_layer.get_shape().as_list()[1:])
			weights = self._weight_variable('weights', [dim]+list(self.outshape))
			biases = self._bias_variable('biases', self.outshape)
			output = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)
		return output

	def evaluate(self, eval_input):
		# Check sanity of input
		Instance.evaluate(self, eval_input)
		eval_input_ = eval_input
		if (self.PreparedFor>eval_input.shape[0]):
			eval_input_ =np.copy(eval_input)
			eval_input_.resize(([self.PreparedFor]+self.inshape))
		# pad with zeros
		eval_labels = np.zeros(tuple([self.PreparedFor]+list(self.outshape)))  # dummy labels
		batch_data = self.PrepareData([eval_input_, eval_labels])
		#embeds_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.embeds_placeholder, self.labels_placeholder)
		tmp = np.array(self.sess.run([self.output], feed_dict=feed_dict))
		if (not np.all(np.isfinite(tmp))):
			LOGGER.error("TFsession returned garbage")
			LOGGER.error("TFInputs"+str(eval_input)) #If it's still a problem here use tf.Print version of the graph.
		if (self.PreparedFor>eval_input.shape[0]):
			return tmp[:eval_input.shape[0]]
		return tmp

	def Save(self):
		self.summary_op =None
		self.summary_writer=None
		Instance.Save(self)
		return

	def loss_op(self, output, labels):
		diff  = tf.slice(tf.sub(output, labels),[0,self.outshape[0]-3],[-1,-1])
		# this only compares direct displacement predictions.
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def train_step(self,step):
		Ncase_train = self.TData.NTrainCasesInScratch()
		start_time = time.time()
		train_loss =  0.0
		total_correct = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data=self.PrepareData(self.TData.GetTrainBatch(self.element,  self.batch_size)) #advances the case pointer in TData...
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			_, total_loss_value, loss_value = self.sess.run([self.train_op, self.total_loss, self.loss], feed_dict=feed_dict)
			train_loss = train_loss + loss_value
		duration = time.time() - start_time
		#self.print_training(step, train_loss, total_correct, Ncase_train, duration)
		self.print_training(step, train_loss, Ncase_train, duration)
		return

	def test(self, step):
		Ncase_test = self.TData.NTestCasesInScratch()
		test_loss =  0.0
		test_start_time = time.time()
		#for ministep in range (0, int(Ncase_test/self.batch_size)):
		batch_data=self.PrepareData(self.TData.GetTestBatch(self.element,  self.batch_size))#, ministep)
		feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
		preds, total_loss_value, loss_value  = self.sess.run([self.output, self.total_loss,  self.loss],  feed_dict=feed_dict)
		self.TData.EvaluateTestBatch(batch_data[1],preds, self.tformer)
		test_loss = test_loss + loss_value
		duration = time.time() - test_start_time
		LOGGER.info("testing...")
		self.print_training(step, test_loss,  Ncase_test, duration)
		return test_loss, feed_dict

	def PrepareData(self, batch_data):

		#for i in range(self.batch_size):
		#	ds=GRIDS.Rasterize(batch_data[0][i])
		#	GridstoRaw(ds, GRIDS.NPts, "Inp"+str(i))

		if (batch_data[0].shape[0]==self.batch_size):
			batch_data=[batch_data[0].reshape(batch_data[0].shape[0],GRIDS.NGau,GRIDS.NGau,GRIDS.NGau,1), batch_data[1]]
		elif (batch_data[0].shape[0] < self.batch_size):
			LOGGER.info("Resizing... ")
			batch_data=[batch_data[0].resize(self.batch_size,GRIDS.NGau,GRIDS.NGau,GRIDS.NGau,1), batch_data[1].resize((self.batch_size,  batch_data[1].shape[1]))]
#			batch_data=[batch_data[0], batch_data[1].reshape((batch_data[1].shape[0],1))]
#			tmp_input = np.copy(batch_data[0])
#			tmp_output = np.copy(batch_data[1])
#			tmp_input.resize((self.batch_size,  batch_data[0].shape[1]))
#			tmp_output.resize((self.batch_size,  batch_data[1].shape[1]))
#			batch_data=[ tmp_input, tmp_output]
		return batch_data


class Instance_KRR(Instance):
	def __init__(self, TData_, ele_ = 1 , Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		self.NetType = "KRR"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = PARAMS["networks_directory"]+self.name
		self.krr = None
		return

	def evaluate(self, eval_input):
		return self.krr.predict(eval_input)

	def Save(self):
		self.summary_op =None
		self.summary_writer=None
		return

	def train(self,n_step):
		from sklearn.kernel_ridge import KernelRidge
		self.krr = KernelRidge(alpha=0.001, kernel='rbf')
		# Here we should use as much data as the kernel method can actually take.
		# probly on the order of 100k cases.
		ti,to = self.TData.GetTrainBatch(self.element,  10000)
		self.krr.fit(ti,to)
		self.test(0)
		return

	def test(self, step):
		Ncase_test = self.TData.NTestCasesInScratch()
		test_loss =  0.0
		ti,to = self.TData.GetTestBatch(self.element,  self.batch_size)
		preds  = self.krr.predict(ti)
		self.TData.EvaluateTestBatch(to,preds, self.tformer)
		return None, None

	def basis_opt_run(self):
		from sklearn.kernel_ridge import KernelRidge
		self.krr = KernelRidge(alpha=0.001, kernel='rbf')
		# Here we should use as much data as the kernel method can actually take.
		# probly on the order of 100k cases.
		ti,to = self.TData.GetTrainBatch(self.element,  10000)
		self.krr.fit(ti,to)
		Ncase_test = self.TData.NTestCasesInScratch()
		test_loss =  0.0
		ti,to = self.TData.GetTestBatch(self.element,  10000)
		preds = self.krr.predict(ti)
		return self.TData.EvaluateTestBatch(to,preds, self.tformer, Opt=True)

	def PrepareData(self, batch_data):
		raise Exception("NYI")
		return
