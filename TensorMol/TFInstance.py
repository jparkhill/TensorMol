from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from TensorMol.TensorData import *
import numpy as np
import math, pickle
import time, os, sys
import os.path
if (HAS_TF):
	import tensorflow as tf

class Instance:
	"""
	Manages a persistent training network instance
	"""
	def __init__(self, TData_, ele_ = 1 , Name_=None):
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
		self.summary_writer = None
		# The parameters below belong to tensorflow and its graph
		# all tensorflow variables cannot be pickled they are populated by Prepare
		self.PreparedFor=0

		self.path='./networks/'
		self.normalize = False
		if (Name_ !=  None):
			self.name = Name_
			#self.QueryAvailable() # Should be a sanity check on the data files.
			self.Load() # Network still cannot be used until it is prepared.
			print("raised network: ", self.train_dir)
			return

		self.element = ele_
		self.TData = TData_
		if (not os.path.isdir(self.path)):
			os.mkdir(self.path)
		self.chk_file = ''

		self.learning_rate = PARAMS["learning_rate"]
		self.momentum = PARAMS["momentum"]
		self.max_steps = PARAMS["max_steps"]
		self.batch_size = PARAMS["batch_size"]

		print("self.learning_rate", self.learning_rate)
		print("self.batch_size", self.batch_size)

		self.NetType = "None"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = './networks/'+self.name
		if (ele_ != 0):
			self.TData.LoadElementToScratch(ele_)
			self.TData.PrintStatus()
			if (self.TData.dig.name=="SymFunc"):
				# Kun: NOOO please keep it homogeneous! LoadElementToScratch() should do this.
				self.TData.NormalizeInputs(ele_)  # let me just normolize it here for sym functions...needs a flag in future
				self.normalize=True
			self.inshape = self.TData.dig.eshape
			self.outshape = self.TData.dig.lshape
		return

	def __del__(self):
		if (self.sess != None):
			self.sess.close()
		self.Clean()

	def evaluate(self, eval_input):
		# Check sanity of input
		if (self.normalize):
			eval_input=self.TData.ApplyNormalize(eval_input, self.element)
		if (not np.all(np.isfinite(eval_input))):
			print("WTF, you trying to feed me, garbage?")
			raise Exception("bad digest.")
		if (self.PreparedFor<eval_input.shape[0]):
			self.Prepare(eval_input,eval_input.shape[0])
		return

	#Seems like train_prepare is used instead of this, is this function deprecated?
	def Prepare(self, eval_input, Ncase=1250):
		self.Clean()
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:0'):
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
			self.output = self.inference(self.embeds_placeholder)
			self.saver = tf.train.Saver()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
			metafiles = [x for x in os.listdir(self.train_dir) if (x.count('meta')>0)]
			if (len(metafiles)>0):
				most_recent_meta_file=metafiles[0]
				print("Restoring training from Meta file: ",most_recent_meta_file)
				self.sess = tf.Session()
				self.saver = tf.train.import_meta_graph(self.train_dir+"/"+most_recent_meta_file)
				self.saver.restore(sess, tf.train.latest_checkpoint(self.train_dir))
				# self.saver.restore(self.sess, self.train_dir+'/'+most_recent_chk_file)
		self.PreparedFor = Ncase
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
		return

	def SaveAndClose(self):
		print("Saving TFInstance...")
		self.save_chk(99999)
		if (self.TData!=None):
			self.TData.CleanScratch()
		self.Clean()
		#print("Going to pickle...\n",[(attr,type(ins)) for attr,ins in self.__dict__.items()])
		f=open(self.path+self.name+".tfn","wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
		return

	def variable_summaries(var):
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

	# one of these two routines need to be removed I think. -JAP
	def save_chk(self,  step, feed_dict=None):  # this can be included in the Instance
		cmd="rm  "+self.train_dir+"/"+self.name+"-chk-*"
		os.system(cmd)
		checkpoint_file_mini = os.path.join(self.train_dir,self.name+'-chk-'+str(step))
		print("Saving Checkpoint file, ",checkpoint_file_mini)
		self.saver.save(self.sess, checkpoint_file_mini)
		return

#this isn't really the correct way to load()
# only the local class members (not any TF objects should be unpickled.)
	def Load(self):
		print ("Unpickling TFInstance...")
		f = open(self.path+self.name+".tfn","rb")
		tmp=pickle.load(f)
		# This is just to use an updated version of evaluate and should be removed after I re-train...
		tmp.pop('evaluate',None)
		self.Clean()
		# All this shit should be deleteable after re-training.
		self.__dict__.update(tmp)
		f.close()
		chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
		if (len(chkfiles)>0):
			self.chk_file = chkfiles[0]
		else:
			print("Network not found... Traindir:", self.train_dir)
			print("Traindir contents: ", os.listdir(self.train_dir))
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
		var = tf.Variable(tf.truncated_normal(var_shape, stddev=var_stddev), name=var_name)
		if var_wd is not None:
			weight_decay = tf.mul(tf.nn.l2_loss(var), var_wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

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
			print("I was fed shit")
			raise Exception("DontEatShit")
		if (not np.all(np.isfinite(batch_data[1]))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict = {embeds_pl: batch_data[0], labels_pl: batch_data[1],}
		return feed_dict

	def inference(self, images):
		"""Build the MNIST model up to where it may be used for inference.
		Args:
		images: Images placeholder, from inputs().
		hidden1_units: Size of the first hidden layer.
		hidden2_units: Size of the second hidden layer.
		Returns:
		softmax_linear: Output tensor with the computed logits.
		"""
		# Hidden 1
		hidden1_units = PARAMS["hidden1"]
		hidden2_units = PARAMS["hidden2"]
		hidden3_units = PARAMS["hidden3"]
		with tf.name_scope('hidden1'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=list(self.inshape)+[hidden1_units], var_stddev= 0.4 / math.sqrt(float(self.inshape[0])), var_wd= 0.00)
				biases = tf.Variable(tf.zeros([hidden1_units]),
				name='biases')
				hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
				#tf.summary.scalar('min/' + weights.name, tf.reduce_min(weights))
				#tf.summary.histogram(weights.name, weights)
		# Hidden 2
		with tf.name_scope('hidden2'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev= 0.4 / math.sqrt(float(hidden1_units)), var_wd= 0.00)
				biases = tf.Variable(tf.zeros([hidden2_units]),
				name='biases')
				hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

		# Hidden 3
		with tf.name_scope('hidden3'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev= 0.4 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
				biases = tf.Variable(tf.zeros([hidden3_units]),
				name='biases')
				hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

		# Linear
		with tf.name_scope('regression_linear'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden3_units]+ list(self.outshape), var_stddev= 0.4 / math.sqrt(float(hidden3_units)), var_wd= 0.00)
				biases = tf.Variable(tf.zeros(self.outshape),
				name='biases')
				output = tf.matmul(hidden3, weights) + biases
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
		#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def train(self, mxsteps, continue_training= False):
		self.TData.LoadElementToScratch(self.element)
		self.train_prepare(continue_training)
		test_freq = 40
		mini_test_loss = 100000000 # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss, feed_dict = self.test(step)
				if (test_loss < mini_test_loss):
					mini_test_loss = test_loss
					self.save_chk(step,feed_dict)
		self.SaveAndClose()
		return

	def train_step(self,step):
		raise Exception("Cannot Train base...")
		return

	def train_prepare(self,  continue_training =False):
		"""Train for a number of steps."""
		with tf.Graph().as_default():
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.embeds_placeholder)
			self.total_loss, self.loss = self.loss_op(self.output, self.labels_placeholder)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
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
			print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(denom*self.batch_size)))
		else:
			print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  test loss: ", "%.10f"%(float(loss)/(denom*self.batch_size)))
		return

class Instance_fc_classify(Instance):
	def __init__(self, TData_, ele_ = 1 , Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		self.hidden1 = 500
		self.hidden2 = 500
		self.hidden3 = 500
		self.NetType = "fc_classify"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = './networks/'+self.name
		self.prob = None
#		self.inshape = self.TData.scratch_inputs.shape[1]
		self.correct = None
		self.summary_op =None
		self.summary_writer=None

	def evaluation(self, output, labels):
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
			print("TFsession returned garbage")
			print("TFInputs",eval_input) #If it's still a problem here use tf.Print version of the graph.
		if (self.PreparedFor>eval_input.shape[0]):
			return tmp[:eval_input.shape[0]]
		return tmp

	def Prepare(self, eval_input, Ncase=1250):
		Instance.Prepare(self)
		print("Preparing a ",self.NetType,"Instance")
		self.prob = None
		self.correct = None
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:0'):
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
			self.output = self.inference(self.embeds_placeholder)
			self.correct = self.evaluation(self.output, self.labels_placeholder)
			self.prob = self.justpreds(self.output)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
			chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
			if (len(chkfiles)>0):
				most_recent_chk_file=chkfiles[0]
				print("Restoring training from Checkpoint: ",most_recent_chk_file)
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
		inputs_pl = tf.placeholder(tf.float32, shape=(batch_size,self.inshape)) # JAP : Careful about the shapes... should be flat for now.
		outputs_pl = tf.placeholder(tf.float32, shape=(batch_size))
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
			print("testing...")
			self.print_training(step, test_loss, test_correct, Ncase_test, duration)
		return test_loss, feed_dict

class Instance_fc_sqdiff(Instance):
	def __init__(self, TData_, ele_ = 1 , Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		# 256*512*512 gives [-0.01215208 -0.0064384   0.00562539] Average output (direct) [-0.01867021 -0.00484998  0.01150864] after 2k epochs.
		self.hidden1 = 512
		self.hidden2 = 512
		self.hidden3 = 512
		self.NetType = "fc_sqdiff"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = './networks/'+self.name
		self.summary_op =None
		self.summary_writer=None

	def evaluate(self, eval_input):
		# Check sanity of input
		Instance.evaluate(self, eval_input)
		eval_input_ = eval_input
		if (self.PreparedFor>eval_input.shape[0]):
			eval_input_ =np.copy(eval_input)
			eval_input_.resize(([self.PreparedFor]+self.inshape))
			# pad with zeros
		eval_labels = np.zeros(tuple([self.PreparedFor]+list(self.outshape)))  # dummy labels
		batch_data = [eval_input_, eval_labels]
		#embeds_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.embeds_placeholder, self.labels_placeholder)
		tmp = np.array(self.sess.run([self.output], feed_dict=feed_dict))
		if (not np.all(np.isfinite(tmp))):
			print("TFsession returned garbage")
			print("TFInputs",eval_input) #If it's still a problem here use tf.Print version of the graph.
		if (self.PreparedFor>eval_input.shape[0]):
			return tmp[:eval_input.shape[0]]
		return tmp

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
		inputs_pl = tf.placeholder(tf.float32, shape=tuple([batch_size]+list(self.inshape)))
		outputs_pl = tf.placeholder(tf.float32, shape=tuple([batch_size]+list(self.outshape)))
		return inputs_pl, outputs_pl

	def loss_op(self, output, labels):
		diff  = tf.sub(output, labels)
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
		self.TData.EvaluateTestBatch(batch_data[1],preds)
		test_loss = test_loss + loss_value
		duration = time.time() - test_start_time
		print("testing...")
		self.print_training(step, test_loss,  Ncase_test, duration)
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

class Instance_3dconv_sqdiff(Instance):
	''' Let's see if a 3d-convolutional network improves the learning rate on the Gaussian grids. '''
	def __init__(self, TData_, ele_ = 1 , Name_=None):
		Instance.__init__(self, TData_, ele_, Name_)
		self.NetType = "3conv_sqdiff"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
		self.train_dir = './networks/'+self.name
		self.summary_op =None
		self.summary_writer=None

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
		inputs_pl = tf.placeholder(tf.float32, shape=tuple([batch_size,GRIDS.NGau,GRIDS.NGau,GRIDS.NGau,1]))
		outputs_pl = tf.placeholder(tf.float32, shape=tuple([batch_size]+list(self.outshape)))
		return inputs_pl, outputs_pl

	def _weight_variable(self, name, shape):
		return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.01))

	def _bias_variable(self, name, shape):
		return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.01, dtype=tf.float32))

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
			print("TFsession returned garbage")
			print("TFInputs",eval_input) #If it's still a problem here use tf.Print version of the graph.
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
		self.TData.EvaluateTestBatch(batch_data[1],preds)
		test_loss = test_loss + loss_value
		duration = time.time() - test_start_time
		print("testing...")
		self.print_training(step, test_loss,  Ncase_test, duration)
		return test_loss, feed_dict

	def PrepareData(self, batch_data):

		#for i in range(self.batch_size):
		#	ds=GRIDS.Rasterize(batch_data[0][i])
		#	GridstoRaw(ds, GRIDS.NPts, "Inp"+str(i))

		if (batch_data[0].shape[0]==self.batch_size):
			batch_data=[batch_data[0].reshape(batch_data[0].shape[0],GRIDS.NGau,GRIDS.NGau,GRIDS.NGau,1), batch_data[1]]
		elif (batch_data[0].shape[0] < self.batch_size):
			print("Resizing... ")
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
		self.train_dir = './networks/'+self.name
		self.summary_op =None
		self.summary_writer=None
		self.krr = None
		from sklearn.kernel_ridge import KernelRidge
		return

	def evaluate(self, eval_input):
		return self.TData.dig.unscld(self.krr.predict(eval_input))

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
		test_start_time = time.time()
		ti,to = self.TData.GetTestBatch(self.element,  self.batch_size)
		preds  = self.krr.predict(ti)
		self.TData.EvaluateTestBatch(to,preds)
		return None, None

	def PrepareData(self, batch_data):
		raise Exception("NYI")
		return
