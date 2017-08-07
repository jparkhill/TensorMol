from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TensorMol.TFInstance import *
from TensorMol.TensorMolData import *
import numpy as np
import cPickle as pickle
import math
import time
import os.path
if (HAS_TF):
	import tensorflow as tf
import os
import sys

#
# These work Moleculewise the versions without the mol prefix work atomwise.
# but otherwise the behavior of these is the same as TFInstance etc.
#

class MolInstance(Instance):
	def __init__(self, TData_,  Name_=None, Trainable_=True):
		Instance.__init__(self, TData_, 0, Name_)
		self.AssignActivation()
		#self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		self.name = "Mol_"+self.TData.name+"_ANI1_Sym_Direct_"+str(self.TData.order)+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.Trainable = Trainable_
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		self.tformer.Print()
		self.TData.PrintStatus()
		self.inshape =  self.TData.dig.eshape  # use the flatted version
		self.outshape = self.TData.dig.lshape    # use the flatted version
		LOGGER.info("MolInstance.inshape %s MolInstance.outshape %s", str(self.inshape) , str(self.outshape))
		self.batch_size = PARAMS["batch_size"]
		self.summary_op =None
		self.summary_writer=None
		return

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
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev= 1.0 / math.sqrt(float(self.inshape[0])), var_wd= 0.00)
					biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
					hiddens.append(self.activation_function(tf.matmul(inputs, weights) + biases))
					tf.scalar_summary('min/' + weights.name, tf.reduce_min(weights))
					tf.histogram_summary(weights.name, weights)
			else:
				with tf.name_scope('hidden'+str(i+1)):
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev= 1.0 / math.sqrt(float(self.HiddenLayers[i-1])), var_wd= 0.00)
					biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec),name='biases')
					hiddens.append(self.activation_function(tf.matmul(hiddens[-1], weights) + biases))
		with tf.name_scope('regression_linear'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], self.outshape], var_stddev= 1.0 / math.sqrt(float(self.HiddenLayers[-1])), var_wd= 0.00)
			biases = tf.Variable(tf.zeros(self.outshape, dtype=self.tf_prec), name='biases')
			output = tf.matmul(hiddens[-1], weights) + biases
		return output

	def train(self, mxsteps, continue_training= False):
		LOGGER.info("running the TFMolInstance.train()")
		self.TrainPrepare(continue_training)
		test_freq = PARAMS["test_freq"]
		mini_test_loss = float('inf') # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					self.save_chk(step)
		self.SaveAndClose()
		return

	def train_step(self,step):
		""" I don't think the base class should be train-able. Remove? JAP """
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		total_correct = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data=self.TData.GetTrainBatch( self.batch_size) #advances the case pointer in TData...
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			_, total_loss_value, loss_value, prob_value, correct_num  = self.sess.run([self.train_op, self.total_loss, self.loss, self.prob, self.correct], feed_dict=feed_dict)
			train_loss = train_loss + loss_value
			total_correct = total_correct + correct_num
		duration = time.time() - start_time
		self.print_training(step, train_loss, total_correct, Ncase_train, duration)
		return

	def save_chk(self, step):  # We need to merge this with the one in TFInstance
		self.chk_file = os.path.join(self.train_dir,self.name+'-chk-'+str(step))
		LOGGER.info("Saving Checkpoint file, in the TFMoInstance", self.chk_file)
		self.saver.save(self.sess,  self.chk_file)
		return

	def Load(self):
		print ("Unpickling TFInstance...")
		f = open(self.path+self.name+".tfn","rb")
		import TensorMol.PickleTM
		tmp = TensorMol.PickleTM.UnPickleTM(f)
		self.Clean()
		self.__dict__.update(tmp)
		f.close()
		print("self.chk_file:", self.chk_file)
		return

	def SaveAndClose(self):
		if (self.TData!=None):
			self.TData.CleanScratch()
		print("Saving TFInstance...")
		self.Clean()
		#print("Going to pickle...\n",[(attr,type(ins)) for attr,ins in self.__dict__.items()])
		f=open(self.path+self.name+".tfn","wb")
		pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		return

class MolInstance_fc_classify(MolInstance):
	def __init__(self, TData_,  Name_=None, Trainable_=True):
		"""
		Translation of the outputs to meaningful numbers is handled by the digester and Tensordata
		"""
		self.NetType = "fc_classify"
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		LOGGER.debug("Instance.__init__: "+self.name)
		self.train_dir = './networks/'+self.name
		self.prob = None
		self.correct = None

	def n_correct(self, output, labels):
		"""
		This should average over the classifier output.
		"""
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
		MolInstance.evaluate(self, eval_input)
		eval_input_ = eval_input
		if (self.PreparedFor>eval_input.shape[0]):
			eval_input_ =np.copy(eval_input)
			eval_input_.resize((self.PreparedFor,eval_input.shape[1]))
			# pad with zeros
		eval_labels = np.zeros(self.PreparedFor)  # dummy labels
		batch_data = [eval_input_, eval_labels]
		#images_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.embeds_placeholder,self.labels_placeholder)
		tmp = (np.array(self.sess.run([self.prob], feed_dict=feed_dict))[0,:eval_input.shape[0],1])
		if (not np.all(np.isfinite(tmp))):
			print("TFsession returned garbage")
			print("TFInputs",eval_input) #If it's still a problem here use tf.Print version of the graph.
		if (self.PreparedFor>eval_input.shape[0]):
			return tmp[:eval_input.shape[0]]
		return tmp

	def Prepare(self, eval_input, Ncase=125000):
		self.Clean()
		print("Preparing a ",self.NetType,"MolInstance")
		self.prob = None
		self.correct = None
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:0'):
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
			self.output = self.inference(self.embeds_placeholder, self.hidden1, self.hidden2, self.hidden3)
			self.correct = self.n_correct(self.output, self.labels_placeholder)
			self.prob = self.justpreds(self.output)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
		self.PreparedFor = Ncase
		return

	def SaveAndClose(self):
		self.prob = None
		self.correct = None
		self.summary_op =None
		self.summary_writer=None
		MolInstance.SaveAndClose(self)
		return

	def placeholder_inputs(self, batch_size):
		"""
		Generate placeholder variables to represent the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded data in the .run() loop.

		Args:
			batch_size: The batch size will be baked into both placeholders.
		Returns:
			images_placeholder: Images placeholder.
			labels_placeholder: Labels placeholder.
		"""
		# Note that the shapes of the placeholders match the shapes of the full
		# image and label tensors, except the first dimension is now batch_size
		# rather than the full size of the train or test data sets.
		inputs_pl = tf.placeholder(self.tf_prec, shape=(batch_size,self.inshape)) # JAP : Careful about the shapes... should be flat for now.
		outputs_pl = tf.placeholder(self.tf_prec, shape=(batch_size))
		return inputs_pl, outputs_pl

	def justpreds(self, output):
		"""
		Calculates the loss from the logits and the labels.

		Args:
			logits: Logits tensor, float - [batch_size, NUM_CLASSES].
			labels: Labels tensor, int32 - [batch_size].

		Returns:
			loss: Loss tensor of type float.
		"""
		prob = tf.nn.softmax(output)
		return prob

	def loss_op(self, output, labels):
		"""
		Calculates the loss from the logits and the labels.

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

	def TrainPrepare(self,  continue_training =False):
		"""Train for a number of steps."""
		with tf.Graph().as_default():
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.embeds_placeholder, self.hidden1, self.hidden2, self.hidden3)
			self.total_loss, self.loss, self.prob = self.loss_op(self.output, self.labels_placeholder)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.merge_all_summaries()
			init = tf.initialize_all_variables()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			try: # I think this may be broken
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
				print("Restore Failed",Ex)
				pass
			self.summary_writer = tf.train.SummaryWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def test(self, step):
		Ncase_test = self.TData.NTest
		test_loss =  0.0
		test_correct = 0.
		test_start_time = time.time()
		test_loss = None
		feed_dict = None
		for  ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data=self.TData.GetTestBatch(  self.batch_size, ministep)
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			loss_value, prob_value, test_correct_num = self.sess.run([ self.loss, self.prob, self.correct],  feed_dict=feed_dict)
			test_loss = test_loss + loss_value
			test_correct = test_correct + test_correct_num
			duration = time.time() - test_start_time
			print("testing...")
			self.print_training(step, test_loss, test_correct, Ncase_test, duration)
		return test_loss

class MolInstance_fc_sqdiff(MolInstance):
	def __init__(self, TData_,  Name_=None, Trainable_=True):
		self.NetType = "fc_sqdiff"
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		self.hidden1 = PARAMS["hidden1"]
		self.hidden2 = PARAMS["hidden2"]
		self.hidden3 = PARAMS["hidden3"]
		self.inshape = np.prod(self.TData.dig.eshape)
		self.outshape = np.prod(self.TData.dig.lshape)
		self.summary_op =None
		self.summary_writer=None

	def evaluate(self, eval_input):
		# Check sanity of input
		MolInstance.evaluate(self, eval_input)
		eval_input_ = eval_input
		if (self.PreparedFor>eval_input.shape[0]):
			eval_input_ =np.copy(eval_input)
			eval_input_.resize((self.PreparedFor,eval_input.shape[1]))
			# pad with zeros
		eval_labels = np.zeros((self.PreparedFor,1))  # dummy labels
		batch_data = [eval_input_, eval_labels]
		#images_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.embeds_placeholder,self.labels_placeholder)
		tmp, gradient =  (self.sess.run([self.output, self.gradient], feed_dict=feed_dict))
		if (not np.all(np.isfinite(tmp))):
			print("TFsession returned garbage")
			print("TFInputs",eval_input) #If it's still a problem here use tf.Print version of the graph.
		if (self.PreparedFor>eval_input.shape[0]):
			return tmp[:eval_input.shape[0]], gradient[:eval_input.shape[0]]
		return tmp, gradient

	def Prepare(self, eval_input, Ncase=125000):
		self.Clean()
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default():
				self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
				self.output = self.inference(self.embeds_placeholder, self.hidden1, self.hidden2, self.hidden3)
				print ("type of self.embeds_placeholder:", type(self.embeds_placeholder))
				self.gradient = tf.gradients(self.output, self.embeds_placeholder)[0]
				self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
				self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
				self.saver.restore(self.sess, self.chk_file)
		self.PreparedFor = Ncase
		return

	def SaveAndClose(self):
		self.summary_op =None
		self.summary_writer=None
		self.check=None
		self.label_pl = None
		self.mats_pl = None
		self.inp_pl = None
		MolInstance.SaveAndClose(self)
		return

	def placeholder_inputs(self, batch_size):
		"""
		Generate placeholder variables to represent the input tensors.

		Args:
			batch_size: The batch size will be baked into both placeholders.
		Returns:
			inputs_pl: Input placeholder.
			outputs_pl: Outputs placeholder.
		"""
		# Note that the shapes of the placeholders match the shapes of the full
		# image and label tensors, except the first dimension is now batch_size
		# rather than the full size of the train or test data sets.
		inputs_pl = tf.placeholder(self.tf_prec, shape=(batch_size,self.inshape)) # JAP : Careful about the shapes... should be flat for now.
		outputs_pl = tf.placeholder(self.tf_prec, shape=(batch_size, self.outshape))
		return inputs_pl, outputs_pl

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def test(self, step):
		Ncase_test = self.TData.NTest
		test_loss =  0.0
		test_correct = 0.
		test_start_time = time.time()
		for  ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data=self.TData.GetTestBatch( self.batch_size, ministep)
			batch_data=self.PrepareData(batch_data)
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			total_loss_value, loss_value, output_value  = self.sess.run([self.total_loss,  self.loss, self.output],  feed_dict=feed_dict)
			test_loss = test_loss + loss_value
			duration = time.time() - test_start_time
		print("testing...")
		self.print_training(step, test_loss,  Ncase_test, duration)
		return test_loss

	def TrainPrepare(self,  continue_training =False):
		"""Train for a number of steps."""
		with tf.Graph().as_default():
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.embeds_placeholder, self.hidden1, self.hidden2, self.hidden3)
			self.total_loss, self.loss = self.loss_op(self.output, self.labels_placeholder)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.merge_all_summaries()
			init = tf.initialize_all_variables()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			try: # I think this may be broken
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
				print("Restore Failed",Ex)
				pass
			self.summary_writer = tf.train.SummaryWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
			return

	def PrepareData(self, batch_data):
		if (batch_data[0].shape[0]==self.batch_size):
			batch_data=[batch_data[0], batch_data[1].reshape((batch_data[1].shape[0],self.outshape))]
		elif (batch_data[0].shape[0] < self.batch_size):
			batch_data=[batch_data[0], batch_data[1].reshape((batch_data[1].shape[0],self.outshape))]
			tmp_input = np.copy(batch_data[0])
			tmp_output = np.copy(batch_data[1])
			tmp_input.resize((self.batch_size,  batch_data[0].shape[1]))
			tmp_output.resize((self.batch_size,  batch_data[1].shape[1]))
			batch_data=[ tmp_input, tmp_output]
		return batch_data

	def train_step(self,step):
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data=self.TData.GetTrainBatch( self.batch_size) #advances the case pointer in TData...
			batch_data=self.PrepareData(batch_data)
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			_, total_loss_value, loss_value  = self.sess.run([self.train_op, self.total_loss, self.loss], feed_dict=feed_dict)
			train_loss = train_loss + loss_value
			duration = time.time() - start_time
		self.print_training(step, train_loss, Ncase_train, duration)
		return

class MolInstance_fc_sqdiff_BP(MolInstance_fc_sqdiff):
	"""
		An instance of A fully connected Behler-Parinello network.
		Which requires a TensorMolData to train/execute.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True):
		"""
		Raise a Behler-Parinello TensorFlow instance.

		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "fc_sqdiff_BP"
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.
		self.inshape = np.prod(self.TData.dig.eshape)
		LOGGER.info("MolInstance_fc_sqdiff_BP.inshape: %s",str(self.inshape))
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		LOGGER.info("MolInstance_fc_sqdiff_BP.eles: %s",str(self.eles))
		LOGGER.info("MolInstance_fc_sqdiff_BP.inshape.n_eles: %i",self.n_eles)
		self.MeanStoich = self.TData.MeanStoich # Average stoichiometry of a molecule.
		self.MeanNumAtoms = np.sum(self.MeanStoich)
		self.inp_pl=None
		self.mats_pl=None
		self.label_pl=None
		self.batch_size_output = 0

	def Clean(self):
		Instance.Clean(self)
		self.inp_pl=None
		self.check = None
		self.mats_pl=None
		self.label_pl=None
		self.atom_outputs = None
		return

	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		LOGGER.info("self.MeanNumAtoms: %i",self.MeanNumAtoms)
		# allow for 120% of required output space, since it's cheaper than input space to be padded by zeros.
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		#self.TData.CheckBPBatchsizes(self.batch_size, self.batch_size_output)
		LOGGER.info("Assigned batch input size: %i",self.batch_size)
		LOGGER.info("Assigned batch output size: %i",self.batch_size_output) #Number of molecules.
		with tf.Graph().as_default():
			self.inp_pl=[]
			self.mats_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.batch_size_output])))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			try: # I think this may be broken
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
				print("Restore Failed",Ex)
				pass
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		#tf.Print(diff, [diff], message="This is diff: ",first_n=10000000,summarize=100000000)
		#tf.Print(labels, [labels], message="This is labels: ",first_n=10000000,summarize=100000000)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def inference(self, inp_pl, mats_pl):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp_pl: a list of (num_of atom type X flattened input shape) matrix of input cases.
			mats_pl: a list of (num_of atom type X batchsize) matrices which linearly combines the elements
		Returns:
			The BP graph output
		"""
		# convert the index matrix from bool to float
		branches=[]
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3
		output = tf.zeros([self.batch_size_output], dtype=self.tf_prec)
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		print("Norms:", nrm1,nrm2,nrm3)
		LOGGER.info("Layer initial Norms: %f %f %f", nrm1,nrm2,nrm3)
		#print(inp_pl)
		#tf.Print(inp_pl, [inp_pl], message="This is input: ",first_n=10000000,summarize=100000000)
		#tf.Print(bnds_pl, [bnds_pl], message="bnds_pl: ",first_n=10000000,summarize=100000000)
		#tf.Print(mats_pl, [mats_pl], message="mats_pl: ",first_n=10000000,summarize=100000000)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp_pl[e]
			mats = mats_pl[e]
			shp_in = tf.shape(inputs)
			if (PARAMS["CheckLevel"]>2):
				tf.Print(tf.to_float(shp_in), [tf.to_float(shp_in)], message="Element "+str(e)+"input shape ",first_n=10000000,summarize=100000000)
				mats_shape = tf.shape(mats)
				tf.Print(tf.to_float(mats_shape), [tf.to_float(mats_shape)], message="Element "+str(e)+"mats shape ",first_n=10000000,summarize=100000000)
			if (PARAMS["CheckLevel"]>3):
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
				#tf.Print(tf.to_float(shp_out), [tf.to_float(shp_out)], message="This is outshape: ",first_n=10000000,summarize=100000000)
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				tmp = tf.matmul(rshp,mats)
				output = tf.add(output,tmp)
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
		for e in range(len(self.eles)):
			if (not np.all(np.isfinite(batch_data[0][e]),axis=(0,1))):
				print("I was fed shit1")
				raise Exception("DontEatShit")
			if (not np.all(np.isfinite(batch_data[1][e]),axis=(0,1))):
				print("I was fed shit3")
				raise Exception("DontEatShit")
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit4")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip(self.inp_pl+self.mats_pl+[self.label_pl], batch_data[0]+batch_data[1]+[batch_data[2]])}
		return feed_dict

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
			batch_data = self.TData.GetTrainBatch(self.batch_size,self.batch_size_output)
			actual_mols  = np.count_nonzero(batch_data[2])
			dump_, dump_2, total_loss_value, loss_value, mol_output = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output], feed_dict=self.fill_feed_dict(batch_data))
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
			#print ("ministep:", ministep)
			batch_data=self.TData.GetTestBatch(self.batch_size,self.batch_size_output)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = np.count_nonzero(batch_data[2])
			preds, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols

		#print("preds:", preds[0][:actual_mols], " accurate:", batch_data[2][:actual_mols])
		duration = time.time() - start_time
		#print ("preds:", preds, " label:", batch_data[2])
		#print ("diff:", preds - batch_data[2])
		print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		#self.TData.dig.EvaluateTestOutputs(batch_data[2],preds)
		return test_loss

	def test_after_training(self, step):   # testing in the training
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		test_loss =  0.0
		start_time = time.time()
		Ncase_test = self.TData.NTest
		num_of_mols = 0
		all_atoms = []
		bond_length = []
		for i in range (0, len(self.eles)):
			all_atoms.append([])
			bond_length.append([])
		all_mols_nn = []
		all_mols_acc = []
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data=self.TData.GetTestBatch(self.batch_size,self.batch_size_output)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = np.count_nonzero(batch_data[2])
			preds, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols

			print ("actual_mols:", actual_mols)
			all_mols_nn += list(preds[np.nonzero(preds)])
			all_mols_acc += list(batch_data[2][np.nonzero(batch_data[2])])
			#print ("length:", len(atom_outputs))
			for atom_index in range (0,len(self.eles)):
				all_atoms[atom_index] += list(atom_outputs[atom_index][0])
				bond_length[atom_index] += list(1.0/batch_data[0][atom_index][:,-1])
				#print ("atom_index:", atom_index, len(atom_outputs[atom_index][0]))
		test_result = dict()
		test_result['atoms'] = all_atoms
		test_result['nn'] = all_mols_nn
		test_result['acc'] = all_mols_acc
		test_result['length'] = bond_length
		#f = open("test_result_energy_cleaned_connectedbond_angle_for_test_writting_all_mol.dat","wb")
		#pickle.dump(test_result, f)
		#f.close()
		#print("preds:", preds[0][:actual_mols], " accurate:", batch_data[2][:actual_mols])
		duration = time.time() - start_time
		#print ("preds:", preds, " label:", batch_data[2])
		#print ("diff:", preds - batch_data[2])
		print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		#self.TData.dig.EvaluateTestOutputs(batch_data[2],preds)
		return test_loss

	def print_training(self, step, loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f", step, duration, (float(loss)/(Ncase)))
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f", step, duration, (float(loss)/(Ncase)))
		return

	def continue_training(self, mxsteps):
		self.EvalPrepare()
		test_loss = self.test(-1)
		test_freq = 1
		mini_test_loss = test_loss
		for step in  range (0, mxsteps+1):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					self.save_chk(step)
		self.SaveAndClose()
		return

	def evaluate(self, batch_data, IfGrad=True):   #this need to be modified
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size_output = nmol
		if not self.sess:
			print ("loading the session..")
			self.EvalPrepare()
		feed_dict=self.fill_feed_dict(batch_data)

		#mol_output, total_loss_value, loss_value, atom_outputs, gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
		#for i in range (0, batch_data[0][-1][-1].shape[0]):
                #        print("i:", i)
                #        import copy
                #        new_batch_data=copy.deepcopy(batch_data)
                #        #new_batch_data = list(batch_data)
                #        new_batch_data[0][-1][-1][i] += 0.01
                #        feed_dict=self.fill_feed_dict(new_batch_data)
		#	new_mol_output, total_loss_value, loss_value, new_atom_outputs, new_gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
                #        print ("new_charge_gradient: ", gradient[-1][-1][i],  new_gradient[-1][-1][i], " numerical: ", (new_atom_outputs[-1][-1][-1]- atom_outputs[-1][-1][-1])/0.01)

		if (IfGrad):
			mol_output, total_loss_value, loss_value, atom_outputs, gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
			#print ("atom_outputs:", atom_outputs)
			return mol_output, atom_outputs, gradient
		else:
			mol_output, total_loss_value, loss_value, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs],  feed_dict=feed_dict)
			return mol_output, atom_outputs

	def EvalPrepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.mats_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None, self.batch_size_output])))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl)
			#self.gradient = tf.gradients(self.atom_outputs, self.inp_pl)
			self.gradient = tf.gradients(self.output, self.inp_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return

	def Prepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		print("I am pretty sure this is depreciated and should be removed. ")
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.mats_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.batch_size_output])))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return

class MolInstance_fc_sqdiff_BP_WithGrad(MolInstance_fc_sqdiff_BP):
	"""
	An instance of A fully connected Behler-Parinello network.
	Which requires a TensorMolData_BP to train/execute.
	This simultaneously constrains the gradient

	Energy Inputs have dimension
	[eles][atom case][Descriptor Dimension]
	The inference is done elementwise.

	Gradient inputs have dimension:
	[eles][atom case][Descriptor Dimension][Max(n3)]

	The desired outputs have dimension:
	max(3n)+1 (energy and all derivatives, where max n3 is determined by training data.)

	the molecular gradient is constructed by breaking up dE/dRx
	dE/dRx  = \sum_atoms dE_atom/dRx = \sum_atoms dE_atom/dRx
	the sum over atoms is done with the index matrices
	after dE_atom/dRx is made elementwise in this way:
	dE_atom/dRy = dE_atom/dD_i * dD_i/dRy

	So dE_atom/dRy has dimension MaxN3
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True):
		"""
		Raise a Behler-Parinello TensorFlow instance.

		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		MolInstance_fc_sqdiff_BP.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "fc_sqdiff_BP_WithGrad"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.grad_pl = None
		self.MaxN3 = None
		self.GradWeight = PARAMS["GradWeight"]
		if (TData_ != None):
			self.MaxN3 = TData_.MaxN3 # This is only a barrier for training.
			self.GradWeight *= 1.0/self.MaxN3 # relative weight of the nuclear gradient loss.
			LOGGER.info("MolInstance_fc_sqdiff_BP_WithGrad.MaxN3: %i", self.MaxN3)
		LOGGER.info("MolInstance_fc_sqdiff_BP_WithGrad.GradWeight: %f", self.GradWeight)

	def Clean(self):
		MolInstance_fc_sqdiff_BP.Clean(self)
		self.grad_pl=None
		return

	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		self.MaxN3 = self.TData.MaxN3
		print("self.MeanNumAtoms: ",self.MeanNumAtoms)
		# allow for 120% of required output space, since it's cheaper than input space to be padded by zeros.
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		#self.TData.CheckBPBatchsizes(self.batch_size, self.batch_size_output)
		LOGGER.debug("Assigned batch input size: %i",self.batch_size)
		LOGGER.debug("Assigned batch output size: %i",self.batch_size_output)
		LOGGER.debug("Inshape: %i",self.inshape)
		LOGGER.debug("Gradshape: %i %i",self.inshape,self.MaxN3)
		with tf.Graph().as_default():
			self.inp_pl=[]
			self.grad_pl=[]
			self.mats_pl=[]
			for e in range(self.n_eles):
				self.inp_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.inshape])))
				self.grad_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.inshape,self.MaxN3])))
				self.mats_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.batch_size_output])))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size_output,self.MaxN3+1]))
			self.output, self.atom_outputs, self.grads = self.inference()
			#self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.grads, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			#self.summary_op = tf.summary.merge_all()
			#init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			try: # I think this may be broken
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
				print("Restore Failed",Ex)
				pass
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def inference(self):
		"""
		A separate inference routine should be made for
		evaluation purposes because of MaxN3. This one is for training, specifically.
		"""
		branches=[]
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3
		output = tf.zeros([self.batch_size_output], dtype=self.tf_prec)
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		grads = tf.zeros([self.batch_size_output, self.MaxN3], dtype=self.tf_prec)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = self.inp_pl[e]
			mats = self.mats_pl[e]
			shp_in = tf.shape(inputs)
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
				#tf.Print(tf.to_float(shp_out), [tf.to_float(shp_out)], message="This is outshape: ",first_n=10000000,summarize=100000000)
				rshp = tf.reshape(cut,[1,shp_out[0]])
				drshp = tf.gradients(rshp,inputs)
				atom_outputs.append(rshp)
				tmp = tf.matmul(rshp,mats)
				output = tf.add(output,tmp)
				# This loop calculates the force on each atom and sums it to the force on the molecule.
				#  dE_atom/dRy = dE_atom/dD_i * dD_i/dRy, as a  padded with zeros up to MaxN3
				#  dE_atom/dD_i
				#drshp = tf.Print(drshp, [tf.to_float(tf.shape(drshp))], message="Element "+str(e)+"dE_atom/dD_i shape ",first_n=10000000,summarize=100000000)
				dAtomdRy = tf.tensordot(drshp, self.grad_pl[e],axes=[[1],[1]]) # => Atoms X Grad
				#dAtomdRy = tf.Print(dAtomdRy, [tf.to_float(tf.shape(dAtomdRy))], message="Element "+str(e)+"dAtomdRy shape ",first_n=10000000,summarize=100000000)
				dMoldRy = tf.tensordot(dAtomdRy,mats,axes=[[0],[0]]) #  => Grad X Mols
				#dMoldRy = tf.Print(dMoldRy, [tf.to_float(tf.shape(tmp))], message="Element "+str(e)+"dE_atom/dRy ",first_n=10000000,summarize=100000000)
				dtmp = tf.transpose(dMoldRy) # we want to sum over atoms and end up with (mol X cart)
				grads = tf.add(grads,dtmp) # Sum over element types.
		tf.verify_tensor_all_finite(output,"Nan in output!!!")
		#tf.Print(output, [output], message="This is output: ",first_n=10000000,summarize=100000000)
		return output, atom_outputs, grads

	def loss_op(self, output, grads, labels):
		"""
		Args:
			output: energies of molecules.
			grads: gradients of molecules (MaxN3)
			labels: energy, gradients.
		Returns:
			l2 loss on the energies + self.GradWeight*l2 loss on gradients.
		"""
		Enlabels = tf.slice(labels,[0,0],[-1,1])
		Gradlabels = tf.slice(labels,[0,1],[-1,-1])
		Ediff  = tf.subtract(output, Enlabels)
		Gdiff  = tf.subtract(grads, Gradlabels)
		#tf.Print(diff, [diff], message="This is diff: ",first_n=10000000,summarize=100000000)
		#tf.Print(labels, [labels], message="This is labels: ",first_n=10000000,summarize=100000000)
		Eloss = tf.nn.l2_loss(Ediff)
		Gloss = tf.scalar_mul(self.GradWeight,tf.nn.l2_loss(Gdiff))
		loss = tf.add(Eloss,Gloss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def train_step(self, step):
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep: ", ministep, " Ncase_train:", Ncase_train, " self.batch_size", self.batch_size)
			batch_data = self.TData.GetTrainBatch(self.batch_size,self.batch_size_output)
			actual_mols  = np.count_nonzero(batch_data[3])
			feeddict={i:d for i,d in zip(self.inp_pl+self.grad_pl+self.mats_pl+[self.label_pl],batch_data[0]+batch_data[1]+batch_data[2]+[batch_data[3]])}
			dump_2, total_loss_value, loss_value, mol_output = self.sess.run([self.train_op, self.total_loss, self.loss, self.output], feed_dict=feeddict)
			train_loss = train_loss + loss_value
			duration = time.time() - start_time
			num_of_mols += actual_mols
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
			#print ("ministep:", ministep)
			batch_data=self.TData.GetTestBatch(self.batch_size,self.batch_size_output)
			feeddict={i:d for i,d in zip(self.inp_pl+self.grad_pl+self.mats_pl+[self.label_pl],batch_data[0]+batch_data[1]+batch_data[2]+[batch_data[3]])}
			actual_mols  = np.count_nonzero(batch_data[3])
			preds, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs],  feed_dict=feeddict)
			test_loss += loss_value
			num_of_mols += actual_mols
		#print("preds:", preds[0][:actual_mols], " accurate:", batch_data[2][:actual_mols])
		duration = time.time() - start_time
		#print ("preds:", preds, " label:", batch_data[2])
		#print ("diff:", preds - batch_data[2])
		print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		#self.TData.dig.EvaluateTestOutputs(batch_data[2],preds)
		return test_loss

	def EvalPrepare(self):
		raise Exception("NYI")


class MolInstance_fc_sqdiff_BP_Update(MolInstance_fc_sqdiff_BP):
	"""
		An instance of A updated version of fully connected Behler-Parinello network.
		What Kun means is that this version doesn't need an index matrix just an index list.
		Which requires a TensorMolData to train/execute.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True):
		"""
		Raise a Behler-Parinello TensorFlow instance.

		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "fc_sqdiff_BP_Update"
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.
		self.inshape = np.prod(self.TData.dig.eshape)
		LOGGER.info("MolInstance_fc_sqdiff_BP_Update.inshape: %s",str(self.inshape))
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		LOGGER.info("MolInstance_fc_sqdiff_BP_Update.eles: %s",str(self.eles))
		LOGGER.info("MolInstance_fc_sqdiff_BP_Update.inshape.n_eles: %i",self.n_eles)
		self.MeanStoich = self.TData.MeanStoich # Average stoichiometry of a molecule.
		self.MeanNumAtoms = np.sum(self.MeanStoich)
		self.inp_pl=None
		self.index_pl=None
		self.label_pl=None
		self.batch_size_output = 0
		self.gradient = None

	def Clean(self):
		Instance.Clean(self)
		self.inp_pl=None
		self.check = None
		self.index_pl=None
		self.label_pl=None
		self.atom_outputs = None
		self.gradient = None
		return

	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		LOGGER.info("self.MeanNumAtoms: %i",self.MeanNumAtoms)
		# allow for 120% of required output space, since it's cheaper than input space to be padded by zeros.
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		#self.TData.CheckBPBatchsizes(self.batch_size, self.batch_size_output)
		LOGGER.info("Assigned batch input size: %i",self.batch_size)
		LOGGER.info("Assigned batch output size: %i",self.batch_size_output) #Number of molecules.
		with tf.Graph().as_default():
			self.inp_pl=[]
			self.index_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.inshape])))
				self.index_pl.append(tf.placeholder(tf.int64, shape=tuple([None])))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.index_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		#tf.Print(diff, [diff], message="This is diff: ",first_n=10000000,summarize=100000000)
		#tf.Print(labels, [labels], message="This is labels: ",first_n=10000000,summarize=100000000)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def inference(self, inp_pl, index_pl):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp_pl: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index_pl: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		# convert the index matrix from bool to float
		branches=[]
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3
		output = tf.zeros([self.batch_size_output], dtype=self.tf_prec)
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		print("Norms:", nrm1,nrm2,nrm3)
		LOGGER.info("Layer initial Norms: %f %f %f", nrm1,nrm2,nrm3)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp_pl[e]
			shp_in = tf.shape(inputs)
			index = index_pl[e]
			if (PARAMS["CheckLevel"]>2):
				tf.Print(tf.to_float(shp_in), [tf.to_float(shp_in)], message="Element "+str(e)+"input shape ",first_n=10000000,summarize=100000000)
				index_shape = tf.shape(index)
				tf.Print(tf.to_float(index_shape), [tf.to_float(index_shape)], message="Element "+str(e)+"index shape ",first_n=10000000,summarize=100000000)
			if (PARAMS["CheckLevel"]>3):
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
				#tf.Print(tf.to_float(shp_out), [tf.to_float(shp_out)], message="This is outshape: ",first_n=10000000,summarize=100000000)
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				rshpflat = tf.reshape(cut,[shp_out[0]])
				range_index = tf.range(tf.cast(shp_out[0], tf.int64), dtype=tf.int64)
				sparse_index =tf.stack([index, range_index], axis=1)
				sp_atomoutputs = tf.SparseTensor(sparse_index, rshpflat, dense_shape=[tf.cast(self.batch_size_output, tf.int64), tf.cast(shp_out[0], tf.int64)])
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
		for e in range(len(self.eles)):
			if (not np.all(np.isfinite(batch_data[0][e]),axis=(0,1))):
				print("I was fed shit1")
				raise Exception("DontEatShit")
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit4")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip(self.inp_pl+self.index_pl+[self.label_pl], batch_data[0]+batch_data[1]+[batch_data[2]])}
		return feed_dict

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
		pre_output = np.zeros((self.batch_size_output),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep: ", ministep, " Ncase_train:", Ncase_train, " self.batch_size", self.batch_size)
			batch_data = self.TData.GetTrainBatch(self.batch_size,self.batch_size_output)
			actual_mols  = np.count_nonzero(batch_data[2])
			#print ("index:", batch_data[1][0][:40], batch_data[1][1][:20])
			dump_, dump_2, total_loss_value, loss_value, mol_output, atom_outputs  = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output,  self.atom_outputs], feed_dict=self.fill_feed_dict(batch_data))
			#np.set_printoptions(threshold=np.nan)
			#print ("self.atom_outputs", atom_outputs[0][0][:40], "\n", atom_outputs[1][0][:20], atom_outputs[1].shape, "\n  mol_outputs:", mol_output[:20], mol_output.shape)
			#print ("self.gradient:", gradient)
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
			#print ("ministep:", ministep)
			batch_data=self.TData.GetTestBatch(self.batch_size,self.batch_size_output)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = np.count_nonzero(batch_data[2])
			preds, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols

		#print("preds:", preds[0][:actual_mols], " accurate:", batch_data[2][:actual_mols])
		duration = time.time() - start_time
		#print ("preds:", preds, " label:", batch_data[2])
		#print ("diff:", preds - batch_data[2])
		print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		#self.TData.dig.EvaluateTestOutputs(batch_data[2],preds)
		return test_loss

	def test_after_training(self, step):   # testing in the training
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		test_loss =  0.0
		start_time = time.time()
		Ncase_test = self.TData.NTest
		num_of_mols = 0
		all_atoms = []
		bond_length = []
		for i in range (0, len(self.eles)):
			all_atoms.append([])
			bond_length.append([])
		all_mols_nn = []
		all_mols_acc = []
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data=self.TData.GetTestBatch(self.batch_size,self.batch_size_output)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = np.count_nonzero(batch_data[2])
			preds, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols

			print ("actual_mols:", actual_mols)
			all_mols_nn += list(preds[np.nonzero(preds)])
			all_mols_acc += list(batch_data[2][np.nonzero(batch_data[2])])
			#print ("length:", len(atom_outputs))
			for atom_index in range (0,len(self.eles)):
				all_atoms[atom_index] += list(atom_outputs[atom_index][0])
				bond_length[atom_index] += list(1.0/batch_data[0][atom_index][:,-1])
				#print ("atom_index:", atom_index, len(atom_outputs[atom_index][0]))
		test_result = dict()
		test_result['atoms'] = all_atoms
		test_result['nn'] = all_mols_nn
		test_result['acc'] = all_mols_acc
		test_result['length'] = bond_length
		#f = open("test_result_energy_cleaned_connectedbond_angle_for_test_writting_all_mol.dat","wb")
		#pickle.dump(test_result, f)
		#f.close()
		#print("preds:", preds[0][:actual_mols], " accurate:", batch_data[2][:actual_mols])
		duration = time.time() - start_time
		#print ("preds:", preds, " label:", batch_data[2])
		#print ("diff:", preds - batch_data[2])
		print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		#self.TData.dig.EvaluateTestOutputs(batch_data[2],preds)
		return test_loss

	def print_training(self, step, loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f", step, duration, (float(loss)/(Ncase)))
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f", step, duration, (float(loss)/(Ncase)))
		return

	def continue_training(self, mxsteps):
		self.EvalPrepare()
		test_loss = self.test(-1)
		test_freq = 1
		mini_test_loss = test_loss
		for step in  range (0, mxsteps+1):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					self.save_chk(step)
		self.SaveAndClose()
		return

	def evaluate(self, batch_data, IfGrad=True):   #this need to be modified
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size_output = nmol
		if not self.sess:
			print ("loading the session..")
			self.EvalPrepare()
		feed_dict=self.fill_feed_dict(batch_data)

		#mol_output, total_loss_value, loss_value, atom_outputs, gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
		#for i in range (0, batch_data[0][-1][-1].shape[0]):
                #        print("i:", i)
                #        import copy
                #        new_batch_data=copy.deepcopy(batch_data)
                #        #new_batch_data = list(batch_data)
                #        new_batch_data[0][-1][-1][i] += 0.01
                #        feed_dict=self.fill_feed_dict(new_batch_data)
		#	new_mol_output, total_loss_value, loss_value, new_atom_outputs, new_gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
                #        print ("new_charge_gradient: ", gradient[-1][-1][i],  new_gradient[-1][-1][i], " numerical: ", (new_atom_outputs[-1][-1][-1]- atom_outputs[-1][-1][-1])/0.01)

		if (IfGrad):
			mol_output, total_loss_value, loss_value, atom_outputs, gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
			#print ("atom_outputs:", atom_outputs)
			return mol_output, atom_outputs, gradient
		else:
			mol_output, total_loss_value, loss_value, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs],  feed_dict=feed_dict)
			return mol_output, atom_outputs

	def EvalPrepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.index_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.inshape])))
				self.index_pl.append(tf.placeholder(tf.int64, shape=tuple([None])))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.index_pl)
			#self.gradient = tf.gradients(self.atom_outputs, self.inp_pl)
			self.gradient = tf.gradients(self.output, self.inp_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return

	def Prepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		print("I am pretty sure this is depreciated and should be removed. ")
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.index_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(self.tf_prec, shape=tuple([None,self.inshape])))
				self.index_pl.append(tf.placeholder(tf.int64, shape=tuple([None])))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.index_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return
