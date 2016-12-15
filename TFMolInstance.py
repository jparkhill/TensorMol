from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TFInstance import *
from TensorMolData import *
import numpy as np
import math,pickle
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
	def __init__(self, TData_,  Name_=None):
		Instance.__init__(TData_, 0, Name)
		self.learning_rate = 0.0001
		#self.learning_rate = 0.0001 # for adam
		#self.learning_rate = 0.00001 # for adadelta 
		#self.learning_rate = 0.000001 # 1st sgd
		#self.learning_rate = 0.0000001  #Pickle do not like to pickle  module, replace all the FLAGS with self.
		self.momentum = 0.9
		self.max_steps = 10000
		self.batch_size = 1000 # This is just the train batch size.
		self.name = "Mol"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.TData.LoadDataToScratch(True)
		self.TData.PrintStatus()
		self.normalize= False
		self.inshape =  self.TData.dig.eshape  # use the flatted version
		self.outshape = self.TData.dig.lshape    # use the flatted version
		print ("inshape", self.inshape, "outshape", self.outshape)
		print ("std of the output", (self.TData.scratch_outputs.reshape(self.TData.scratch_outputs.shape[0])).std())
		return

	def inference(self, images, hidden1_units, hidden2_units, hidden3_units):
		"""Build the MNIST model up to where it may be used for inference.
		Args:
		images: Images placeholder, from inputs().
		hidden1_units: Size of the first hidden layer.
		hidden2_units: Size of the second hidden layer.
		Returns:
		softmax_linear: Output tensor with the computed logits.
		"""
		# Hidden 1
		with tf.name_scope('hidden1'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev= 1 / math.sqrt(float(self.inshape)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden1_units]),
			name='biases')
			hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
			tf.scalar_summary('min/' + weights.name, tf.reduce_min(weights))
			tf.histogram_summary(weights.name, weights)
		# Hidden 2
		with tf.name_scope('hidden2'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev= 1 / math.sqrt(float(hidden1_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden2_units]),
			name='biases')
			hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

		# Linear
		with tf.name_scope('regression_linear'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, self.outshape], var_stddev= 1 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
				biases = tf.Variable(tf.zeros([self.outshape]),
				name='biases')
				output = tf.matmul(hidden2, weights) + biases
		return output

	def train(self, mxsteps, continue_training= False):
		self.train_prepare(continue_training)
		test_freq = 10
		mini_test_loss = 100000000 # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss, feed_dict = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					self.save_chk(step, test_loss, feed_dict)  # this method is kind of shitty written 
		self.sess.close()
		self.Save()
		return
	
	def train_step(self,step):
		""" I don't think the base class should be
			train-able. Remove? JAP """
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

class MolInstance_fc_classify(MolInstance):
	def __init__(self, TData_,  Name_=None):
		MolInstance.__init__(self, TData_,  Name_)
		self.hidden1 = 200
		self.hidden2 = 200
		self.hidden3 = 200
		self.NetType = "fc_classify"
		self.prob = None
#		self.inshape = self.TData.scratch_inputs.shape[1] 
		self.correct = None
		self.summary_op =None
		self.summary_writer=None
		self.name = "Mol"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType

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
		super().Prepare(self)
		print("Preparing a ",self.NetType,"MolInstance")
		self.prob = None
		self.correct = None
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:0'):
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
			self.output = self.inference(self.embeds_placeholder, self.hidden1, self.hidden2, self.hidden3)
			self.correct = self.evaluation(self.output, self.labels_placeholder)
			self.prob = self.justpreds(self.output)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
			self.saver.restore(self.sess, self.chk_file)
		self.PreparedFor = Ncase
		return

	def Save(self):
		self.prob = None
		self.correct = None
		self.summary_op =None
		self.summary_writer=None
		MolInstance.Save(self)
		return

	
	def placeholder_inputs(self, batch_size):
		"""Generate placeholder variables to represent the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded data in the .run() loop, below.
		Args:
		batch_size: The batch size will be baked into both placeholders.
		Returns:
		images_placeholder: Images placeholder.
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

	def train_prepare(self,  continue_training =False):
		"""Train for a number of steps."""
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.embeds_placeholder, self.hidden1, self.hidden2, self.hidden3)
			self.total_loss, self.loss, self.prob = self.loss_op(self.output, self.labels_placeholder)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.merge_all_summaries()
			init = tf.initialize_all_variables()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
			try: # I think this may be broken 
				chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
				if (len(chkfiles)>0):
					most_recent_chk_file=chkfiles[0]
					print("Restoring training from Checkpoint: ",most_recent_chk_file)
					self.saver.restore(self.sess, self.train_dir+'/'+most_recent_chk_file)
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
			if (self.Test_TData!=None):
				batch_data= self.Test_TData.LoadData()
				predicts=self.evaluate(batch_data[0])
				print("another testing ...")
				print (batch_data[1], predicts) 
		return test_loss, feed_dict


class MolInstance_fc_sqdiff(MolInstance):
	def __init__(self, TData_,  Name_=None):
		MolInstance.__init__(self, TData_,  Name_)
		self.hidden1 = 500
		self.hidden2 = 500
		self.hidden3 = 500
		self.NetType = "fc_sqdiff"
#		self.inshape = self.TData.scratch_inputs.shape[1] 
		self.summary_op =None
		self.summary_writer=None
		self.name = "Mol"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType

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
		MolInstance.Prepare(self)
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:0'):
				self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
				self.output = self.inference(self.embeds_placeholder, self.hidden1, self.hidden2, self.hidden3)
				print ("type of self.embeds_placeholder:", type(self.embeds_placeholder))
				self.gradient = tf.gradients(self.output, self.embeds_placeholder)[0]
				self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
				self.saver = tf.train.Saver()
				self.saver.restore(self.sess, self.chk_file)
		self.PreparedFor = Ncase
		return

	def Save(self):
		self.summary_op =None
		self.summary_writer=None
		MolInstance.Save(self)
		return

	def placeholder_inputs(self, batch_size):
		"""Generate placeholder variables to represent the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded data in the .run() loop, below.
		Args:
		batch_size: The batch size will be baked into both placeholders.
		Returns:
		images_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
		"""
		# Note that the shapes of the placeholders match the shapes of the full
		# image and label tensors, except the first dimension is now batch_size
		# rather than the full size of the train or test data sets.
		inputs_pl = tf.placeholder(tf.float32, shape=(batch_size,self.inshape)) # JAP : Careful about the shapes... should be flat for now.
		outputs_pl = tf.placeholder(tf.float32, shape=(batch_size, self.outshape))
		return inputs_pl, outputs_pl

	def loss_op(self, output, labels):
		diff  = tf.sub(output, labels)
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
		if (self.Test_TData!=None):
			batch_data= self.Test_TData.LoadData()
			if (self.normalize):
				norm_output=self.TData.ApplyNormalize(batch_data[1])
			batch_data = [batch_data[0] , norm_output]
			actual_size = batch_data[0].shape[0]
			batch_data = self.PrepareData(batch_data)
			feed_dict = self.fill_feed_dict(batch_data, self.embeds_placeholder, self.labels_placeholder)
			predicts  = self.sess.run([self.output],  feed_dict=feed_dict)
			print("another testing ...")
			for i in range (0, actual_size):
				print (batch_data[1][i], predicts[0][i])
		print ("input:",batch_data[0], "predict:",output_value, "accu:",batch_data[1])
		return test_loss, feed_dict

	def train_prepare(self,  continue_training =False):
		"""Train for a number of steps."""
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.embeds_placeholder, self.labels_placeholder = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.embeds_placeholder, self.hidden1, self.hidden2, self.hidden3)
			self.total_loss, self.loss = self.loss_op(self.output, self.labels_placeholder)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.merge_all_summaries()
			init = tf.initialize_all_variables()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
			try: # I think this may be broken 
				chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
				if (len(chkfiles)>0):
					most_recent_chk_file=chkfiles[0]
					print("Restoring training from Checkpoint: ",most_recent_chk_file)
					self.saver.restore(self.sess, self.train_dir+'/'+most_recent_chk_file)
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
	def __init__(self, TData_, Name_=None):
		MolInstance.__init__(self, TData_,  Name_)
		self.inshape =  self.TData.dig.eshape[1]
		
		self.eletypes = self.TData.ElementTypes()
		self.MeanStoich = self.TData.MeanStoich() # Average stoichiometry of a molecule.
		self.MeanNumAtoms = np.sum(self.MeanStoich)
		#Here we should check if the number of max number of atoms in a mol exceeds input case but we will be lazy.
		self.input_case = self.batch_size * self.aver_atom_per_mol
		
		self.hidden1 = 100
		self.hidden2 = 100
		self.hidden3 = 500
		self.H_length = None  # start with a random int for inference
		self.O_length = None  # start with a random int for inference
		self.C_length = None  # start with a random int for inference
		self.index_matrix = None
		self.NetType = "fc_sqdiff_BP"
		self.summary_op =None
		self.summary_writer=None
		self.name = "Mol"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType

	def inference(self, images, index_mat, H_length, C_length, O_length, hidden1_units, hidden2_units):
		# convert the index matrix from bool to float
		index_mat = tf.cast(index_mat,tf.float32)
		# define the Hydrogen network
		with tf.name_scope('H_hidden1'):
			H_inputs = tf.slice(images, [0,0], [H_length, self.inshape]) # debug the indexing.  The tf.slice is kind of weired
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev= 1 / math.sqrt(float(self.inshape)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden1_units]),
			name='biases')
			H_hidden1 = tf.nn.relu(tf.matmul(H_inputs, weights) + biases)

		with tf.name_scope('H_hidden2'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev= 1 / math.sqrt(float(hidden1_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden2_units]),
			name='biases')
			H_hidden2 = tf.nn.relu(tf.matmul(H_hidden1, weights) + biases)

		with tf.name_scope('H_regression_linear'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, self.outshape], var_stddev= 1 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([self.outshape]),
			name='biases')
			H_output = tf.matmul(H_hidden2, weights) + biases
			H_output = tf.reshape(H_output, [1, H_length])  # this needs to be replaced by the natom
		
			H_index_mat = tf.slice(index_mat, [0,0], [H_length, self.batch_size])

			H_output = tf.matmul(H_output, H_index_mat) 
			H_output = tf.reshape(H_output, [self.batch_size, 1]) # this needs to be replaced by the nmol	
	
		# define the Carbon newtork
		with tf.name_scope('C_hidden1'):
			C_inputs = tf.slice(images, [H_length,0], [C_length, self.inshape])
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev= 1 / math.sqrt(float(self.inshape)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden1_units]),
			name='biases')
			C_hidden1 = tf.nn.relu(tf.matmul(C_inputs, weights) + biases)

		with tf.name_scope('C_hidden2'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev= 1 / math.sqrt(float(hidden1_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden2_units]),
			name='biases')
			C_hidden2 = tf.nn.relu(tf.matmul(C_hidden1, weights) + biases)

		with tf.name_scope('C_regression_linear'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, self.outshape], var_stddev= 1 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([self.outshape]),
			name='biases')
			C_output = tf.matmul(C_hidden2, weights) + biases
			C_output = tf.reshape(C_output, [1, C_length])  # this needs to be replace by the natom

			C_index_mat = tf.slice(index_mat, [H_length,0],[C_length, self.batch_size])

			C_output = tf.matmul(C_output, C_index_mat)
			C_output = tf.reshape(C_output, [self.batch_size, 1])


		# define the Oxygen newtork
		with tf.name_scope('O_hidden1'):
			O_inputs = tf.slice(images, [H_length+C_length, 0], [O_length, self.inshape])
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev= 1 / math.sqrt(float(self.inshape)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden1_units]),
			name='biases')
			O_hidden1 = tf.nn.relu(tf.matmul(O_inputs, weights) + biases)

		with tf.name_scope('O_hidden2'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev= 1 / math.sqrt(float(hidden1_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([hidden2_units]),
			name='biases')
			O_hidden2 = tf.nn.relu(tf.matmul(O_hidden1, weights) + biases)

		with tf.name_scope('O_regression_linear'):
			weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, self.outshape], var_stddev= 1 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
			biases = tf.Variable(tf.zeros([self.outshape]),
			name='biases')
			O_output = tf.matmul(O_hidden2, weights) + biases
			O_output = tf.reshape(O_output, [1, O_length])  # this needs to be replace by the natom
			O_index_mat = tf.slice(index_mat, [H_length+C_length, 0],[O_length, self.batch_size])
			O_output = tf.matmul(O_output, O_index_mat)
			O_output = tf.reshape(O_output, [self.batch_size, 1])

		with tf.name_scope('sum_up'):
				H_C_output = tf.add(H_output, C_output)
				output = tf.add(H_C_output, O_output)

		return output, H_output, C_output, O_output, H_inputs, C_inputs, O_inputs



	def PrepareData(self, raw_data): # for debug purpose, this only works for system with two kinds of element: H and O
		H_index_matrix = raw_data[3][1]  # 
		C_index_matrix = raw_data[3][6]  # 
		O_index_matrix = raw_data[3][8]
		
		H_length = raw_data[2][1]  
		C_length = raw_data[2][6]
		O_length = raw_data[2][8]

		index_matrix = np.zeros((self.input_case, self.batch_size),dtype=bool)
		index_matrix[0:H_length, :] = H_index_matrix
		index_matrix[H_length:H_length + C_length,  :] = C_index_matrix
		index_matrix[H_length + C_length:H_length + C_length + O_length, :] = O_index_matrix
	
		return [raw_data[0], raw_data[1]], [H_length, C_length, O_length], index_matrix
		
	def fill_feed_dict(self, batch_data, atom_length, index_matrix, images_pl, labels_pl, index_mat_pl, H_length_pl, C_length_pl, O_length_pl):
                # Create the feed_dict for the placeholders filled with the next
                # `batch size` examples.
                images_feed = batch_data[0]
                labels_feed = batch_data[1]
		H_length_feed = atom_length[0] # debug, shitty way to write it.
		C_length_feed = atom_length[1]
		O_length_feed = atom_length[2]  # debug, shitty way to write it.
		index_feed = index_matrix
                # Don't eat shit. 
                if (not np.all(np.isfinite(images_feed),axis=(0,1))):
                        print("I was fed shit")
                        raise Exception("DontEatShit")
                if (not np.all(np.isfinite(labels_feed))):
                        print("I was fed shit")
                        raise Exception("DontEatShit")
                feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
                index_mat_pl: index_feed,
	 	H_length_pl: H_length_feed,
		C_length_pl: C_length_feed,
                O_length_pl: O_length_feed,
                }
                return feed_dict

        def placeholder_inputs(self, batch_size):
                # rather than the full size of the train or test data sets.
                inputs_pl = tf.placeholder(tf.float32, shape=(self.input_case, self.inshape)) # JAP : Careful about the shapes... should be flat for now.
                outputs_pl = tf.placeholder(tf.float32, shape=(batch_size, self.outshape))
		index_mat_pl = tf.placeholder(tf.bool, shape=((self.input_case, batch_size)))
		H_pl = tf.placeholder("int32")
		C_pl = tf.placeholder("int32")
                O_pl = tf.placeholder("int32")
                return inputs_pl, outputs_pl, index_mat_pl, H_pl, C_pl, O_pl 

	def train_prepare(self,  continue_training =False):
                """Train for a number of steps."""
                with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
                        self.embeds_placeholder, self.labels_placeholder, self.index_matrix, self.H_length, self.C_length, self.O_length = self.placeholder_inputs(self.batch_size)
                        self.output, self.H_output, self.C_output, self.O_output, self.H_input, self.C_input, self.O_input = self.inference(self.embeds_placeholder, self.index_matrix, self.H_length, self.C_length, self.O_length, self.hidden1, self.hidden2)
                        self.total_loss, self.loss = self.loss_op(self.output, self.labels_placeholder)
                        self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
                        self.summary_op = tf.merge_all_summaries()
                        init = tf.initialize_all_variables()
                        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                        self.saver = tf.train.Saver()
                        try: # I think this may be broken 
                                chkfiles = [x for x in os.listdir(self.train_dir) if (x.count('chk')>0 and x.count('meta')==0)]
                                if (len(chkfiles)>0):
                                        most_recent_chk_file=chkfiles[0]
                                        print("Restoring training from Checkpoint: ",most_recent_chk_file)
                                        self.saver.restore(self.sess, self.train_dir+'/'+most_recent_chk_file)
                        except Exception as Ex:
                                print("Restore Failed",Ex)
                                pass
                        self.summary_writer = tf.train.SummaryWriter(self.train_dir, self.sess.graph)
                        self.sess.run(init)
                        return

	def train_step(self, step):
		Ncase_train = self.TData.NTrain
		print ("NTraing:", Ncase_train)
		start_time = time.time()
                train_loss =  0.0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			raw_data=self.TData.GetTrainBatch(self.input_case, self.batch_size) # batch_data strucutre: inputs (self.input_case*self.eshape), outputs (self.batch_size*self.lshape), number_atom_per_ele (dic[1(H)]=2000, dic[8(0)]=1000), index_matrix(dic[1(H)]: number_atom_per_ele[1(H)]*self.batch_size)
			batch_data, atom_length, index_matrix=self.PrepareData(raw_data)
			feed_dict = self.fill_feed_dict(batch_data, atom_length, index_matrix, self.embeds_placeholder, self.labels_placeholder, self.index_matrix, self.H_length, self.C_length, self.O_length)
			_, total_loss_value, loss_value, tmp_mol_output, tmp_H_output, tmp_C_output, tmp_O_output, tmp_H_input, tmp_C_input, tmp_O_input  = self.sess.run([self.train_op, self.total_loss, self.loss, self.output, self.H_output, self.C_output, self.O_output, self.H_input, self.C_input, self.O_input], feed_dict=feed_dict)
                        train_loss = train_loss + loss_value
                duration = time.time() - start_time
		#print ("self.H_length, self.O_length", self.H_length, self.O_length)
		print ("ministep:", ministep)
		print ("accu:", batch_data[1],  "Mol:", tmp_mol_output,"H:", tmp_H_output, "C:", tmp_C_output, "O:", tmp_O_output)
		#print ("input:", raw_data[0], "output:", raw_data[1])
                self.print_training(step, train_loss, Ncase_train, duration)
                return


	def test(self, step):
                Ncase_test = self.TData.NTest
                test_loss =  0.0
                test_correct = 0.
                test_start_time = time.time()
                for  ministep in range (0, int(Ncase_test/self.batch_size)):
                        raw_data=self.TData.GetTestBatch( self.input_case, self.batch_size)
			batch_data, atom_length, index_matrix=self.PrepareData(raw_data)
			feed_dict = self.fill_feed_dict(batch_data, atom_length, index_matrix, self.embeds_placeholder, self.labels_placeholder, self.index_matrix, self.H_length, self.C_length, self.O_length)
                        total_loss_value, loss_value, output_value  = self.sess.run([self.total_loss,  self.loss, self.output],  feed_dict=feed_dict)
                        test_loss = test_loss + loss_value
                        duration = time.time() - test_start_time
                print("testing...")
                self.print_training(step, test_loss,  Ncase_test, duration, Train=False)
		return test_loss, feed_dict

