from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
# Manages a persistent training network instance
# To evaluate a property over many molecules or many points in a large molecule. 
#

class Instance:
	def __init__(self, TData_,  Name_=None, Test_TData_=None):
		self.path='./networks/'
		if (Name_ !=  None):
			self.name = Name_
			#self.QueryAvailable() # Should be a sanity check on the data files.
			self.Load() # Network still cannot be used until it is prepared.
			print("raised network: ", self.train_dir, "  path:",  self.chk_file)
			return
		
		self.TData = TData_
		self.Test_TData = Test_TData_
		if (not os.path.isdir(self.path)):
			os.mkdir(self.path)
		#	self.checkpoint_file_mini =self.path+self.name
		self.chk_file = ''
		self.learning_rate = 0.0001 # for adam
		#self.learning_rate = 0.00001 # for adadelta 
		#self.learning_rate = 0.000001 # 1st sgd
		#self.learning_rate = 0.0000001  #Pickle do not like to pickle  module, replace all the FLAGS with self.
		self.momentum = 0.9
		self.max_steps = 100000
		self.group = 2
		self.y_group = 2
		self.batch_size = 8*400 # This is just the train batch size.
		self.NetType = "None"
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.TData.LoadDataToScratch( True)
		self.TData.PrintStatus()

		self.normalize=1
	
		self.inshape =  self.TData.dig.eshape  # use the flatted version

		self.outshape = self.TData.dig.lshape    # use the flatted version
  
		print ("inshape", self.inshape, "outshape", self.outshape)
#		self.inshape = self.TData.scratch_inputs.shape[1]  # For now assume the inputs are flat... Change this in the future.

		if (self.normalize):
			self.TData.NormalizeOutputs()

		#dist_out = np.zeros((self.TData.scratch_outputs.shape[0],2))
                #dist_out[:,0] = 1/(self.TData.scratch_inputs[:,2])
                #dist_out[:,1] = self.TData.scratch_outputs.reshape(self.TData.scratch_outputs.shape[0])
		#np.savetxt("dist_out.dat", dist_out)
		print ("std of the output", (self.TData.scratch_outputs.reshape(self.TData.scratch_outputs.shape[0])).std())

		# The parameters below belong to tensorflow and its graph
		# all tensorflow variables cannot be pickled they are populated by Prepare
		self.PreparedFor=0
		self.sess = None
		self.loss = None
		self.output = None
		self.train_op = None
		self.total_loss = None
		self.images_placeholder_1 = None
		self.images_placeholder_2 = None
		self.labels_placeholder = None
		self.index_mat = None
		self.saver = None
		self.gradient =None
		return

	def __del__(self):
		if (self.sess != None):
			self.sess.close()

	def evaluate(self, eval_input):
		# Check sanity of input
		if (not np.all(np.isfinite(eval_input),axis=(0,1))): 
			print("WTF, you trying to feed me garbage?") 
			raise Exception("bad digest.")
		if (self.PreparedFor<eval_input.shape[0]):
			self.Prepare(eval_input,eval_input.shape[0])
		return 

# This should really be called prepare for evaluation...
# Since we do training once we don't really need the same thing.
	def Prepare(self):
		self.Clean()
		return

	def Clean(self):
		self.sess = None
		self.loss = None
		self.output = None
		self.total_loss = None
		self.train_op = None
		self.images_placeholder_1 = None
		self.images_placeholder_2 = None
		self.labels_placeholder = None
		self.index_mat = None
		self.saver = None
		self.PreparedFor = 0
		return

	def Save(self):
		print("Saving TFInstance...")
		if (self.TData!=None):
			self.TData.CleanScratch()
		if (self.Test_TData!=None):
			self.Test_TData.CleanScratch()
		self.Clean()
		f=open(self.path+self.name+".tfn","wb")
		pickle.dump(self.__dict__, f, protocol=1)
		f.close()
		return

	# one of these two routines need to be removed I think.
	def save_chk(self,  step, test_loss, feed_dict=None):  # this can be included in the Instance
		cmd="rm  "+self.train_dir+"/"+self.name+"-chk-*"
		os.system(cmd)
		checkpoint_file_mini = os.path.join(self.train_dir, self.name+'-chk-'+str(step)+"-"+str(float(test_loss)))
		self.saver.save(self.sess, checkpoint_file_mini)
		self.chk_file = checkpoint_file_mini
		if (self.summary_op!=None and self.summary_writer!=None and feed_dict!=None):
			self.summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
			self.summary_writer.add_summary(self.summary_str, step)
			self.summary_writer.flush()
		return

#this isn't really the correct way to load()
# only the local class members (not any TF objects should be unpickled.)
	def Load(self):
		print ("Unpickling Instance...")
		f = open(self.path+self.name+".tfn","rb")
		tmp=pickle.load(f)
		# This is just to use an updated version of evaluate and should be removed after I re-train...
		tmp.pop('evaluate',None)
		self.Clean()
		# All this shit should be deleteable after re-training.
		self.__dict__.update(tmp)
		self.group = 1
		f.close()
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
		"""Generate placeholder variables to represent the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded data in the .run() loop, below.
		Args:
		batch_size: The batch size will be baked into both placeholders.
		Returns:
		images_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
		"""
		return

	def fill_feed_dict(self, batch_data, index, images_pl_1, images_pl_2, labels_pl, index_mat):
		"""Fills the feed_dict for training the given step.
		A feed_dict takes the form of:
		feed_dict = {
		<placeholder>: <tensor of values to be passed for placeholder>,
		....
		}
		Args:
		data_set: The set of images and labels, from input_data.read_data_sets()
		images_pl: The images placeholder, from placeholder_inputs().
		labels_pl: The labels placeholder, from placeholder_inputs().
		Returns:
		feed_dict: The feed dictionary mapping from placeholders to values.
		"""
		# Create the feed_dict for the placeholders filled with the next
		# `batch size` examples.
		images_feed_1 = batch_data[0][:, :batch_data[0].shape[1]/2]
		images_feed_2 = batch_data[0][:, batch_data[0].shape[1]/2:]
		labels_feed = batch_data[1]
		# Don't eat shit. 
		if (not (np.all(np.isfinite(images_feed_1),axis=(0,1)) and np.all(np.isfinite(images_feed_2),axis=(0,1)))): 
			print("I was fed shit") 
			raise Exception("DontEatShit")
		if (not np.all(np.isfinite(labels_feed))): 
			print("I was fed shit") 
			raise Exception("DontEatShit")
		feed_dict = {
		images_pl_1: images_feed_1,
		images_pl_2: images_feed_2,
		labels_pl: labels_feed,
		index_mat: index,
		}
		return feed_dict


	def inference(self, images_1, images_2, index_mat,  hidden1_units, hidden2_units, hidden3_units):
		"""Build the MNIST model up to where it may be used for inference.
		Args:
		images: Images placeholder, from inputs().
		hidden1_units: Size of the first hidden layer.
		hidden2_units: Size of the second hidden layer.
		Returns:
		softmax_linear: Output tensor with the computed logits.
		"""
		# Hidden 1
		index_mat = tf.cast(index_mat,tf.float32)
		with tf.name_scope('hidden1'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev= 1 / math.sqrt(float(self.inshape)), var_wd= 0.00)
				biases = tf.Variable(tf.zeros([hidden1_units]),
				name='biases')
				hidden1 = tf.nn.relu(tf.matmul(images_1, weights) + biases)
				tf.scalar_summary('min/' + weights.name, tf.reduce_min(weights))
				tf.histogram_summary(weights.name, weights)
		# Hidden 2
		with tf.name_scope('hidden2'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev= 1 / math.sqrt(float(hidden1_units)), var_wd= 0.00)
				biases = tf.Variable(tf.zeros([hidden2_units]),
				name='biases')
				hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
				

		# Hidden 3
#		with tf.name_scope('hidden3'):
#				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev= 0.5 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
#				biases = tf.Variable(tf.zeros([hidden3_units]),
#				name='biases')
#				hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)


		# Linear
		with tf.name_scope('regression_linear_1'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, self.outshape], var_stddev= 1 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
				biases = tf.Variable(tf.zeros([self.outshape]),
				name='biases')
				output_1 = tf.matmul(hidden2, weights) + biases
				#output_1 = tf.reshape(output_1, [-1, self.group])  # this is for Kun's scheme
				#output_1 = tf.reduce_sum(output_1, 1, keep_dims=True) # Kun's scheme
				output_1 = tf.reshape(output_1, [1, int(self.batch_size/2)])  # this needs to be replaced by the natom
				output_1= tf.matmul(output_1, index_mat) 
				output_1 = tf.reshape(output_1, [int(self.batch_size/4), 1]) # this needs to be replaced by the nmol	
	
		with tf.name_scope('hidden3'):
                                weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev= 1 / math.sqrt(float(self.inshape)), var_wd= 0.00)
                                biases = tf.Variable(tf.zeros([hidden1_units]),
                                name='biases')
                                hidden3 = tf.nn.relu(tf.matmul(images_2, weights) + biases)
                # Hidden 2
                with tf.name_scope('hidden4'):
                                weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev= 1 / math.sqrt(float(hidden1_units)), var_wd= 0.00)
                                biases = tf.Variable(tf.zeros([hidden2_units]),
                                name='biases')
                                hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases)

                with tf.name_scope('regression_linear_2'):
                                weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, self.outshape], var_stddev= 1 / math.sqrt(float(hidden2_units)), var_wd= 0.00)
                                biases = tf.Variable(tf.zeros([self.outshape]),
                                name='biases')
                                output_2 = tf.matmul(hidden4, weights) + biases
                                #output_2 = tf.reshape(output_2, [-1, self.group])
                                #output_2 = tf.reduce_sum(output_2, 1, keep_dims=True)
				output_2 = tf.reshape(output_2, [1, int(self.batch_size/2)])  # this needs to be replace by the natom
                                output_2= tf.matmul(output_2, index_mat)
                                output_2 = tf.reshape(output_2, [int(self.batch_size/4), 1])
			
		with tf.name_scope('sum_up'):
				output = tf.add(output_1, output_2)

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
		tf.scalar_summary(loss.op.name, loss)
		#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
		#optimizer = tf.train.AdadeltaOptimizer(learning_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

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
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		total_correct = 0
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data=self.TData.GetTrainBatch( self.batch_size) #advances the case pointer in TData...
			feed_dict = self.fill_feed_dict(batch_data, self.images_placeholder, self.labels_placeholder)
			_, total_loss_value, loss_value, prob_value, correct_num  = self.sess.run([self.train_op, self.total_loss, self.loss, self.prob, self.correct], feed_dict=feed_dict)
			train_loss = train_loss + loss_value
			total_correct = total_correct + correct_num
		duration = time.time() - start_time
		self.print_training(step, train_loss, total_correct, Ncase_train, duration)
		return
	
	def train_prepare(self, continue_training=False):
		return

	def test(self,step):
		return 

	def print_training(self, step, loss, Ncase, duration):
		denom = max((int(Ncase/self.batch_size)),1)
		print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(denom*self.batch_size)))
		return 

class Instance_fc_classify(Instance):
	def __init__(self, TData_,  Name_=None, Test_TData_=None):
		Instance.__init__(self, TData_,  Name_, Test_TData_)
		self.hidden1 = 200
		self.hidden2 = 200
		self.hidden3 = 200
		self.NetType = "fc_classify"
		self.prob = None
#		self.inshape = self.TData.scratch_inputs.shape[1] 
		self.correct = None
		self.summary_op =None
		self.summary_writer=None
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType

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
		#images_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.images_placeholder,self.labels_placeholder)
		tmp = (np.array(self.sess.run([self.prob], feed_dict=feed_dict))[0,:eval_input.shape[0],1])
		if (not np.all(np.isfinite(tmp))):
			print("TFsession returned garbage")
			print("TFInputs",eval_input) #If it's still a problem here use tf.Print version of the graph. 
		if (self.PreparedFor>eval_input.shape[0]):
			return tmp[:eval_input.shape[0]]
		return tmp
		
	def Prepare(self, eval_input, Ncase=125000):
		super().Prepare(self)
		print("Preparing a ",self.NetType,"Instance")
		self.prob = None
		self.correct = None
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:0'):
			self.images_placeholder, self.labels_placeholder = self.placeholder_inputs(Ncase)
			self.output = self.inference(self.images_placeholder, self.hidden1, self.hidden2, self.hidden3)
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
		Instance.Save(self)
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
			self.images_placeholder, self.labels_placeholder = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.images_placeholder, self.hidden1, self.hidden2, self.hidden3)
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
			feed_dict = self.fill_feed_dict(batch_data, self.images_placeholder, self.labels_placeholder)
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


class Instance_fc_sqdiff(Instance):
	def __init__(self, TData_,  Name_=None, Test_TData_=None):
		Instance.__init__(self, TData_,  Name_, Test_TData_)
		self.hidden1 = 500
		self.hidden2 = 500
		self.hidden3 = 500
		self.NetType = "fc_sqdiff"
#		self.inshape = self.TData.scratch_inputs.shape[1] 
		self.summary_op =None
		self.summary_writer=None
		self.name = self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType

	def evaluate(self, eval_input):
		# Check sanity of input
		Instance.evaluate(self, eval_input)
		eval_input_ = eval_input
		if (self.PreparedFor>eval_input.shape[0]):
			eval_input_ =np.copy(eval_input)
			eval_input_.resize((self.PreparedFor,eval_input.shape[1]))
			# pad with zeros
		eval_labels = np.zeros((self.PreparedFor,1))  # dummy labels
		batch_data = [eval_input_, eval_labels]

		batch_data[0] = (batch_data[0]).reshape((-1, batch_data[0].shape[1]*2))
                batch_data[1] = (batch_data[1].reshape((-1, self.group*self.y_group, 1))).sum(axis=1)
		feed_dict = self.fill_feed_dict(batch_data, self.images_placeholder_1,  self.images_placeholder_2, self.labels_placeholder)

		#images_placeholder, labels_placeholder = self.placeholder_inputs(Ncase) Made by Prepare()
		feed_dict = self.fill_feed_dict(batch_data,self.images_placeholder_1,self.images_placeholder_2,self.labels_placeholder)
		tmp, gradient =  (self.sess.run([self.output, self.gradient], feed_dict=feed_dict))
		if (not np.all(np.isfinite(tmp))):
			print("TFsession returned garbage")
			print("TFInputs",eval_input) #If it's still a problem here use tf.Print version of the graph.
		if (self.PreparedFor>eval_input.shape[0]):
			return tmp[:eval_input.shape[0]], gradient[:eval_input.shape[0]]
		return tmp, gradient
	
	def Prepare(self, eval_input, Ncase=125000):
		Instance.Prepare(self)
		# Always prepare for at least 125,000 cases which is a 50x50x50 grid.
		eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:0'):
				self.images_placeholder_1, self.images_placeholder_2,  self.labels_placeholder = self.placeholder_inputs(Ncase)
				self.output = self.inference(self.images_placeholder_1, self.images_placeholder_2, self.hidden1, self.hidden2, self.hidden3)
				print ("type of self.images_placeholder:", type(self.images_placeholder_1))
				self.gradient = tf.gradients(self.output, self.images_placeholder_1)[0]
				self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
				self.saver = tf.train.Saver()
				self.saver.restore(self.sess, self.chk_file)
		self.PreparedFor = Ncase
		return

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
		images_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
		"""
		# Note that the shapes of the placeholders match the shapes of the full
		# image and label tensors, except the first dimension is now batch_size
		# rather than the full size of the train or test data sets.
		inputs_pl_1 = tf.placeholder(tf.float32, shape=(batch_size/self.y_group, self.inshape)) # JAP : Careful about the shapes... should be flat for now.
		inputs_pl_2 = tf.placeholder(tf.float32, shape=(batch_size/self.y_group, self.inshape))
		outputs_pl = tf.placeholder(tf.float32, shape=(batch_size/self.group/self.y_group, self.outshape))
		index_mat_pl = tf.placeholder(tf.bool, shape=(batch_size/self.y_group, int(batch_size/self.group/self.y_group))) # batch_size/2 is the number of molecule here	
		return inputs_pl_1, inputs_pl_2, outputs_pl, index_mat_pl

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

			# here trying to define a matrix: Natom by Nmolecule
                        mol_index = []  #suppose we have a list (or list of list) define which molecule the atom belong to, for simplicit we define water dimer as a molecule here..
                        for i in range (0, int(self.batch_size/self.y_group)):
                                mol_index.append(int(i/self.group))
                        nmol = int(self.batch_size/self.group/self.y_group) #water dimer is a  molecule here.. 
                        index_mat = np.zeros((self.batch_size/self.y_group, nmol),dtype=np.bool)# here we build the molecule index matrix
                        for i in range (0, int(self.batch_size/self.y_group)):
                                index_mat[i][mol_index[i]]=1


			batch_data[0] = (batch_data[0]).reshape((-1, batch_data[0].shape[1]*2))
			batch_data[1] = (batch_data[1].reshape((-1,self.group*self.y_group,1))).sum(axis=1)
			feed_dict = self.fill_feed_dict(batch_data, index_mat, self.images_placeholder_1,  self.images_placeholder_2, self.labels_placeholder, self.index_mat)
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
			feed_dict = self.fill_feed_dict(batch_data, self.images_placeholder, self.labels_placeholder)
			predicts  = self.sess.run([self.output],  feed_dict=feed_dict)
			print("another testing ...")
			for i in range (0, actual_size):
				print (batch_data[1][i], predicts[0][i])
		print ("input:",batch_data[0], "predict:",output_value, "accu shape:",batch_data[1], "predict shape", output_value.shape, "accu shape:", batch_data[1].shape)
		return test_loss, feed_dict

	def train_prepare(self,  continue_training =False):
		"""Train for a number of steps."""
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.images_placeholder_1, self.images_placeholder_2, self.labels_placeholder, self.index_mat = self.placeholder_inputs(self.batch_size)
			self.output = self.inference(self.images_placeholder_1, self.images_placeholder_2, self.index_mat, self.hidden1, self.hidden2, self.hidden3)
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
		index_time = 0.0
           	for ministep in range (0, int(Ncase_train/self.batch_size)):
            		batch_data=self.TData.GetTrainBatch( self.batch_size) #advances the case pointer in TData...
			batch_data=self.PrepareData(batch_data)
		
			index_start_time = time.time()	
			# here trying to define a matrix: Natom by Nmolecule
                        mol_index = []  #suppose we have a list (or list of list) define which molecule the atom belong to, for simplicit we define water dimer as a molecule here..
                        for i in range (0, int(self.batch_size/self.y_group)):
				mol_index.append(int(i/self.group))
			nmol = int(self.batch_size/self.group/self.y_group) #water dimer is a  molecule here.. 
			index_mat = np.zeros((self.batch_size/self.y_group, nmol),dtype=np.bool)# here we build the molecule index matrix
			for i in range (0, int(self.batch_size/self.y_group)):
				index_mat[i][mol_index[i]]=True
			index_time +=  time.time()-index_start_time

			batch_data[0] = (batch_data[0]).reshape((-1, batch_data[0].shape[1]*2))
			batch_data[1] = (batch_data[1].reshape((-1, self.group*self.y_group, 1))).sum(axis=1)
            		feed_dict = self.fill_feed_dict(batch_data, index_mat, self.images_placeholder_1,  self.images_placeholder_2, self.labels_placeholder, self.index_mat)
            		_, total_loss_value, loss_value  = self.sess.run([self.train_op, self.total_loss, self.loss], feed_dict=feed_dict)
             		train_loss = train_loss + loss_value
            	duration = time.time() - start_time
		print ("time to generate the mol index matrix:", index_time)
             	self.print_training(step, train_loss, Ncase_train, duration)
		return
