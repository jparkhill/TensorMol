"""
These instances are re-writes of the convoluted instances found in TFMolInstanceDirect.

I would still like the following changes:
- Independence from any manager.
- Inheritance from a re-written instance base class.
- Removal of any dependence on TensorMolData
- Removal of any dependence on TFInstance.

But at least these are a first step.  JAP 12/2017.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .TFInstance import *
from ..Containers.TensorMolData import *
from .TFMolInstance import *
from ..ForceModels.ElectrostaticsTF import *
from ..ForceModifiers.Neighbors import *
from ..TFDescriptors.RawSymFunc import *
from tensorflow.python.client import timeline
import time
import threading

class MolInstance_DirectBP_EandG_SymFunction(MolInstance_fc_sqdiff_BP):
	"""
	Behler Parinello Scheme with energy and gradient training.
	NO Electrostatic embedding.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
			Trainable_: True for training, False for evalution
			ForceType_: Deprecated
		"""
		self.SFPa = None
		self.SFPr = None
		self.Ra_cut = None
		self.Rr_cut = None
		self.HasANI1PARAMS = False
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.MaxNAtoms = self.TData.MaxNAtoms
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
		self.eles_pairs = []
		for i in range (len(self.eles)):
			for j in range(i, len(self.eles)):
				self.eles_pairs.append([self.eles[i], self.eles[j]])
		self.eles_pairs_np = np.asarray(self.eles_pairs)
		if not self.HasANI1PARAMS:
			self.SetANI1Param()
		self.HiddenLayers = PARAMS["HiddenLayers"]
		self.batch_size = PARAMS["batch_size"]
		print ("self.activation_function_type: ", self.activation_function_type)
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		self.xyzs_pl = None
		self.Zs_pl = None
		self.label_pl = None
		self.grads_pl = None
		self.sess = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None
		self.learning_rate = PARAMS["learning_rate"]
		self.suffix = PARAMS["NetNameSuffix"]
		self.SetANI1Param()
		self.run_metadata = None

		self.GradScalar = PARAMS["GradScalar"]
		self.EnergyScalar = PARAMS["EnergyScalar"]
		self.TData.ele = self.eles_np
		self.TData.elep = self.eles_pairs_np

		self.NetType = "RawBP_EandG"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+self.suffix
		self.train_dir = './networks/'+self.name
		self.keep_prob = np.asarray(PARAMS["KeepProb"])
		self.nlayer = len(PARAMS["KeepProb"]) - 1
		self.monitor_mset =  PARAMS["MonitorSet"]

	def SetANI1Param(self, prec=np.float64):
		"""
		Generate ANI1 symmetry function parameter tensor.
		"""
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
		self.inshape_withencode = int(self.inshape + AN1_num_r_Rs)
		#self.inshape = int(len(self.eles)*AN1_num_r_Rs)
		p1 = np.tile(np.reshape(thetas,[AN1_num_a_As,1,1]),[1,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(rs,[1,AN1_num_a_Rs,1]),[AN1_num_a_As,1,1])
		SFPa2 = np.concatenate([p1,p2],axis=2)
		self.SFPa2 = np.transpose(SFPa2, [2,0,1])
		p1_new = np.reshape(rs_R,[AN1_num_r_Rs,1])
		self.SFPr2 = np.transpose(p1_new, [1,0])
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]
		self.HasANI1PARAMS = True
		print ("self.inshape:", self.inshape)

	def Clean(self):
		"""
		Clean Instance for pickle saving.
		"""
		Instance.Clean(self)
		#self.tf_prec = None
		self.xyzs_pl, self.Zs_pl, self.label_pl, self.grads_pl, self.natom_pl = None, None, None, None, None
		self.check, self.options, self.run_metadata = None, None, None
		self.atom_outputs = None
		self.energy_loss = None
		self.Scatter_Sym, self.Sym_Index = None, None
		self.Radp_pl, self.Angt_pl = None, None
		self.Elabel_pl = None
		self.Etotal, self.Ebp, self.Ebp_atom = None, None, None
		self.gradient = None
		self.total_loss_dipole, self.energy_loss, self.grads_loss = None, None, None
		self.train_op_dipole, self.train_op_EandG = None, None
		self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG = None, None, None, None
		self.Radp_Ele_pl, self.Angt_Elep_pl = None, None
		self.mil_jk_pl, self.mil_j_pl = None, None
		self.keep_prob_pl = None
		return


	def TrainPrepare(self,  continue_training =False):
		"""
		Define Tensorflow graph for training.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]),name="DesEnergy")
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="DesGrads")
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Release(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl)
			self.Etotal, self.Ebp, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.keep_prob_pl)
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			self.total_loss, self.loss, self.energy_loss, self.grads_loss = self.loss_op(self.Etotal, self.gradient, self.Elabel_pl, self.grads_pl, self.natom_pl)
			tf.summary.scalar("loss", self.loss)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
			config.gpu_options.per_process_gpu_memory_fraction = 0.90
			self.sess = tf.Session(config=config)
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess.run(init)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			if (PARAMS["Profiling"]>0):
				print("logging with FULL TRACE")
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
				self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
			self.sess.graph.finalize()

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
			raise Exception("Please check your inputs")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.grads_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.mil_j_pl] + [self.mil_jk_pl] + [self.natom_pl] + [self.keep_prob_pl], batch_data)}
		return feed_dict

	def energy_inference(self, inp, indexs, xyzs, keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating energy.

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements.
			xyzs: xyz coordinates of atoms.
			keep_prob: dropout prob of each layer.
		Returns:
			The BP graph energy output
		"""
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Ebranches=[]
		output = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		atom_outputs = []
		with tf.name_scope("EnergyNet"):
			for e in range(len(self.eles)):
				Ebranches.append([])
				inputs = inp[e]
				shp_in = tf.shape(inputs)
				index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(inputs, keep_prob[i]), weights) + biases))
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[i]), weights) + biases))
				with tf.name_scope(str(self.eles[e])+'_regression_linear'):
					shp = tf.shape(inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Ebranches[-1].append(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Ebranches[-1][-1])
					cut = tf.slice(Ebranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					atom_outputs.append(rshp)
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(index, [0,1], [shp_out[0],1])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
					output = tf.add(output, ToAdd)
				tf.verify_tensor_all_finite(output,"Nan in output!!!")
			bp_energy = tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size])
		total_energy = tf.identity(bp_energy)
		return total_energy, bp_energy, output


	def loss_op(self, energy, energy_grads, Elabels, grads, natom):
		"""
		losss function that includes dipole loss, energy loss and gradient loss.
		"""
		maxatom=tf.cast(tf.shape(energy_grads)[2], self.tf_prec)
		energy_diff  = tf.multiply(tf.subtract(energy, Elabels,name="EnDiff"), natom*maxatom)
		energy_loss = tf.nn.l2_loss(energy_diff,name="EnL2")
		grads_diff = tf.multiply(tf.subtract(energy_grads, grads,name="GradDiff"), tf.reshape(natom*maxatom, [1, self.batch_size, 1, 1]))
		grads_loss = tf.nn.l2_loss(grads_diff,name="GradL2")
		EandG_loss = tf.add(tf.multiply(energy_loss, self.EnergyScalar), tf.multiply(grads_loss, self.GradScalar),name="MulLoss")
		loss = tf.identity(EandG_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss

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
		optimizer = tf.train.AdamOptimizer(learning_rate,name="Adam")
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step, name="trainop")
		return train_op

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size.
		Training object including dipole, energy and gradient

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
			batch_data = self.TData.GetTrainBatch(self.batch_size) + [self.keep_prob]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, total_loss_value, loss_value, energy_loss, grads_loss, Etotal = self.sess.run([self.train_op, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.Etotal], feed_dict=self.fill_feed_dict(batch_data))
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, num_of_mols, duration)
		return

	def test(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_test = self.TData.NTest
		start_time = time.time()
		test_loss =  0.0
		test_energy_loss = 0.0
		test_grads_loss = 0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.TData.GetTestBatch(self.batch_size) + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, loss_value, energy_loss, grads_loss, Etotal = self.sess.run([self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.Etotal], feed_dict=self.fill_feed_dict(batch_data))
			test_loss = test_loss + loss_value
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		print ("testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, num_of_mols, duration, False)
		return test_loss

	def train(self, mxsteps, continue_training= False):
		"""
		This the training loop for the united model.
		"""
		LOGGER.info("running the TFMolInstance.train()")
		self.TrainPrepare(continue_training)
		test_freq = PARAMS["test_freq"]
		mini_dipole_test_loss = float('inf') # some big numbers
		mini_energy_test_loss = float('inf')
		mini_test_loss = float('inf')
		for step in  range (0, mxsteps):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				if self.monitor_mset != None:
					self.InTrainEval(self.monitor_mset, self.Rr_cut, self.Ra_cut, step=step)
				test_loss = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					self.save_chk(step)
		self.SaveAndClose()
		return


	def profile_step(self, step):
		"""
		Perform a single profiling step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		time_print_mini = time.time()
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			t_mini = time.time()
			batch_data = self.TData.GetTrainBatch(self.batch_size) + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, Etotal = self.sess.run([self.train_op, self.Etotal], feed_dict=self.fill_feed_dict(batch_data), options=self.options, run_metadata=self.run_metadata)
			print ("inference time:", time.time() - t)
			self.summary_writer.add_run_metadata(self.run_metadata, 'minstep%d' % ministep)
			duration = time.time() - start_time
			num_of_mols += actual_mols
			fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			chrome_trace = fetched_timeline.generate_chrome_trace_format()
			with open('timeline_step_%d.json' % ministep, 'w') as f:
				f.write(chrome_trace)
		return

	def profile(self):
		"""
		This profiles a training step.
		"""
		LOGGER.info("running the TFMolInstance.train()")
		self.TrainPrepare(False)
		self.profile_step(1)
		return

	def InTrainEval(self, mol_set, Rr_cut, Ra_cut, step=0):
		"""
		Evaluted the network during training.
		"""
		nmols = len(mol_set.mols)
		for i in range(nmols, self.batch_size):
			mol_set.mols.append(mol_set.mols[-1])
		nmols = len(mol_set.mols)
		dummy_energy = np.zeros((nmols))
		dummy_dipole = np.zeros((nmols, 3))
		xyzs = np.zeros((nmols, self.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((nmols), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = mol.NAtoms()
		NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
		rad_p_ele, ang_t_elep, mil_j, mil_jk = NL.buildPairsAndTriplesWithEleIndexLinear(Rr_cut, Ra_cut, self.eles_np, self.eles_pairs_np)
		batch_data = [xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, mil_j, mil_jk, 1.0/natom]
		feed_dict=self.fill_feed_dict(batch_data + [np.ones(self.nlayer+1)])
		Etotal, Ebp, Ebp_atom, gradient= self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.gradient], feed_dict=feed_dict)
		monitor_data = [Etotal, Ebp, Ebp_atom, gradient]
		f = open(self.name+"_monitor_"+str(step)+".dat","wb")
		pickle.dump(monitor_data, f)
		f.close()
		print ("calculating monitoring set..")
		return Etotal, Ebp, Ebp_atom, gradient

	def print_training(self, step, loss, energy_loss, grads_loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f  energy_loss: %.10f  grad_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)))
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f energy_loss: %.10f  grad_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)))
		return

	def evaluate(self, batch_data):
		"""
		Evaluate the energy, atom energies, and IfGrad = True the gradients
		of this Direct Behler-Parinello graph.
		"""
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		self.activation_function_type = PARAMS["NeuronType"]
		self.AssignActivation()
		#print ("self.activation_function:\n\n", self.activation_function)
		if (batch_data[0].shape[1] != self.MaxNAtoms):
			self.MaxNAtoms = batch_data[0].shape[1]
			self.batch_size = nmol
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare()
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
		if not self.sess:
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare()
		feed_dict=self.fill_feed_dict(batch_data+[np.ones(self.nlayer+1)])
		Etotal, Ebp, Ebp_atom, gradient = self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.gradient], feed_dict=feed_dict)
		return Etotal, Ebp, Ebp_atom, gradient

	def EvalPrepare(self,  continue_training =False):
		"""
		Generate Tensorflow graph of evalution.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]),name="DesEnergy")
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="DesGrads")
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Release(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl)
			self.Etotal, self.Ebp, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.keep_prob_pl)
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			if (PARAMS["Profiling"]>0):
				print("logging with FULL TRACE")
				self.summary_writer = tf.summary.FileWriter('./networks/PROFILE', self.sess.graph)
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
				self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
			self.sess.graph.finalize()

	def energy_inference_periodic(self, inp, indexs, xyzs, keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating the energy of periodic system

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Ebranches=[]
		output = tf.zeros([self.batch_size, self.nreal], dtype=self.tf_prec)
		atom_outputs = []
		with tf.name_scope("EnergyNet"):
			for e in range(len(self.eles)):
				Ebranches.append([])
				inputs = inp[e]
				shp_in = tf.shape(inputs)
				index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(inputs, keep_prob[i]), weights) + biases))
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[i]), weights) + biases))
				with tf.name_scope(str(self.eles[e])+'_regression_linear'):
					shp = tf.shape(inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Ebranches[-1].append(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Ebranches[-1][-1])
					cut = tf.slice(Ebranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					atom_outputs.append(rshp)
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(index, [0,1], [shp_out[0],1])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.nreal]),[self.batch_size, self.nreal])
					output = tf.add(output, ToAdd)
				tf.verify_tensor_all_finite(output,"Nan in output!!!")
			bp_energy = tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size])
		total_energy = tf.identity(bp_energy, cc_energy)
		return total_energy, bp_energy, output


	def fill_feed_dict_periodic(self, batch_data):
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
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.grads_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl]  + [self.mil_j_pl]  + [self.mil_jk_pl] + [self.natom_pl] + [self.keep_prob_pl], batch_data)}
		return feed_dict

	@TMTiming("EvalPeriodic")
	def evaluate_periodic(self, batch_data, nreal, DoForce = True):
		"""
		Evaluate the energy, atom energies, and IfGrad = True the gradients
		of this Direct Behler-Parinello graph for a periodic system.
		"""
		nmol = batch_data[2].shape[0]
		self.nreal = nreal
		self.activation_function_type = PARAMS["NeuronType"]
		self.AssignActivation()
		if (batch_data[0].shape[1] != self.MaxNAtoms):
			self.MaxNAtoms = batch_data[0].shape[1]
			self.batch_size = nmol
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare_Periodic()
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
		if not self.sess:
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare_Periodic()
		t0 = time.time()
		feed_dict=self.fill_feed_dict_periodic(batch_data+[np.ones(self.nlayer+1)])
		if (DoForce):
			Etotal, Ebp, Ebp_atom, gradient = self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.gradient], feed_dict=feed_dict)
			return Etotal, Ebp, Ebp_atom, gradient
		else:
			Etotal = self.sess.run(self.Etotal, feed_dict=feed_dict)
			return Etotal

	def EvalPrepare_Periodic(self,  continue_training =False):
		"""
		Generate Tensorlfow graph for evalution of periodic system.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]),name="DesEnergy")
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="DesGrads")
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Periodic(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl, self.nreal)
			self.Etotal, self.Ebp, self.Ebp_atom = self.energy_inference_periodic(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.keep_prob_pl)
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			if (PARAMS["Profiling"]>0):
				print("logging with FULL TRACE")
				self.summary_writer = tf.summary.FileWriter('./networks/PROFILE', self.sess.graph)
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
				self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
			self.sess.graph.finalize()

class MolInstance_DirectBP_EE_SymFunction(MolInstance_fc_sqdiff_BP):
	"""
	Electrostatic embedding Behler Parinello with van der waals interaction implemented with Grimme C6 scheme.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
			Trainable_: True for training, False for evalution
			ForceType_: Deprecated
		"""
		self.SFPa = None
		self.SFPr = None
		self.Ra_cut = None
		self.Rr_cut = None
		self.HasANI1PARAMS = False
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.MaxNAtoms = self.TData.MaxNAtoms
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
		self.eles_pairs = []
		for i in range (len(self.eles)):
			for j in range(i, len(self.eles)):
				self.eles_pairs.append([self.eles[i], self.eles[j]])
		self.eles_pairs_np = np.asarray(self.eles_pairs)
		if not self.HasANI1PARAMS:
			self.SetANI1Param()
		self.HiddenLayers = PARAMS["HiddenLayers"]
		self.batch_size = PARAMS["batch_size"]
		print ("self.activation_function_type: ", self.activation_function_type)
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		self.xyzs_pl = None
		self.Zs_pl = None
		self.label_pl = None
		self.grads_pl = None
		self.sess = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None

		self.GradScalar = PARAMS["GradScalar"]
		self.EnergyScalar = PARAMS["EnergyScalar"]
		self.DipoleScalar = PARAMS["DipoleScalar"]
		self.Ree_on  = PARAMS["EECutoffOn"]
		self.Ree_off  = PARAMS["EECutoffOff"]
		self.DSFAlpha = PARAMS["DSFAlpha"]
		self.learning_rate_dipole = PARAMS["learning_rate_dipole"]
		self.learning_rate_energy = PARAMS["learning_rate_energy"]
		self.suffix = PARAMS["NetNameSuffix"]
		self.SetANI1Param()
		self.run_metadata = None

		self.Training_Traget = "Dipole"
		self.TData.ele = self.eles_np
		self.TData.elep = self.eles_pairs_np

		self.Training_Traget = "Dipole"
		self.vdw_R = np.zeros(self.n_eles)
		self.C6 = np.zeros(self.n_eles)
		for i, ele in enumerate(self.eles):
			self.C6[i] = C6_coff[ele]* (BOHRPERA*10.0)**6.0 / JOULEPERHARTREE # convert into a.u.
			self.vdw_R[i] = atomic_vdw_radius[ele]*BOHRPERA

		if self.Ree_on != 0.0:
			raise Exception("EECutoffOn should equal to zero in DSF_elu")
		self.elu_width = PARAMS["Elu_Width"]
		self.elu_shift = DSF(self.elu_width*BOHRPERA, self.Ree_off*BOHRPERA, self.DSFAlpha/BOHRPERA)
		self.elu_alpha = DSF_Gradient(self.elu_width*BOHRPERA, self.Ree_off*BOHRPERA, self.DSFAlpha/BOHRPERA)

		self.NetType = "RawBP_EE_SymFunction"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+self.suffix
		self.train_dir = './networks/'+self.name
		self.keep_prob = np.asarray(PARAMS["KeepProb"])
		self.nlayer = len(PARAMS["KeepProb"]) - 1
		self.monitor_mset =  PARAMS["MonitorSet"]

	def SetANI1Param(self, prec=np.float64):
		"""
		Generate ANI1 symmetry function paramter tensor.
		"""
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
		self.inshape_withencode = int(self.inshape + AN1_num_r_Rs)
		#self.inshape = int(len(self.eles)*AN1_num_r_Rs)
		p1 = np.tile(np.reshape(thetas,[AN1_num_a_As,1,1]),[1,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(rs,[1,AN1_num_a_Rs,1]),[AN1_num_a_As,1,1])
		SFPa2 = np.concatenate([p1,p2],axis=2)
		self.SFPa2 = np.transpose(SFPa2, [2,0,1])
		p1_new = np.reshape(rs_R,[AN1_num_r_Rs,1])
		self.SFPr2 = np.transpose(p1_new, [1,0])
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]
		self.HasANI1PARAMS = True
		print ("self.inshape:", self.inshape)

	def Clean(self):
		"""
		Clean Instance for pickle saving.
		"""
		Instance.Clean(self)
		#self.tf_prec = None
		self.xyzs_pl, self.Zs_pl, self.label_pl, self.grads_pl = None, None, None, None
		self.check, self.options, self.run_metadata = None, None, None
		self.atom_outputs = None
		self.energy_loss, self.grads_loss, self.dipole_loss = None, None, None
		self.Scatter_Sym, self.Sym_Index = None, None
		self.Radp_pl, self.Angt_pl = None, None
		self.Elabel_pl, self.Dlabel_pl = None, None
		self.Reep_pl, self.natom_pl, self.AddEcc_pl = None, None, None
		self.Etotal, self.Ebp, self.Ecc, self.Ebp_atom, self.Evdw  = None, None, None, None, None
		self.dipole, self.charge = None, None
		self.energy_wb, self.dipole_wb = None, None
		self.gradient = None
		self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = None, None, None, None, None
		self.train_op_dipole, self.train_op_EandG = None, None
		self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = None, None, None, None, None
		self.Radius_Qs_Encode, self.Radius_Qs_Encode_Index = None, None
		self.Radp_Ele_pl, self.Angt_Elep_pl = None, None
		self.mil_jk_pl, self.mil_j_pl = None, None
		self.elu_width, self.elu_shift, self.elu_alpha = None, None, None
		self.keep_prob_pl = None
		return


	def TrainPrepare(self,  continue_training =False):
		"""
		Define Tensorflow graph for training.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]),name="DesEnergy")
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]),name="DesDipoles")
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="DesGrads")
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]),name="RadialElectros")
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			elu_width  = tf.Variable(self.elu_width, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			C6 = tf.Variable(self.C6,trainable=False, dtype = self.tf_prec)
			vdw_R = tf.Variable(self.vdw_R,trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Release(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl)
			self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, elu_width, Ree_off, self.Reep_pl, self.AddEcc_pl, self.keep_prob_pl)
			self.Radp_pl  = self.Radp_Ele_pl[:,:3]
			self.Etotal, self.Ebp, self.Evdw,  self.energy_wb, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.Ecc, self.xyzs_pl, self.Zs_pl, Ele, C6, vdw_R, self.Reep_pl, Ree_on, Ree_off, self.keep_prob_pl)
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl, self.natom_pl)
			self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl, self.natom_pl)
			self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl, self.natom_pl)
			tf.summary.scalar("loss", self.loss)
			tf.summary.scalar("loss_dip", self.loss_dipole)
			tf.summary.scalar("loss_EG", self.loss_EandG)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)
			self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
			config.gpu_options.per_process_gpu_memory_fraction = 0.90
			self.sess = tf.Session(config=config)
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess.run(init)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			if (PARAMS["Profiling"]>0):
				print("logging with FULL TRACE")
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
				self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
			self.sess.graph.finalize()

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
			raise Exception("Please check your inputs")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.Dlabel_pl] + [self.grads_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.Reep_pl] + [self.mil_j_pl] + [self.mil_jk_pl] + [self.natom_pl] + [self.AddEcc_pl] + [self.keep_prob_pl], batch_data)}
		return feed_dict

	def energy_inference(self, inp, indexs,  cc_energy, xyzs, Zs, eles, c6, R_vdw, Reep, EE_cuton, EE_cutoff, keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating energy.

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements.
			cc_energy: System Coulomb energy.
			xyzs: xyz coordinates of atoms.
			Zs: atomic number of atoms.
			eles: list of element type.
			c6: Grimmer C6 coefficient.
			R_vdw: Van der waals cutoff.
			Reep: Atom index of vdw pairs.
			EE_cuton: Where Coulomb is turned on.
			EE_cutoff: Where Coulomb is turned off.
			keep_prob: dropout prob of each layer.
		Returns:
			The BP graph energy output
		"""
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Ebranches=[]
		output = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		atom_outputs = []
		with tf.name_scope("EnergyNet"):
			for e in range(len(self.eles)):
				Ebranches.append([])
				inputs = inp[e]
				shp_in = tf.shape(inputs)
				index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(inputs, keep_prob[i]), weights) + biases))
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[i]), weights) + biases))
				with tf.name_scope(str(self.eles[e])+'_regression_linear'):
					shp = tf.shape(inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Ebranches[-1].append(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Ebranches[-1][-1])
					cut = tf.slice(Ebranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					atom_outputs.append(rshp)
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(index, [0,1], [shp_out[0],1])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
					output = tf.add(output, ToAdd)
				tf.verify_tensor_all_finite(output,"Nan in output!!!")
			bp_energy = tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size])
		total_energy = tf.add(bp_energy, cc_energy)
		vdw_energy = TFVdwPolyLR(xyzsInBohr, Zs, eles, c6, R_vdw, EE_cuton*BOHRPERA, Reep)
		total_energy_with_vdw = tf.add(total_energy, vdw_energy)
		energy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EnergyNet")
		return total_energy_with_vdw, bp_energy, vdw_energy, energy_vars, output

	def dipole_inference(self, inp, indexs, xyzs, natom, Elu_Width, EE_cutoff, Reep, AddEcc, keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating dipole.

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements.
			xyzs: xyz coordinates of atoms.
			natom: 1/(max of number of atoms in the set).
			Elu_Width: Width of the elu version of the Coulomb interaction.
			EE_cutoff: Where Coulomb is turned off
			Reep: Atom index of vdw pairs.
			AddEcc: Whether add Coulomb energy to the total energy
			keep_prob: dropout prob of each layer.
		Returns:
			The BP graph charge and dipole  output
		"""
		# convert the index matrix from bool to float
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Dbranches=[]
		atom_outputs_charge = []
		output_charge = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		dipole_wb = []
		with tf.name_scope("DipoleNet"):
			for e in range(len(self.eles)):
				Dbranches.append([])
				charge_inputs = inp[e]
				charge_shp_in = tf.shape(charge_inputs)
				charge_index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1_charge'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(charge_inputs, keep_prob[i]), weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)+"_charge"):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[i]), weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
				with tf.name_scope(str(self.eles[e])+'_regression_linear_charge'):
					charge_shp = tf.shape(charge_inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					dipole_wb.append(weights)
					dipole_wb.append(biases)
					Dbranches[-1].append(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Dbranches[-1][-1])
					cut = tf.slice(Dbranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					atom_outputs_charge.append(rshp)
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(charge_index, [0,1], [shp_out[0],1])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
					output_charge = tf.add(output_charge, ToAdd)
			tf.verify_tensor_all_finite(output_charge,"Nan in output!!!")
			netcharge = tf.reshape(tf.reduce_sum(output_charge, axis=1), [self.batch_size])
			delta_charge = tf.multiply(netcharge, natom)
			delta_charge_tile = tf.tile(tf.reshape(delta_charge,[self.batch_size,1]),[1, self.MaxNAtoms])
			scaled_charge =  tf.subtract(output_charge, delta_charge_tile)
			flat_dipole = tf.multiply(tf.reshape(xyzsInBohr,[self.batch_size*self.MaxNAtoms, 3]), tf.reshape(scaled_charge,[self.batch_size*self.MaxNAtoms, 1]))
			dipole = tf.reduce_sum(tf.reshape(flat_dipole,[self.batch_size, self.MaxNAtoms, 3]), axis=1)
		def f1(): return TFCoulombEluSRDSFLR(xyzsInBohr, scaled_charge, Elu_Width*BOHRPERA, Reep, tf.cast(self.DSFAlpha, self.tf_prec), tf.cast(self.elu_alpha,self.tf_prec), tf.cast(self.elu_shift,self.tf_prec))
		def f2(): return  tf.zeros([self.batch_size], dtype=self.tf_prec)
		cc_energy = tf.cond(AddEcc, f1, f2)
		return  cc_energy, dipole, scaled_charge, dipole_wb

	def loss_op(self, energy, energy_grads, dipole, Elabels, grads, Dlabels, natom):
		"""
		losss function that includes dipole loss, energy loss and gradient loss.
		"""
		maxatom=tf.cast(tf.shape(energy_grads)[2], self.tf_prec)
		energy_diff  = tf.multiply(tf.subtract(energy, Elabels,name="EnDiff"), natom*maxatom)
		energy_loss = tf.nn.l2_loss(energy_diff,name="EnL2")
		grads_diff = tf.multiply(tf.subtract(energy_grads, grads,name="GradDiff"), tf.reshape(natom*maxatom, [1, self.batch_size, 1, 1]))
		grads_loss = tf.nn.l2_loss(grads_diff,name="GradL2")
		dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels,name="DipoleDiff"), tf.reshape(natom*maxatom,[self.batch_size,1]))
		dipole_loss = tf.nn.l2_loss(dipole_diff,name="DipL2")
		EandG_loss = tf.add(tf.multiply(energy_loss, self.EnergyScalar), tf.multiply(grads_loss, self.GradScalar),name="MulLoss")
		loss = tf.add(EandG_loss, tf.multiply(dipole_loss, self.DipoleScalar))
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

	def loss_op_dipole(self, energy, energy_grads, dipole, Elabels, grads, Dlabels, natom):
		"""
		losss function that includes dipole loss.
		"""
		maxatom=tf.cast(tf.shape(energy_grads)[2], self.tf_prec)
		energy_diff  = tf.multiply(tf.subtract(energy, Elabels), natom*maxatom)
		energy_loss = tf.nn.l2_loss(energy_diff)
		grads_diff = tf.multiply(tf.subtract(energy_grads, grads), tf.reshape(natom*maxatom, [1, self.batch_size, 1, 1]))
		grads_loss = tf.nn.l2_loss(grads_diff)
		dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels), tf.reshape(natom*maxatom,[self.batch_size,1]))
		dipole_loss = tf.nn.l2_loss(dipole_diff)
		EandG_loss = tf.add(tf.multiply(energy_loss, self.EnergyScalar), tf.multiply(grads_loss, self.GradScalar))
		loss = tf.identity(dipole_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

	def loss_op_EandG(self, energy, energy_grads, dipole, Elabels, grads, Dlabels, natom):
		"""
		losss function that includes energy loss and gradient loss.
		"""
		maxatom=tf.cast(tf.shape(energy_grads)[2], self.tf_prec)
		energy_diff  = tf.multiply(tf.subtract(energy, Elabels), natom*maxatom)
		energy_loss = tf.nn.l2_loss(energy_diff)
		grads_diff = tf.multiply(tf.subtract(energy_grads, grads), tf.reshape(natom*maxatom, [1, self.batch_size, 1, 1]))
		grads_loss = tf.nn.l2_loss(grads_diff)
		dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels), tf.reshape(natom*maxatom,[self.batch_size,1]))
		dipole_loss = tf.nn.l2_loss(dipole_diff)
		EandG_loss = tf.add(tf.multiply(energy_loss, self.EnergyScalar), tf.multiply(grads_loss, self.GradScalar))
		loss = tf.identity(EandG_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

	def training(self, loss, learning_rate, momentum, update_var=None):
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
		optimizer = tf.train.AdamOptimizer(learning_rate,name="Adam")
		#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		if update_var == None:
			train_op = optimizer.minimize(loss, global_step=global_step, name="trainop")
		else:
			train_op = optimizer.minimize(loss, global_step=global_step, var_list=update_var, name="trainop")
		return train_op

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size.
		Training object including dipole, energy and gradient

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		train_dipole_loss = 0.0
		train_grads_loss = 0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data = self.TData.GetTrainBatch(self.batch_size)+[PARAMS["AddEcc"]] + [self.keep_prob]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.train_op, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss, self.Etotal, self.Ecc,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		return


	def train_step_EandG(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size
		Training object including energy and dipole.

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		train_dipole_loss = 0.0
		train_grads_loss = 0.0
		num_of_mols = 0

		print_per_mini = 100
		print_loss = 0.0
		print_energy_loss = 0.0
		print_dipole_loss = 0.0
		print_grads_loss = 0.0
		print_time = 0.0
		time_print_mini = time.time()
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			t_mini = time.time()
			batch_data = self.TData.GetTrainBatch(self.batch_size)+[PARAMS["AddEcc"]] + [self.keep_prob]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, Evdw, mol_dipole, atom_charge = self.sess.run([self.train_op_EandG, self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG, self.Etotal, self.Ecc, self.Evdw,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			print_loss += loss_value
			print_energy_loss += energy_loss
			print_grads_loss += grads_loss
			print_dipole_loss += dipole_loss
			if (ministep%print_per_mini == 0 and ministep!=0):
				print ("ministep... time:", (time.time() - time_print_mini)/print_per_mini ,  " loss_value: ",  print_loss/print_per_mini, " energy_loss:", print_energy_loss/print_per_mini, " grads_loss:", print_grads_loss/print_per_mini, " dipole_loss:", print_dipole_loss/print_per_mini)
				print_loss = 0.0
				print_energy_loss = 0.0
				print_dipole_loss = 0.0
				print_grads_loss = 0.0
				print_time = 0.0
				time_print_mini = time.time()
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		return


	def train_step_dipole(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size
		Training object including dipole.

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		train_energy_loss = 0.0
		train_dipole_loss = 0.0
		train_grads_loss = 0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		print_per_mini = 100
		print_loss = 0.0
		print_energy_loss = 0.0
		print_dipole_loss = 0.0
		print_grads_loss = 0.0
		print_time = 0.0
		time_print_mini = time.time()
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			t_mini = time.time()
			batch_data = self.TData.GetTrainBatch(self.batch_size) + [False] + [self.keep_prob]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.train_op_dipole, self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole, self.Etotal, self.Ecc,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			print_loss += loss_value
			print_energy_loss += energy_loss
			print_grads_loss += grads_loss
			print_dipole_loss += dipole_loss
			if (ministep%print_per_mini == 0 and ministep!=0):
				print ("ministep... time:", (time.time() - time_print_mini)/print_per_mini ,  " loss_value: ",  print_loss/print_per_mini, " energy_loss:", print_energy_loss/print_per_mini, " grads_loss:", print_grads_loss/print_per_mini, " dipole_loss:", print_dipole_loss/print_per_mini)
				print_loss = 0.0
				print_energy_loss = 0.0
				print_dipole_loss = 0.0
				print_grads_loss = 0.0
				print_time = 0.0
				time_print_mini = time.time()
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		return

	def test(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_test = self.TData.NTest
		start_time = time.time()
		test_loss =  0.0
		test_energy_loss = 0.0
		test_dipole_loss = 0.0
		test_grads_loss = 0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.TData.GetTestBatch(self.batch_size)+[PARAMS["AddEcc"]] + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			test_loss = test_loss + loss_value
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
			test_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		print ("testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, test_dipole_loss, num_of_mols, duration, False)
		return test_loss

	def test_dipole(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_test = self.TData.NTest
		start_time = time.time()
		test_loss =  0.0
		test_energy_loss = 0.0
		test_dipole_loss = 0.0
		test_grads_loss = 0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.TData.GetTestBatch(self.batch_size)+[False] + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			test_loss = test_loss + loss_value
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
			test_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		print ("testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, test_dipole_loss, num_of_mols, duration, False)
		return  test_loss

	def test_EandG(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_test = self.TData.NTest
		start_time = time.time()
		test_loss =  0.0
		test_energy_loss = 0.0
		test_dipole_loss = 0.0
		test_grads_loss = 0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.TData.GetTestBatch(self.batch_size)+[PARAMS["AddEcc"]] + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			test_loss = test_loss + loss_value
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
			test_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		print ("testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, test_dipole_loss, num_of_mols, duration, False)
		return  test_loss

	def train(self, mxsteps, continue_training= False):
		"""
		This the training loop for the united model.
		"""
		LOGGER.info("running the TFMolInstance.train()")
		self.TrainPrepare(continue_training)
		test_freq = PARAMS["test_freq"]
		mini_dipole_test_loss = float('inf') # some big numbers
		mini_energy_test_loss = float('inf')
		mini_test_loss = float('inf')
		for step in  range (0, mxsteps):
			if self.Training_Traget == "EandG":
				self.train_step_EandG(step)
				if step%test_freq==0 and step!=0 :
					if self.monitor_mset != None:
						self.InTrainEval(self.monitor_mset, self.Rr_cut, self.Ra_cut, self.Ree_off, step=step)
					test_energy_loss = self.test_EandG(step)
					if test_energy_loss < mini_energy_test_loss:
						mini_energy_test_loss = test_energy_loss
						self.save_chk(step)
			elif self.Training_Traget == "Dipole":
				self.train_step_dipole(step)
				if step%test_freq==0 and step!=0 :
					if self.monitor_mset != None:
						self.InTrainEval(self.monitor_mset, self.Rr_cut, self.Ra_cut, self.Ree_off, step=step)
					test_dipole_loss = self.test_dipole(step)
					if test_dipole_loss < mini_dipole_test_loss:
						mini_dipole_test_loss = test_dipole_loss
						self.save_chk(step)
						if step >= PARAMS["SwitchEpoch"]:
							self.Training_Traget = "EandG"
							print ("Switching to Energy and Gradient Learning...")
			else:
				self.train_step(step)
				if step%test_freq==0 and step!=0 :
					if self.monitor_mset != None:
						self.InTrainEval(self.monitor_mset, self.Rr_cut, self.Ra_cut, self.Ree_off, step=step)
					test_loss = self.test(step)
					if test_loss < mini_test_loss:
						mini_test_loss = test_loss
						self.save_chk(step)
		self.SaveAndClose()
		return

	def profile_step(self, step):
		"""
		Perform a single profiling step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		time_print_mini = time.time()
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			t_mini = time.time()
			batch_data = self.TData.GetTrainBatch(self.batch_size) + [PARAMS["AddEcc"]] + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.train_op_dipole, self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole, self.Etotal, self.Ecc,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data), options=self.options, run_metadata=self.run_metadata)
			print ("inference time:", time.time() - t)
			self.summary_writer.add_run_metadata(self.run_metadata, 'minstep%d' % ministep)
			duration = time.time() - start_time
			num_of_mols += actual_mols
			fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			chrome_trace = fetched_timeline.generate_chrome_trace_format()
			with open('timeline_step_%d.json' % ministep, 'w') as f:
				f.write(chrome_trace)
		return

	def profile(self):
		"""
		This profiles a training step.
		"""
		LOGGER.info("running the TFMolInstance.train()")
		self.TrainPrepare(False)
		self.profile_step(1)
		return

	def InTrainEval(self, mol_set, Rr_cut, Ra_cut, Ree_cut, step=0):
		"""
		Evaluted the network during training.
		"""
		nmols = len(mol_set.mols)
		for i in range(nmols, self.batch_size):
			mol_set.mols.append(mol_set.mols[-1])
		nmols = len(mol_set.mols)
		dummy_energy = np.zeros((nmols))
		dummy_dipole = np.zeros((nmols, 3))
		xyzs = np.zeros((nmols, self.MaxNAtoms, 3), dtype = np.float64)
		dummy_grads = np.zeros((nmols, self.MaxNAtoms, 3), dtype = np.float64)
		Zs = np.zeros((nmols, self.MaxNAtoms), dtype = np.int32)
		natom = np.zeros((nmols), dtype = np.int32)
		for i, mol in enumerate(mol_set.mols):
			xyzs[i][:mol.NAtoms()] = mol.coords
			Zs[i][:mol.NAtoms()] = mol.atoms
			natom[i] = mol.NAtoms()
		NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
		rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(Rr_cut, Ra_cut, self.eles_np, self.eles_pairs_np)
		NLEE = NeighborListSet(xyzs, natom, False, False,  None)
		rad_eep = NLEE.buildPairs(Ree_cut)
		batch_data = [xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom]
		feed_dict=self.fill_feed_dict(batch_data+[PARAMS["AddEcc"]]+[np.ones(self.nlayer+1)])
		Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient= self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.Ecc, self.Evdw, self.dipole, self.charge, self.gradient], feed_dict=feed_dict)
		monitor_data = [Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient]
		f = open(self.name+"_monitor_"+str(step)+".dat","wb")
		pickle.dump(monitor_data, f)
		f.close()
		print ("calculating monitoring set..")
		return Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient

	def print_training(self, step, loss, energy_loss, grads_loss, dipole_loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f  energy_loss: %.10f  grad_loss: %.10f, dipole_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)), (float(dipole_loss)/(Ncase)))
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f energy_loss: %.10f  grad_loss: %.10f, dipole_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)), (float(dipole_loss)/(Ncase)))
		return

	def evaluate(self, batch_data):
		"""
		Evaluate the energy, atom energies, and IfGrad = True the gradients
		of this Direct Behler-Parinello graph.
		"""
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		self.activation_function_type = PARAMS["NeuronType"]
		self.AssignActivation()
		#print ("self.activation_function:\n\n", self.activation_function)
		if (batch_data[0].shape[1] != self.MaxNAtoms):
			self.MaxNAtoms = batch_data[0].shape[1]
			self.batch_size = nmol
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare()
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
		if not self.sess:
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare()
		feed_dict=self.fill_feed_dict(batch_data+[PARAMS["AddEcc"]]+[np.ones(self.nlayer+1)])
		Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.Ecc, self.Evdw, self.dipole, self.charge, self.gradient], feed_dict=feed_dict)
		return Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient

	def EvalPrepare(self,  continue_training =False):
		"""
		Generate Tensorflow graph of evalution.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]),name="DesEnergy")
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]),name="DesDipoles")
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="DesGrads")
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]),name="RadialElectros")
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			elu_width  = tf.Variable(self.elu_width, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			C6 = tf.Variable(self.C6,trainable=False, dtype = self.tf_prec)
			vdw_R = tf.Variable(self.vdw_R,trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Release(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl)
			self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, elu_width, Ree_off, self.Reep_pl, self.AddEcc_pl, self.keep_prob_pl)
			self.Radp_pl  = self.Radp_Ele_pl[:,:3]
			self.Etotal, self.Ebp, self.Evdw,  self.energy_wb, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.Ecc, self.xyzs_pl, self.Zs_pl, Ele, C6, vdw_R, self.Reep_pl, Ree_on, Ree_off, self.keep_prob_pl)
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			self.bp_gradient  = tf.gradients(self.Ebp, self.xyzs_pl, name="BPGrad")
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			if (PARAMS["Profiling"]>0):
				print("logging with FULL TRACE")
				self.summary_writer = tf.summary.FileWriter('./networks/PROFILE', self.sess.graph)
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
				self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
			self.sess.graph.finalize()

	def energy_inference_periodic(self, inp, indexs, cc_energy, xyzs, Zs, eles, c6, R_vdw, Reep_e1e2, EE_cuton, EE_cutoff, keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating the energy of periodic system

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Ebranches=[]
		output = tf.zeros([self.batch_size, self.nreal], dtype=self.tf_prec)
		atom_outputs = []
		with tf.name_scope("EnergyNet"):
			for e in range(len(self.eles)):
				Ebranches.append([])
				inputs = inp[e]
				shp_in = tf.shape(inputs)
				index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(inputs, keep_prob[i]), weights) + biases))
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[i]), weights) + biases))
				with tf.name_scope(str(self.eles[e])+'_regression_linear'):
					shp = tf.shape(inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Ebranches[-1].append(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Ebranches[-1][-1])
					cut = tf.slice(Ebranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					atom_outputs.append(rshp)
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(index, [0,1], [shp_out[0],1])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.nreal]),[self.batch_size, self.nreal])
					output = tf.add(output, ToAdd)
				tf.verify_tensor_all_finite(output,"Nan in output!!!")
			bp_energy = tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size])
		total_energy = tf.add(bp_energy, cc_energy)
		vdw_energy = TFVdwPolyLRWithEle(xyzsInBohr, Zs, eles, c6, R_vdw, EE_cuton*BOHRPERA, Reep_e1e2)/2.0
		total_energy_with_vdw = tf.add(total_energy, vdw_energy)
		energy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EnergyNet")
		return total_energy_with_vdw, bp_energy, vdw_energy, energy_vars, output

	def dipole_inference_periodic(self, inp, indexs, xyzs, natom, Elu_Width, EE_cutoff, Reep, AddEcc, keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating the atom charges of periodic system.

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		# convert the index matrix from bool to float
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		xyzs_real = xyzsInBohr[:,:self.nreal]
		Dbranches=[]
		atom_outputs_charge = []
		output_charge = tf.zeros([self.batch_size, self.nreal], dtype=self.tf_prec)
		dipole_wb = []
		with tf.name_scope("DipoleNet"):
			for e in range(len(self.eles)):
				Dbranches.append([])
				charge_inputs = inp[e]
				charge_shp_in = tf.shape(charge_inputs)
				charge_index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1_charge'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(charge_inputs, keep_prob[i]), weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)+"_charge"):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[i]), weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
				with tf.name_scope(str(self.eles[e])+'_regression_linear_charge'):
					charge_shp = tf.shape(charge_inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					dipole_wb.append(weights)
					dipole_wb.append(biases)
					Dbranches[-1].append(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Dbranches[-1][-1])
					cut = tf.slice(Dbranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					atom_outputs_charge.append(rshp)
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(charge_index, [0,1], [shp_out[0],1])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.nreal]),[self.batch_size, self.nreal])
					output_charge = tf.add(output_charge, ToAdd)
			tf.verify_tensor_all_finite(output_charge,"Nan in output!!!")
			netcharge = tf.reshape(tf.reduce_sum(output_charge, axis=1), [self.batch_size])
			delta_charge = tf.multiply(netcharge, natom)
			delta_charge_tile = tf.tile(tf.reshape(delta_charge,[self.batch_size,1]),[1, self.nreal])
			scaled_charge =  tf.subtract(output_charge, delta_charge_tile)
			flat_dipole = tf.multiply(tf.reshape(xyzs_real,[self.batch_size*self.nreal, 3]), tf.reshape(scaled_charge,[self.batch_size*self.nreal, 1]))
			dipole = tf.reduce_sum(tf.reshape(flat_dipole,[self.batch_size, self.nreal, 3]), axis=1)

		ntess = tf.cast(tf.div(self.MaxNAtoms, self.nreal), dtype=tf.int32)
		scaled_charge_all = tf.tile(scaled_charge, [1, ntess])
		def f1(): return TFCoulombEluSRDSFLR(xyzsInBohr, scaled_charge_all, Elu_Width*BOHRPERA, Reep, tf.cast(self.DSFAlpha, self.tf_prec), tf.cast(self.elu_alpha,self.tf_prec), tf.cast(self.elu_shift,self.tf_prec))
		def f2(): return  tf.zeros([self.batch_size], dtype=self.tf_prec)
		cc_energy = tf.cond(AddEcc, f1, f2)/2.0
		return  cc_energy, dipole, scaled_charge_all, dipole_wb

	def fill_feed_dict_periodic(self, batch_data):
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
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.Dlabel_pl] + [self.grads_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.Reep_e1e2_pl] + [self.mil_j_pl]  + [self.mil_jk_pl] + [self.natom_pl] + [self.AddEcc_pl] + [self.keep_prob_pl], batch_data)}
		return feed_dict

	@TMTiming("EvalPeriodic")
	def evaluate_periodic(self, batch_data, nreal,DoForce = True):
		"""
		Evaluate the energy, atom energies, and IfGrad = True the gradients
		of this Direct Behler-Parinello graph for a periodic system.
		"""
		nmol = batch_data[2].shape[0]
		self.nreal = nreal
		self.activation_function_type = PARAMS["NeuronType"]
		self.AssignActivation()
		if (batch_data[0].shape[1] != self.MaxNAtoms):
			self.MaxNAtoms = batch_data[0].shape[1]
			self.batch_size = nmol
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare_Periodic()
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
		if not self.sess:
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare_Periodic()
		t0 = time.time()
		feed_dict=self.fill_feed_dict_periodic(batch_data+[PARAMS["AddEcc"]]+[np.ones(self.nlayer+1)])
		if (DoForce):
			Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.Ecc, self.Evdw, self.dipole, self.charge, self.gradient], feed_dict=feed_dict)
			return Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient
		else:
			Etotal = self.sess.run(self.Etotal, feed_dict=feed_dict)
			return Etotal

	def EvalPrepare_Periodic(self,  continue_training =False):
		"""
		Generate Tensorlfow graph for evalution of periodic system.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]),name="DesEnergy")
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]),name="DesDipoles")
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="DesGrads")
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_e1e2_pl=tf.placeholder(tf.int64, shape=tuple([None,5]),name="RadialElectros")
			self.Reep_pl = self.Reep_e1e2_pl[:,:3]
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			elu_width  = tf.Variable(self.elu_width, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			C6 = tf.Variable(self.C6,trainable=False, dtype = self.tf_prec)
			vdw_R = tf.Variable(self.vdw_R,trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Periodic(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl, self.nreal)
			self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference_periodic(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, elu_width, Ree_off, self.Reep_pl, self.AddEcc_pl, self.keep_prob_pl)
			self.Radp_pl  = self.Radp_Ele_pl[:,:3]
			self.Etotal, self.Ebp, self.Evdw,  self.energy_wb, self.Ebp_atom = self.energy_inference_periodic(self.Scatter_Sym, self.Sym_Index, self.Ecc, self.xyzs_pl, self.Zs_pl, Ele, C6, vdw_R, self.Reep_e1e2_pl, Ree_on, Ree_off, self.keep_prob_pl)
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			if (PARAMS["Profiling"]>0):
				print("logging with FULL TRACE")
				self.summary_writer = tf.summary.FileWriter('./networks/PROFILE', self.sess.graph)
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
				self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
			self.sess.graph.finalize()


class MolInstance_DirectBP_Charge_SymFunction(MolInstance_fc_sqdiff_BP):
	"""
	Electrostatic embedding Behler Parinello with van der waals interaction implemented with Grimme C6 scheme.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
			Trainable_: True for training, False for evalution
			ForceType_: Deprecated
		"""
		self.SFPa = None
		self.SFPr = None
		self.Ra_cut = None
		self.Rr_cut = None
		self.HasANI1PARAMS = False
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.MaxNAtoms = self.TData.MaxNAtoms
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
		self.eles_pairs = []
		for i in range (len(self.eles)):
			for j in range(i, len(self.eles)):
				self.eles_pairs.append([self.eles[i], self.eles[j]])
		self.eles_pairs_np = np.asarray(self.eles_pairs)
		if not self.HasANI1PARAMS:
			self.SetANI1Param()
		self.HiddenLayers = PARAMS["HiddenLayers"]
		self.batch_size = PARAMS["batch_size"]
		print ("self.activation_function_type: ", self.activation_function_type)
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		self.xyzs_pl = None
		self.Zs_pl = None
		self.sess = None
		self.total_loss = None
		self.dipole_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None
		self.learning_rate_dipole = PARAMS["learning_rate_dipole"]
		self.suffix = PARAMS["NetNameSuffix"]
		self.Ree_off  = PARAMS["EECutoffOff"]
		self.SetANI1Param()
		self.run_metadata = None
		self.Training_Traget = "Dipole"
		self.TData.ele = self.eles_np
		self.TData.elep = self.eles_pairs_np
		self.Training_Traget = "Dipole"

		self.NetType = "RawBP_Charge_SymFunction"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+self.suffix
		self.train_dir = './networks/'+self.name
		self.keep_prob = np.asarray(PARAMS["KeepProb"])
		self.nlayer = len(PARAMS["KeepProb"]) - 1
		self.monitor_mset =  PARAMS["MonitorSet"]
	
		print ("self.eles_pairs:",self.eles_pairs)
		

	def SetANI1Param(self, prec=np.float64):
		"""
		Generate ANI1 symmetry function paramter tensor.
		"""
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
		self.inshape_withencode = int(self.inshape + AN1_num_r_Rs)
		#self.inshape = int(len(self.eles)*AN1_num_r_Rs)
		p1 = np.tile(np.reshape(thetas,[AN1_num_a_As,1,1]),[1,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(rs,[1,AN1_num_a_Rs,1]),[AN1_num_a_As,1,1])
		SFPa2 = np.concatenate([p1,p2],axis=2)
		self.SFPa2 = np.transpose(SFPa2, [2,0,1])
		p1_new = np.reshape(rs_R,[AN1_num_r_Rs,1])
		self.SFPr2 = np.transpose(p1_new, [1,0])
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]
		self.HasANI1PARAMS = True
		print ("self.inshape:", self.inshape)

	def Clean(self):
		"""
		Clean Instance for pickle saving.
		"""
		Instance.Clean(self)
		#self.tf_prec = None
		self.xyzs_pl, self.Zs_pl = None, None
		self.check, self.options, self.run_metadata = None, None, None
		self.atom_outputs = None
		self.dipole_loss = None
		self.Scatter_Sym, self.Sym_Index = None, None
		self.Radp_pl, self.Angt_pl = None, None
		self.Dlabel_pl = None
		self.Reep_pl, self.Reep_e1e2_pl, self.natom_pl = None, None, None
		self.dipole, self.charge = None, None
		self.dipole_wb = None, None
		self.total_loss_dipole, self.loss_dipole = None, None
		self.train_op_dipole = None
		self.Radp_Ele_pl, self.Angt_Elep_pl = None, None
		self.mil_jk_pl, self.mil_j_pl = None, None
		self.keep_prob_pl = None
		self.eleneg  = None
		return


	def TrainPrepare(self,  continue_training =False):
		"""
		Define Tensorflow graph for training.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]),name="DesDipoles")
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.mil_j_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_e1e2_pl=tf.placeholder(tf.int64, shape=tuple([None,5]),name="RadialElectros")
			self.Reep_pl = self.Reep_e1e2_pl[:,:3]
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle_Release(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_j_pl, self.mil_jk_pl)
			#self.dipole, self.charge = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, self.Reep_e1e2_pl,  self.keep_prob_pl)
			self.eleneg = self.eleneg_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, self.keep_prob_pl)
			self.ini_charge = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
			self.dipole, self.charge = self.pairwise_charge_exchange(self.Sym_Index, self.xyzs_pl, self.Zs_pl, self.eleneg, self.ini_charge, Ele, Elep, self.Reep_e1e2_pl, self.keep_prob_pl)
			self.total_loss, self.dipole_loss = self.loss_op(self.dipole, self.Dlabel_pl, self.natom_pl)
			tf.summary.scalar("loss", self.total_loss)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
			config.gpu_options.per_process_gpu_memory_fraction = 0.90
			self.sess = tf.Session(config=config)
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.sess.run(init)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			if (PARAMS["Profiling"]>0):
				print("logging with FULL TRACE")
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
				self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
			self.sess.graph.finalize()

	def fill_feed_dict(self, batch_data):
		"""
		Fill the tensorflow feed dictionary.

		Args:
			batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
			and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

		Returns:
			Filled feed dictionary.
		"""
		feed_dict={i: d for i, d in zip([self.xyzs_pl] + [self.Zs_pl] + [self.Dlabel_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.Reep_e1e2_pl] + [self.mil_j_pl] + [self.mil_jk_pl] + [self.natom_pl] + [self.keep_prob_pl], batch_data)}
		return feed_dict


	def dipole_inference(self, inp, indexs, xyzs, natom, Reep_e1e2,  keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating dipole.

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements.
			xyzs: xyz coordinates of atoms.
			natom: 1/(max of number of atoms in the set).
			Elu_Width: Width of the elu version of the Coulomb interaction.
			EE_cutoff: Where Coulomb is turned off
			Reep: Atom index of vdw pairs.
			AddEcc: Whether add Coulomb energy to the total energy
			keep_prob: dropout prob of each layer.
		Returns:
			The BP graph charge and dipole  output
		"""
		# convert the index matrix from bool to float
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Dbranches=[]
		atom_outputs_charge = []
		output_charge = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		dipole_wb = []
		with tf.name_scope("DipoleNet"):
			for e in range(len(self.eles)):
				Dbranches.append([])
				charge_inputs = inp[e]
				charge_shp_in = tf.shape(charge_inputs)
				charge_index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1_charge'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(charge_inputs, keep_prob[i]), weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)+"_charge"):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[i]), weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
				with tf.name_scope(str(self.eles[e])+'_regression_linear_charge'):
					charge_shp = tf.shape(charge_inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					dipole_wb.append(weights)
					dipole_wb.append(biases)
					Dbranches[-1].append(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Dbranches[-1][-1])
					cut = tf.slice(Dbranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					atom_outputs_charge.append(rshp)
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(charge_index, [0,1], [shp_out[0],1])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
					output_charge = tf.add(output_charge, ToAdd)
			tf.verify_tensor_all_finite(output_charge,"Nan in output!!!")
			netcharge = tf.reshape(tf.reduce_sum(output_charge, axis=1), [self.batch_size])
			delta_charge = tf.multiply(netcharge, natom)
			delta_charge_tile = tf.tile(tf.reshape(delta_charge,[self.batch_size,1]),[1, self.MaxNAtoms])
			scaled_charge =  tf.subtract(output_charge, delta_charge_tile)
			flat_dipole = tf.multiply(tf.reshape(xyzsInBohr,[self.batch_size*self.MaxNAtoms, 3]), tf.reshape(scaled_charge,[self.batch_size*self.MaxNAtoms, 1]))
			dipole = tf.reduce_sum(tf.reshape(flat_dipole,[self.batch_size, self.MaxNAtoms, 3]), axis=1)

		return  dipole, scaled_charge

	def eleneg_inference(self, inp, indexs, xyzs, natom, keep_prob):
		"""
		Builds a Behler-Parinello graph for calculating electronic negativity for atoms.

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements.
			xyzs: xyz coordinates of atoms.
			keep_prob: dropout prob of each layer.
		Returns:
			The BP graph electronic negativity  output
		"""
		# convert the index matrix from bool to float
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Dbranches=[]
		output_eleneg = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		with tf.name_scope("EleNegNet"):
			for e in range(len(self.eles)):
				Dbranches.append([])
				eleneg_inputs = inp[e]
				eleneg_shp_in = tf.shape(eleneg_inputs)
				eleneg_index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1_eleneg'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(eleneg_inputs, keep_prob[i]), weights) + biases))
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)+"_eleneg"):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[i]), weights) + biases))
				with tf.name_scope(str(self.eles[e])+'_regression_linear_eleneg'):
					eleneg_shp = tf.shape(eleneg_inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Dbranches[-1].append(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Dbranches[-1][-1])
					cut = tf.slice(Dbranches[-1][-1],[0,0],[shp_out[0],1])
					rshp = tf.reshape(cut,[1,shp_out[0]])
					rshpflat = tf.reshape(cut,[shp_out[0]])
					atom_indice = tf.slice(eleneg_index, [0,1], [shp_out[0],1])
					ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
					output_eleneg = tf.add(output_eleneg, ToAdd)
			tf.verify_tensor_all_finite(output_eleneg,"Nan in output!!!")
		return  output_eleneg


	def pairwise_charge_exchange(self, indexs, xyzs, Zs, eleneg, charge, Ele, Elep, Reep_e1e2, keep_prob):
		"""
		Pairwise charge exchange model.

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements.
			xyzs: xyz coordinates of atoms.
			ini_charge: initial charge
			eleneg: electronic negativity determined by the BP network.
			keep_prob: dropout prob of each layer.
		Returns:
			The BP graph electronic negativity  output
		"""
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Cbranches = []
		pair_indexs = []
		npair, pair_dim = Reep_e1e2.get_shape().as_list()
		ele1 = tf.gather_nd(tf.reshape(Ele,[2]), tf.reshape(Reep_e1e2[:,3],[-1,1]))
		ele2 = tf.gather_nd(tf.reshape(Ele,[2]), tf.reshape(Reep_e1e2[:,4],[-1,1]))
		ele12 = tf.transpose(tf.stack([ele1, ele2]))
		with tf.name_scope("ChargeNet"):
			delta_charge = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
			for pair in range(len(self.eles_pairs)):
				mask = tf.reduce_all(tf.equal(Elep[pair], ele12), 1)
				masked  = tf.boolean_mask(Reep_e1e2, mask)
				neg1 = tf.gather_nd(eleneg, masked[:,:2])
				neg2 = tf.gather_nd(eleneg, tf.transpose(tf.stack([masked[:,0], masked[:,2]])))
				charge1 = tf.gather_nd(charge, masked[:,:2])
				charge2 = tf.gather_nd(charge, tf.transpose(tf.stack([masked[:,0], masked[:,2]])))
				dist = TFDistancesLinear(xyzs, masked[:,:3])
				pair_indexs.append(masked)
				charge_inputs = tf.transpose(tf.stack([neg1, charge1, neg2, charge2, 1.0/dist]))
				Cbranches.append([])
				charge_shp_in = tf.shape(charge_inputs)
					
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(pair)+'_hidden1_charge'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[5, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(5.0)), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Cbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(charge_inputs, keep_prob[i]), weights) + biases))
					else:
						with tf.name_scope(str(pair)+'_hidden'+str(i+1)+"_charge"):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Cbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Cbranches[-1][-1], keep_prob[i]), weights) + biases))
				with tf.name_scope(str(pair)+'_regression_linear_charge'):
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Cbranches[-1].append(tf.matmul(tf.nn.dropout(Cbranches[-1][-1], keep_prob[-1]), weights) + biases)
					shp_out = tf.shape(Cbranches[-1][-1])
					rshp = tf.reshape(Cbranches[-1][-1],[-1])
					delta_charge += tf.scatter_nd(masked[:,:2], rshp, [self.batch_size, self.MaxNAtoms])
					delta_charge -= tf.scatter_nd(tf.transpose(tf.stack([masked[:,0], masked[:,2]])), rshp, [self.batch_size, self.MaxNAtoms])	
			charge += delta_charge
			dipole = tf.reduce_sum(tf.multiply(xyzsInBohr, tf.reshape(charge,[self.batch_size,self.MaxNAtoms,1])), axis=1)	
		return  dipole, charge


	def loss_op(self, dipole, Dlabels, natom):
		"""
		losss function that includes dipole loss, energy loss and gradient loss.
		"""
		dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels,name="DipoleDiff"), tf.reshape(natom*100.0,[self.batch_size,1]))
		dipole_loss = tf.nn.l2_loss(dipole_diff,name="DipL2")
		tf.add_to_collection('losses', dipole_loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), dipole_loss


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
		optimizer = tf.train.AdamOptimizer(learning_rate,name="Adam")
		#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step, name="trainop")
		return train_op

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size.
		Training object including dipole, energy and gradient

		Args:
			step: the index of this step.
		"""
		Ncase_train = self.TData.NTrain
		start_time = time.time()
		train_loss =  0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			batch_data = self.TData.GetTrainBatch(self.batch_size)+ [self.keep_prob]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, total_loss_value, dipole_loss, dipole, charge = self.sess.run([self.train_op, self.total_loss, self.dipole_loss, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			train_loss = train_loss + dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#print ("dipole:", dipole, dipole.shape, dipole[0], np.sum(dipole[0]))
			#print ("charge:", charge, charge.shape, charge[0], np.sum(charge[0]))
		self.print_training(step, train_loss, num_of_mols, duration)
		return


	def test(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_test = self.TData.NTest
		start_time = time.time()
		test_loss =  0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.TData.GetTestBatch(self.batch_size) + [np.ones(self.nlayer+1)]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, dipole_loss, mol_dipole, atom_charge = self.sess.run([self.total_loss, self.dipole_loss, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			test_loss = test_loss + dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
		print ("testing...")
		print ("charge:", atom_charge, atom_charge.shape, atom_charge[0], np.sum(atom_charge[0]))
		self.print_training(step, test_loss, num_of_mols, duration, False)
		return test_loss

	def train(self, mxsteps, continue_training= False):
		"""
		This the training loop for the united model.
		"""
		LOGGER.info("running the TFMolInstance.train()")
		self.TrainPrepare(continue_training)
		test_freq = PARAMS["test_freq"]
		mini_dipole_test_loss = float('inf') # some big numbers
		mini_energy_test_loss = float('inf')
		mini_test_loss = float('inf')
		for step in  range (0, mxsteps):
			self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					self.save_chk(step)
		self.SaveAndClose()
		return


	def print_training(self, step, dipole_loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  dipole_loss: %.10f", step, duration,  (float(dipole_loss)/(Ncase)))
		else:
			LOGGER.info("step: %7d  duration: %.5f  dipole_loss: %.10f", step, duration,  (float(dipole_loss)/(Ncase)))
		return
