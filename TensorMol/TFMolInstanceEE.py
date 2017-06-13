from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TensorMol.TFInstance import *
from TensorMol.TensorMolData import *
from TensorMol.TFMolInstance import *
from TensorMol.ElectrostaticsTF import *

class MolInstance_EE(MolInstance_fc_sqdiff_BP):
	"""
		An electrostatically-embedded Behler Parinello
		--------------------
		A network with both energy and charge branches.
		Loss is : l2(Energies)+l2(Dipole)
		Network Yields: E_T,E_BP,E_electrostatic,Charges
		Only neutral systems supported as of now.

		This also uses TF's new scatter/gathers/einsum
		and Einsum to reduce the overhead of the BP scheme.

		The whole energy is differentiable and can yield forces.
		Requires TensorMolData_EE

		E_\text{electrostatic} The electrostatic energy is attenuated to only exist
		outside PARAMS["EECutoff"]
		# A couple cutoff schemes are available
		PARAMS["EESwitchFunc"]=='Cos':
			1/r -> (0.5*(cos(PI*r/EECutoff)+1))/r (if r<Cutoff else 0)
		PARAMS["EESwitchFunc"]=='Tanh'
			1/r => 0.5*(Tanh[(x - EECutoff)/EEdr] + 1)/r
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True):
		"""
		Raise a Behler-Parinello TensorFlow instance.

		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "fc_sqdiff_BPEE"
		MolInstance_fc_sqdiff_BP.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "fc_sqdiff_BPEE"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+str(self.TData.order)+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		self.TData.LoadDataToScratch(self.tformer)
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		print("self.MeanNumAtoms: ",self.MeanNumAtoms)
		# allow for 120% of required output space, since it's cheaper than input space to be padded by zeros.
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		#self.TData.CheckBPBatchsizes(self.batch_size, self.batch_size_output)
		LOGGER.info("Assigned batch input size: ",self.batch_size)
		LOGGER.info("Assigned batch output size: ",self.batch_size_output)
		with tf.Graph().as_default():
			self.inp_pl=[]
			self.mats_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.batch_size_output])))
			self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			#self.summary_op = tf.summary.merge_all()
			#init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
#			try: # I think this may be broken
#				metafiles = [x for x in os.listdir(self.train_dir) if (x.count('meta')>0)]
#				if (len(metafiles)>0):
#					most_recent_meta_file=metafiles[0]
#					LOGGER.info("Restoring training from Metafile: "+most_recent_meta_file)
#					#Set config to allow soft device placement for temporary fix to known issue with Tensorflow up to version 0.12 atleast - JEH
#					config = tf.ConfigProto(allow_soft_placement=True)
#					self.sess = tf.Session(config=config)
#					self.saver = tf.train.import_meta_graph(self.train_dir+'/'+most_recent_meta_file)
#					self.saver.restore(self.sess, tf.train.latest_checkpoint(self.train_dir))
#			except Exception as Ex:
#				print("Restore Failed",Ex)
#				pass
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def loss_op(self, output, E_mats, Q_mats, labels):
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

	def inference(self, inp_pl, mats_pl):
		"""
		Builds a Behler-Parinello graph which also matches monopole,3-dipole, and 9-quadropole elements.

		- It has the shape of two energy networks in parallel.
		- One network produces the energy, the other produces the charge on an atom.
		- The charges are constrained to reproduce the molecular multipoles.
		- The energy and charge are together constrained to produce the molecular energy.
		- The same atomic linear transformation is used to produce the charges as the energy.
		- All multipoles have the form of a dot product with atomic charges. That's pre-computed externally
		The attenuated coulomb energy has the form of a per-molecule vector-matrix-vector product.
		And it is generated as well by this routine.

		Args:
				inp_pl: a list of (num_of atom type X flattened input shape) matrix of input cases.
				mats_pl: a list of (num_of atom type X batchsize) matrices which linearly combines the elements to give molecular outputs.
				mul_pl: Multipole inputs (see Mol::GenerateMultipoleInputs)
		Returns:
			Atom BP Energy, Atom Charges
			I'm thinking about doing the contractions for the multipoles and electrostatic energy loss in loss_op... haven't settled on it yet.

		"""
		# convert the index matrix from bool to float
		branches=[]
		atom_outputs = []
		charge_branches=[]
		charge_atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3
		output = tf.zeros([self.batch_size_output])
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		LOGGER.info("Norms: %f,%f,%f", nrm1,nrm2,nrm3)
		#print(inp_pl)
		#tf.Print(inp_pl, [inp_pl], message="This is input: ",first_n=10000000,summarize=100000000)
		#tf.Print(bnds_pl, [bnds_pl], message="bnds_pl: ",first_n=10000000,summarize=100000000)
		#tf.Print(mats_pl, [mats_pl], message="mats_pl: ",first_n=10000000,summarize=100000000)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp_pl[e]
			mats = mats_pl[e]
			shp_in = tf.shape(inputs)
			if (PARAMS["check_level"]>2):
				tf.Print(tf.to_float(shp_in), [tf.to_float(shp_in)], message="Element "+str(e)+"input shape ",first_n=10000000,summarize=100000000)
				mats_shape = tf.shape(mats)
				tf.Print(tf.to_float(mats_shape), [tf.to_float(mats_shape)], message="Element "+str(e)+"mats shape ",first_n=10000000,summarize=100000000)
			if (PARAMS["check_level"]>3):
				tf.Print(tf.to_float(inputs), [tf.to_float(inputs)], message="This is input shape ",first_n=10000000,summarize=100000000)
			with tf.name_scope(str(self.eles[e])+'_hidden_1'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev=nrm1, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(inputs, weights) + biases))
				#
				charge_weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev=nrm1, var_wd=0.001)
				charge_biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
				charge_branches[-1].append(tf.nn.relu(tf.matmul(inputs, charge_weights) + charge_biases))
			with tf.name_scope(str(self.eles[e])+'_hidden_2'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev=nrm2, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(branches[-1][-1], weights) + biases))
				#
				charge_weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev=nrm2, var_wd=0.001)
				charge_biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
				charge_branches[-1].append(tf.nn.relu(tf.matmul(branches[-1][-1], charge_weights) + charge_biases))
			with tf.name_scope(str(self.eles[e])+'_hidden_3'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev=nrm3, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden3_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(branches[-1][-1], weights) + biases))
				#
				charge_weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev=nrm3, var_wd=0.001)
				charge_biases = tf.Variable(tf.zeros([hidden3_units]), name='biases')
				charge_branches[-1].append(tf.nn.relu(tf.matmul(charge_branches[-1][-1], charge_weights) + charge_biases))
			with tf.name_scope(str(self.eles[e])+'_regression_linear'):
				shp = tf.shape(inputs)
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden3_units, 1], var_stddev=nrm4, var_wd=None)
				biases = tf.Variable(tf.zeros([1]), name='biases')
				branches[-1].append(tf.matmul(branches[-1][-1], weights) + biases)
				#
				charge_weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden3_units, 1], var_stddev=nrm4, var_wd=None)
				charge_biases = tf.Variable(tf.zeros([1]), name='biases')
				charge_branches[-1].append(tf.matmul(charge_branches[-1][-1], charge_weights) + charge_biases)
				#
				shp_out = tf.shape(branches[-1][-1])
				cut = tf.slice(branches[-1][-1],[0,0],[shp_out[0],1])
				#tf.Print(tf.to_float(shp_out), [tf.to_float(shp_out)], message="This is outshape: ",first_n=10000000,summarize=100000000)
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				#
				charge_shp_out = tf.shape(charge_branches[-1][-1])
				charge_cut = tf.slice(charge_branches[-1][-1],[0,0],[charge_shp_out[0],1])
				#tf.Print(tf.to_float(shp_out), [tf.to_float(shp_out)], message="This is outshape: ",first_n=10000000,summarize=100000000)
				charge_rshp = tf.reshape(charge_cut,[1,charge_shp_out[0]])
				charge_atom_outputs.append(charge_rshp)
				#
				tmp = tf.matmul(rshp,mats)
				output = tf.add(output,tmp)
		tf.verify_tensor_all_finite(output,"Nan in output!!!")
		#tf.Print(output, [output], message="This is output: ",first_n=10000000,summarize=100000000)
		return output, atom_outputs, atom_charge_outputs

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
		return test_loss, feed_dict


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
		return test_loss, feed_dict


	def print_training(self, step, loss, Ncase, duration, Train=True):
		if Train:
			print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		else:
			print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  test loss: ", "%.10f"%(float(loss)/(NCase)))
		return

	def evaluate(self, batch_data):   #this need to be modified
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size_output = nmol
		self.Eval_Prepare()
		feed_dict=self.fill_feed_dict(batch_data)
		preds, total_loss_value, loss_value, mol_output, atom_outputs, gradient = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
		return mol_output, atom_outputs, gradient

	def Eval_Prepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.mats_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None, self.batch_size_output])))
			self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl)
			self.gradient = tf.gradients(self.output, self.inp_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return

	def Prepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.mats_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.batch_size_output])))
			self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output]))
			self.output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return

class MolInstance_BP_Dipole(MolInstance_fc_sqdiff_BP):
	"""
		Calculate the Dipole of Molecules
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
		self.learning_rate = 0.0001
		#self.learning_rate = 0.00001
		self.momentum = 0.95
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
			# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.
			self.inshape = np.prod(self.TData.dig.eshape)
		# HACK something was up with the shapes in kun's saved network...
		#I do not need that..
		#self.inshape = self.inshape[0]
		print("MolInstance_BP_Dipole.inshape: ",self.inshape)
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		self.MeanStoich = self.TData.MeanStoich # Average stoichiometry of a molecule.
		self.MeanNumAtoms = np.sum(self.MeanStoich)
		self.AtomBranchNames=[] # a list of the layers named in each atom branch

		self.netcharge_output = None
		self.dipole_output = None
		self.inp_pl=None
		self.mats_pl=None
		self.coords = None
		self.label_pl=None

		# self.batch_size is still the number of inputs in a batch.
		self.batch_size = 10000
		self.batch_size_output = 0
		self.hidden1 = 100
		self.hidden2 = 100
		self.hidden3 = 100
		self.summary_op =None
		self.summary_writer=None

	def Clean(self):
		MolInstance_fc_sqdiff_BP.Clean(self)
		self.coords_pl = None
		self.netcharge_output = None
		self.dipole_output = None
		return

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		print("self.MeanNumAtoms: ",self.MeanNumAtoms)
		# allow for 120% of required output space, since it's cheaper than input space to be padded by zeros.
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		#self.TData.CheckBPBatchsizes(self.batch_size, self.batch_size_output)
		print("Assigned batch input size: ",self.batch_size)
		print("Assigned batch output size in BP_Dipole:",self.batch_size_output)
		with tf.Graph().as_default():
			self.inp_pl=[]
			self.mats_pl=[]
			self.coords_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.batch_size_output])))
				self.coords_pl.append(tf.placeholder(tf.float32, shape=tuple([None, 3])))
			self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 4]))
			self.netcharge_output, self.dipole_output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl, self.coords_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.netcharge_output, self.dipole_output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			#self.summary_op = tf.summary.merge_all()
			#init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
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

	def loss_op(self, netcharge_output, dipole_output, labels):
		"""
		total_loss = scaler*l2(netcharge) + l2(dipole)
		"""
		charge_labels = tf.slice(labels, [0, 0], [self.batch_size_output,1])
		dipole_labels = tf.slice(labels, [0, 1], [self.batch_size_output,3])
		charge_diff  = tf.subtract(netcharge_output, charge_labels)
		dipole_diff  = tf.subtract(dipole_output, dipole_labels)
		charge_loss = tf.nn.l2_loss(charge_diff)
		dipole_loss = tf.nn.l2_loss(dipole_diff)
		loss = tf.add(charge_loss, dipole_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def inference(self, inp_pl, mats_pl, coords_pl):
		"""
		Builds a Behler-Parinello graph which also matches monopole,3-dipole, and 9-quadropole elements.

		- It has the shape of two energy networks in parallel.
		- One network produces the energy, the other produces the charge on an atom.
		- The charges are constrained to reproduce the molecular multipoles.
		- The energy and charge are together constrained to produce the molecular energy.
		- The same atomic linear transformation is used to produce the charges as the energy.
		- All multipoles have the form of a dot product with atomic charges. That's pre-computed externally
		The attenuated coulomb energy has the form of a per-molecule vector-matrix-vector product.
		And it is generated as well by this routine.

		Args:
				inp_pl: a list of (num_of atom type X flattened input shape) matrix of input cases.
				mats_pl: a list of (num_of atom type X batchsize) matrices which linearly combines the elements to give molecular outputs.
				mul_pl: Multipole inputs (see Mol::GenerateMultipoleInputs)
		Returns:
			Atom BP Energy, Atom Charges
			I'm thinking about doing the contractions for the multipoles and electrostatic energy loss in loss_op... haven't settled on it yet.

		"""
		# convert the index matrix from bool to float
		branches=[]
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3
		netcharge_output = tf.zeros([self.batch_size_output, 1])
		dipole_output = tf.zeros([self.batch_size_output, 3])
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		LOGGER.info("Norms: %f,%f,%f", nrm1,nrm2,nrm3)
		#print(inp_pl)
		#tf.Print(inp_pl, [inp_pl], message="This is input: ",first_n=10000000,summarize=100000000)
		#tf.Print(bnds_pl, [bnds_pl], message="bnds_pl: ",first_n=10000000,summarize=100000000)
		#tf.Print(mats_pl, [mats_pl], message="mats_pl: ",first_n=10000000,summarize=100000000)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp_pl[e]
			mats = mats_pl[e]
			coords = coords_pl[e]
			shp_in = tf.shape(inputs)
			shp_coords = tf.shape(coords)
			if (PARAMS["check_level"]>2):
				tf.Print(tf.to_float(shp_in), [tf.to_float(shp_in)], message="Element "+str(e)+"input shape ",first_n=10000000,summarize=100000000)
				mats_shape = tf.shape(mats)
				tf.Print(tf.to_float(mats_shape), [tf.to_float(mats_shape)], message="Element "+str(e)+"mats shape ",first_n=10000000,summarize=100000000)
				tf.Print(tf.to_float(shp_coords), [tf.to_float(shp_coords)], message="Element "+str(e)+"coords shape ",first_n=10000000,summarize=100000000)
			if (PARAMS["check_level"]>3):
				tf.Print(tf.to_float(inputs), [tf.to_float(inputs)], message="This is input shape ",first_n=10000000,summarize=100000000)
			with tf.name_scope(str(self.eles[e])+'_hidden_1'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev=nrm1, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(inputs, weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_hidden_2'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev=nrm2, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_hidden_3'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev=nrm3, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden3_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_regression_linear'):
				shp = tf.shape(inputs)
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden3_units, 1], var_stddev=nrm4, var_wd=None)
				biases = tf.Variable(tf.zeros([1]), name='biases')
				branches[-1].append(tf.matmul(branches[-1][-1], weights) + biases)
				shp_out = tf.shape(branches[-1][-1])
				cut = tf.slice(branches[-1][-1],[0,0],[shp_out[0],1])
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				coords_rshp = tf.transpose(coords)
				coords_rshp_shape = tf.shape(coords_rshp)

				dipole_tmp = tf.multiply(rshp, coords_rshp)
				dipole_tmp = tf.reshape(dipole_tmp,[3, shp_out[0]])
				netcharge = tf.matmul(rshp,mats)
				dipole = tf.matmul(dipole_tmp, mats)
				netcharge = tf.transpose(netcharge)
				dipole = tf.transpose(dipole)
				netcharge_output = tf.add(netcharge_output, netcharge)
				dipole_output = tf.add(dipole_output, dipole)
		tf.verify_tensor_all_finite(netcharge_output,"Nan in output!!!")
		tf.verify_tensor_all_finite(dipole_output,"Nan in output!!!")
		#tf.Print(output, [output], message="This is output: ",first_n=10000000,summarize=100000000)
		return netcharge_output, dipole_output, atom_outputs

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
			if (not np.all(np.isfinite(batch_data[2][e]),axis=(0,1))):
				print("I was fed shit3")
				raise Exception("DontEatShit")
		if (not np.all(np.isfinite(batch_data[3]),axis=(0,1))):
			print("I was fed shit4")
			raise Exception("DontEatShit")
		#feed_dict={i: d for i, d in zip(self.inp_pl+self.mats_pl + self.coords_pl, batch_data[0]+batch_data[1] +  batch_data[2])}
		feed_dict={i: d for i, d in zip(self.inp_pl+self.mats_pl+self.coords_pl+[self.label_pl], batch_data[0]+batch_data[1]+ batch_data[2] + [batch_data[3]])}
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
			#new_coords = np.zeros((batch_data[4].shape[0], batch_data[4].shape[1]+1))
			#new_coords[:,0] = 0
			#new_coords[:,1:] = batch_data[4]
			#batch_data[4] = new_coords
			#batch_data = batch_data[:3] + [batch_data[4]]
			#print ("checking shape:", batch_data[2][0].shape, batch_data[2][1].shape, batch_data[2][2].shape, batch_data[2][3].shape)
			#print ("checking shape, input:", batch_data[0][0].shape, batch_data[0][1].shape, batch_data[0][2].shape, batch_data[0][3].shape)
			actual_mols  = np.count_nonzero(np.any(batch_data[3][1:], axis=1))
			dump_, dump_2, total_loss_value, loss_value, netcharge_output, dipole_output = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.netcharge_output, self.dipole_output], feed_dict=self.fill_feed_dict(batch_data))
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
			actual_mols  = np.count_nonzero(np.any(batch_data[3][1:], axis=1))
			total_loss_value, loss_value, netcharge_output, dipole_output, atom_outputs = self.sess.run([self.total_loss, self.loss, self.netcharge_output, self.dipole_output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols
		print ("testing result:")
		print ("acurrate charge, dipole:", batch_data[3][:20])
		print ("predict dipole", dipole_output[:20])
		#print ("charge sum:", netcharge_output)
		#print ("charges: ",  atom_outputs)
		duration = time.time() - start_time
		#print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		#self.TData.dig.EvaluateTestOutputs(batch_data[2],preds)
		return test_loss, feed_dict

	def print_training(self, step, loss, Ncase, duration, Train=True):
		if Train:
			print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		else:
			print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  test loss: ", "%.10f"%(float(loss)/(NCase)))
		return

	def evaluate(self, batch_data):   #this need to be modified
		# Check sanity of input
		nmol = batch_data[3].shape[0]
		LOGGER.debug("nmol: %i", batch_data[3].shape[0])
		self.batch_size_output = nmol
		if not self.sess:
			LOGGER.info("loading the session..")
			self.Eval_Prepare()
		feed_dict=self.fill_feed_dict(batch_data)
		netcharge, dipole, total_loss_value, loss_value,  atom_outputs = self.sess.run([self.netcharge_output, self.dipole_output, self.total_loss, self.loss, self.atom_outputs],  feed_dict=feed_dict)
		return netcharge, dipole/AUPERDEBYE, atom_outputs

	def Eval_Prepare(self):
		if (isinstance(self.inshape,tuple)):
			if (len(self.inshape)>1):
				raise Exception("My input should be flat")
			else:
				self.inshape = self.inshape[0]
		#eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.mats_pl=[]
			self.coords_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None, self.batch_size_output])))
				self.coords_pl.append(tf.placeholder(tf.float32, shape=tuple([None, 3])))
			self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 4]))
			self.netcharge_output, self.dipole_output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl, self.coords_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.netcharge_output, self.dipole_output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return

	def Prepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.mats_pl=[]
			self.coords = []
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.batch_size_output])))
				self.coords_pl.append(tf.placeholder(tf.float32, shape=tuple([None, 3])))
			self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 4]))
			self.netcharge_output, self.dipole_output, self.atom_outputs = self.inference(self.inp_pl, self.mats_pl, self.coords_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.netcharge_output, self.dipole_out, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return




class MolInstance_BP_Dipole_2(MolInstance_BP_Dipole):
	"""
		Calculate the Dipole of Molecules
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
		#self.learning_rate = 0.1
		self.learning_rate = 0.0001
		#self.learning_rate = 0.00001
		self.momentum = 0.95
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
			# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.
			self.inshape = np.prod(self.TData.dig.eshape)
		# HACK something was up with the shapes in kun's saved network...
		#I do not need that..
		#self.inshape = self.inshape[0]
		print("MolInstance_BP_Dipole.inshape: ",self.inshape)
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		self.MeanStoich = self.TData.MeanStoich # Average stoichiometry of a molecule.
		self.MeanNumAtoms = np.sum(self.MeanStoich)
		self.AtomBranchNames=[] # a list of the layers named in each atom branch

		self.netcharge_output = None
		self.dipole_output = None
		self.inp_pl=None
		self.mats_pl=None
		self.coords = None
		self.label_pl=None
		self.natom_pl = None
		self.charge_gradient = None
		self.output_list = None
		self.unscaled_atom_outputs = None

		# self.batch_size is still the number of inputs in a batch.
		self.batch_size = 10000
		self.batch_size_output = 0
		self.hidden1 = 500
		self.hidden2 = 500
		self.hidden3 = 500
		self.summary_op =None
		self.summary_writer=None


	def Clean(self):
		MolInstance_BP_Dipole.Clean(self)
		self.natom_pl = None
		self.net_charge = None
		self.charge_gradient = None
		self.unscaled_atom_outputs = None
		self.output_list = None
		return

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		print("self.MeanNumAtoms: ",self.MeanNumAtoms)
		# allow for 120% of required output space, since it's cheaper than input space to be padded by zeros.
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		#self.TData.CheckBPBatchsizes(self.batch_size, self.batch_size_output)
		print("Assigned batch input size: ",self.batch_size)
		print("Assigned batch output size in BP_Dipole:",self.batch_size_output)
		with tf.Graph().as_default():
			self.inp_pl=[]
			self.mats_pl=[]
			self.coords_pl=[]
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.batch_size_output])))
				self.coords_pl.append(tf.placeholder(tf.float32, shape=tuple([None, 3])))
			self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 3]))
			self.natom_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 1]))
			self.dipole_output, self.atom_outputs, self.unscaled_atom_outputs, self.net_charge  = self.inference(self.inp_pl, self.mats_pl, self.coords_pl, self.natom_pl)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.dipole_output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver()
#			try: # I think this may be broken
#				metafiles = [x for x in os.listdir(self.train_dir) if (x.count('meta')>0)]
#				if (len(metafiles)>0):
#					most_recent_meta_file=metafiles[0]
#					LOGGER.info("Restoring training from Metafile: "+most_recent_meta_file)
#					#Set config to allow soft device placement for temporary fix to known issue with Tensorflow up to version 0.12 atleast - JEH
#					config = tf.ConfigProto(allow_soft_placement=True)
#					self.sess = tf.Session(config=config)
#					self.saver = tf.train.import_meta_graph(self.train_dir+'/'+most_recent_meta_file)
#					self.saver.restore(self.sess, tf.train.latest_checkpoint(self.train_dir))
#			except Exception as Ex:
#				print("Restore Failed",Ex)
#				pass
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def loss_op(self, dipole_output, labels):
		"""
		total_loss =  l2(dipole)
		"""
		dipole_diff  = tf.subtract(dipole_output, labels)
		loss = tf.nn.l2_loss(dipole_diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def inference(self, inp_pl, mats_pl, coords_pl, natom_pl):
		"""
		Builds a Behler-Parinello graph which also matches monopole,3-dipole, and 9-quadropole elements.

		- It has the shape of two energy networks in parallel.
		- One network produces the energy, the other produces the charge on an atom.
		- The charges are constrained to reproduce the molecular multipoles.
		- The energy and charge are together constrained to produce the molecular energy.
		- The same atomic linear transformation is used to produce the charges as the energy.
		- All multipoles have the form of a dot product with atomic charges. That's pre-computed externally
		The attenuated coulomb energy has the form of a per-molecule vector-matrix-vector product.
		And it is generated as well by this routine.

		Args:
				inp_pl: a list of (num_of atom type X flattened input shape) matrix of input cases.
				mats_pl: a list of (num_of atom type X batchsize) matrices which linearly combines the elements to give molecular outputs.
				mul_pl: Multipole inputs (see Mol::GenerateMultipoleInputs)
		Returns:
			Atom BP Energy, Atom Charges
			I'm thinking about doing the contractions for the multipoles and electrostatic energy loss in loss_op... haven't settled on it yet.

		"""
		# convert the index matrix from bool to float
		branches=[]
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3
		netcharge_output = tf.zeros([self.batch_size_output, 1])
		scaled_netcharge_output = tf.zeros([1, self.batch_size_output])
		dipole_output = tf.zeros([self.batch_size_output, 3])
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		LOGGER.info("Norms: %f,%f,%f", nrm1,nrm2,nrm3)
		#print(inp_pl)
		#tf.Print(inp_pl, [inp_pl], message="This is input: ",first_n=10000000,summarize=100000000)
		#tf.Print(bnds_pl, [bnds_pl], message="bnds_pl: ",first_n=10000000,summarize=100000000)
		#tf.Print(mats_pl, [mats_pl], message="mats_pl: ",first_n=10000000,summarize=100000000)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp_pl[e]
			mats = mats_pl[e]
			coords = coords_pl[e]
			shp_in = tf.shape(inputs)
			shp_coords = tf.shape(coords)
			if (PARAMS["check_level"]>2):
				tf.Print(tf.to_float(shp_in), [tf.to_float(shp_in)], message="Element "+str(e)+"input shape ",first_n=10000000,summarize=100000000)
				mats_shape = tf.shape(mats)
				tf.Print(tf.to_float(mats_shape), [tf.to_float(mats_shape)], message="Element "+str(e)+"mats shape ",first_n=10000000,summarize=100000000)
				tf.Print(tf.to_float(shp_coords), [tf.to_float(shp_coords)], message="Element "+str(e)+"coords shape ",first_n=10000000,summarize=100000000)
			if (PARAMS["check_level"]>3):
				tf.Print(tf.to_float(inputs), [tf.to_float(inputs)], message="This is input shape ",first_n=10000000,summarize=100000000)
			with tf.name_scope(str(self.eles[e])+'_hidden_1'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev=nrm1, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(inputs, weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_hidden_2'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev=nrm2, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_hidden_3'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev=nrm3, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden3_units]), name='biases')
				branches[-1].append(tf.nn.relu(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_regression_linear'):
				shp = tf.shape(inputs)
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden3_units, 1], var_stddev=nrm4, var_wd=None)
				biases = tf.Variable(tf.zeros([1]), name='biases')
				branches[-1].append(tf.matmul(branches[-1][-1], weights) + biases)
				shp_out = tf.shape(branches[-1][-1])
				cut = tf.slice(branches[-1][-1],[0,0],[shp_out[0],1])
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				netcharge = tf.matmul(rshp,mats)
				netcharge = tf.transpose(netcharge)
				netcharge_output = tf.add(netcharge_output, netcharge)

		delta_charge = tf.multiply(netcharge_output, natom_pl)
		delta_charge = tf.transpose(delta_charge)
		#total_atom = 0
		scaled_charge_list = []
		#for e in range(len(self.eles)):
		#	total_atom = total_atom + tf.shape(atom_outputs[e])[1]
		#scaled_charge_list = tf.zeros([total_atom])
		#pointer = 0
		for e in range(len(self.eles)):
			mats = mats_pl[e]
			shp_out = tf.shape(atom_outputs[e])
			coords = coords_pl[e]
			trans_mats = tf.transpose(mats)
			ele_delta_charge = tf.matmul(delta_charge, trans_mats)
			scaled_charge = tf.subtract(atom_outputs[e], ele_delta_charge)
			#num_rows, natom =scaled_charge.get_shape().as_list()
			#scaled_charge_rshp = tf.reshape(scaled_charge, [shp_out[1]])
			#indices = range(pointer, pointer+num_rows)
			#scaled_charge_list = tf.scatter_update(scaled_charge_list, indices, scaled_charge_rshp) 
			scaled_charge_list.append(scaled_charge)
			scaled_netcharge = tf.matmul(scaled_charge,mats)
			scaled_netcharge_output = tf.add(scaled_netcharge_output, scaled_netcharge)
			coords_rshp = tf.transpose(coords)
			dipole_tmp = tf.multiply(scaled_charge, coords_rshp)
			dipole_tmp = tf.reshape(dipole_tmp, [3, shp_out[1]])
			dipole = tf.matmul(dipole_tmp, mats)
			dipole = tf.transpose(dipole)
			dipole_output = tf.add(dipole_output, dipole)
			#pointer = pointer + natom
		tf.verify_tensor_all_finite(netcharge_output,"Nan in output!!!")
		tf.verify_tensor_all_finite(dipole_output,"Nan in output!!!")
		#tf.Print(output, [output], message="This is output: ",first_n=10000000,summarize=100000000)
		#return  dipole_output, scaled_charge_list,  scaled_netcharge_output#atom_outputs
		return  dipole_output, scaled_charge_list, atom_outputs, scaled_netcharge_output#atom_outputs

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
			if (not np.all(np.isfinite(batch_data[2][e]),axis=(0,1))):
				print("I was fed shit3")
				raise Exception("DontEatShit")
		#if (not np.all(np.isfinite(batch_data[3]),axis=(0,1))):
		#	print("I was fed shit4")
		#	raise Exception("DontEatShit")
		if (not np.all(np.isfinite(batch_data[4]),axis=(0,1))):
                        print("I was fed shit5")
                        raise Exception("DontEatShit")
		#feed_dict={i: d for i, d in zip(self.inp_pl+self.mats_pl + self.coords_pl, batch_data[0]+batch_data[1] +  batch_data[2])}
		feed_dict={i: d for i, d in zip(self.inp_pl+self.mats_pl+self.coords_pl+[self.natom_pl]+[self.label_pl], batch_data[0]+batch_data[1]+ batch_data[2] + [batch_data[3]] + [batch_data[4]])}
		#print ("batch_data", batch_data)
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
			#print (batch_data)
			#print ("checking shape:", batch_data[2][0].shape, batch_data[2][1].shape, batch_data[2][2].shape, batch_data[2][3].shape)
			#print ("checking shape, input:", batch_data[0][0].shape, batch_data[0][1].shape, batch_data[0][2].shape, batch_data[0][3].shape)
			actual_mols  = np.count_nonzero(np.any(batch_data[3][1:], axis=1))
			dump_, dump_2, total_loss_value, loss_value,  dipole_output, atom_outputs, net_charge = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.dipole_output, self.atom_outputs, self.net_charge],  feed_dict=self.fill_feed_dict(batch_data))


			#dump_2, total_loss_value, loss_value, dipole_output = self.sess.run([self.train_op, self.total_loss, self.loss,  self.dipole_output], feed_dict=self.fill_feed_dict(batch_data))
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
			actual_mols  = np.count_nonzero(np.any(batch_data[3][1:], axis=1))
			total_loss_value, loss_value,  dipole_output, atom_outputs, net_charge  = self.sess.run([self.total_loss, self.loss, self.dipole_output, self.atom_outputs, self.net_charge],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols
		#print ("net charge:", net_charge)
		#print ("predict charge:", atom_outputs[0])
		print ("acurrate charge, dipole:", batch_data[4][:20], " dipole shape:", batch_data[4].shape)
		print ("predict dipole", dipole_output[:20])
		#print ("charge sum:", netcharge_output)
		#print ("charges: ",  atom_outputs)
		duration = time.time() - start_time
		#print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		#self.TData.dig.EvaluateTestOutputs(batch_data[2],preds)
		return test_loss, feed_dict

	def print_training(self, step, loss, Ncase, duration, Train=True):
		if Train:
			print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		else:
			print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  test loss: ", "%.10f"%(float(loss)/(NCase)))
		return

	def evaluate(self, batch_data, IfChargeGrad =  False):   #this need to be modified
		# Check sanity of input
		nmol = batch_data[4].shape[0]
		LOGGER.debug("nmol: %i", batch_data[4].shape[0])
		self.batch_size_output = nmol
		if not self.sess:
			LOGGER.info("loading the session..")
			self.Eval_Prepare()
		feed_dict=self.fill_feed_dict(batch_data)
		if not IfChargeGrad:
			output_list  = self.sess.run([ self.output_list],  feed_dict=feed_dict)
			return   output_list[0]/AUPERDEBYE, output_list[1]
		else:
			#dipole, total_loss_value, loss_value,  atom_outputs, charge_gradient = self.sess.run([ self.dipole_output, self.total_loss, self.loss, self.atom_outputs, self.charge_gradient],  feed_dict=feed_dict)
			output_list, charge_gradient = self.sess.run([  self.output_list, self.charge_gradient],  feed_dict=feed_dict)
                        #print ("scaled_atom_outputs:", output_list[1], "unscaled_atom_outputs:", output_list[2], " charge_gradient:", charge_gradient, "length of charge_gradient:", len(charge_gradient))
                        return   output_list[0]/AUPERDEBYE, output_list[1], charge_gradient

	def Eval_Prepare(self):
		if (isinstance(self.inshape,tuple)):
			if (len(self.inshape)>1):
				raise Exception("My input should be flat")
			else:
				self.inshape = self.inshape[0]
		#eval_labels = np.zeros(Ncase)  # dummy labels
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
                        self.inp_pl=[]
                        self.mats_pl=[]
                        self.coords_pl=[]
                        for e in range(len(self.eles)):
                                self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
                                self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.batch_size_output])))
                                self.coords_pl.append(tf.placeholder(tf.float32, shape=tuple([None, 3])))
                        self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 3]))
                        self.natom_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 1]))
			self.output_list   = self.inference(self.inp_pl, self.mats_pl, self.coords_pl, self.natom_pl)
			self.charge_gradient = tf.gradients(self.output_list[2], self.inp_pl)  # gradient of unscaled_charge respect to input
                        self.check = tf.add_check_numerics_ops()
                        #self.total_loss, self.loss = self.loss_op(self.dipole_output, self.label_pl)
                        #self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return

	def Prepare(self):
		#eval_labels = np.zeros(Ncase)  # dummy labels
		self.MeanNumAtoms = self.TData.MeanNumAtoms
		self.batch_size_output = int(1.5*self.batch_size/self.MeanNumAtoms)
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.inp_pl=[]
			self.mats_pl=[]
			self.coords = []
			for e in range(len(self.eles)):
				self.inp_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.inshape])))
				self.mats_pl.append(tf.placeholder(tf.float32, shape=tuple([None,self.batch_size_output])))
				self.coords_pl.append(tf.placeholder(tf.float32, shape=tuple([None, 3])))
			self.label_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 3]))
			self.natom_pl = tf.placeholder(tf.float32, shape=tuple([self.batch_size_output, 1]))
			self.dipole_output, self.atom_outputs, self.unscaled_atom_outputs, self.net_charge = self.inference(self.inp_pl, self.mats_pl, self.coords_pl, self.natom_pl)
			#self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.dipole_out, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver.restore(self.sess, self.chk_file)
		return


