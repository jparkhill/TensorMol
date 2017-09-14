"""
	These instances work directly on raw coordinate, atomic number data.
	They either generate their own descriptor or physical model.
	They are also simplified relative to the usual MolInstance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TensorMol.TFInstance import *
from TensorMol.TensorMolData import *
from TensorMol.TFMolInstance import *
from TensorMol.ElectrostaticsTF import *
from TensorMol.Neighbors import *
from TensorMol.RawEmbeddings import *
from tensorflow.python.client import timeline
import threading

class MolInstance_DirectForce_tmp(MolInstance_fc_sqdiff_BP):
	"""
	An instance which can evaluate and optimize some model force field.
	The force routines are in ElectrostaticsTF.py
	The force routines can take some parameters described here.
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True, ForceType_="LJ"):
		"""
		Args:
				TData_: A TensorMolData instance.
				Name_: A name for this instance.
		"""
		self.NetType = "LJE"
		self.TData = TData_
		self.MaxNAtoms = TData_.MaxNAtoms
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.batch_size_output = 10000
		self.inp_pl=None
		self.frce_pl=None
		self.sess = None
		self.ForceType = ForceType_
		self.forces = None
		self.energies = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None
		self.LJe = None
		self.LJr = None
		self.Deq = None
		self.dbg1 = None
		self.dbg2 = None
		self.batch_data = self.TData.RawBatch(nmol=30000)
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.

	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
		        continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.LJe = tf.placeholder(tf.float64, shape=(),name="Ee_pl")
			self.LJr = tf.placeholder(tf.float64, shape=(),name="Re_pl")
			# self.Ee_pl = tf.constant(0.316, dtype=tf.float32)
			# self.Re_pl = tf.constant(1.0, dtype=tf.float32)
			self.inp_shp = tf.shape(self.batch_data[0])
			self.nmol = self.inp_shp[0]
			self.maxnatom = self.inp_shp[1]
			self.XYZs = tf.to_float(tf.slice(self.batch_data[0],[0,0,1],[-1,-1,-1]))
			self.REns = tf.convert_to_tensor(self.batch_data[1][:,0,0],dtype=tf.float64)
			self.Zs = tf.cast(tf.reshape(tf.slice(self.batch_data[0],[0,0,0],[-1,-1,1]),[self.nmol,self.maxnatom,1]),tf.int64)
			self.Ens = LJEnergies(self.XYZs, self.Zs, self.LJe, self.LJr)
			self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.Ens, self.REns)))
			# params = (XYZs, Zs, REns)
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.sess.run(init)

	def LJER(self,inp_pl,E_pl,R_pl):
		"""
		with the current LJe, and LJr.

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
		"""
		# separate out the Z from the XYZ.
		inp_shp = tf.shape(inp_pl)
		nmol = inp_shp[0]
		maxnatom = inp_shp[1]
		XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int64)
		#self.LJe = tf.Print(self.LJe,[self.LJe],"LJe",1000,1000)
		#self.LJr = tf.Print(self.LJr,[self.LJr],"LJr",1000,1000)
		LJe2 = E_pl*E_pl
		LJr2 = R_pl*R_pl
		#LJe2 = tf.Print(LJe2,[LJe2],"LJe2",1000,1000)
		#LJr2 = tf.Print(LJr2,[LJr2],"LJr2",1000,1000)
		Ens = LJEnergies(XYZs, Zs, LJe2, LJr2)
		#Ens = tf.Print(Ens,[Ens],"Energies",5000,5000)
		return Ens

	def LJFrc(self, params):
		"""
		Compute forces for a batch of molecules
		with the current LJe, and LJr.

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
		"""
		feeddict = {self.LJe:params[0], self.LJr:params[1]}
		result = self.sess.run(self.mae,feed_dict=feeddict)
		return result

class MolInstance_DirectBP(MolInstance_fc_sqdiff_BP):
	"""
	An Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True, ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP"
		self.TData = TData_
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		self.MaxNAtoms = TData_.MaxNAtoms
		self.batch_size_output = 4096
		self.inp_pl=None
		self.frce_pl=None
		self.sess = None
		self.ForceType = ForceType_
		self.forces = None
		self.energies = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None

	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		#with tf.Graph().as_default():

		return

	def loss_op(self, output, labels):
		"""
		The loss operation of this model is complicated
		Because you have to construct the electrostatic energy moleculewise,
		and the mulitpoles.

		Emats and Qmats are constructed to accerate this process...
		"""
		output = tf.Print(output,[output],"Comp'd",1000,1000)
		labels = tf.Print(labels,[labels],"Desired",1000,1000)
		diff  = tf.subtract(output, labels)
		#tf.Print(diff, [diff], message="This is diff: ",first_n=10000000,summarize=100000000)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

	def LJFrc(self, inp_pl):
		"""
		Compute forces for a batch of molecules
		with the current LJe, and LJr.

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
		"""
		# separate out the Z from the XYZ.
		inp_shp = tf.shape(inp_pl)
		nmol = inp_shp[0]
		maxnatom = inp_shp[1]
		XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int64)
		#self.LJe = tf.Print(self.LJe,[self.LJe],"LJe",1000,1000)
		#self.LJr = tf.Print(self.LJr,[self.LJr],"LJr",1000,1000)
		LJe2 = self.LJe*self.LJe
		LJr2 = self.LJr*self.LJr
		#LJe2 = tf.Print(LJe2,[LJe2],"LJe2",1000,1000)
		#LJr2 = tf.Print(LJr2,[LJr2],"LJr2",1000,1000)
		Ens = LJEnergies(XYZs, Zs, LJe2, LJr2)
		#Ens = tf.Print(Ens,[Ens],"Energies",5000,5000)
		frcs = -1.0*(tf.gradients(Ens, XYZs)[0])
		return Ens, frcs

	def HarmFrc(self, inp_pl):
		"""
		Compute Harmonic Forces with equilibrium distance matrix
		Deqs, and force constant matrix, Keqs

		Args:
			inp_pl: placeholder for the NMol X MaxNatom X 4 tensor of Z,x,y,z
		"""
		# separate out the Z from the XYZ.
		inp_shp = tf.shape(inp_pl)
		nmol = inp_shp[0]
		maxnatom = inp_shp[1]
		XYZs = tf.slice(inp_pl,[0,0,1],[-1,-1,-1])
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int64)
		ZZeroTensor = tf.cast(tf.where(tf.equal(Zs,0),tf.ones_like(Zs),tf.zeros_like(Zs)),self.tf_prec)
		# Construct a atomic number masks.
		Zshp = tf.shape(Zs)
		Zzij1 = tf.tile(ZZeroTensor,[1,1,Zshp[1]]) # mol X atom X atom.
		Zzij2 = tf.transpose(Zzij1,perm=[0,2,1]) # mol X atom X atom.
		Deqs = tf.ones((nmol,maxnatom,maxnatom),dtype=self.tf_prec)
		Keqs = 0.001*tf.ones((nmol,maxnatom,maxnatom),dtype=self.tf_prec)
		K = HarmKernels(XYZs, Deqs, Keqs)
		K = tf.where(tf.equal(Zzij1,1.0),tf.zeros_like(K,dtype=self.tf_prec),K)
		K = tf.where(tf.equal(Zzij2,1.0),tf.zeros_like(K,dtype=self.tf_prec),K)
		Ens = tf.reduce_sum(K,[1,2])
		frcs = -1.0*(tf.gradients(Ens, XYZs)[0])
		#frcs = tf.Print(frcs,[frcs],"Forces",1000,1000)
		return Ens, frcs

	def EvalForce(self,m):
		Ins = self.TData.dig.Emb(m,False,False)
		Ins = Ins.reshape(tuple([1]+list(Ins.shape)))
		feeddict = {self.inp_pl:Ins}
		En,Frc = self.sess.run([self.energies, self.forces],feed_dict=feeddict)
		return En, JOULEPERHARTREE*Frc[0] # Returns energies and forces.

	def print_training(self, step, loss, Ncase, duration, Train=True):
		print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		return

	def Prepare(self):
		self.TrainPrepare()
		return

	def train_step(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		Ncase_train = len(self.TData.set.mols)
		start_time = time.time()
		train_loss =  0.0
		num_of_mols = 0
		for ministep in range (0, int(Ncase_train/self.batch_size_output)):
			#print ("ministep: ", ministep, " Ncase_train:", Ncase_train, " self.batch_size", self.batch_size)
			batch_data = self.TData.RawBatch()
			if (not np.all(np.isfinite(batch_data[0]))):
				print("Bad Batch...0 ")
			if (not np.all(np.isfinite(batch_data[1]))):
				print("Bad Batch...1 ")
			feeddict={i:d for i,d in zip([self.inp_pl,self.frce_pl],[batch_data[0],batch_data[1]])}
			dump_2, total_loss_value, loss_value = self.sess.run([self.train_op, self.total_loss, self.loss], feed_dict=feeddict)
			train_loss = train_loss + loss_value
			duration = time.time() - start_time
			num_of_mols += self.batch_size_output
			#print ("atom_outputs:", atom_outputs, " mol outputs:", mol_output)
			#print ("atom_outputs shape:", atom_outputs[0].shape, " mol outputs", mol_output.shape)
		#print("train diff:", (mol_output[0]-batch_data[2])[:actual_mols], np.sum(np.square((mol_output[0]-batch_data[2])[:actual_mols])))
		#print ("train_loss:", train_loss, " Ncase_train:", Ncase_train, train_loss/num_of_mols)
		#print ("diff:", mol_output - batch_data[2], " shape:", mol_output.shape)
		self.print_training(step, train_loss, num_of_mols, duration)
		return

	def train(self, mxsteps=10000):
		self.TrainPrepare()
		LOGGER.info("MolInstance_LJForce.train()")
		test_freq = PARAMS["test_freq"]
		mini_test_loss = float('inf') # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
		self.SaveAndClose()
		return

class MolInstance_DirectBP_NoGrad(MolInstance_fc_sqdiff_BP):
	"""
	An Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	Do not use gradient in training
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_noGrad"
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		#if (Name_ != None):
		#	return
		self.SFPa = None
		self.SFPr = None
		self.Ra_cut = None
		self.Rr_cut = None
		self.MaxNAtoms = self.TData.MaxNAtoms
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
		self.eles_pairs = []
		for i in range (len(self.eles)):
			for j in range(i, len(self.eles)):
				self.eles_pairs.append([self.eles[i], self.eles[j]])
		self.eles_pairs_np = np.asarray(self.eles_pairs)
		self.SetANI1Param()
		self.batch_size = PARAMS["batch_size"]
		self.NetType = "RawBP_noGrad"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		self.xyzs_pl = None
		self.Zs_pl = None
		self.label_pl = None
		self.sess = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None

	def SetANI1Param(self, prec=np.float64):
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
		p1 = np.tile(np.reshape(thetas,[AN1_num_a_As,1,1]),[1,AN1_num_a_Rs,1])
		p2 = np.tile(np.reshape(rs,[1,AN1_num_a_Rs,1]),[AN1_num_a_As,1,1])
		SFPa2 = np.concatenate([p1,p2],axis=2)
		self.SFPa2 = np.transpose(SFPa2, [2,0,1])
		p1_new = np.reshape(rs_R,[AN1_num_r_Rs,1])
		self.SFPr2 = np.transpose(p1_new, [1,0])
		self.zeta = PARAMS["AN1_zeta"]
		self.eta = PARAMS["AN1_eta"]

		print ("self.inshape:", self.inshape)

	def Clean(self):
		Instance.Clean(self)
		self.xyzs_pl=None
		self.check = None
		self.Zs_pl=None
		self.label_pl=None
		self.atom_outputs = None
		self.Scatter_Sym = None
		self.Sym_Index = None
		self.options = None
		self.run_metadata = None
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
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update2(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2, self.Rr_cut, Elep, self.SFPa2,self.zeta, self.eta, self.Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Rr_cut_tf = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			#self.Ra_cut_tf = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut_tf, Elep, self.SFPa, self.Ra_cut_tf)
			#tf.verify_tensor_all_finite(self.Scatter_Sym[0], "Nan in output!!! 0 ")
			#tf.verify_tensor_all_finite(self.Scatter_Sym[1], "Nan in output!!! 1")
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.output, self.xyzs_pl)
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
			#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			#self.run_metadata = tf.RunMetadata()
		return

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

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
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3

		output = tf.zeros([self.batch_size], dtype=self.tf_prec)
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		print("Norms:", nrm1,nrm2,nrm3)
		LOGGER.info("Layer initial Norms: %f %f %f", nrm1,nrm2,nrm3)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp[e]
			shp_in = tf.shape(inputs)
			index = tf.cast(indexs[e], tf.int64)
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
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				rshpflat = tf.reshape(cut,[shp_out[0]])
				range_index = tf.range(tf.cast(shp_out[0], tf.int64), dtype=tf.int64)
				sparse_index =tf.stack([index, range_index], axis=1)
				sp_atomoutputs = tf.SparseTensor(sparse_index, rshpflat, dense_shape=[tf.cast(self.batch_size, tf.int64), tf.cast(shp_out[0], tf.int64)])
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
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.label_pl], [batch_data[0]]+[batch_data[1]]+[batch_data[2]])}
		return feed_dict

	def print_training(self, step, loss, Ncase, duration, Train=True):
		print("step: ", "%7d"%step, "  duration: ", "%.5f"%duration,  "  train loss: ", "%.10f"%(float(loss)/(Ncase)))
		return

	def Prepare(self):
		self.TrainPrepare()
		return

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
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep:", ministep)
			batch_data = self.TData.GetTrainBatch(self.batch_size)
			actual_mols  = self.batch_size
			t = time.time()
			dump_, dump_2, total_loss_value, loss_value, mol_output, atom_outputs, gradient = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output,  self.atom_outputs, self.gradient], feed_dict=self.fill_feed_dict(batch_data))
			#dump_, dump_2, total_loss_value, loss_value, mol_output, atom_outputs, gradient = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output,  self.atom_outputs, self.gradient], feed_dict=self.fill_feed_dict(batch_data), options=self.options, run_metadata=self.run_metadata)
			#print ("gradient:", gradient[0][:4])

			#print ("gradient:", np.sum(gradient[0]))
			#print ("gradient:", np.sum(np.isinf(gradient[0])))
			#print ("gradient:", np.where(np.isinf(gradient[0]) == True))
			train_loss = train_loss + loss_value
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
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
			batch_data=self.TData.GetTestBatch(self.batch_size)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = self.batch_size
			preds, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols
		duration = time.time() - start_time
		print( "testing...")
		self.print_training(step, test_loss, num_of_mols, duration)
		return test_loss

	def print_training(self, step, loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f", step, duration, (float(loss)/(Ncase)))
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f", step, duration, (float(loss)/(Ncase)))
		return

	def evaluate(self, batch_data, IfGrad=True):   #this need to be modified
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
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
		#       new_mol_output, total_loss_value, loss_value, new_atom_outputs, new_gradient = self.sess.run([self.output,self.total_loss, self.loss, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
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
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update2(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2, self.Rr_cut, Elep, self.SFPa2,self.zeta, self.eta, self.Ra_cut)
			#elf.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Rr_cut_tf = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			#self.Ra_cut_tf = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut_tf, Elep, self.SFPa, self.Ra_cut_tf)
			#tf.verify_tensor_all_finite(self.Scatter_Sym[0], "Nan in output!!! 0 ")
			#tf.verify_tensor_all_finite(self.Scatter_Sym[1], "Nan in output!!! 1")
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.output, self.xyzs_pl)
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			#self.run_metadata = tf.RunMetadata()
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

class MolInstance_DirectBPBond_NoGrad(MolInstance_fc_sqdiff_BP):
	"""
	An Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	Do not use gradient in training
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "BPPairPotential"
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		# if (Name_ != None):
			# return
		self.MaxNAtoms = self.TData.MaxNAtoms
		self.eles = self.TData.eles
		self.n_eles = len(self.eles)
		self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
		self.eles_pairs = []
		for i in range (len(self.eles)):
			for j in range(i, len(self.eles)):
				self.eles_pairs.append([self.eles[i], self.eles[j]])
		self.eles_pairs_np = np.asarray(self.eles_pairs)
		self.batch_size = PARAMS["batch_size"]
		self.NetType = "RawBPBond_noGrad"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		if (self.Trainable):
			self.TData.LoadDataToScratch(self.tformer)
		self.xyzs_pl = None
		self.Zs_pl = None
		self.label_pl = None
		self.sess = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None
		self.profiling = PARAMS["Profiling"]

	def Clean(self):
		Instance.Clean(self)
		self.Zxyzs_pl=None
		self.check = None
		self.BondIdxMatrix_pl=None
		self.RList=None
		self.label_pl=None
		self.atom_outputs = None
		self.Scatter_Sym = None
		self.Sym_Index = None
		self.options = None
		self.run_metadata = None
		return


	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.Zxyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,4]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.BondIdxMatrix_pl = tf.placeholder(tf.int32, shape=tuple([None,3]))
			ElemPairs = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int32)
			self.RList, MolIdxList = TFBond(self.Zxyzs_pl, self.BondIdxMatrix_pl, ElemPairs)
			self.output, self.atom_outputs = self.inference(self.RList, MolIdxList)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
			if self.profiling:
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
		return

	def loss_op(self, output, labels):
		diff  = tf.subtract(output, labels)
		loss = tf.nn.l2_loss(diff)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss

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
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3

		output = tf.zeros([self.batch_size], dtype=self.tf_prec)
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		print("Norms:", nrm1,nrm2,nrm3)
		LOGGER.info("Layer initial Norms: %f %f %f", nrm1,nrm2,nrm3)
		for e in range(len(self.eles_pairs)):
			branches.append([])
			inputs = tf.reshape(inp[e], [tf.shape(inp[e])[0],1])
			shp_in = tf.shape(inputs)
			index = tf.cast(indexs[e], tf.int64)
			if (PARAMS["CheckLevel"]>2):
				tf.Print(tf.to_float(shp_in), [tf.to_float(shp_in)], message="Element "+str(e)+"input shape ",first_n=10000000,summarize=100000000)
				index_shape = tf.shape(index)
				tf.Print(tf.to_float(index_shape), [tf.to_float(index_shape)], message="Element "+str(e)+"index shape ",first_n=10000000,summarize=100000000)
			if (PARAMS["CheckLevel"]>3):
				tf.Print(tf.to_float(inputs), [tf.to_float(inputs)], message="This is input shape ",first_n=10000000,summarize=100000000)
			with tf.name_scope(str(self.eles_pairs[e][0])+str(self.eles_pairs[e][1])+'_hidden_1'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, hidden1_units], var_stddev=nrm1, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden1_units], dtype=self.tf_prec), name='biases')
				branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
			with tf.name_scope(str(self.eles_pairs[e][0])+str(self.eles_pairs[e][1])+'_hidden_2'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden1_units, hidden2_units], var_stddev=nrm2, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden2_units], dtype=self.tf_prec), name='biases')
				branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles_pairs[e][0])+str(self.eles_pairs[e][1])+'_hidden_3'):
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden2_units, hidden3_units], var_stddev=nrm3, var_wd=0.001)
				biases = tf.Variable(tf.zeros([hidden3_units], dtype=self.tf_prec), name='biases')
				branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
				#tf.Print(branches[-1], [branches[-1]], message="This is layer 2: ",first_n=10000000,summarize=100000000)
			with tf.name_scope(str(self.eles_pairs[e][0])+str(self.eles_pairs[e][1])+'_regression_linear'):
				shp = tf.shape(inputs)
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[hidden3_units, 1], var_stddev=nrm4, var_wd=None)
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
				branches[-1].append(tf.matmul(branches[-1][-1], weights) + biases)
				shp_out = tf.shape(branches[-1][-1])
				cut = tf.slice(branches[-1][-1],[0,0],[shp_out[0],1])
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				rshpflat = tf.reshape(cut,[shp_out[0]])
				range_index = tf.range(tf.cast(shp_out[0], tf.int64), dtype=tf.int64)
				sparse_index = tf.stack([index[:,0], range_index], axis=1)
				sp_atomoutputs = tf.SparseTensor(sparse_index, rshpflat, dense_shape=[tf.cast(self.batch_size, tf.int64), tf.cast(shp_out[0], tf.int64)])
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
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.Zxyzs_pl]+[self.BondIdxMatrix_pl]+[self.label_pl], [batch_data[0]]+[batch_data[1]]+[batch_data[2]])}
		return feed_dict

	def Prepare(self):
		self.TrainPrepare()
		return

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
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep:", ministep)
			batch_data = self.TData.RawBatch(nmol=self.batch_size)
			actual_mols  = self.batch_size
			t = time.time()
			if self.profiling:
				dump_, dump_2, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output,  self.atom_outputs], feed_dict=self.fill_feed_dict(batch_data), options=self.options, run_metadata=self.run_metadata)
				fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
				chrome_trace = fetched_timeline.generate_chrome_trace_format()
				with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
					f.write(chrome_trace)
			else:
				dump_, dump_2, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output,  self.atom_outputs], feed_dict=self.fill_feed_dict(batch_data))
			train_loss = train_loss + loss_value
			duration = time.time() - start_time
			num_of_mols += actual_mols
		self.PrintTrain(step, train_loss, num_of_mols, duration)
		return

	def test(self, step):
		"""
		Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

		Args:
			step: the index of this step.
		"""
		print( "testing...")
		test_loss =  0.0
		start_time = time.time()
		Ncase_test = self.TData.NTest
		num_of_mols = 0
		for ministep in range (0, int(Ncase_test/self.batch_size)):
			batch_data = self.TData.RawBatch(nmol=self.batch_size)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = self.batch_size
			dump_, dump_2, total_loss_value, loss_value, mol_output, atom_outputs, labels = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.output,  self.atom_outputs, self.label_pl], feed_dict=feed_dict)
			# preds, total_loss_value, loss_value, mol_output, atom_outputs = self.sess.run([self.output,self.total_loss, self.loss, self.output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols
		duration = time.time() - start_time
		self.PrintTest(mol_output, labels, test_loss, num_of_mols, duration)
		return test_loss

	def PrintTest(self, output, labels, loss, Ncase, duration):
		for i in range(50):
			LOGGER.info("Label: %.5f   Prediction: %.5f", labels[i], output[i])
		LOGGER.info("Duration: %.5f  Test Loss: %.10f", duration, (float(loss)/(Ncase)))
		return

	def PrintTrain(self, step, loss, Ncase, duration, Train=True):
		LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f", step, duration, (float(loss)/(Ncase)))
		return

	def Evaluate(self):   #this need to be modified
		if not self.sess:
			print("loading the session..")
			self.EvalPrepare()
		self.TData.ReloadSet()
		self.TData.raw_it = iter(self.TData.set.mols)
		Ncase_train = self.TData.NTrain
		AtomOutputs = []
		for ministep in xrange(0, int(Ncase_train/self.batch_size)):
			batch_data = self.TData.RawBatch(nmol=self.batch_size)
			feed_dict=self.fill_feed_dict(batch_data)
			actual_mols  = self.batch_size
			_, mol_output, atom_outputs, RList = self.sess.run([self.check, self.output, self.atom_outputs, self.RList], feed_dict=feed_dict)
			energy_distance = []
			for i in range(len(atom_outputs)):
				energy_distance.append(np.stack([atom_outputs[i][0,:], RList[i]], axis=1))
			if ministep == 0:
				AtomOutputs = energy_distance
			else:
				for i in range(len(AtomOutputs)):
					AtomOutputs[i] = np.append(AtomOutputs[i], energy_distance[i],axis=0)
		return AtomOutputs

	def EvalPrepare(self):
		with tf.Graph().as_default(), tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			self.Zxyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.TData.set.MaxNAtoms(),4]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.BondIdxMatrix_pl = tf.placeholder(tf.int32, shape=tuple([None,3]))
			ElemPairs = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int32)
			self.RList, MolIdxList = TFBond(self.Zxyzs_pl, self.BondIdxMatrix_pl, ElemPairs)
			self.output, self.atom_outputs = self.inference(self.RList, MolIdxList)
			self.check = tf.add_check_numerics_ops()
			self.total_loss, self.loss = self.loss_op(self.output, self.label_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			#self.run_metadata = tf.RunMetadata()
		return


class MolInstance_DirectBP_Grad(MolInstance_fc_sqdiff_BP):
	"""
	An Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	Do not use gradient in training
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True, ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_Grad"
		self.SFPa = None
		self.SFPr = None
		self.Ra_cut = None
		self.Rr_cut = None
		self.HasANI1PARAMS = False
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		#if (Name_ != None):
		#	return
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
		self.GradScalar = PARAMS["GradScalar"]
		self.NetType = "RawBP_Grad"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		print ("self.activation_function_type: ", self.activation_function_type)
		self.train_dir = './networks/'+self.name
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
		print ("self.hidden1:",self.hidden1, " self.hidden2:", self.hidden2, " self.hidden3:", self.hidden3)

	def SetANI1Param(self, prec=np.float64):
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
		Instance.Clean(self)
		self.xyzs_pl=None
		self.check = None
		self.Zs_pl=None
		self.label_pl=None
		self.grads_pl = None
		self.atom_outputs = None
		self.energy_loss = None
		self.grads_loss = None
		self.Scatter_Sym = None
		self.Sym_Index = None
		self.options = None
		self.run_metadata = None
		return

	def loss_op(self, output, nn_grads, labels, grads):
		energy_diff  = tf.subtract(output, labels)
		energy_loss = tf.nn.l2_loss(energy_diff)
		grads_diff = tf.subtract(nn_grads, grads)
		grads_loss = tf.nn.l2_loss(grads_diff)
		#loss = tf.multiply(grads_loss, energy_loss)
		loss = tf.add(energy_loss, tf.multiply(grads_loss, self.GradScalar))
		#loss = tf.identity(energy_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss


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
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3

		output = tf.zeros([self.batch_size], dtype=self.tf_prec)
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		print("Norms:", nrm1,nrm2,nrm3)
		LOGGER.info("Layer initial Norms: %f %f %f", nrm1,nrm2,nrm3)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp[e]
			shp_in = tf.shape(inputs)
			index = tf.cast(indexs[e], tf.int64)
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
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				rshpflat = tf.reshape(cut,[shp_out[0]])
				range_index = tf.range(tf.cast(shp_out[0], tf.int64), dtype=tf.int64)
				sparse_index =tf.stack([index, range_index], axis=1)
				sp_atomoutputs = tf.SparseTensor(sparse_index, rshpflat, dense_shape=[tf.cast(self.batch_size, tf.int64), tf.cast(shp_out[0], tf.int64)])
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
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.label_pl] + [self.grads_pl], [batch_data[0]]+[batch_data[1]]+[batch_data[2]] + [batch_data[3]])}
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
		train_energy_loss = 0.0
		train_grads_loss = 0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep:", ministep)
			batch_data = self.TData.GetTrainBatch(self.batch_size)
			actual_mols  = self.batch_size
			t = time.time()
			#dump_, total_loss_value, loss_value, energy_loss, grads_loss,  mol_output, atom_outputs   = self.sess.run([self.train_op, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.output,  self.atom_outputs], feed_dict=self.fill_feed_dict(batch_data))
			#dump_, dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  mol_output, atom_outputs   = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.output,  self.atom_outputs], feed_dict=self.fill_feed_dict(batch_data))
			dump_, dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  mol_output, atom_outputs = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.output,  self.atom_outputs], feed_dict=self.fill_feed_dict(batch_data))
			#print ("loss_value:", loss_value, "grads_loss:", grads_loss, "energy_loss:", energy_loss, "\n self.SFPr2_vary :", SFPr2_vary )
			#print ("all time:", time.time() - t0, " get batch time:", batchtime)
			#print ("loss_value:", loss_value, "grads_loss:", grads_loss, "energy_loss:", energy_loss)
			#print ("SFPr2:", SFPr2_vary, "\n SFPa2:", SFPa2_vary)
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
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
			preds, total_loss_value, loss_value, energy_loss, grads_loss, mol_output, atom_outputs = self.sess.run([self.output, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.output, self.atom_outputs],  feed_dict=feed_dict)
			test_loss += loss_value
			num_of_mols += actual_mols
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
		duration = time.time() - start_time
		print( "testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, num_of_mols, duration)
		return test_loss

	def print_training(self, step, loss, energy_loss, grads_loss, Ncase, duration, Train=True):
		if Train:
			LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f  energy_loss: %.10f  grad_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)))
		else:
			LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f energy_loss: %.10f  grad_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)))
		return

	def evaluate(self, batch_data, IfGrad=True):
		"""
		Evaluate the energy, atom energies, and IfGrad = True the gradients
		of this Direct Behler-Parinello graph.
		"""
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		self.MaxNAtoms = batch_data[0].shape[1]
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
		if not self.sess:
			print ("loading the session..")
			self.EvalPrepare()
		feed_dict=self.fill_feed_dict(batch_data)
		mol_output, atom_outputs, gradient = self.sess.run([self.output, self.atom_outputs, self.gradient],  feed_dict=feed_dict)
		return mol_output, atom_outputs, gradient

	def Prepare(self):
		self.TrainPrepare()
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
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable=False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable=False, dtype = self.tf_prec)
			#self.SFPr2_vary = tf.Variable(self.SFPr2, trainable= True, dtype = self.tf_prec)
			Rr_cut   = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut   = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta   = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta   = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update2(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update2(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut, Elep, self.SFPa, self.Ra_cut)
			#self.Rr_cut_tf = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			#self.Ra_cut_tf = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, self.Rr_cut_tf, Elep, self.SFPa, self.Ra_cut_tf)
			#tf.verify_tensor_all_finite(self.Scatter_Sym[0], "Nan in output!!! 0 ")
			#tf.verify_tensor_all_finite(self.Scatter_Sym[1], "Nan in output!!! 1")
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.check = tf.add_check_numerics_ops()
			self.gradient = tf.gradients(self.output, self.xyzs_pl)
			self.total_loss, self.loss, self.energy_loss, self.grads_loss = self.loss_op(self.output, self.gradient, self.label_pl, self.grads_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
			self.sess.graph.finalize()
			#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			#self.run_metadata = tf.RunMetadata()
		return

	def EvalPrepare(self):
		"""
		Doesn't generate the training operations or losses.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable=False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable=False, dtype = self.tf_prec)
			Rr_cut   = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut   = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta   = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta   = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update2(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut)
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.output, self.xyzs_pl)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		print("Prepared for Evaluation...")
		return

	def continue_training(self, mxsteps):
		self.EvalPrepare()
		test_loss = self.test(-1)
		test_freq = 1
		mini_test_loss = test_loss
		for step in  range (0, mxsteps+1):
			SFPr2_vary, SFPra_vary = self.train_step(step)
			if step%test_freq==0 and step!=0 :
				test_loss = self.test(step)
				if test_loss < mini_test_loss:
					mini_test_loss = test_loss
					self.save_chk(step)
		self.SaveAndClose()
		return

class MolInstance_DirectBP_Grad_noGradTrain(MolInstance_DirectBP_Grad):
	"""
	An Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	Do not use gradient in training
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_Grad_noGradTrain"
		MolInstance_DirectBP_Grad.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_Grad_noGradTrain"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name

	def loss_op(self, output, nn_grads, labels, grads):
		energy_diff  = tf.subtract(output, labels)
		energy_loss = tf.nn.l2_loss(energy_diff)
		grads_diff = tf.subtract(nn_grads, grads)
		grads_loss = tf.nn.l2_loss(grads_diff)
		#loss = tf.add(energy_loss, grads_loss)
		loss = tf.identity(energy_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss

class MolInstance_DirectBP_Grad_NewIndex(MolInstance_DirectBP_Grad):
	"""
	An Update version of Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	index_pl holds both the index of molecule and the index of each atom
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_Grad_Update"
		MolInstance_DirectBP_Grad.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_Grad_Update"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name

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
		atom_outputs = []
		hidden1_units=self.hidden1
		hidden2_units=self.hidden2
		hidden3_units=self.hidden3

		output = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		nrm1=1.0/(10+math.sqrt(float(self.inshape)))
		nrm2=1.0/(10+math.sqrt(float(hidden1_units)))
		nrm3=1.0/(10+math.sqrt(float(hidden2_units)))
		nrm4=1.0/(10+math.sqrt(float(hidden3_units)))
		print("Norms:", nrm1,nrm2,nrm3)
		LOGGER.info("Layer initial Norms: %f %f %f", nrm1,nrm2,nrm3)
		for e in range(len(self.eles)):
			branches.append([])
			inputs = inp[e]
			shp_in = tf.shape(inputs)
			index = tf.cast(indexs[e], tf.int64)
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
				rshp = tf.reshape(cut,[1,shp_out[0]])
				atom_outputs.append(rshp)
				rshpflat = tf.reshape(cut,[shp_out[0]])
				atom_indice = tf.slice(index, [0,1], [shp_out[0],1])
				ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
				output = tf.add(output, ToAdd)
			tf.verify_tensor_all_finite(output,"Nan in output!!!")
			#tf.Print(output, [output], message="This is output: ",first_n=10000000,summarize=100000000)
		return tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size]), atom_outputs

	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable=False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable=False, dtype = self.tf_prec)
			#self.SFPr2_vary = tf.Variable(self.SFPr2, trainable= True, dtype = self.tf_prec)
			Rr_cut   = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut   = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta   = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta   = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Update_Scatter(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut)
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.output, self.xyzs_pl)
			self.total_loss, self.loss, self.energy_loss, self.grads_loss = self.loss_op(self.output, self.gradient, self.label_pl, self.grads_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
			#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			#self.run_metadata = tf.RunMetadata()
		return

class MolInstance_DirectBP_Grad_Linear(MolInstance_DirectBP_Grad):
	"""
	An Update version of Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	index_pl holds both the index of molecule and the index of each atom
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_Grad_Linear"
		MolInstance_DirectBP_Grad.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_Grad_Linear"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name

	def Clean(self):
		MolInstance_DirectBP_Grad.Clean(self)
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
		# Don't eat shit.
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.label_pl] + [self.grads_pl] + [self.Radp_pl] + [self.Angt_pl], [batch_data[0]]+[batch_data[1]]+[batch_data[2]] + [batch_data[3]] + [batch_data[4]] + [batch_data[5]])}
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
						weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
						biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
				else:
					with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
						weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
						biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_regression_linear'):
				shp = tf.shape(inputs)
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
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
			#tf.Print(output, [output], message="This is output: ",first_n=10000000,summarize=100000000)
		return tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size]), atom_outputs

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
		print ("batch_data:  ", batch_data)
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
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.Angt_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.gradient  = tf.gradients(self.output, self.xyzs_pl)
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
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.label_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.Angt_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			#SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			#SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut   = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut   = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			zeta   = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta   = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.output, self.atom_outputs = self.inference(self.Scatter_Sym, self.Sym_Index)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.output, self.xyzs_pl)
			self.total_loss, self.loss, self.energy_loss, self.grads_loss = self.loss_op(self.output, self.gradient, self.label_pl, self.grads_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
			if (self.FindLastCheckpoint() != False):
				self.saver.restore(self.sess, self.FindLastCheckpoint())
		return

class MolInstance_DirectBP_Grad_Linear_EmbOpt(MolInstance_DirectBP_Grad):
	"""
	An Update version of Instance which does a direct Behler Parinello
	Using Output from RawEmbeddings.py
	index_pl holds both the index of molecule and the index of each atom
	"""
	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_Grad_Linear"
		MolInstance_DirectBP_Grad.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_Grad_Linear"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.TData.ele = self.eles_np
		self.TData.elep = self.eles_pairs_np

	def compute_normalization_constants(self):
		batch_data = self.TData.GetTrainBatch(self.batch_size)
		self.TData.ScratchPointer = 0
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

	def SetANI1Param(self, prec=np.float64):
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
		self.inshape = int(AN1_num_r_Rs + AN1_num_a_Rs*AN1_num_a_As)
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
		MolInstance_DirectBP_Grad.Clean(self)
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
		# Don't eat shit.
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
						weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
						biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
				else:
					with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
						weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
						biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
						branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
			with tf.name_scope(str(self.eles[e])+'_regression_linear'):
				shp = tf.shape(inputs)
				weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
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

	def save_chk(self, step):  # We need to merge this with the one in TFInstance
		self.chk_file = os.path.join(self.train_dir,self.name+'-chk-'+str(step))
		LOGGER.info("Saving Checkpoint file in the TFMoInstance")
		self.saver.save(self.sess,  self.chk_file)
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

class MolInstance_DirectBP_EE(MolInstance_DirectBP_Grad_Linear):
	"""
	Electrostatic embedding Behler-Parinello scheme
	"""

	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_EE"
		MolInstance_DirectBP_Grad.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_EE"
		self.GradScalar = PARAMS["GradScalar"]
		self.DipoleScalar = PARAMS["DipoleScalar"]
		self.Ree_on  = PARAMS["EECutoffOn"]
		self.Ree_off  = PARAMS["EECutoffOff"]
		self.DSFAlpha = PARAMS["DSFAlpha"]
		self.learning_rate_dipole = PARAMS["learning_rate_dipole"]
		self.learning_rate_energy = PARAMS["learning_rate_energy"]
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.Training_Traget = "Dipole"
		self.SetANI1Param()
		self.run_metadata = None

	def Clean(self):
		MolInstance_DirectBP_Grad_Linear.Clean(self)
		self.Elabel_pl = None
		self.Dlabel_pl = None
		self.Radp_pl = None
		self.Angt_pl = None
		self.Reep_pl = None
		self.natom_pl = None
		self.AddEcc_pl = None
		self.Etotal = None
		self.Ebp = None
		self.Ecc = None
		self.dipole = None
		self.charge = None
		self.energy_wb = None
		self.dipole_wb = None
		self.dipole_loss = None
		self.gradient = None
		self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = None, None, None, None, None
		self.train_op_dipole, self.train_op_EandG = None, None
		self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = None, None, None, None, None
		self.run_metadata = None
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
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.Angt_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			#SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			#SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			#with tf.device('/cpu:0'):
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.Etotal, self.Ebp, self.Ecc, self.dipole, self.charge, self.energy_wb, self.dipole_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl, self.AddEcc_pl)
			#self.Etotal,  self.energy_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl,name="BPEnGrad")
			#self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl,name="BPEnGrad", colocate_gradients_with_ops=True)

			self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)

			self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)

			self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)

			#self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG = self.loss_op_EandG_test(self.Etotal, self.gradient, self.Elabel_pl, self.grads_pl)
			#self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)

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
		return

	def loss_op(self, energy, energy_grads, dipole, Elabels, grads, Dlabels):
		energy_diff  = tf.subtract(energy, Elabels,name="EnDiff")
		energy_loss = tf.nn.l2_loss(energy_diff,name="EnL2")
		grads_diff = tf.subtract(energy_grads, grads,name="GradDiff")
		grads_loss = tf.nn.l2_loss(grads_diff,name="GradL2")
		dipole_diff = tf.subtract(dipole, Dlabels,name="DipoleDiff")
		dipole_loss = tf.nn.l2_loss(dipole_diff,name="DipL2")
		#loss = tf.multiply(grads_loss, energy_loss)
		EandG_loss = tf.add(energy_loss, tf.multiply(grads_loss, self.GradScalar),name="MulLoss")
		loss = tf.add(EandG_loss, tf.multiply(dipole_loss, self.DipoleScalar))
		#loss = tf.identity(dipole_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

	def loss_op_dipole(self, energy, energy_grads, dipole, Elabels, grads, Dlabels):
		energy_diff  = tf.subtract(energy, Elabels)
		energy_loss = tf.nn.l2_loss(energy_diff)
		grads_diff = tf.subtract(energy_grads, grads)
		grads_loss = tf.nn.l2_loss(grads_diff)
		dipole_diff = tf.subtract(dipole, Dlabels)
		dipole_loss = tf.nn.l2_loss(dipole_diff)
		#loss = tf.multiply(grads_loss, energy_loss)
		EandG_loss = tf.add(energy_loss, tf.multiply(grads_loss, self.GradScalar))
		loss = tf.identity(dipole_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

	def loss_op_EandG(self, energy, energy_grads, dipole, Elabels, grads, Dlabels):
		energy_diff  = tf.subtract(energy, Elabels)
		energy_loss = tf.nn.l2_loss(energy_diff)
		grads_diff = tf.subtract(energy_grads, grads)
		grads_loss = tf.nn.l2_loss(grads_diff)
		dipole_diff = tf.subtract(dipole, Dlabels)
		dipole_loss = tf.nn.l2_loss(dipole_diff)
		#loss = tf.multiply(grads_loss, energy_loss)
		EandG_loss = tf.add(energy_loss, tf.multiply(grads_loss, self.GradScalar))
		#loss = tf.add(EandG_loss, tf.multiply(dipole_loss, self.DipoleScalar))
		loss = tf.identity(EandG_loss)
		#loss = tf.identity(energy_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

	def loss_op_EandG_test(self, energy, energy_grads, Elabels, grads):
		energy_diff  = tf.subtract(energy, Elabels)
		energy_loss = tf.nn.l2_loss(energy_diff)
		grads_diff = tf.subtract(energy_grads, grads)
		grads_loss = tf.nn.l2_loss(grads_diff)
		EandG_loss = tf.add(energy_loss, tf.multiply(grads_loss, self.GradScalar))
		loss = tf.identity(EandG_loss)
		tf.add_to_collection('losses', loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss

	def inference(self, inp, indexs, xyzs, natom, EE_cuton, EE_cutoff, Reep, AddEcc):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		# convert the index matrix from bool to float
		Ebranches=[]
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
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
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Ebranches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Ebranches[-1].append(self.activation_function(tf.matmul(Ebranches[-1][-1], weights) + biases))
				with tf.name_scope(str(self.eles[e])+'_regression_linear'):
					shp = tf.shape(inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Ebranches[-1].append(tf.matmul(Ebranches[-1][-1], weights) + biases)
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
		energy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EnergyNet")
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
							Dbranches[-1].append(self.activation_function(tf.matmul(charge_inputs, weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)+"_charge"):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(Dbranches[-1][-1], weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
				with tf.name_scope(str(self.eles[e])+'_regression_linear_charge'):
					charge_shp = tf.shape(charge_inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					dipole_wb.append(weights)
					dipole_wb.append(biases)
					Dbranches[-1].append(tf.matmul(Dbranches[-1][-1], weights) + biases)
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
		#cc_energy = TFCoulombErfLR(xyzsInBohr, scaled_charge, EE_cuton, Reep)
		#cc_energy =tf.zeros([self.batch_size], dtype=self.tf_prec)
		def f1(): return TFCoulombPolyLR(xyzsInBohr, scaled_charge, EE_cuton*BOHRPERA, Reep)
		#def f1(): return TFCoulombErfLR(xyzsInBohr, scaled_charge, EE_cuton*BOHRPERA, Reep)
		#def f1(): return  TFCoulombErfSRDSFLR(xyzsInBohr, scaled_charge, EE_cuton*BOHRPERA, EE_cutoff*BOHRPERA, Reep, self.DSFAlpha)
		def f2(): return  tf.zeros([self.batch_size], dtype=self.tf_prec)
		cc_energy = tf.cond(AddEcc, f1, f2)
		#dipole_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DipoleNet")
		total_energy = tf.add(bp_energy, cc_energy)
		return total_energy, bp_energy, cc_energy, dipole, scaled_charge, energy_vars, dipole_wb


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
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.Dlabel_pl] + [self.grads_pl] + [self.Radp_pl] + [self.Angt_pl] + [self.Reep_pl] + [self.natom_pl] + [self.AddEcc_pl], batch_data)}
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
		train_energy_loss = 0.0
		train_dipole_loss = 0.0
		train_grads_loss = 0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep:", ministep)
			batch_data = self.TData.GetTrainBatch(self.batch_size)+[True]
			actual_mols  = self.batch_size
			t = time.time()
			dump_, dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.check, self.train_op, self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss, self.Etotal, self.Ecc,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			#print ("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#print ("Etotal:", Etotal, " Ecc:", Ecc)
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
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
			#print ("ministep:", ministep)
			batch_data = self.TData.GetTestBatch(self.batch_size)+[True]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			#print ("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			test_loss = test_loss + loss_value
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
			test_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		print ("testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, test_dipole_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
		return test_loss

	def train_step_dipole(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

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
			#print ("ministep:", ministep)
			t_mini = time.time()
			batch_data = self.TData.GetTrainBatch(self.batch_size) + [False]
			actual_mols  = self.batch_size
			t = time.time()
			dump_, dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.check, self.train_op_dipole, self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole, self.Etotal, self.Ecc,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			print ("ministep:  ", ministep, "mini step time dipole:", time.time() - t_mini )
			print ("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#LOGGER.debug("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#max_index = np.argmax(np.sum(abs(batch_data[3]-mol_dipole),axis=1))
			#LOGGER.debug("real dipole:\n", batch_data[3][max_index], "\nmol_dipole:\n", mol_dipole[max_index], "\n xyz:", batch_data[0][max_index], batch_data[1][max_index])
			#print ("Etotal:", Etotal[:20], " Ecc:", Ecc[:20])
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
		return


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
			#print ("ministep:", ministep)
			batch_data = self.TData.GetTestBatch(self.batch_size)+[False]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			#print ("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			test_loss = test_loss + loss_value
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
			test_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		print ("testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, test_dipole_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
		return  test_loss

	def train_step_EandG(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

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
			t_mini = time.time()
			batch_data = self.TData.GetTrainBatch(self.batch_size)+[True]
			actual_mols  = self.batch_size
			t = time.time()
			dump_, dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.check, self.train_op_EandG, self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG, self.Etotal, self.Ecc,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			print ("ministep:  ", ministep, "mini step time EandG:", time.time() - t_mini)
			#print ("Ecc:", Ecc[:20])
			#for k, ecc in enumerate(list(Ecc)):
			#	if ecc > 0.05:
			#		print ("Ecc:", ecc)
			#		np.savetxt("test_charge.dat", atom_charge[k])
			#		np.savetxt("test_xyz.dat", batch_data[0][k])
			#		raise Exception("end now")
			print ("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#print ("Etotal:", Etotal, " Ecc:", Ecc)
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
		return

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
			#print ("ministep:", ministep)
			batch_data = self.TData.GetTestBatch(self.batch_size)+[True]
			actual_mols  = self.batch_size
			t = time.time()
			total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			#print ("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			test_loss = test_loss + loss_value
			test_energy_loss += energy_loss
			test_grads_loss += grads_loss
			test_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		print ("testing...")
		self.print_training(step, test_loss, test_energy_loss, test_grads_loss, test_dipole_loss, num_of_mols, duration)
		return  test_loss

	def print_training(self, step, loss, energy_loss, grads_loss, dipole_loss, Ncase, duration, Train=True):
	    if Train:
	        LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f  energy_loss: %.10f  grad_loss: %.10f, dipole_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)), (float(dipole_loss)/(Ncase)))
	    else:
	        LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f energy_loss: %.10f  grad_loss: %.10f, dipole_loss: %.10f", step, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)), (float(dipole_loss)/(Ncase)))
	    return

	def EvalPrepare(self):
		"""
		Load pretrained network and build graph for evaluation
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.Angt_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			#SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			#SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut   = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut   = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.Etotal, self.Ebp, self.Ecc, self.dipole, self.charge, self.energy_wb, self.dipole_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl, self.AddEcc_pl)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, "TotalEnGrad")

			self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)

			self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)

			self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)

			self.summary_op = tf.summary.merge_all()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		print("Prepared for Evaluation...")
		return

	def evaluate(self, batch_data):
		"""
		Evaluate the energy, atom energies, and IfGrad = True the gradients
		of this Direct Behler-Parinello graph.
		"""
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		self.MaxNAtoms = batch_data[0].shape[1]
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
		if not self.sess:
			print ("loading the session..")
			self.EvalPrepare()
		feed_dict=self.fill_feed_dict(batch_data+[False])
		Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = self.sess.run([self.Etotal, self.Ebp, self.Ecc, self.dipole, self.charge, self.gradient], feed_dict=feed_dict)
		return Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient

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
					test_energy_loss = self.test_EandG(step)
					if test_energy_loss < mini_energy_test_loss:
						mini_energy_test_loss = test_energy_loss
						self.save_chk(step)
			elif self.Training_Traget == "Dipole":
				self.train_step_dipole(step)
				if step%test_freq==0 and step!=0 :
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
		train_loss =  0.0
		train_energy_loss = 0.0
		train_dipole_loss = 0.0
		train_grads_loss = 0.0
		num_of_mols = 0
		pre_output = np.zeros((self.batch_size),dtype=np.float64)
		for ministep in range (0, int(Ncase_train/self.batch_size)):
			#print ("ministep:", ministep)
			batch_data = self.TData.GetTrainBatch(self.batch_size)+[False]
			actual_mols  = self.batch_size
			t = time.time()
			dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.train_op_EandG, self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data), options=self.options, run_metadata=self.run_metadata)
			#print ("Ecc:", Ecc[:20])
			#for k, ecc in enumerate(list(Ecc)):
			#	if ecc > 0.05:
			#		print ("Ecc:", ecc)
			#		np.savetxt("test_charge.dat", atom_charge[k])
			#		np.savetxt("test_xyz.dat", batch_data[0][k])
			#		raise Exception("end now")
			print ("inference time:", time.time() - t)
			print ("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#print ("Etotal:", Etotal, " Ecc:", Ecc)
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			self.summary_writer.add_run_metadata(self.run_metadata, 'minstep%d' % ministep)
			duration = time.time() - start_time
			num_of_mols += actual_mols
			fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			chrome_trace = fetched_timeline.generate_chrome_trace_format()
			with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
				f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
		return

	def profile(self):
		"""
		This profiles a training step.
		"""
		LOGGER.info("running the TFMolInstance.train()")
		self.TrainPrepare(False)
		self.profile_step(1)
		return

	def continue_training(self, mxsteps):
		self.EvalPrepare()
		#test_loss = self.test(-1)
		self.test(-1)
		test_loss = float('inf')
		test_freq = PARAMS["test_freq"]
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


class MolInstance_DirectBP_EE_ChargeEncode(MolInstance_DirectBP_EE):
	"""
	Electrostatic embedding Behler Parinello
	"""

	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_EE_ChargeEnCode"
		MolInstance_DirectBP_EE.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_EE_ChargeEncode"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.Training_Traget = "Dipole"

	def Clean(self):
		MolInstance_DirectBP_EE.Clean(self)
		self.Radius_Qs_Encode = None
		self.Radius_Qs_Encode_Index = None

	def SetANI1Param(self, prec=np.float64):
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

	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="InputCoords")
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]),name="InputZs")
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]),name="DesEnergy")
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]),name="DesDipoles")
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]),name="DesGrads")
			self.Radp_pl=tf.placeholder(tf.int64, shape=tuple([None,3]),name="RadialPairs")
			self.Angt_pl=tf.placeholder(tf.int64, shape=tuple([None,4]),name="AngularTriples")
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]),name="RadialElectros")
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			#SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			#SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
#			with tf.name_scope("MakeDescriptors"):
			#with tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			#with tf.device('/cpu:0'):
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl, self.AddEcc_pl)
			self.Radius_Qs_Encode, self.Radius_Qs_Encode_Index = TFSymSet_Radius_Scattered_Linear_Qs(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, eta,  self.Radp_pl, self.charge)
#			with tf.name_scope("behler"):
			self.Etotal, self.Ebp,  self.energy_wb = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.Radius_Qs_Encode, self.Ecc)
			#self.Etotal,  self.energy_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			#self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad", colocate_gradients_with_ops=True)
#			with tf.name_scope("losses"):
			self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			tf.summary.scalar("loss", self.loss)
			tf.summary.scalar("loss_dip", self.loss_dipole)
			tf.summary.scalar("loss_EG", self.loss_EandG)

#			with tf.name_scope("training"):
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum, )
			self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)
			self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			# please do not use the totality of the GPU memory
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

	def dipole_inference(self, inp, indexs, xyzs, natom, EE_cuton, EE_cutoff, Reep, AddEcc):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
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
							Dbranches[-1].append(self.activation_function(tf.matmul(charge_inputs, weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)+"_charge"):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
							Dbranches[-1].append(self.activation_function(tf.matmul(Dbranches[-1][-1], weights) + biases))
							dipole_wb.append(weights)
							dipole_wb.append(biases)
				with tf.name_scope(str(self.eles[e])+'_regression_linear_charge'):
					charge_shp = tf.shape(charge_inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					dipole_wb.append(weights)
					dipole_wb.append(biases)
					Dbranches[-1].append(tf.matmul(Dbranches[-1][-1], weights) + biases)
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

		def f1(): return TFCoulombPolyLR(xyzsInBohr, scaled_charge, EE_cuton*BOHRPERA, Reep)
		#def f1(): return TFCoulombErfLR(xyzsInBohr, scaled_charge, EE_cuton*BOHRPERA, Reep)
		#def f1(): return  TFCoulombErfSRDSFLR(xyzsInBohr, scaled_charge, EE_cuton*BOHRPERA, EE_cutoff*BOHRPERA, Reep, self.DSFAlpha)
		def f2(): return  tf.zeros([self.batch_size], dtype=self.tf_prec)
		cc_energy = tf.cond(AddEcc, f1, f2)
		#dipole_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DipoleNet")
		return  cc_energy, dipole, scaled_charge, dipole_wb


	def energy_inference(self, inp, indexs, charge_encode, cc_energy):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		# convert the index matrix from bool to float
		Ebranches=[]
		output = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		atom_outputs = []
		with tf.name_scope("EnergyNet"):
			for e in range(len(self.eles)):
				Ebranches.append([])
				raw_inputs = inp[e]
				encode_inputs = charge_encode[e]
				inputs = tf.concat([encode_inputs, raw_inputs], axis=1)
				shp_in = tf.shape(inputs)
				index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape_withencode, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape_withencode))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(Ebranches[-1][-1], weights) + biases))
				with tf.name_scope(str(self.eles[e])+'_regression_linear'):
					shp = tf.shape(inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Ebranches[-1].append(tf.matmul(Ebranches[-1][-1], weights) + biases)
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
		energy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EnergyNet")
		total_energy = tf.add(bp_energy, cc_energy)
		return total_energy, bp_energy, energy_vars


	def EvalPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.Angt_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			#SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			#SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl, self.AddEcc_pl)
			self.Radius_Qs_Encode, self.Radius_Qs_Encode_Index = TFSymSet_Radius_Scattered_Linear_Qs(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, eta,  self.Radp_pl, self.charge)
			self.Etotal, self.Ebp,  self.energy_wb = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.Radius_Qs_Encode, self.Ecc)
			#self.Etotal,  self.energy_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl)
			self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)

			self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)

			self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)

			#self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG = self.loss_op_EandG_test(self.Etotal, self.gradient, self.Elabel_pl, self.grads_pl)
			#self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)

			self.summary_op = tf.summary.merge_all()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		print("Prepared for Evaluation...")


class MolInstance_DirectBP_EE_Update(MolInstance_DirectBP_EE):
	"""
	Electrostatic embedding Behler-Parinello scheme.
	This version prebuild the mijkl and mil_jk in python.
	"""

	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		MolInstance_DirectBP_EE.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_EE_Update"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.TData.ele = self.eles_np
		self.TData.elep = self.eles_pairs_np
		self.SetANI1Param()

	def Clean(self):
		MolInstance_DirectBP_Grad_Linear.Clean(self)
		self.Elabel_pl = None
		self.Dlabel_pl = None
		self.Radp_Ele_pl = None
		self.Angt_Elep_pl = None
		self.mil_jk_pl = None
		self.Reep_pl = None
		self.natom_pl = None
		self.AddEcc_pl = None
		self.Etotal = None
		self.Ebp = None
		self.Ecc = None
		self.dipole = None
		self.charge = None
		self.energy_wb = None
		self.dipole_wb = None
		self.dipole_loss = None
		self.gradient = None
		self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = None, None, None, None, None
		self.train_op_dipole, self.train_op_EandG = None, None
		self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = None, None, None, None, None
		self.run_metadata = None
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
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			#SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			#SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			#with tf.device('/cpu:0'):
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_tmp(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl)
			self.Etotal, self.Ebp, self.Ecc, self.dipole, self.charge, self.energy_wb, self.dipole_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl, self.AddEcc_pl)
			#self.Etotal,  self.energy_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl,name="BPEnGrad")
			#self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl,name="BPEnGrad", colocate_gradients_with_ops=True)

			self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)

			self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)

			self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)

			#self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG = self.loss_op_EandG_test(self.Etotal, self.gradient, self.Elabel_pl, self.grads_pl)
			#self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)

			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			config=tf.ConfigProto(allow_soft_placement=True)
			#config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
			#config.gpu_options.per_process_gpu_memory_fraction = 0.90
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
		# Don't eat shit.
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.Dlabel_pl] + [self.grads_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.Reep_pl] + [self.mil_jk_pl] + [self.natom_pl] + [self.AddEcc_pl], batch_data)}
		return feed_dict


	def evaluate(self, batch_data):
		"""
		Evaluate the energy, atom energies, and IfGrad = True the gradients
		of this Direct Behler-Parinello graph.
		"""
		# Check sanity of input
		nmol = batch_data[2].shape[0]
		self.MaxNAtoms = batch_data[0].shape[1]
		LOGGER.debug("nmol: %i", batch_data[2].shape[0])
		self.batch_size = nmol
		if not self.sess:
			print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
			print ("loading the session..")
			self.EvalPrepare()
		feed_dict=self.fill_feed_dict(batch_data+[True])
		Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient = self.sess.run([self.Etotal, self.Ebp, self.Ecc, self.dipole, self.charge, self.gradient], feed_dict=feed_dict)
		return Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient


	def EvalPrepare(self):
		"""
		Load pretrained network and build graph for evaluation
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([self.batch_size, self.MaxNAtoms]))
			self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, 3]))
			self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([self.batch_size, self.MaxNAtoms,3]))
			self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
			self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]))
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
			#with tf.device('/cpu:0'):
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_tmp(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl)
			self.Etotal, self.Ebp, self.Ecc, self.dipole, self.charge, self.energy_wb, self.dipole_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl, self.AddEcc_pl)
			#self.Etotal,  self.energy_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl,name="BPEnGrad")
			#self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			#self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum)

			#self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			#self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)

			#self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			#self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)

			self.summary_op = tf.summary.merge_all()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, self.chk_file)
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		print("Prepared for Evaluation...")
		return


class MolInstance_DirectBP_EE_ChargeEncode_Update(MolInstance_DirectBP_EE_ChargeEncode):
	"""
	Electrostatic embedding Behler Parinello
	"""

	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		MolInstance_DirectBP_EE_ChargeEncode.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_EE_ChargeEncode_Update"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.Training_Traget = "Dipole"
		self.TData.ele = self.eles_np
		self.TData.elep = self.eles_pairs_np

	def Clean(self):
		MolInstance_DirectBP_EE_ChargeEncode.Clean(self)
		self.Radp_Ele_pl = None
		self.Angt_Elep_pl = None
		self.mil_jk_pl = None


	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

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
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]),name="RadialElectros")
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			#SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			#SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
#			with tf.name_scope("MakeDescriptors"):
			#with tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			#with tf.device('/cpu:0'):
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl)
			self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl, self.AddEcc_pl)
			self.Radp_pl  = self.Radp_Ele_pl[:,:3]
			self.Radius_Qs_Encode, self.Radius_Qs_Encode_Index = TFSymSet_Radius_Scattered_Linear_Qs(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, eta,  self.Radp_pl, self.charge)
#			with tf.name_scope("behler"):
			self.Etotal, self.Ebp,  self.energy_wb = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.Radius_Qs_Encode, self.Ecc)
			#self.Etotal,  self.energy_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			#self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad", colocate_gradients_with_ops=True)
#			with tf.name_scope("losses"):
			self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			tf.summary.scalar("loss", self.loss)
			tf.summary.scalar("loss_dip", self.loss_dipole)
			tf.summary.scalar("loss_EG", self.loss_EandG)

#			with tf.name_scope("training"):
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum, )
			self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)
			self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			# please do not use the totality of the GPU memory
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
		# Don't eat shit.
		if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
			print("I was fed shit")
			raise Exception("DontEatShit")
		feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.Dlabel_pl] + [self.grads_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.Reep_pl] + [self.mil_jk_pl] + [self.natom_pl] + [self.AddEcc_pl], batch_data)}
		return feed_dict

class MolInstance_DirectBP_EE_ChargeEncode_Update_vdw(MolInstance_DirectBP_EE_ChargeEncode_Update):
	"""
	Electrostatic embedding Behler Parinello with van der waals interaction implemented with Grimmer C6 scheme.
	"""

	def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		MolInstance_DirectBP_EE_ChargeEncode_Update.__init__(self, TData_,  Name_, Trainable_)
		self.NetType = "RawBP_EE_ChargeEncode_Update_vdw"
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		self.train_dir = './networks/'+self.name
		self.Training_Traget = "Dipole"
		self.vdw_R = np.zeros(self.n_eles)
		self.C6 = np.zeros(self.n_eles)
		for i, ele in enumerate(self.eles):
			self.C6[i] = C6_coff[ele]* (BOHRPERA*10.0)**6.0 / JOULEPERHARTREE # convert into a.u.
			self.vdw_R[i] = atomic_vdw_radius[ele]*BOHRPERA



	def TrainPrepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

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
			self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]),name="RadialElectros")
			self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([self.batch_size]))
			self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
			Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
			Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
			#SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
			#SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
			SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
			SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
			Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
			Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
			Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
			Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
			C6 = tf.Variable(self.C6,trainable=False, dtype = self.tf_prec)
			vdw_R = tf.Variable(self.vdw_R,trainable=False, dtype = self.tf_prec)
			#self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
#			with tf.name_scope("MakeDescriptors"):
			#with tf.device('/job:localhost/replica:0/task:0/gpu:1'):
			#with tf.device('/cpu:0'):
			self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl)
			self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl, self.AddEcc_pl)
			self.Radp_pl  = self.Radp_Ele_pl[:,:3]
			self.Radius_Qs_Encode, self.Radius_Qs_Encode_Index = TFSymSet_Radius_Scattered_Linear_Qs(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, eta,  self.Radp_pl, self.charge)
#			with tf.name_scope("behler"):
			self.Etotal, self.Ebp, self.Evdw,  self.energy_wb = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.Radius_Qs_Encode, self.Ecc, self.xyzs_pl, self.Zs_pl, Ele, C6, vdw_R, self.Reep_pl, Ree_on, Ree_off)
			#self.Etotal,  self.energy_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl)
			self.check = tf.add_check_numerics_ops()
			self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
			#self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad", colocate_gradients_with_ops=True)
#			with tf.name_scope("losses"):
			self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl)
			tf.summary.scalar("loss", self.loss)
			tf.summary.scalar("loss_dip", self.loss_dipole)
			tf.summary.scalar("loss_EG", self.loss_EandG)

#			with tf.name_scope("training"):
			self.train_op = self.training(self.total_loss, self.learning_rate, self.momentum, )
			self.train_op_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole, self.momentum, self.dipole_wb)
			self.train_op_EandG = self.training(self.total_loss_EandG, self.learning_rate_energy, self.momentum, self.energy_wb)
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			# please do not use the totality of the GPU memory
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



	def energy_inference(self, inp, indexs, charge_encode, cc_energy, xyzs, Zs, eles, c6, R_vdw, Reep, EE_cuton, EE_cutoff):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		# convert the index matrix from bool to float
		xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
		Ebranches=[]
		output = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
		atom_outputs = []
		with tf.name_scope("EnergyNet"):
			for e in range(len(self.eles)):
				Ebranches.append([])
				raw_inputs = inp[e]
				encode_inputs = charge_encode[e]
				inputs = tf.concat([encode_inputs, raw_inputs], axis=1)
				shp_in = tf.shape(inputs)
				index = tf.cast(indexs[e], tf.int64)
				for i in range(len(self.HiddenLayers)):
					if i == 0:
						with tf.name_scope(str(self.eles[e])+'_hidden1'):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape_withencode, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape_withencode))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
					else:
						with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
							weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
							biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
							Ebranches[-1].append(self.activation_function(tf.matmul(Ebranches[-1][-1], weights) + biases))
				with tf.name_scope(str(self.eles[e])+'_regression_linear'):
					shp = tf.shape(inputs)
					weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
					biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
					Ebranches[-1].append(tf.matmul(Ebranches[-1][-1], weights) + biases)
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
		return total_energy_with_vdw, bp_energy, vdw_energy, energy_vars

	def Clean(self):
		MolInstance_DirectBP_EE_ChargeEncode_Update.Clean(self)
		self.Evdw = None

	def train_step_EandG(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

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
			batch_data = self.TData.GetTrainBatch(self.batch_size)+[True]
			actual_mols  = self.batch_size
			t = time.time()
			dump_, dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, Evdw, mol_dipole, atom_charge = self.sess.run([self.check, self.train_op_EandG, self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG, self.Etotal, self.Ecc, self.Evdw,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			print_loss += loss_value
			print_energy_loss += energy_loss
			print_grads_loss += grads_loss
			print_dipole_loss += dipole_loss
			if (ministep%print_per_mini == 0 and ministep!=0):
				print ("time:", (time.time() - time_print_mini)/print_per_mini ,  " loss_value: ",  print_loss/print_per_mini, " energy_loss:", print_energy_loss/print_per_mini, " grads_loss:", print_grads_loss/print_per_mini, " dipole_loss:", print_dipole_loss/print_per_mini)
				print_loss = 0.0
				print_energy_loss = 0.0
				print_dipole_loss = 0.0
				print_grads_loss = 0.0
				print_time = 0.0
				time_print_mini = time.time()
			
				#print ("Etotal:", Etotal, " Ecc:", Ecc, "Evdw:", Evdw)
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
		return


	def train_step_dipole(self, step):
		"""
		Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

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
			#print ("ministep:", ministep)
			t_mini = time.time()
			batch_data = self.TData.GetTrainBatch(self.batch_size) + [False]
			actual_mols  = self.batch_size
			t = time.time()
			dump_, dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.check, self.train_op_dipole, self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole, self.Etotal, self.Ecc,  self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
			#print ("ministep:  ", ministep, "mini step time dipole:", time.time() - t_mini )
			#print ("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			print_loss += loss_value
			print_energy_loss += energy_loss
			print_grads_loss += grads_loss
			print_dipole_loss += dipole_loss
			if (ministep%print_per_mini == 0 and ministep!=0):
				print ("time:", (time.time() - time_print_mini)/print_per_mini ,  " loss_value: ",  print_loss/print_per_mini, " energy_loss:", print_energy_loss/print_per_mini, " grads_loss:", print_grads_loss/print_per_mini, " dipole_loss:", print_dipole_loss/print_per_mini)
				print_loss = 0.0
				print_energy_loss = 0.0
				print_dipole_loss = 0.0
				print_grads_loss = 0.0
				print_time = 0.0
				time_print_mini = time.time()
			#LOGGER.debug("loss_value: ", loss_value, " energy_loss:", energy_loss, " grads_loss:", grads_loss, " dipole_loss:", dipole_loss)
			#max_index = np.argmax(np.sum(abs(batch_data[3]-mol_dipole),axis=1))
			#LOGGER.debug("real dipole:\n", batch_data[3][max_index], "\nmol_dipole:\n", mol_dipole[max_index], "\n xyz:", batch_data[0][max_index], batch_data[1][max_index])
			#print ("Etotal:", Etotal[:20], " Ecc:", Ecc[:20])
			#print ("energy_wb[1]:", energy_wb[1], "\ndipole_wb[1]", dipole_wb[1])
			#print ("charge:", atom_charge )
			train_loss = train_loss + loss_value
			train_energy_loss += energy_loss
			train_grads_loss += grads_loss
			train_dipole_loss += dipole_loss
			duration = time.time() - start_time
			num_of_mols += actual_mols
			#fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
			#chrome_trace = fetched_timeline.generate_chrome_trace_format()
			#with open('timeline_step_%d_tm_nocheck_h2o.json' % ministep, 'w') as f:
			#       f.write(chrome_trace)
		#print ("gradients:", gradients)
		#print ("labels:", batch_data[2], "\n", "predcits:",mol_output)
		self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration)
		#self.print_training(step, train_loss,  num_of_mols, duration)
		return
