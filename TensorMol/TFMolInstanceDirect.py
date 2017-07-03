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
from TensorMol.RawEmbeddings import *


class MolInstance_DirectForce(MolInstance_fc_sqdiff_BP):
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
		self.NetType = "LJForce"
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
		self.LJe = None
		self.LJr = None
		self.Deq = None
		self.dbg1 = None
		self.dbg2 = None
		# Using multidimensional inputs creates all sorts of issues; for the time being only support flat inputs.

	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.inp_pl=tf.placeholder(self.tf_prec, shape=tuple([None,self.MaxNAtoms,4]))
			self.frce_pl = tf.placeholder(self.tf_prec, shape=tuple([None,self.MaxNAtoms,3])) # Forces.
			if (self.ForceType=="LJ"):
				self.LJe = tf.Variable(0.316*tf.ones([8,8]),trainable=True)
				self.LJr = tf.Variable(tf.ones([8,8]),trainable=True)
				# These are squared later to keep them positive.
				self.energies, self.forces = self.LJFrc(self.inp_pl)
				self.total_loss, self.loss = self.loss_op(self.forces, self.frce_pl)
				self.train_op = self.training(self.total_loss, PARAMS["learning_rate"], PARAMS["momentum"])
				self.saver = tf.train.Saver()
			elif (self.ForceType=="Harm"):
				self.energies, self.forces = self.HarmFrc(self.inp_pl)
			else:
				raise Exception("Unknown Kernel")
			self.summary_op = tf.summary.merge_all()
			init = tf.global_variables_initializer()
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
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
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
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
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
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
		self.train_prepare()
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
		self.train_prepare()
		LOGGER.info("MolInstance_LJForce.train()")
		test_freq = PARAMS["test_freq"]
		mini_test_loss = float('inf') # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
		self.SaveAndClose()
		return

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

	def train_prepare(self,  continue_training =False):
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
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
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
		Zs = tf.cast(tf.reshape(tf.slice(inp_pl,[0,0,0],[-1,-1,1]),[nmol,maxnatom,1]),tf.int32)
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
		self.train_prepare()
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
		self.train_prepare()
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
	def __init__(self, TData_, Name_=None, Trainable_=True, ForceType_="LJ"):
		"""
		Args:
			TData_: A TensorMolData instance.
			Name_: A name for this instance.
		"""
		self.NetType = "RawBP_noGrad"
		self.SetANI1Param()
		MolInstance.__init__(self, TData_,  Name_, Trainable_)
		self.name = "Mol_"+self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType
		LOGGER.debug("Raised Instance: "+self.name)
		self.train_dir = './networks/'+self.name
		self.MaxNAtoms = TData_.MaxNAtoms
		self.eles = self.TData.eles
                self.n_eles = len(self.eles)
		self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
		self.eles_pairs = []
		for i in range (len(self.eles)):
			for j in range(i, len(self.eles)):
				self.eles_pairs.append([self.eles[i], self.eles[j]])
		self.eles_pairs_np = np.asarray(self.eles_pairs)
		self.SFPa = None
		self.SFPr = None
		self.batch_size = 1000
		self.xyzs_pl = None
		self.Zs_pl = None
		self.labels_pl = None	
		self.sess = None
		self.total_loss = None
		self.loss = None
		self.train_op = None
		self.summary_op = None
		self.saver = None
		self.summary_writer = None



        def SetANI1Param(self, prec=np.float64):
                zetas = np.array([[PARAMS["AN1_zeta"]]], dtype = prec)
                etas = np.array([[PARAMS["AN1_eta"]]], dtype = prec)
                AN1_num_a_As = PARAMS["AN1_num_a_As"]
		AN1_num_a_Rs = PARAMS["AN1_num_a_Rs"]
                thetas = np.array([ 2.0*Pi*i/AN1_num_a_As for i in range (0, AN1_num_a_As)], dtype = prec)
                AN1_a_Rc = PARAMS["AN1_a_Rc"]
                rs =  np.array([ AN1_a_Rc*i/AN1_num_a_Rs for i in range (0, AN1_num_a_Rs)], dtype = prec)
                Ra_cut = AN1_a_Rc
                # Create a parameter tensor. 4 x nzeta X neta X ntheta X nr 
                p1 = np.tile(np.reshape(zetas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
                p2 = np.tile(np.reshape(etas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
                p3 = np.tile(np.reshape(thetas,[1,1,AN1_num_a_As,1,1]),[1,1,1,AN1_num_a_Rs,1])
                p4 = np.tile(np.reshape(rs,[1,1,1,AN1_num_a_Rs,1]),[1,1,AN1_num_a_As,1,1])
                SFPa = np.concatenate([p1,p2,p3,p4],axis=4)
                self.SFPa = np.transpose(SFPa, [4,0,1,2,3])

                etas_R = np.array([[PARAMS["AN1_eta"]]], dtype = prec)
                AN1_num_r_Rs = PARAMS["AN1_num_r_Rs"]
                AN1_r_Rc = PARAMS["AN1_r_Rc"]
                rs_R =  np.array([ AN1_r_Rc*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)], dtype = prec)
                Rr_cut = AN1_r_Rc
                # Create a parameter tensor. 2 x  neta X nr 
                p1_R = np.tile(np.reshape(etas_R,[1,1,1]),[1,AN1_num_r_Rs,1])
                p2_R = np.tile(np.reshape(rs_R,[1,AN1_num_r_Rs,1]),[1,1,1])
                SFPr = np.concatenate([p1_R,p2_R],axis=2)
                self.SFPr = np.transpose(SFPr, [2,0,1])


	def train_prepare(self,  continue_training =False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(self.prec, shape=tuple([self.batch_size, self.MaxAtoms,3]))
                        self.Zs_pl=tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.MaxAtoms]))
			self.labels_pl = tf.placeholder(self.prec, shape=tuple([self.batch_size]))
			Ele = tf.Variable(self.eles_np, dtype = tf.int32)
                        Elep = tf.Variable(self.ele_pairs_np, dtype = tf.int32)
			SFPa = tf.Variable(self.SFPa, self.prec) 
			SFPr = tf.Variable(self.SFPr, self.prec)
			Ra_cut = PARAMS["AN1_a_Rc"]
                        Rr_cut = PARAMS["AN1_r_Rc"]
			self.SymOutput  = TFSymSet(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr, Rr_cut, self.Elep, self.SFPa, Ra_cut)
                        self.Scatter_Sym, self.Sym_Index = NNInterface(self.xyzs_pl, self.Zs_pl, Ele, self.SymOutput)
                        init = tf.global_variables_initializer()
                        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                        self.sess.run(init)
                        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        self.run_metadata = tf.RunMetadata()

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
                feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.labels_pl], [batch_data[0]]+[batch_data[1]]+[batch_data[2]])}
                return feed_dict


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
		self.train_prepare()
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
		self.train_prepare()
		LOGGER.info("MolInstance_LJForce.train()")
		test_freq = PARAMS["test_freq"]
		mini_test_loss = float('inf') # some big numbers
		for step in  range (0, mxsteps):
			self.train_step(step)
		self.SaveAndClose()
		return
