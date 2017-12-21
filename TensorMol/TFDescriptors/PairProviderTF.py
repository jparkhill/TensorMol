from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TensorMol.ElectrostaticsTF import *
from .RawEmbeddings import *

class PairProvider:
	def __init__(self,nmol_,natom_):
		"""
		Returns pairs within a cutoff.

		Args:
			natom: number of atoms in a molecule.
			nmol: number of molecules in the batch.
		"""
		self.nmol = nmol_
		self.natom = natom_
		self.sess = None
		self.xyzs_pl = None
		self.nnz_pl = None
		self.rng_pl = None
		self.Prepare()
		return

	def PairsWithinCutoff(self,xyzs_pl,rng_pl,nnz_pl):
		"""
		It's up to whatever routine calls this to chop out
		all the indices which are larger than the number of atoms in this molecule.
		"""
		nrawpairs = self.nmol*self.natom*self.natom
		Ds = TFDistances(xyzs_pl)
		Ds = tf.matrix_band_part(Ds, 0, -1) # Extract upper triangle
		Ds = tf.where(tf.equal(Ds,0.0),10000.0*tf.ones_like(Ds),Ds)
		inds = tf.reshape(AllDoublesSet(tf.tile(tf.reshape(tf.range(self.natom),[1,self.natom]),[self.nmol,1])),[self.nmol*self.natom*self.natom,3])
		# inds has shape nmol X natom X natom X 3
		nnzs = tf.reshape(tf.tile(tf.reshape(nnz_pl,[self.nmol,1,1]),[1,self.natom,self.natom]),[self.nmol*self.natom*self.natom,1])
		a1 = tf.slice(inds,[0,1],[-1,1])
		a2 = tf.slice(inds,[0,2],[-1,1])
		prs = tf.reshape(tf.less(Ds,rng_pl),[self.nmol*self.natom*self.natom,1]) # Shape nmol X natom X natom X 1
		msk0 = tf.reshape(tf.logical_and(tf.logical_and(tf.less(a1,nnzs),tf.less(a2,nnzs)),prs),[nrawpairs])
		inds = tf.boolean_mask(inds,msk0)
		a1 = tf.slice(inds,[0,1],[-1,1])
		a2 = tf.slice(inds,[0,2],[-1,1])
		nodiag = tf.not_equal(a1,a2)
		msk = tf.reshape(nodiag,[tf.shape(inds)[0]])
		return tf.boolean_mask(inds,msk)

	def Prepare(self):
		with tf.Graph().as_default():
			self.xyzs_pl=tf.placeholder(tf.float64, shape=tuple([None,self.natom,3]))
			self.rng_pl=tf.placeholder(tf.float64, shape=())
			self.nnz_pl=tf.placeholder(tf.int32, shape=(None))
			init = tf.global_variables_initializer()
			self.GetPairs = self.PairsWithinCutoff(self.xyzs_pl,self.rng_pl,self.nnz_pl)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			#self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
			self.sess.run(init)
		return

	def __call__(self, xpl, rngpl, nnz):
		"""
		Returns the nonzero pairs.
		"""
		tmp = self.sess.run([self.GetPairs], feed_dict = {self.xyzs_pl:xpl,self.rng_pl:rngpl,self.nnz_pl: nnz})
		return tmp[0]
