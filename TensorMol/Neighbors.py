"""
Linear Scaling Atom Neighbor List Generators.
see also: http://scipy-cookbook.readthedocs.io/items/KDTree_example.html
Depending on cutoffs and density these scale to >20,000 atoms
"""

import numpy as np

def kdtree( data_, leafsize=10 ):
	"""
	build a kd-tree for O(n log n) nearest neighbour search

	input:
		data:       2D ndarray, shape =(ndim,ndata), preferentially C order
		leafsize:   max. number of data points to leave in a leaf

	output:
		kd-tree:    list of tuples
	"""
	data = data_.copy()
	ndim = data.shape[0]
	ndata = data.shape[1]
	# find bounding hyper-rectangle
	hrect = np.zeros((2,data.shape[0]))
	hrect[0,:] = data.min(axis=1)
	hrect[1,:] = data.max(axis=1)
	# create root of kd-tree
	idx = np.argsort(data[0,:], kind='mergesort')
	data[:,:] = data[:,idx]
	splitval = data[0,ndata/2]
	left_hrect = hrect.copy()
	right_hrect = hrect.copy()
	left_hrect[1, 0] = splitval
	right_hrect[0, 0] = splitval
	tree = [(None, None, left_hrect, right_hrect, None, None)]
	stack = [(data[:,:ndata/2], idx[:ndata/2], 1, 0, True),
		(data[:,ndata/2:], idx[ndata/2:], 1, 0, False)]

	# recursively split data in halves using hyper-rectangles:
	while stack:
		# pop data off stack
		data, didx, depth, parent, leftbranch = stack.pop()
		ndata = data.shape[1]
		nodeptr = len(tree)

		# update parent node
		_didx, _data, _left_hrect, _right_hrect, left, right = tree[parent]
		tree[parent] = (_didx, _data, _left_hrect, _right_hrect, nodeptr, right) if leftbranch \
			else (_didx, _data, _left_hrect, _right_hrect, left, nodeptr)
		# insert node in kd-tree

		# leaf node?
		if ndata <= leafsize:
			_didx = didx.copy()
			_data = data.copy()
			leaf = (_didx, _data, None, None, 0, 0)
			tree.append(leaf)

		# not a leaf, split the data in two
		else:
			splitdim = depth % ndim
			idx = np.argsort(data[splitdim,:], kind='mergesort')
			data[:,:] = data[:,idx]
			didx = didx[idx]
			nodeptr = len(tree)
			stack.append((data[:,:ndata/2], didx[:ndata/2], depth+1, nodeptr, True))
			stack.append((data[:,ndata/2:], didx[ndata/2:], depth+1, nodeptr, False))
			splitval = data[splitdim,ndata/2]
			if leftbranch:
				left_hrect = _left_hrect.copy()
				right_hrect = _left_hrect.copy()
			else:
				left_hrect = _right_hrect.copy()
				right_hrect = _right_hrect.copy()
			left_hrect[1, splitdim] = splitval
			right_hrect[0, splitdim] = splitval
			# append node to tree
			tree.append((None, None, left_hrect, right_hrect, None, None))
	return tree

def intersect(hrect, r2, centroid):
	"""
	checks if the hyperrectangle hrect intersects with the
	hypersphere defined by centroid and r2
	"""
	maxval = hrect[1,:]
	minval = hrect[0,:]
	p = centroid.copy()
	idx = p < minval
	p[idx] = minval[idx]
	idx = p > maxval
	p[idx] = maxval[idx]
	return ((p-centroid)**2).sum() < r2

def radius_search(tree, datapoint, radius):
	""" find all points within radius of datapoint """
	stack = [tree[0]]
	inside = []
	while stack:
		leaf_idx, leaf_data, left_hrect, \
			right_hrect, left, right = stack.pop()
		# leaf
		if leaf_idx is not None:
			param=leaf_data.shape[0]
			distance = np.sqrt(((leaf_data - datapoint.reshape((param,1)))**2).sum(axis=0))
			near = np.where(distance<=radius)
			if len(near[0]):
				idx = leaf_idx[near]
				inside += idx.tolist()
		else:
			if intersect(left_hrect, radius, datapoint):
				stack.append(tree[left])

			if intersect(right_hrect, radius, datapoint):
				stack.append(tree[right])
	return inside

class NeighborList:
	"""
	TODO: incremental tree and neighborlist updates.
	"""
	def __init__(self,x_,rcut_=5.0):
		"""
		Builds or updates a neighbor list of atoms within rcut_
		using n*Log(n) kd-tree.

		Args:
			x_: coordinate array
			rcut_: distance cutoff.
		"""
		self.natom = x_.shape[0]
		self.x = x_.copy()
		self.r = rcut_
		self.pairs = None
		self.triples = None

	def buildPairs(self):
		"""
		pairs is a number of nonzero atom pairs X 2 tensor.
		"""
		tree = kdtree(self.x.T)
		pairs = []
		for i in range(self.natom):
			pairs = pairs+[[i,k] for k in radius_search(tree,self.x[i],self.r) if i != k]
		print "Found ", len(pairs), "nonzero pairs"
		return np.array(pairs)

	def Update(x_,rcut_=5.0):
		self.x = x_.copy()
		self.pairs, self.triples = self.buildPairsAndTriples(rcut_)
		return

	def buildPairsAndTriples(self, rcut=5.0):
		"""
		Returns the nonzero pairs, triples in self.x within the cutoff.
        Triples are non-repeating ie: no 1,2,2 or 2,2,2 etc. but unordered

		Args:
			rcut: the cutoff for pairs and triples
		Returns:
			pair matrix (npair X 2)
			triples matrix (ntrip X 3)
		"""
		tree = kdtree(self.x.T)
		pair = []
		for i in range(self.natom):
			pair = pair+[[k for k in radius_search(tree,self.x[i],rcut) if i != k]]
		npairi = map(len,pair)
		npair = sum(npairi)
		ntrip = sum(map(lambda x: x*(x-1) if x>0 else 0, npairi))
		p=np.zeros((npair,2),dtype = np.uint32)
		t=np.zeros((ntrip,3),dtype = np.uint32)
		pp = 0
		tp = 0
		for i in range(self.natom):
			for j in pair[i]:
				p[pp,0]=i
				p[pp,1]=j
				pp = pp+1
				for k in pair[i]:
					if (k!=j):
						t[tp,0]=i
						t[tp,1]=j
						t[tp,2]=k
						tp=tp+1
		return p,t
