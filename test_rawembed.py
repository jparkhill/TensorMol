from TensorMol import *
from TensorMol.NN_MBE import *
from TensorMol.MBE_Opt import *
from TensorMol.RawEmbeddings import *
from TensorMol.Neighbors import *
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if (0):
	import numpy as np
	Pi = 3.1415
	xyz_array = np.array([[[0.,0.,0.],[1.0,0.,0.],[0.,0.,5.],[1.,1.,5.],[0,0,0],[0,0,0],[0,0,0]],[[0.,0.,0.],[1.0,0.,0.],[0.,0.,5.],[1.,1.,5.],[0,0,0],[0,0,0],[0,0,0]]], dtype=np.float64)
	xyz = tf.Variable(xyz_array,trainable=False)
	Z_array = np.array([[1,1,7,8,0,0,0],[1,1,7,8,0,0,0]], dtype = np.int32)
	Z = tf.Variable(Z_array,trainable=False)
	
	eleps = tf.Variable([[1,1],[1,7],[1,8],[7,8],[7,7]], dtype=tf.int32)
	zetas = tf.Variable([[8.0]], dtype=tf.float64)
	etas = tf.Variable([[4.0]], dtype=tf.float64)
	AN1_num_a_As = 8
	thetas = tf.Variable([ 2.0*Pi*i/AN1_num_a_As for i in range (0, AN1_num_a_As)], dtype=tf.float64)
	AN1_num_r_Rs = 22
	AN1_r_Rc = 4.6
	AN1_a_Rc = 3.1
	rs = tf.Variable([ AN1_r_Rc*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)], dtype=tf.float64)
	Ra_cut = AN1_r_Rc
	# Create a parameter tensor. 4 x nzeta X neta X ntheta X nr 
	p1 = tf.tile(tf.reshape(zetas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_r_Rs,1])
	p2 = tf.tile(tf.reshape(etas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_r_Rs,1])
	p3 = tf.tile(tf.reshape(thetas,[1,1,AN1_num_a_As,1,1]),[1,1,1,AN1_num_r_Rs,1])
	p4 = tf.tile(tf.reshape(rs,[1,1,1,AN1_num_r_Rs,1]),[1,1,AN1_num_a_As,1,1])
	SFPa = tf.concat([p1,p2,p3,p4],axis=4)
	SFPa = tf.transpose(SFPa, perm=[4,0,1,2,3])
	
	eles = tf.Variable([[1],[7],[8]], dtype=tf.int32)
	etas_R = tf.Variable([[4.0]], dtype=tf.float64)
	AN1_num_r_Rs = 22
	AN1_r_Rc = 4.6
	rs_R = tf.Variable([ AN1_r_Rc*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)], dtype=tf.float64)
	Rr_cut = AN1_r_Rc
	# Create a parameter tensor. 2 x  neta X nr 
	p1_R = tf.tile(tf.reshape(etas_R,[1,1,1]),[1,AN1_num_r_Rs,1])
	p2_R = tf.tile(tf.reshape(rs_R,[1,AN1_num_r_Rs,1]),[1,1,1])
	SFPr = tf.concat([p1_R,p2_R],axis=2)
	SFPr = tf.transpose(SFPr, perm=[2,0,1])
	
	
	# A test of the above.
	init = tf.global_variables_initializer()
	with tf.Session() as session:
	        session.run(init)
	        GM = session.run(tf.gradients(TFSymSet(xyz, Z, eles, SFPr, Rr_cut, eleps, SFPa, Ra_cut), xyz))
	        print (GM, GM[0].shape)

if (0):
	mset = MSet("H2O_augmented_more_cutoff5")
        mset.Load()
	SymMaker = ANISym(mset)
	SymMaker.Generate_ANISYM()

if (0):
	mset = MSet("Neigbor_test")	
	mset.ReadXYZ("Neigbor_test")
	xyzs = np.zeros((1, 6, 3),dtype=np.float64)
	Zs = np.zeros((1, 6), dtype=np.int64)
	nnz_atom = np.zeros((1), dtype=np.int64)
	for i, mol in enumerate(mset.mols):
                xyzs[i][:mol.NAtoms()] = mol.coords
                Zs[i][:mol.NAtoms()] = mol.atoms
		nnz_atom[i] = mol.NAtoms()
	print xyzs, nnz_atom, Zs
	NL = NeighborListSet(xyzs, nnz_atom, True, True, Zs)
	rad_p, ang_t = NL.buildPairsAndTriples(4.6, 3.1)
	print ("rad_p:", rad_p, " ang_t:", ang_t)

if (1):
	mset=MSet("NeigborMB_test")
	mset.ReadXYZ("NeigborMB_test")
	MBEterms = MBNeighbors(mset.mols[0].coords, mset.mols[0].atoms, [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]])
	MBEterms.Update(mset.mols[0].coords, 10.0, 10.0)
	print MBEterms.singC
	print MBEterms.pairC
	print MBEterms.tripC
	#print MBEterms.pairs
	#print MBEterms.trips
	#print MBEterms.pairz
	#print MBEterms.tripz
