from TensorMol import *
from TensorMol.NN_MBE import *
from TensorMol.MBE_Opt import *
from TensorMol.RawEmbeddings import *
from TensorMol import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
Pi = 3.1415
xyz = tf.Variable([[[0.,0.,0.],[1.0,0.,0.],[0.,0.,5.],[1.,1.,5.],[0,0,0],[0,0,0],[0,0,0]],[[0.,0.,0.],[1.0,0.,0.],[0.,0.,5.],[1.,1.,5.],[0,0,0],[0,0,0],[0,0,0]]],trainable=False)
Z = tf.Variable([[1,1,7,8,0,0,0],[1,1,7,8,0,0,0]],trainable=False)

eleps = tf.Variable([[1,1],[1,7],[1,8],[7,8],[7,7]])
zetas = tf.Variable([[8.0]])
etas = tf.Variable([[4.0]])
AN1_num_a_As = 8
thetas = tf.Variable([ 2.0*Pi*i/AN1_num_a_As for i in range (0, AN1_num_a_As)])
AN1_num_r_Rs = 22
AN1_r_Rc = 4.6
AN1_a_Rc = 3.1
rs = tf.Variable([ AN1_r_Rc*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)])
Ra_cut = AN1_r_Rc
# Create a parameter tensor. 4 x nzeta X neta X ntheta X nr 
p1 = tf.tile(tf.reshape(zetas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_r_Rs,1])
p2 = tf.tile(tf.reshape(etas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_r_Rs,1])
p3 = tf.tile(tf.reshape(thetas,[1,1,AN1_num_a_As,1,1]),[1,1,1,AN1_num_r_Rs,1])
p4 = tf.tile(tf.reshape(rs,[1,1,1,AN1_num_r_Rs,1]),[1,1,AN1_num_a_As,1,1])
SFPa = tf.concat([p1,p2,p3,p4],axis=4)
SFPa = tf.transpose(SFPa, perm=[4,0,1,2,3])

eles = tf.Variable([[1],[7],[8]])
etas_R = tf.Variable([[4.0]])
AN1_num_r_Rs = 22
AN1_r_Rc = 4.6
rs_R = tf.Variable([ AN1_r_Rc*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)])
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
        GM = session.run(TFSymSet(xyz, Z, eles, SFPr, Rr_cut, eleps, SFPa, Ra_cut))
        print (GM, GM.shape)

