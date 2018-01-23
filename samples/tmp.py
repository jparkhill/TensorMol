import numpy as np
import tensorflow as tf
from TensorMol.RawEmbeddings import *

sample_xyzs = tf.tile(tf.expand_dims(tf.constant([[0.000, 0.000, 0.000],[0.757, 0.586, 0.000],[-0.757, 0.586, 0.000]], dtype=tf.float32), axis=0), [1000000, 1, 1])

rotation_params = tf.concat([np.pi * tf.expand_dims(tf.tile(tf.linspace(0.0, 2.0, 100), [10000]), axis=1),
		np.pi * tf.reshape(tf.tile(tf.expand_dims(tf.linspace(0.0, 2.0, 100), axis=1), [1,10000]), [1000000,1]),
		tf.reshape(tf.tile(tf.expand_dims(tf.expand_dims(tf.linspace(0.2, 1.8, 100), axis=1),
		axis=2), [100,1,100]), [1000000,1])], axis=1)

rotated_xyzs = tf_random_rotate(sample_xyzs, rotation_params)
grads = tf.gradients(rotated_xyzs, rotation_params)[0]
bad_params_idx = tf.where(tf.is_inf(grads))
bad_params = tf.gather_nd(rotation_params, bad_params_idx)
large_grad_idx = tf.where(tf.greater(tf.abs(grads), 1))
large_grad = tf.gather_nd(rotation_params, large_grad_idx)

z=tf.Variable(0.0)
r=tf.sqrt(z)
gradr=tf.gradients(r, z)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	bad_p = sess.run(gradr)

print bad_p

