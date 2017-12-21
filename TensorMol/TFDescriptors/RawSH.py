"""
Raw => various descriptors in Tensorflow code.

The Raw format is a batch of rank three tensors.
mol X MaxNAtoms X 4
The final dim is atomic number, x,y,z (Angstrom)

https://www.youtube.com/watch?v=h2zgB93KANE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..ForceModifiers.Neighbors import *
from ..Containers.TensorData import *
from ..ForceModels.ElectrostaticsTF import * # Why is this imported here?
from tensorflow.python.client import timeline
import numpy as np
import math, time
from tensorflow.python.framework import function
if (HAS_TF):
	import tensorflow as tf

# John H. this is horrible/awful.
data_precision = eval(PARAMS["tf_prec"])

def tf_pairs_list(xyzs, Zs, r_cutoff, element_pairs):
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	distance_tensor = tf.norm(delta_xyzs,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(distance_tensor, r_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
	permutation_identity_mask = tf.where(tf.less(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, permutation_identity_mask)), tf.int32)
	pair_distances = tf.expand_dims(tf.gather_nd(distance_tensor, pair_indices), axis=1)
	pair_elements = tf.stack([tf.gather_nd(Zs, pair_indices[:,0:2]), tf.gather_nd(Zs, pair_indices[:,0:3:2])], axis=-1)
	element_pair_mask = tf.cast(tf.where(tf.logical_or(tf.reduce_all(tf.equal(tf.expand_dims(pair_elements, axis=1), tf.expand_dims(element_pairs, axis=0)), axis=2),
						tf.reduce_all(tf.equal(tf.expand_dims(pair_elements, axis=1), tf.expand_dims(element_pairs[:,::-1], axis=0)), axis=2))), tf.int32)
	num_element_pairs = element_pairs.get_shape().as_list()[0]
	element_pair_distances = tf.dynamic_partition(pair_distances, element_pair_mask[:,1], num_element_pairs)
	mol_indices = tf.dynamic_partition(pair_indices[:,0], element_pair_mask[:,1], num_element_pairs)
	return element_pair_distances, mol_indices

def tf_triples_list(xyzs, Zs, r_cutoff, element_triples):
	num_mols = Zs.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	distance_tensor = tf.norm(delta_xyzs,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(distance_tensor, r_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
	permutation_identity_mask = tf.where(tf.less(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, permutation_identity_mask)), tf.int32)
	mol_pair_indices = tf.dynamic_partition(pair_indices, pair_indices[:,0], num_mols)
	mol_triples_indices = []
	tmp = []
	for i in xrange(num_mols):
		mol_common_atom_indices = tf.where(tf.reduce_all(tf.equal(tf.expand_dims(mol_pair_indices[i][:,0:2], axis=0), tf.expand_dims(mol_pair_indices[i][:,0:2], axis=1)), axis=2))
		permutation_pairs_mask = tf.where(tf.less(mol_common_atom_indices[:,0], mol_common_atom_indices[:,1]))
		mol_common_atom_indices = tf.squeeze(tf.gather(mol_common_atom_indices, permutation_pairs_mask), axis=1)
		tmp.append(mol_common_atom_indices)
		mol_triples_indices.append(tf.concat([tf.gather(mol_pair_indices[i], mol_common_atom_indices[:,0]), tf.expand_dims(tf.gather(mol_pair_indices[i], mol_common_atom_indices[:,1])[:,2], axis=1)], axis=1))
	triples_indices = tf.concat(mol_triples_indices, axis=0)
	triples_distances = tf.stack([tf.gather_nd(distance_tensor, triples_indices[:,:3]),
						tf.gather_nd(distance_tensor, tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1)),
						tf.gather_nd(distance_tensor, tf.concat([triples_indices[:,0:1], triples_indices[:,2:]], axis=1))], axis=-1)
	cos_thetas = tf.stack([(tf.square(triples_distances[:,0]) + tf.square(triples_distances[:,1]) - tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,0] * triples_distances[:,1]),
						(tf.square(triples_distances[:,0]) - tf.square(triples_distances[:,1]) + tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,0] * triples_distances[:,2]),
						(-tf.square(triples_distances[:,0]) + tf.square(triples_distances[:,1]) + tf.square(triples_distances[:,2])) \
							/ (2 * triples_distances[:,1] * triples_distances[:,2])], axis=-1)
	cos_thetas = tf.where(tf.greater_equal(cos_thetas, 1.0), tf.ones_like(cos_thetas) * (1.0 - 1.0e-24), cos_thetas)
	cos_thetas = tf.where(tf.less_equal(cos_thetas, -1.0), -1.0 * tf.ones_like(cos_thetas) * (1.0 - 1.0e-24), cos_thetas)
	triples_angles = tf.acos(cos_thetas)
	triples_distances_angles = tf.concat([triples_distances, triples_angles], axis=1)
	triples_elements = tf.stack([tf.gather_nd(Zs, triples_indices[:,0:2]), tf.gather_nd(Zs, triples_indices[:,0:3:2]), tf.gather_nd(Zs, triples_indices[:,0:4:3])], axis=-1)
	sorted_triples_elements, _ = tf.nn.top_k(triples_elements, k=3)
	element_triples_mask = tf.cast(tf.where(tf.reduce_all(tf.equal(tf.expand_dims(sorted_triples_elements, axis=1), tf.expand_dims(element_triples, axis=0)), axis=2)), tf.int32)
	num_element_triples = element_triples.get_shape().as_list()[0]
	element_triples_distances_angles = tf.dynamic_partition(triples_distances_angles, element_triples_mask[:,1], num_element_triples)
	mol_indices = tf.dynamic_partition(triples_indices[:,0], element_triples_mask[:,1], num_element_triples)
	return element_triples_distances_angles, mol_indices

def matrix_power(matrix, power):
	"""
	Raise a Hermitian Matrix to a possibly fractional power.

	Args:
		matrix (tf.float): Diagonalizable matrix
		power (tf.float): power to raise the matrix to

	Returns:
		matrix_to_power (tf.float): matrix raised to the power

	Note:
		As of tensorflow v1.3, tf.svd() does not have gradients implimented
	"""
	s, U, V = tf.svd(matrix)
	s = tf.maximum(s, tf.pow(10.0, -14.0))
	return tf.matmul(U, tf.matmul(tf.diag(tf.pow(s, power)), tf.transpose(V)))

def matrix_power2(matrix, power):
	"""
	Raises a matrix to a possibly fractional power

	Args:
		matrix (tf.float): Diagonalizable matrix
		power (tf.float): power to raise the matrix to

	Returns:
		matrix_to_power (tf.float): matrix raised to the power
	"""
	matrix_eigenvals, matrix_eigenvecs = tf.self_adjoint_eig(matrix)
	matrix_to_power = tf.matmul(matrix_eigenvecs, tf.matmul(tf.matrix_diag(tf.pow(matrix_eigenvals, power)), tf.transpose(matrix_eigenvecs)))
	return matrix_to_power

def tf_gaussian_overlap(gaussian_params):
	r_nought = gaussian_params[:,0]
	sigma = gaussian_params[:,1]
	scaling_factor = tf.cast(tf.sqrt(np.pi / 2), eval(PARAMS["tf_prec"]))
	exponential_factor = tf.exp(-tf.square(tf.expand_dims(r_nought, axis=0) - tf.expand_dims(r_nought, axis=1))
	/ (2.0 * (tf.square(tf.expand_dims(sigma, axis=0)) + tf.square(tf.expand_dims(sigma, axis=1)))))
	root_inverse_sigma_sum = tf.sqrt((1.0 / tf.expand_dims(tf.square(sigma), axis=0)) + (1.0 / tf.expand_dims(tf.square(sigma), axis=1)))
	erf_numerator = (tf.expand_dims(r_nought, axis=0) * tf.expand_dims(tf.square(sigma), axis=1)
				+ tf.expand_dims(r_nought, axis=1) * tf.expand_dims(tf.square(sigma), axis=0))
	erf_denominator = (tf.sqrt(tf.cast(2.0, eval(PARAMS["tf_prec"]))) * tf.expand_dims(tf.square(sigma), axis=0) * tf.expand_dims(tf.square(sigma), axis=1)
				* root_inverse_sigma_sum)
	erf_factor = 1 + tf.erf(erf_numerator / erf_denominator)
	overlap_matrix = scaling_factor * exponential_factor * erf_factor / root_inverse_sigma_sum
	return overlap_matrix

def tf_gaussians(distance_tensor, Zs, gaussian_params):
	exponent = (tf.square(tf.expand_dims(distance_tensor, axis=-1) - tf.expand_dims(tf.expand_dims(gaussian_params[:,0], axis=0), axis=1))) \
				/ (-2.0 * (gaussian_params[:,1] ** 2))
	gaussian_embed = tf.where(tf.greater(exponent, -25.0), tf.exp(exponent), tf.zeros_like(exponent))
	gaussian_embed *= tf.expand_dims(tf.where(tf.not_equal(distance_tensor, 0), tf.ones_like(distance_tensor),
						tf.zeros_like(distance_tensor)), axis=-1)
	return gaussian_embed

def tf_gaussians_cutoff(distance_tensor, Zs, gaussian_params):
	exponent = (tf.square(tf.expand_dims(distance_tensor, axis=-1) - tf.expand_dims(tf.expand_dims(gaussian_params[:,0], axis=0), axis=1))) \
				/ (-2.0 * (gaussian_params[:,1] ** 2))
	gaussian_embed = tf.where(tf.greater(exponent, -25.0), tf.exp(exponent), tf.zeros_like(exponent))
	gaussian_embed *= tf.expand_dims(tf.where(tf.not_equal(distance_tensor, 0), tf.ones_like(distance_tensor),
						tf.zeros_like(distance_tensor)), axis=-1)
	xi = (distance_tensor - 5.5) / (6.5 - 5.5)
	cutoff_factor = 1 - 3 * tf.square(xi) + 2 * tf.pow(xi, 3.0)
	cutoff_factor = tf.where(tf.greater(distance_tensor, 6.5), tf.zeros_like(cutoff_factor), cutoff_factor)
	cutoff_factor = tf.where(tf.less(distance_tensor, 5.5), tf.ones_like(cutoff_factor), cutoff_factor)
	return gaussian_embed * tf.expand_dims(cutoff_factor, axis=-1)

def tf_spherical_harmonics_0(inverse_distance_tensor):
	return tf.fill(tf.shape(inverse_distance_tensor), tf.constant(0.28209479177387814, dtype=eval(PARAMS["tf_prec"])))

def tf_spherical_harmonics_1(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_0(tf.expand_dims(inverse_distance_tensor, axis=-1))
	l1_harmonics = 0.4886025119029199 * tf.stack([delta_xyzs[...,1], delta_xyzs[...,2], delta_xyzs[...,0]],
										axis=-1) * tf.expand_dims(inverse_distance_tensor, axis=-1)
	return tf.concat([lower_order_harmonics, l1_harmonics], axis=-1)

def tf_spherical_harmonics_2(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_1(delta_xyzs, inverse_distance_tensor)
	l2_harmonics = tf.stack([(-1.0925484305920792 * delta_xyzs[...,0] * delta_xyzs[...,1]),
			(1.0925484305920792 * delta_xyzs[...,1] * delta_xyzs[...,2]),
			(-0.31539156525252005 * (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2]))),
			(1.0925484305920792 * delta_xyzs[...,0] * delta_xyzs[...,2]),
			(0.5462742152960396 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])))], axis=-1) \
			* tf.expand_dims(tf.square(inverse_distance_tensor),axis=-1)
	return tf.concat([lower_order_harmonics, l2_harmonics], axis=-1)

def tf_spherical_harmonics_3(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_2(delta_xyzs, inverse_distance_tensor)
	l3_harmonics = tf.stack([(-0.5900435899266435 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]))),
			(-2.890611442640554 * delta_xyzs[...,0] * delta_xyzs[...,1] * delta_xyzs[...,2]),
			(-0.4570457994644658 * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 4. \
				* tf.square(delta_xyzs[...,2]))),
			(0.3731763325901154 * delta_xyzs[...,2] * (-3. * tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1]) \
				+ 2. * tf.square(delta_xyzs[...,2]))),
			(-0.4570457994644658 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 4. \
				* tf.square(delta_xyzs[...,2]))),
			(1.445305721320277 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2]),
			(0.5900435899266435 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])))], axis=-1) \
				* tf.expand_dims(tf.pow(inverse_distance_tensor,3),axis=-1)
	return tf.concat([lower_order_harmonics, l3_harmonics], axis=-1)

def tf_spherical_harmonics_4(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_3(delta_xyzs, inverse_distance_tensor)
	l4_harmonics = tf.stack([(2.5033429417967046 * delta_xyzs[...,0] * delta_xyzs[...,1] * (-1. * tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]))),
			(-1.7701307697799304 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2]),
			(0.9461746957575601 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2]))),
			(-0.6690465435572892 * delta_xyzs[...,1] * delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) + 3. \
				* tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2]))),
			(0.10578554691520431 * (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 24. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) + 6. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2])))),
			(-0.6690465435572892 * delta_xyzs[...,0] * delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) + 3.
				* tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2]))),
			(-0.47308734787878004 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2]))),
			(1.7701307697799304 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2]),
			(0.6258357354491761 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) \
				+ tf.pow(delta_xyzs[...,1], 4)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,4),axis=-1)
	return tf.concat([lower_order_harmonics, l4_harmonics], axis=-1)

def tf_spherical_harmonics_5(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_4(delta_xyzs, inverse_distance_tensor)
	l5_harmonics = tf.stack([(0.6563820568401701 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4))),
			(8.302649259524166 * delta_xyzs[...,0] * delta_xyzs[...,1] * (-1. * tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2]),
			(0.4892382994352504 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2]))),
			(4.793536784973324 * delta_xyzs[...,0] * delta_xyzs[...,1] * delta_xyzs[...,2] \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2]))),
			(0.45294665119569694 * delta_xyzs[...,1] * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) - 12. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) + 2. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2])))),
			(0.1169503224534236 * delta_xyzs[...,2] * (15. * tf.pow(delta_xyzs[...,0], 4) + 15. * tf.pow(delta_xyzs[...,1], 4) \
				- 40. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) + 10. \
				* tf.square(delta_xyzs[...,0]) * (3. * tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2])))),
			(0.45294665119569694 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) - 12. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) + 2. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2])))),
			(-2.396768392486662 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2] \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2]))),
			(-0.4892382994352504 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2]))),
			(2.0756623148810416 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2]),
			(0.6563820568401701 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + 5. * tf.pow(delta_xyzs[...,1], 4)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,5),axis=-1)
	return tf.concat([lower_order_harmonics, l5_harmonics], axis=-1)

def tf_spherical_harmonics_6(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_5(delta_xyzs, inverse_distance_tensor)
	l6_harmonics = tf.stack([(-1.3663682103838286 * delta_xyzs[...,0] * delta_xyzs[...,1] * (3. * tf.pow(delta_xyzs[...,0], 4) \
				- 10. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + 3. * tf.pow(delta_xyzs[...,1], 4))),
			(2.366619162231752 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2]),
			(2.0182596029148967 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2]))),
			(0.9212052595149236 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) \
				* delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) + 3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2]))),
			(-0.9212052595149236 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) \
				- 16. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. * tf.pow(delta_xyzs[...,2], 4) \
				+ 2. * tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])))),
			(0.5826213625187314 * delta_xyzs[...,1] * delta_xyzs[...,2] * (5. * tf.pow(delta_xyzs[...,0], 4) + 5. * tf.pow(delta_xyzs[...,1], 4) \
				- 20. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4) \
				+ 10. * tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2])))),
			(-0.06356920226762842 * (5. * tf.pow(delta_xyzs[...,0], 6) + 5. * tf.pow(delta_xyzs[...,1], 6) - 90. \
				* tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 120. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) - 16. * tf.pow(delta_xyzs[...,2], 6) + 15. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 6. * tf.square(delta_xyzs[...,2])) + 15. * tf.square(delta_xyzs[...,0]) \
				* (tf.pow(delta_xyzs[...,1], 4) - 12. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. * tf.pow(delta_xyzs[...,2], 4)))),
			(0.5826213625187314 * delta_xyzs[...,0] * delta_xyzs[...,2] * (5. * tf.pow(delta_xyzs[...,0], 4) + 5. \
				* tf.pow(delta_xyzs[...,1], 4) - 20. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 8. \
				* tf.pow(delta_xyzs[...,2], 4) + 10. * tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 2. \
				* tf.square(delta_xyzs[...,2])))),
			(0.4606026297574618 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * (tf.pow(delta_xyzs[...,0], 4) \
				+ tf.pow(delta_xyzs[...,1], 4) - 16. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. \
				* tf.pow(delta_xyzs[...,2], 4) + 2. * tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 8. \
				* tf.square(delta_xyzs[...,2])))),
			(-0.9212052595149236 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2] \
				* (3. * tf.square(delta_xyzs[...,0]) + 3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2]))),
			(-0.5045649007287242 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) \
				+ tf.pow(delta_xyzs[...,1], 4)) * (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2]))),
			(2.366619162231752 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + 5. * tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2]),
			(0.6831841051919143 * (tf.pow(delta_xyzs[...,0], 6) - 15. * tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) \
				+ 15. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 1. * tf.pow(delta_xyzs[...,1], 6)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,6),axis=-1)
	return tf.concat([lower_order_harmonics, l6_harmonics], axis=-1)

def tf_spherical_harmonics_7(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_6(delta_xyzs, inverse_distance_tensor)
	l7_harmonics = tf.stack([(-0.7071627325245962 * delta_xyzs[...,1] * (-7. * tf.pow(delta_xyzs[...,0], 6) + 35. \
				* tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) - 21. * tf.square(delta_xyzs[...,0]) \
				* tf.pow(delta_xyzs[...,1], 4) + tf.pow(delta_xyzs[...,1], 6))),
			(-5.291921323603801 * delta_xyzs[...,0] * delta_xyzs[...,1] * (3. * tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + 3. * tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2]),
			(-0.5189155787202604 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4)) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 12. * tf.square(delta_xyzs[...,2]))),
			(4.151324629762083 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) - 1. \
				* tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) + 3. \
				* tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2]))),
			(-0.15645893386229404 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) \
				* (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 60. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 80. * tf.pow(delta_xyzs[...,2], 4) + 6. * tf.square(delta_xyzs[...,0]) \
				* (tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2])))),
			(-0.4425326924449826 * delta_xyzs[...,0] * delta_xyzs[...,1] * delta_xyzs[...,2] * (15. * tf.pow(delta_xyzs[...,0], 4) \
				+ 15. * tf.pow(delta_xyzs[...,1], 4) - 80. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 48. \
				* tf.pow(delta_xyzs[...,2], 4) + 10. * tf.square(delta_xyzs[...,0]) * (3. * tf.square(delta_xyzs[...,1]) - 8. \
				* tf.square(delta_xyzs[...,2])))),
			(-0.0903316075825173 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 6) + 5. * tf.pow(delta_xyzs[...,1], 6) - 120. \
				* tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 240. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) - 64. * tf.pow(delta_xyzs[...,2], 6) + 15. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])) + 15. * tf.square(delta_xyzs[...,0]) \
				* (tf.pow(delta_xyzs[...,1], 4) - 16. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. \
				* tf.pow(delta_xyzs[...,2], 4)))),
			(0.06828427691200495 * delta_xyzs[...,2] * (-35. * tf.pow(delta_xyzs[...,0], 6) - 35. * tf.pow(delta_xyzs[...,1], 6) \
				+ 210. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) - 168. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) + 16. * tf.pow(delta_xyzs[...,2], 6) - 105. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 2. * tf.square(delta_xyzs[...,2])) - 21. * tf.square(delta_xyzs[...,0]) \
				* (5. * tf.pow(delta_xyzs[...,1], 4) - 20. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) \
				+ 8. * tf.pow(delta_xyzs[...,2], 4)))),
			(-0.0903316075825173 * delta_xyzs[...,0] * (5. * tf.pow(delta_xyzs[...,0], 6) + 5. * tf.pow(delta_xyzs[...,1], 6) \
				- 120. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 240. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) - 64. * tf.pow(delta_xyzs[...,2], 6) + 15. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])) + 15. * tf.square(delta_xyzs[...,0]) \
				* (tf.pow(delta_xyzs[...,1], 4) - 16. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. \
				* tf.pow(delta_xyzs[...,2], 4)))),
			(0.2212663462224913 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * delta_xyzs[...,2] \
				* (15. * tf.pow(delta_xyzs[...,0], 4) + 15. * tf.pow(delta_xyzs[...,1], 4) - 80. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 48. * tf.pow(delta_xyzs[...,2], 4) + 10. * tf.square(delta_xyzs[...,0]) \
				* (3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])))),
			(0.15645893386229404 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) \
				* (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 60. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 80. * tf.pow(delta_xyzs[...,2], 4) + 6. * tf.square(delta_xyzs[...,0]) \
				* (tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2])))),
			(-1.0378311574405208 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) \
				+ tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2] * (3. * tf.square(delta_xyzs[...,0]) \
				+ 3. * tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2]))),
			(-0.5189155787202604 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + 5. * tf.pow(delta_xyzs[...,1], 4)) * (tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]) - 12. * tf.square(delta_xyzs[...,2]))),
			(2.6459606618019005 * (tf.pow(delta_xyzs[...,0], 6) - 15. * tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) \
				+ 15. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 1. * tf.pow(delta_xyzs[...,1], 6)) * delta_xyzs[...,2]),
			(0.7071627325245962 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 6) - 21. * tf.pow(delta_xyzs[...,0], 4) \
				* tf.square(delta_xyzs[...,1]) + 35. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 7. \
				* tf.pow(delta_xyzs[...,1], 6)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,7),axis=-1)
	return tf.concat([lower_order_harmonics, l7_harmonics], axis=-1)

def tf_spherical_harmonics_8(delta_xyzs, inverse_distance_tensor):
	lower_order_harmonics = tf_spherical_harmonics_7(delta_xyzs, inverse_distance_tensor)
	l8_harmonics = tf.stack([(-5.831413281398639 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.pow(delta_xyzs[...,0], 6) \
				- 7. * tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) + 7. * tf.square(delta_xyzs[...,0]) \
				* tf.pow(delta_xyzs[...,1], 4) - 1. * tf.pow(delta_xyzs[...,1], 6))),
			(-2.9157066406993195 * delta_xyzs[...,1] * (-7. * tf.pow(delta_xyzs[...,0], 6) + 35. * tf.pow(delta_xyzs[...,0], 4) \
				* tf.square(delta_xyzs[...,1]) - 21. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) \
				+ tf.pow(delta_xyzs[...,1], 6)) * delta_xyzs[...,2]),
			(1.0646655321190852 * delta_xyzs[...,0] * delta_xyzs[...,1] * (3. * tf.pow(delta_xyzs[...,0], 4) - 10. \
				* tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) + 3. * tf.pow(delta_xyzs[...,1], 4)) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 14. * tf.square(delta_xyzs[...,2]))),
			(-3.449910622098108 * delta_xyzs[...,1] * (5. * tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2] * (tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2]))),
			(-1.9136660990373227 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.square(delta_xyzs[...,0]) - 1. \
				* tf.square(delta_xyzs[...,1])) * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) - 24. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 40. * tf.pow(delta_xyzs[...,2], 4) + 2. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 12. * tf.square(delta_xyzs[...,2])))),
			(-1.2352661552955442 * delta_xyzs[...,1] * (-3. * tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1])) \
				* delta_xyzs[...,2] * (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 20. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. * tf.pow(delta_xyzs[...,2], 4) \
				+ tf.square(delta_xyzs[...,0]) * (6. * tf.square(delta_xyzs[...,1]) - 20. * tf.square(delta_xyzs[...,2])))),
			(0.912304516869819 * delta_xyzs[...,0] * delta_xyzs[...,1] * (tf.pow(delta_xyzs[...,0], 6) + tf.pow(delta_xyzs[...,1], 6) \
				- 30. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 80. * tf.square(delta_xyzs[...,1]) \
				* tf.pow(delta_xyzs[...,2], 4) - 32. * tf.pow(delta_xyzs[...,2], 6) + 3. * tf.pow(delta_xyzs[...,0], 4) \
				* (tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2])) + tf.square(delta_xyzs[...,0]) \
				* (3. * tf.pow(delta_xyzs[...,1], 4) - 60. * tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 80. \
				* tf.pow(delta_xyzs[...,2], 4)))),
			(-0.10904124589877995 * delta_xyzs[...,1] * delta_xyzs[...,2] * (35. * tf.pow(delta_xyzs[...,0], 6) + 35. \
				* tf.pow(delta_xyzs[...,1], 6) - 280. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 336. \
				* tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 4) - 64. * tf.pow(delta_xyzs[...,2], 6) + 35. \
				* tf.pow(delta_xyzs[...,0], 4) * (3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])) + 7. \
				* tf.square(delta_xyzs[...,0]) * (15. * tf.pow(delta_xyzs[...,1], 4) - 80. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 48. * tf.pow(delta_xyzs[...,2], 4)))),
			(0.009086770491564996 * (35. * tf.pow(delta_xyzs[...,0], 8) + 35. * tf.pow(delta_xyzs[...,1], 8) - 1120. \
				* tf.pow(delta_xyzs[...,1], 6) * tf.square(delta_xyzs[...,2]) + 3360. * tf.pow(delta_xyzs[...,1], 4) \
				* tf.pow(delta_xyzs[...,2], 4) - 1792. * tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 6) + 128. \
				* tf.pow(delta_xyzs[...,2], 8) + 140. * tf.pow(delta_xyzs[...,0], 6) * (tf.square(delta_xyzs[...,1]) - 8. \
				* tf.square(delta_xyzs[...,2])) + 210. * tf.pow(delta_xyzs[...,0], 4) * (tf.pow(delta_xyzs[...,1], 4) - 16. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. * tf.pow(delta_xyzs[...,2], 4)) + 28. \
				* tf.square(delta_xyzs[...,0]) * (5. * tf.pow(delta_xyzs[...,1], 6) - 120. * tf.pow(delta_xyzs[...,1], 4) \
				* tf.square(delta_xyzs[...,2]) + 240. * tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 4) - 64. \
				* tf.pow(delta_xyzs[...,2], 6)))),
			(-0.10904124589877995 * delta_xyzs[...,0] * delta_xyzs[...,2] * (35. * tf.pow(delta_xyzs[...,0], 6) + 35. \
				* tf.pow(delta_xyzs[...,1], 6) - 280. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 336. \
				* tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 4) - 64. * tf.pow(delta_xyzs[...,2], 6) + 35. \
				* tf.pow(delta_xyzs[...,0], 4) * (3. * tf.square(delta_xyzs[...,1]) - 8. * tf.square(delta_xyzs[...,2])) + 7. \
				* tf.square(delta_xyzs[...,0]) * (15. * tf.pow(delta_xyzs[...,1], 4) - 80. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 48. * tf.pow(delta_xyzs[...,2], 4)))),
			(-0.4561522584349095 * (tf.square(delta_xyzs[...,0]) - 1. * tf.square(delta_xyzs[...,1])) * (tf.pow(delta_xyzs[...,0], 6) \
				+ tf.pow(delta_xyzs[...,1], 6) - 30. * tf.pow(delta_xyzs[...,1], 4) * tf.square(delta_xyzs[...,2]) + 80. \
				* tf.square(delta_xyzs[...,1]) * tf.pow(delta_xyzs[...,2], 4) - 32. * tf.pow(delta_xyzs[...,2], 6) + 3. \
				* tf.pow(delta_xyzs[...,0], 4) * (tf.square(delta_xyzs[...,1]) - 10. * tf.square(delta_xyzs[...,2])) \
				+ tf.square(delta_xyzs[...,0]) * (3. * tf.pow(delta_xyzs[...,1], 4) - 60. * tf.square(delta_xyzs[...,1]) \
				* tf.square(delta_xyzs[...,2]) + 80. * tf.pow(delta_xyzs[...,2], 4)))),
			(1.2352661552955442 * delta_xyzs[...,0] * (tf.square(delta_xyzs[...,0]) - 3. * tf.square(delta_xyzs[...,1])) \
				* delta_xyzs[...,2] * (3. * tf.pow(delta_xyzs[...,0], 4) + 3. * tf.pow(delta_xyzs[...,1], 4) - 20. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 16. * tf.pow(delta_xyzs[...,2], 4) \
				+ tf.square(delta_xyzs[...,0]) * (6. * tf.square(delta_xyzs[...,1]) - 20. * tf.square(delta_xyzs[...,2])))),
			(0.47841652475933066 * (tf.pow(delta_xyzs[...,0], 4) - 6. * tf.square(delta_xyzs[...,0]) * tf.square(delta_xyzs[...,1]) \
				+ tf.pow(delta_xyzs[...,1], 4)) * (tf.pow(delta_xyzs[...,0], 4) + tf.pow(delta_xyzs[...,1], 4) - 24. \
				* tf.square(delta_xyzs[...,1]) * tf.square(delta_xyzs[...,2]) + 40. * tf.pow(delta_xyzs[...,2], 4) + 2. \
				* tf.square(delta_xyzs[...,0]) * (tf.square(delta_xyzs[...,1]) - 12. * tf.square(delta_xyzs[...,2])))),
			(-3.449910622098108 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 4) - 10. * tf.square(delta_xyzs[...,0]) \
				* tf.square(delta_xyzs[...,1]) + 5. * tf.pow(delta_xyzs[...,1], 4)) * delta_xyzs[...,2] * (tf.square(delta_xyzs[...,0]) \
				+ tf.square(delta_xyzs[...,1]) - 4. * tf.square(delta_xyzs[...,2]))),
			(-0.5323327660595426 * (tf.pow(delta_xyzs[...,0], 6) - 15. * tf.pow(delta_xyzs[...,0], 4) * tf.square(delta_xyzs[...,1]) \
				+ 15. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 1. * tf.pow(delta_xyzs[...,1], 6)) \
				* (tf.square(delta_xyzs[...,0]) + tf.square(delta_xyzs[...,1]) - 14. * tf.square(delta_xyzs[...,2]))),
			(2.9157066406993195 * delta_xyzs[...,0] * (tf.pow(delta_xyzs[...,0], 6) - 21. * tf.pow(delta_xyzs[...,0], 4) \
				* tf.square(delta_xyzs[...,1]) + 35. * tf.square(delta_xyzs[...,0]) * tf.pow(delta_xyzs[...,1], 4) - 7. \
				* tf.pow(delta_xyzs[...,1], 6)) * delta_xyzs[...,2]),
			(0.7289266601748299 * (tf.pow(delta_xyzs[...,0], 8) - 28. * tf.pow(delta_xyzs[...,0], 6) * tf.square(delta_xyzs[...,1]) \
				+ 70. * tf.pow(delta_xyzs[...,0], 4) * tf.pow(delta_xyzs[...,1], 4) - 28. * tf.square(delta_xyzs[...,0]) \
				* tf.pow(delta_xyzs[...,1], 6) + tf.pow(delta_xyzs[...,1], 8)))], axis=-1) \
			* tf.expand_dims(tf.pow(inverse_distance_tensor,8),axis=-1)
	return tf.concat([lower_order_harmonics, l8_harmonics], axis=-1)

def tf_spherical_harmonics(delta_xyzs, distance_tensor, max_l):
	inverse_distance_tensor = tf.where(tf.greater(distance_tensor, 1.e-9), tf.reciprocal(distance_tensor), tf.zeros_like(distance_tensor))
	if max_l == 8:
		harmonics = tf_spherical_harmonics_8(delta_xyzs, inverse_distance_tensor)
	elif max_l == 7:
		harmonics = tf_spherical_harmonics_7(delta_xyzs, inverse_distance_tensor)
	elif max_l == 6:
		harmonics = tf_spherical_harmonics_6(delta_xyzs, inverse_distance_tensor)
	elif max_l == 5:
		harmonics = tf_spherical_harmonics_5(delta_xyzs, inverse_distance_tensor)
	elif max_l == 4:
		harmonics = tf_spherical_harmonics_4(delta_xyzs, inverse_distance_tensor)
	elif max_l == 3:
		harmonics = tf_spherical_harmonics_3(delta_xyzs, inverse_distance_tensor)
	elif max_l == 2:
		harmonics = tf_spherical_harmonics_2(delta_xyzs, inverse_distance_tensor)
	elif max_l == 1:
		harmonics = tf_spherical_harmonics_1(delta_xyzs, inverse_distance_tensor)
	elif max_l == 0:
		harmonics = tf_spherical_harmonics_0(inverse_distance_tensor)
	else:
		raise Exception("Spherical Harmonics only implemented up to l=8. Choose a lower order")
	return harmonics

def tf_gaussian_spherical_harmonics_element(xyzs, Zs, element, gaussian_params, atomic_embed_factors, l_max, labels=None):
	"""
	Encodes atoms into a gaussians and spherical harmonics embedding

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		element (int): element to return embedding/labels for
		gaussian_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		atomic_embed_factors (tf.float): MaxElementNumber tensor of scaling factors for elements
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use (needs implemented)
		labels (tf.Tensor): NMol x MaxNAtoms x label shape tensor of learning targets

	Returns:
		embedding (tf.float): atom embeddings for element
		labels (tf.float): atom labels for element
	"""
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	element_indices = tf.cast(tf.where(tf.equal(Zs, element)), tf.int32)
	num_batch_elements = tf.shape(element_indices)[0]
	element_delta_xyzs = tf.gather_nd(delta_xyzs, element_indices)
	element_Zs = tf.gather(Zs, element_indices[:,0])
	distance_tensor = tf.norm(element_delta_xyzs+1.e-16,axis=2)
	atom_scaled_gaussians, min_eigenval = tf_gaussians(tf.expand_dims(distance_tensor, axis=-1), element_Zs, gaussian_params, atomic_embed_factors)
	spherical_harmonics = tf_spherical_harmonics(element_delta_xyzs, distance_tensor, l_max)
	element_embedding = tf.reshape(tf.einsum('jkg,jkl->jgl', atom_scaled_gaussians, spherical_harmonics),
							[num_batch_elements, tf.shape(gaussian_params)[0] * (l_max + 1) ** 2])
	if labels != None:
		element_labels = tf.gather_nd(labels, element_indices)
		return element_embedding, element_labels, element_indices, min_eigenval
	else:
		return element_embedding, element_indices, min_eigenval

def tf_gaussian_spherical_harmonics(xyzs, Zs, elements, gaussian_params, atomic_embed_factors, l_max):
	"""
	Encodes atoms into a gaussians and spherical harmonics embedding

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		element (int): element to return embedding/labels for
		gaussian_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		atomic_embed_factors (tf.float): MaxElementNumber tensor of scaling factors for elements
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use (needs implemented)
		labels (tf.Tensor): NMol x MaxNAtoms x label shape tensor of learning targets

	Returns:
		embedding (tf.float): atom embeddings for element
		labels (tf.float): atom labels for element
	"""
	num_elements = elements.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	element_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(Zs, axis=-1), tf.reshape(elements, [1, 1, tf.shape(elements)[0]]))), tf.int32)
	distance_tensor = tf.norm(delta_xyzs+1.e-16,axis=3)
	atom_scaled_gaussians = tf_gaussians(distance_tensor, Zs, gaussian_params, atomic_embed_factors, orthogonalize)
	spherical_harmonics = tf_spherical_harmonics(delta_xyzs, distance_tensor, l_max)
	embeddings = tf.reshape(tf.einsum('ijkg,ijkl->ijgl', atom_scaled_gaussians, spherical_harmonics),
							[tf.shape(Zs)[0], tf.shape(Zs)[1], tf.shape(gaussian_params)[0] * (l_max + 1) ** 2])
	embeddings = tf.gather_nd(embeddings, element_indices[:,0:2])
	element_embeddings = tf.dynamic_partition(embeddings, element_indices[:,2], num_elements)
	molecule_indices = tf.dynamic_partition(element_indices[:,0:2], element_indices[:,2], num_elements)
	return element_embeddings, molecule_indices

def tf_gaussian_spherical_harmonics_channel_sparse(xyzs, Zs, elements, gaussian_params, l_max, RadNeighbors):
	"""
	Encodes atoms into a gaussians * spherical harmonics embedding
	Works on a batch of molecules. This is the embedding routine used
	in BehlerParinelloDirectGauSH.

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		element (int): element to return embedding/labels for
		gaussian_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use (needs implemented)
		RadNeighbors: NMol x MaxNAtoms x MaxNeighbors tensor of neighbors within cutoff.

	Returns:
		embedding (tf.float): atom embeddings for element
		molecule_indices (tf.float): mapping between atoms and molecules.
	"""
	num_elements = elements.get_shape().as_list()[0]
	num_molecules = Zs.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	mol_atom_indices = tf.where(tf.not_equal(Zs, 0))
	delta_xyzs = tf.gather_nd(delta_xyzs, mol_atom_indices)
	distance_tensor = tf.norm(delta_xyzs+1.e-16,axis=2)
	gaussians = tf_gaussians_cutoff(distance_tensor, Zs, gaussian_params)
	spherical_harmonics = tf_spherical_harmonics(delta_xyzs, distance_tensor, l_max)
	channel_scatter_bool = tf.gather(tf.equal(tf.expand_dims(Zs, axis=1), tf.reshape(elements, [1, num_elements, 1])), mol_atom_indices[:,0])

	channel_scatter = tf.where(channel_scatter_bool, tf.ones_like(channel_scatter_bool, dtype=PARAMS["tf_prec"]),tf.zeros_like(channel_scatter_bool, dtype=PARAMS["tf_prec"]))

	element_channel_gaussians = tf.expand_dims(gaussians, axis=1) * tf.expand_dims(channel_scatter, axis=-1)
	element_channel_harmonics = tf.expand_dims(spherical_harmonics, axis=1) * tf.expand_dims(channel_scatter, axis=-1)

	embeddings = tf.reshape(tf.einsum('ijkg,ijkl->ijgl', element_channel_gaussians, element_channel_harmonics),
	[tf.shape(mol_atom_indices)[0], -1])

	partition_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(tf.gather_nd(Zs, mol_atom_indices), axis=-1), tf.expand_dims(elements, axis=0)))[:,1], tf.int32)

	element_embeddings = tf.dynamic_partition(embeddings, partition_indices, num_elements)
	molecule_indices = tf.dynamic_partition(mol_atom_indices, partition_indices, num_elements)
	return element_embeddings, molecule_indices


def tf_gaussian_spherical_harmonics_channel(xyzs, Zs, elements, gaussian_params, l_max):
	"""
	Encodes atoms into a gaussians * spherical harmonics embedding
	Works on a batch of molecules. This is the embedding routine used
	in BehlerParinelloDirectGauSH.

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		element (int): element to return embedding/labels for
		gaussian_params (tf.float): NGaussians x 2 tensor of gaussian parameters
		l_max (tf.int32): Scalar for the highest order spherical harmonics to use (needs implemented)
		labels (tf.Tensor): NMol x MaxNAtoms x label shape tensor of learning targets

	Returns:
		embedding (tf.float): atom embeddings for element
		molecule_indices (tf.float): mapping between atoms and molecules.
	"""
	num_elements = elements.get_shape().as_list()[0]
	num_molecules = Zs.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	mol_atom_indices = tf.where(tf.not_equal(Zs, 0))
	delta_xyzs = tf.gather_nd(delta_xyzs, mol_atom_indices)
	distance_tensor = tf.norm(delta_xyzs+1.e-16,axis=2)
	gaussians = tf_gaussians_cutoff(distance_tensor, Zs, gaussian_params)
	spherical_harmonics = tf_spherical_harmonics(delta_xyzs, distance_tensor, l_max)
	channel_scatter_bool = tf.gather(tf.equal(tf.expand_dims(Zs, axis=1), tf.reshape(elements, [1, num_elements, 1])), mol_atom_indices[:,0])

	channel_scatter = tf.where(channel_scatter_bool, tf.ones_like(channel_scatter_bool, dtype=eval(PARAMS["tf_prec"])),tf.zeros_like(channel_scatter_bool, dtype=eval(PARAMS["tf_prec"])))

	element_channel_gaussians = tf.expand_dims(gaussians, axis=1) * tf.expand_dims(channel_scatter, axis=-1)
	element_channel_harmonics = tf.expand_dims(spherical_harmonics, axis=1) * tf.expand_dims(channel_scatter, axis=-1)

	embeddings = tf.reshape(tf.einsum('ijkg,ijkl->ijgl', element_channel_gaussians, element_channel_harmonics),
	[tf.shape(mol_atom_indices)[0], -1])

	partition_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(tf.gather_nd(Zs, mol_atom_indices), axis=-1), tf.expand_dims(elements, axis=0)))[:,1], tf.int32)

	element_embeddings = tf.dynamic_partition(embeddings, partition_indices, num_elements)
	molecule_indices = tf.dynamic_partition(mol_atom_indices, partition_indices, num_elements)
	return element_embeddings, molecule_indices

def tf_random_rotate(xyzs, rotation_params, labels = None, return_matrix = False):
	"""
	Rotates molecules and optionally labels in a uniformly random fashion

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		labels (tf.float, optional): NMol x MaxNAtoms x label shape tensor of learning targets
		return_matrix (bool): Returns rotation tensor if True

	Returns:
		new_xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor of randomly rotated molecules
		new_labels (tf.float): NMol x MaxNAtoms x label shape tensor of randomly rotated learning targets
	"""
	r = tf.sqrt(rotation_params[...,2])
	v = tf.stack([tf.sin(rotation_params[...,1]) * r, tf.cos(rotation_params[...,1]) * r, tf.sqrt(2.0 - rotation_params[...,2])], axis=-1)
	zero_tensor = tf.zeros_like(rotation_params[...,1])

	R1 = tf.stack([tf.cos(rotation_params[...,0]), tf.sin(rotation_params[...,0]), zero_tensor], axis=-1)
	R2 = tf.stack([-tf.sin(rotation_params[...,0]), tf.cos(rotation_params[...,0]), zero_tensor], axis=-1)
	R3 = tf.stack([zero_tensor, zero_tensor, tf.ones_like(rotation_params[...,1])], axis=-1)
	R = tf.stack([R1, R2, R3], axis=-2)
	M = tf.matmul((tf.expand_dims(v, axis=-2) * tf.expand_dims(v, axis=-1)) - tf.eye(3, dtype=eval(PARAMS["tf_prec"])), R)
	new_xyzs = tf.einsum("lij,lkj->lki", M, xyzs)
	if labels != None:
		new_labels = tf.einsum("lij,lkj->lki",M, (xyzs + labels)) - new_xyzs
		if return_matrix:
			return new_xyzs, new_labels, M
		else:
			return new_xyzs, new_labels
	elif return_matrix:
		return new_xyzs, M
	else:
		return new_xyzs

def tf_symmetry_functions(xyzs, Zs, elements, element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta):
	"""
	Encodes atoms into the symmetry function embedding as implemented in the ANI-1 Neural Network (doi: 10.1039/C6SC05720A)

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		num_atoms (tf.int32): NMol number of atoms numpy array
		elements (tf.int32): NElements tensor containing sorted unique atomic numbers present
		element_pairs (tf.int32): NElementPairs x 2 tensor containing sorted unique pairs of atomic numbers present
		radial_cutoff (tf.float): scalar tensor with the cutoff for radial pairs
		angular_cutoff (tf.float): scalar tensor with the cutoff for the angular triples
		radial_rs (tf.float): NRadialGridPoints tensor with R_s values for the radial grid
		angular_rs (tf.float): NAngularGridPoints tensor with the R_s values for the radial part of the angular grid
		theta_s (tf.float): NAngularGridPoints tensor with the theta_s values for the angular grid
		zeta (tf.float): scalar tensor with the zeta parameter for the symmetry functions
		eta (tf.float): scalar tensor with the eta parameter for the symmetry functions

	Returns:
		element_embeddings (list of tf.floats): List of NAtoms x (NRadial_rs x NElements + NAngular_rs x NTheta_s x NElementPairs)
				tensors of the same element type
		mol_indices (list of tf.int32s): List of NAtoms of the same element types with the molecule index of each atom
	"""
	num_molecules = Zs.get_shape().as_list()[0]
	num_elements = elements.get_shape().as_list()[0]
	num_element_pairs = element_pairs.get_shape().as_list()[0]

	radial_embedding, pair_indices, pair_elements = tf_symmetry_functions_radial_grid(xyzs, Zs, radial_cutoff, radial_rs, eta)
	angular_embedding, triples_indices, triples_element, triples_element_pairs = tf_symmetry_function_angular_grid(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta)

	pair_element_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(pair_elements[:,1], axis=-1),
							tf.expand_dims(elements, axis=0))), tf.int32)[:,1]
	triples_elements_indices = tf.cast(tf.where(tf.reduce_all(tf.equal(tf.expand_dims(triples_element_pairs, axis=-2),
									element_pairs), axis=-1)), tf.int32)[:,1]
	radial_scatter_indices = tf.concat([pair_indices, tf.expand_dims(pair_element_indices, axis=1)], axis=1)
	angular_scatter_indices = tf.concat([triples_indices, tf.expand_dims(triples_elements_indices, axis=1)], axis=1)

	radial_molecule_embeddings = tf.dynamic_partition(radial_embedding, pair_indices[:,0], num_molecules)
	radial_atom_indices = tf.dynamic_partition(radial_scatter_indices[:,1:], pair_indices[:,0], num_molecules)
	angular_molecule_embeddings = tf.dynamic_partition(angular_embedding, triples_indices[:,0], num_molecules)
	angular_atom_indices = tf.dynamic_partition(angular_scatter_indices[:,1:], triples_indices[:,0], num_molecules)

	embeddings = []
	mol_atom_indices = []
	for molecule in range(num_molecules):
		atom_indices = tf.cast(tf.where(tf.not_equal(Zs[molecule], 0)), tf.int32)
		molecule_atom_elements = tf.gather_nd(Zs[molecule], atom_indices)
		num_atoms = tf.shape(molecule_atom_elements)[0]
		radial_atom_embeddings = tf.reshape(tf.reduce_sum(tf.scatter_nd(radial_atom_indices[molecule], radial_molecule_embeddings[molecule],
								[num_atoms, num_atoms, num_elements, tf.shape(radial_rs)[0]]), axis=1), [num_atoms, -1])
		angular_atom_embeddings = tf.reshape(tf.reduce_sum(tf.scatter_nd(angular_atom_indices[molecule], angular_molecule_embeddings[molecule],
									[num_atoms, num_atoms, num_atoms, num_element_pairs, tf.shape(angular_rs)[0] * tf.shape(theta_s)[0]]),
									axis=[1,2]), [num_atoms, -1])
		embeddings.append(tf.concat([radial_atom_embeddings, angular_atom_embeddings], axis=1))
		mol_atom_indices.append(tf.concat([tf.fill([num_atoms, 1], molecule), atom_indices], axis=1))

	embeddings = tf.concat(embeddings, axis=0)
	mol_atom_indices = tf.concat(mol_atom_indices, axis=0)
	atom_Zs = tf.gather_nd(Zs, tf.where(tf.not_equal(Zs, 0)))
	atom_Z_indices = tf.cast(tf.where(tf.equal(tf.expand_dims(atom_Zs, axis=1), tf.expand_dims(elements, axis=0)))[:,1], tf.int32)

	element_embeddings = tf.dynamic_partition(embeddings, atom_Z_indices, num_elements)
	mol_indices = tf.dynamic_partition(mol_atom_indices, atom_Z_indices, num_elements)
	return element_embeddings, mol_indices

def tf_symmetry_functions_radial_grid(xyzs, Zs, radial_cutoff, radial_rs, eta, prec=tf.float64):
	"""
	Encodes the radial grid portion of the symmetry functions. Should be called by tf_symmetry_functions_2()

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		num_atoms (np.int32): NMol number of atoms numpy array
		radial_cutoff (tf.float): scalar tensor with the cutoff for radial pairs
		radial_rs (tf.float): NRadialGridPoints tensor with R_s values for the radial grid
		eta (tf.float): scalar tensor with the eta parameter for the symmetry functions

	Returns:
		radial_embedding (tf.float): tensor of radial embeddings for all atom pairs within the radial_cutoff
		pair_indices (tf.int32): tensor of the molecule, atom, and pair atom indices
		pair_elements (tf.int32): tensor of the atomic numbers for the atom and its pair atom
	"""
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	distance_tensor = tf.norm(delta_xyzs + 1.e-16,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.where(tf.logical_and(tf.logical_and(tf.less(distance_tensor, radial_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1)))
	identity_mask = tf.where(tf.not_equal(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, identity_mask)), tf.int32)
	pair_distances = tf.gather_nd(distance_tensor, pair_indices)
	pair_elements = tf.stack([tf.gather_nd(Zs, pair_indices[:,0:2]), tf.gather_nd(Zs, pair_indices[:,0:3:2])], axis=-1)
	gaussian_factor = tf.exp(-eta * tf.square(tf.expand_dims(pair_distances, axis=-1) - tf.expand_dims(radial_rs, axis=0)))
	cutoff_factor = tf.expand_dims(0.5 * (tf.cos(3.14159265359 * pair_distances / radial_cutoff) + 1.0), axis=-1)
	radial_embedding = gaussian_factor * cutoff_factor
	return radial_embedding, pair_indices, pair_elements

def tf_symmetry_function_angular_grid(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta):
	"""
	Encodes the radial grid portion of the symmetry functions. Should be called by tf_symmetry_functions_2()

	Args:
		xyzs (tf.float): NMol x MaxNAtoms x 3 coordinates tensor
		Zs (tf.int32): NMol x MaxNAtoms atomic number tensor
		angular_cutoff (tf.float): scalar tensor with the cutoff for the angular triples
		angular_rs (tf.float): NAngularGridPoints tensor with the R_s values for the radial part of the angular grid
		theta_s (tf.float): NAngularGridPoints tensor with the theta_s values for the angular grid
		zeta (tf.float): scalar tensor with the zeta parameter for the symmetry functions
		eta (tf.float): scalar tensor with the eta parameter for the symmetry functions

	Returns:
		angular_embedding (tf.float): tensor of radial embeddings for all atom pairs within the radial_cutoff
		triples_indices (tf.int32): tensor of the molecule, atom, and triples atom indices
		triples_elements (tf.int32): tensor of the atomic numbers for the atom
		sorted_triples_element_pairs (tf.int32): sorted tensor of the atomic numbers of triples atoms
	"""
	num_mols = Zs.get_shape().as_list()[0]
	delta_xyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
	distance_tensor = tf.norm(delta_xyzs + 1.e-16,axis=3)
	padding_mask = tf.not_equal(Zs, 0)
	pair_indices = tf.cast(tf.where(tf.logical_and(tf.logical_and(tf.less(distance_tensor, angular_cutoff),
					tf.expand_dims(padding_mask, axis=1)), tf.expand_dims(padding_mask, axis=-1))), tf.int32)
	identity_mask = tf.where(tf.not_equal(pair_indices[:,1], pair_indices[:,2]))
	pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, identity_mask)), tf.int32)
	mol_pair_indices = tf.dynamic_partition(pair_indices, pair_indices[:,0], num_mols)
	triples_indices = []
	tmp = []
	for i in xrange(num_mols):
		mol_common_pair_indices = tf.where(tf.equal(tf.expand_dims(mol_pair_indices[i][:,1], axis=1),
									tf.expand_dims(mol_pair_indices[i][:,1], axis=0)))
		mol_triples_indices = tf.concat([tf.gather(mol_pair_indices[i], mol_common_pair_indices[:,0]),
								tf.gather(mol_pair_indices[i], mol_common_pair_indices[:,1])[:,-1:]], axis=1)
		permutation_identity_pairs_mask = tf.where(tf.less(mol_triples_indices[:,2], mol_triples_indices[:,3]))
		mol_triples_indices = tf.squeeze(tf.gather(mol_triples_indices, permutation_identity_pairs_mask))
		triples_indices.append(mol_triples_indices)
	triples_indices = tf.concat(triples_indices, axis=0)

	triples_elements = tf.gather_nd(Zs, triples_indices[:,0:2])
	triples_element_pairs, _ = tf.nn.top_k(tf.stack([tf.gather_nd(Zs, triples_indices[:,0:3:2]),
							tf.gather_nd(Zs, triples_indices[:,0:4:3])], axis=-1), k=2)
	sorted_triples_element_pairs = tf.reverse(triples_element_pairs, axis=[-1])

	triples_distances = tf.stack([tf.gather_nd(distance_tensor, triples_indices[:,:3]), tf.gather_nd(distance_tensor,
						tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1))], axis=1)
	r_ijk_s = tf.square(tf.expand_dims(tf.reduce_sum(triples_distances, axis=1) / 2.0, axis=-1) - tf.expand_dims(angular_rs, axis=0))
	exponential_factor = tf.exp(-eta * r_ijk_s)

	xyz_ij_ik = tf.reduce_sum(tf.gather_nd(delta_xyzs, triples_indices[:,:3]) * tf.gather_nd(delta_xyzs,
						tf.concat([triples_indices[:,:2], triples_indices[:,3:]], axis=1)), axis=1)
	cos_theta = xyz_ij_ik / (triples_distances[:,0] * triples_distances[:,1])
	cos_theta = tf.where(tf.greater_equal(cos_theta, 1.0), tf.ones_like(cos_theta) - 1.0e-16, cos_theta)
	cos_theta = tf.where(tf.less_equal(cos_theta, -1.0), -1.0 * tf.ones_like(cos_theta) - 1.0e-16, cos_theta)
	triples_angle = tf.acos(cos_theta)
	theta_ijk_s = tf.expand_dims(triples_angle, axis=-1) - tf.expand_dims(theta_s, axis=0)
	cos_factor = tf.pow((1 + tf.cos(theta_ijk_s)), zeta)

	cutoff_factor = 0.5 * (tf.cos(3.14159265359 * triples_distances / angular_cutoff) + 1.0)
	scalar_factor = tf.pow(tf.cast(2.0, eval(PARAMS["tf_prec"])), 1.0-zeta)

	angular_embedding = tf.reshape(scalar_factor * tf.expand_dims(cos_factor * tf.expand_dims(cutoff_factor[:,0] * cutoff_factor[:,1], axis=-1), axis=-1) \
						* tf.expand_dims(exponential_factor, axis=-2), [tf.shape(triples_indices)[0], tf.shape(theta_s)[0] * tf.shape(angular_rs)[0]])
	return angular_embedding, triples_indices, triples_elements, sorted_triples_element_pairs

def tf_coulomb_dsf_elu(xyzs, charges, Radpair, elu_width, dsf_alpha, cutoff_dist):
	"""
	A tensorflow linear scaling implementation of the Damped Shifted Electrostatic Force with short range cutoff with elu function (const at short range).
	http://aip.scitation.org.proxy.library.nd.edu/doi/pdf/10.1063/1.2206581
	Batched over molecules.

	Args:
		R: a nmol X maxnatom X 3 tensor of coordinates.
		Qs : nmol X maxnatom X 1 tensor of atomic charges.
		R_srcut: Short Range Erf Cutoff
		R_lrcut: Long Range DSF Cutoff
		Radpair: None zero pairs X 3 tensor (mol, i, j)
		alpha: DSF alpha parameter (~0.2)
	Returns
		Energy of  Mols
	"""
	xyzs *= BOHRPERA
	elu_width *= BOHRPERA
	dsf_alpha /= BOHRPERA
	cutoff_dist *= BOHRPERA
	inp_shp = tf.shape(xyzs)
	num_mol = tf.cast(tf.shape(xyzs)[0], dtype=tf.int64)
	num_pairs = tf.cast(tf.shape(Radpair)[0], tf.int64)
	elu_shift, elu_alpha = tf_dsf_potential(elu_width, cutoff_dist, dsf_alpha, return_grad=True)

	Rij = DifferenceVectorsLinear(xyzs + 1e-16, Radpair)
	RijRij2 = tf.norm(Rij, axis=1)
	qij = tf.gather_nd(charges, Radpair[:,:2]) * tf.gather_nd(charges, Radpair[:,::2])

	coulomb_potential = qij * tf.where(tf.greater(RijRij2, elu_width), tf_dsf_potential(RijRij2, cutoff_dist, dsf_alpha),
						elu_alpha * (tf.exp(RijRij2 - elu_width) - 1.0) + elu_shift)

	range_index = tf.range(num_pairs, dtype=tf.int64)
	mol_index = tf.cast(Radpair[:,0], dtype=tf.int64)
	sparse_index = tf.cast(tf.stack([mol_index, range_index], axis=1), tf.int64)
	sp_atomoutputs = tf.SparseTensor(sparse_index, coulomb_potential, [num_mol, num_pairs])
	return tf.sparse_reduce_sum(sp_atomoutputs, axis=1)

def tf_dsf_potential(dists, cutoff_dist, dsf_alpha, return_grad=False):
	dsf_potential = tf.erfc(dsf_alpha * dists) / dists - tf.erfc(dsf_alpha * cutoff_dist) / cutoff_dist \
					+ (dists - cutoff_dist) * (tf.erfc(dsf_alpha * cutoff_dist) / tf.square(cutoff_dist) \
					+ 2.0 * dsf_alpha * tf.exp(-tf.square(dsf_alpha * cutoff_dist)) / (tf.sqrt(np.pi) * cutoff_dist))
	dsf_potential = tf.where(tf.greater(dists, cutoff_dist), tf.zeros_like(dsf_potential), dsf_potential)
	if return_grad:
		dsf_gradient = -(tf.erfc(dsf_alpha * dists) / tf.square(dists) - tf.erfc(dsf_alpha * cutoff_dist) / tf.square(cutoff_dist) \
	 					+ 2.0 * dsf_alpha / tf.sqrt(np.pi) * (tf.exp(-tf.square(dsf_alpha * dists)) / dists \
						- tf.exp(-tf.square(dsf_alpha * cutoff_dist)) / tf.sqrt(np.pi)))
		return dsf_potential, dsf_gradient
	else:
		return dsf_potential
