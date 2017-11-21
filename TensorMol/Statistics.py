"""
Routines for simple statistical analysis.
"""

from __future__ import absolute_import
from __future__ import print_function

class OnlineEstimator:
	"""
	Simple storage-less Knuth estimator which
	accumulates mean and variance.
	"""
	def __init__(self,x_):
		self.n = 1
		self.mean = x_*0.
		self.m2 = x_*0.
		delta = x_ - self.mean
		self.mean += delta / self.n
		delta2 = x_ - self.mean
		self.m2 += delta * delta2
	def __call__(self, x_):
		self.n += 1
		delta = x_ - self.mean
		self.mean += delta / self.n
		delta2 = x_ - self.mean
		self.m2 += delta * delta2
		return self.mean, self.m2/(self.n-1)
