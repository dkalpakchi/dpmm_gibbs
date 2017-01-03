# -*- coding: utf-8 -*-
# @Author: dmytro
# @Date:   2017-01-02 21:00:09
# @Last Modified by:   dmytro
# @Last Modified time: 2017-01-03 16:33:05
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class DPM(object):
	def __init__(self, x, z, K, mu, sigma2, tau2, alpha=1.0, numiter=100):
		"""
		Args:
		    x (TYPE): Description
		    z (TYPE): The initial clustering
		    K (TYPE): Description
		    mu (TYPE): Description
		    sigma2 (TYPE): Description
		    tau2 (TYPE): Description
		    alpha (float, optional): Description
		    numiter (int, optional): Description
		"""
		self.__K = K
		self.__alpha = alpha
		self.__data = x
		self.__clusters = z
		self.__iter = numiter
		self.__mu = mu
		self.__sigma_sq = sigma2
		self.__tau_sq = tau2
		self.__D = mu.shape[0]
		self.__N = x.shape[0]
		self.__colors = np.array(['b', 'r', 'g', 'm', 'y', 'k', 'c'])

	def gibbs_sample(self):
		plt.scatter(self.__data[:,0], self.__data[:,1], c=self.__colors[self.__clusters])
		plt.show()
		for it in xrange(self.__iter):
			points = range(self.__N)
			np.random.shuffle(points)
			for i in points:
				cluster = self.__clusters[i]
				self.__clusters[i] = -1
				p_xi = np.zeros(self.__K + 1)
				p_z = np.zeros(self.__K + 1)
				for k in xrange(self.__K):
					ind = np.argwhere(self.__clusters == k)
					N_ki = ind.shape[0]
					sigma_sq_n = self.__sigma_sq * self.__tau_sq / (N_ki * self.__tau_sq + self.__sigma_sq)
					mu_n = ((self.__data[ind].sum(axis=0) / self.__sigma_sq + self.__mu / self.__tau_sq) * sigma_sq_n).flatten()
					p_xi[k] = multivariate_normal.pdf(self.__data[i], mean=mu_n, cov=(self.__sigma_sq + sigma_sq_n) * np.eye(self.__D))
					p_z[k] = N_ki / (self.__alpha + self.__N - 1)
				p_xi[self.__K] = multivariate_normal.pdf(self.__data[i], mean=self.__mu, cov=(self.__sigma_sq + self.__tau_sq) * np.eye(self.__D))
				p_z[self.__K] = self.__alpha / (self.__alpha + self.__N - 1)
				pzx = p_z * p_xi
				pzx /= pzx.sum()
				sample = np.random.multinomial(1, pzx)
				z_i = np.argwhere(sample == 1).flatten()[0]
				self.__clusters[i] = z_i
				
				# add new cluster if needed
				if z_i == self.__K:
					self.__K += 1

				# remove empty clusters
				if np.argwhere(self.__clusters == cluster).size == 0:
					self.__K -= 1
					ind = np.argwhere(self.__clusters > cluster)
					self.__clusters[ind] -= 1
			print "Iteration {} has finished with {} clusters".format(it, self.__K)
		print [np.argwhere(self.__clusters == k).size for k in xrange(self.__K)]
		plt.scatter(self.__data[:,0], self.__data[:,1], c=self.__colors[self.__clusters])
		plt.show()
