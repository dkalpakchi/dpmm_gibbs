# -*- coding: utf-8 -*-
# @Author: dmytro
# @Date:   2017-01-02 10:33:56
# @Last Modified by:   Dmytro Kalpakchi
# @Last Modified time: 2017-01-03 16:57:08

import numpy as np
import matplotlib.pyplot as plt
from dpm import DPM


# data generation params
D = 2
K_genuine = 7
colors = np.array(['b', 'r', 'g', 'm', 'y', 'k', 'c'])
N = 100
mu = np.array([0, 0.5])
tau_sq = 9
sigma_sq = 0.5

# data construction
true_z = np.arange(N) % K_genuine # clusters
c = colors[true_z]
theta_k = np.array([np.random.multivariate_normal(mu, tau_sq * np.eye(D)) for _ in xrange(K_genuine)])
x = np.array([np.random.multivariate_normal(th, sigma_sq * np.eye(D)) for th in theta_k[true_z]])

plt.scatter(x[:,0], x[:,1], c=c)
plt.show()

# hyperparameters
alpha = 1.0
K = 1
z = np.arange(N) % K
dpm = DPM(x, z, K, mu, sigma_sq, tau_sq, alpha, numiter=200)
dpm.gibbs_sample()
