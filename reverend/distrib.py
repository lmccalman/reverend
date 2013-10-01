# Reverend -- Practical Bayesian Inference with Kernel Embeddings
# Copyright (C) 2013 Lachlan McCalman
# lachlan@mccalman.info

# Reverend is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Reverend is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Reverend.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy.linalg
import scipy.cluster.vq

def scale(X, mean=None, sd=None):
    """scale input data to mean 0 sd 1"""
    if (mean is None) or (sd is None):
        mean, sd = scale_factors(X)
    X_dash = (X - mean) / sd
    return X_dash


def unscale(X_dash, mean, sd):
    """undo a scaling to mean 0 sd 1"""
    X = X_dash * sd + mean
    return X


def scale_factors(X):
    """the mean and standard deviation along every dimension
    of an array of vectors"""
    assert X.ndim == 2
    mean = np.average(X, axis=0)
    sd = np.array([np.std(X[:, d], axis=0) for d in range(X.shape[1])])
    return mean, sd

def whitening_matrix(X):
    """The matrix of Eigenvectors that whitens the input vector X"""
    assert (X.ndim == 2)
    sigma = np.dot(X.T, X)
    e, m = scipy.linalg.eigh(sigma)
    return m.T

def kmeans(X, Y, frac):
    """Compute K-mean clusters as good initialisers for a reduced set"""
    J = np.concatenate((X,Y), axis=1)
    dx = X.shape[1]
    k = int(X.shape[0] * frac)
    result, cost = scipy.cluster.vq.kmeans(J, k)
    plusX = result[:, 0:dx]
    plusY = result[:, dx:]
    return plusX, plusY


