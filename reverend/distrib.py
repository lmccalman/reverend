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
import matvec

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

def evaluate_Log_GM(points, means, sigma, coefficients):
    assert(points.ndim == 2)
    assert(means.ndim == 2)
    assert(coefficients.ndim == 1)
    assert(coefficients.shape[0] == means.shape[0])
    coefficients = np.maximum(coefficients, 0.0)
    log_scale_factor = -1*np.log(float( sigma * np.sqrt( 2. * np.pi)))
    p = points.reshape((points.shape[0],1, -1))
    q = means.reshape((1,means.shape[0], -1))
    deltas_squared = np.sum((p-q) ** 2,axis=-1)
    exp_coeffs = -1 * deltas_squared / float(2. * (sigma * sigma))
    #now find the min power of this
    max_power = np.amax(exp_coeffs,axis=1)
    #and subtract it off
    adj_exp_coeffs = exp_coeffs - max_power[:,np.newaxis]
    adj_probs = coefficients * np.exp(adj_exp_coeffs)
    log_sum_adj_probs = np.log(np.sum(adj_probs, axis=1) + 1e-100)
    log_probs = (log_scale_factor + max_power + log_sum_adj_probs) 
    return log_probs

def LogGaussianPDF(mean, covariance):
    inv_cov = matvec.svd_inverse(covariance)
    det_cov = np.linalg.det(covariance)
    assert(det_cov > 0.0)
    n = mean.shape[0]
    scale_factor = 1.0 / np.sqrt( (2 * np.pi) ** n * det_cov)
    
    def f(x):
        assert (x.ndim == 2)
        delta = (x - mean)
        log_r = np.log(scale_factor) + -0.5 * np.dot(delta.T, np.dot(inv_cov, delta))
        return log_r
    return f

def main():
    X = np.linspace(-1, 1, 100)
    Y = np.linspace(-1, 1, 100)
    x_grid, y_grid = np.mgrid[-1:1:100*1j, -1:1:100*1j]
    prob = np.zeros((X.shape[0], Y.shape[0]))
    mu = np.array([0.5, 0.5])[:, np.newaxis]
    sigma = np.array([[0.2, 0.], [0., 0.1]])
    lpdf = LogGaussianPDF(mu, sigma)
    for i in xrange(X.shape[0]):
        for j in xrange(Y.shape[0]):
            x = np.array([X[i], Y[j]])[:, np.newaxis]
            prob[i, j] = np.exp(lpdf(x))
    import matplotlib.pylab as pl
    pl.figure()
    pl.imshow(prob.T, origin='lower')
    pl.show()

if __name__ == "__main__":
    main()

