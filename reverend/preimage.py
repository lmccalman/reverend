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
import distrib
from functools import partial

def MAP_estimate(y_initial, Y, weights, sigma):
    assert(weights.ndim == 1)
    tolerance = 0.000001
    max_iterations = 100
    s2 = 2 * sigma * sigma
    d = Y.shape[1]
    a = weights.ravel()
    #initialisation
    # if y_initial == None:
        # z = np.random.multivariate_normal(np.zeros(d), np.eye(d))
    # else:
    z = y_initial
    for i in xrange(max_iterations):
        prez = z
        t1 = np.sum(Y * Y, axis=1)[..., np.newaxis]
        t2 = 2 * np.dot(Y, z)[..., np.newaxis]
        t3 = np.dot(z, z)
        gxz = np.exp((-t1 + t2 - t3) / s2)
        r = np.dot(a, gxz)
        if r.any() < tolerance:
            z = np.random.multivariate_normal(np.zeros(d), np.eye(d))
            continue
        p = np.dot(a, Y * gxz)
        z = p / r
        if (np.dot(z - prez, z - prez) < tolerance):
            break
    return z

def posterior_embedding_image(weights, Y, Y_s, sigma_y):
    image = np.zeros((weights.shape[0], Y_s.shape[0]))
    for i, w in enumerate(weights):
        image[i] = distrib.evaluate_Log_GM(Y_s, Y, sigma_y, w.flatten())
    return np.exp(image)

def multistart_MAP_estimate(Y, weights, sigma, y_initial_array):
    estimator = partial(MAP_estimate, Y=Y, weights=weights, sigma=sigma)
    results = np.array(map(estimator, y_initial_array))
    probs = distrib.evaluate_Log_GM(results, Y, sigma, weights)
    best_idx = np.argmax(probs)
    return results[best_idx]
    #now we have to evaluate all these points to check which is the best
