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

def nystrom_approximation(G, rank, columns):
    """Drineas, Mahoney 2005, G_dash = C W_p C.T"""
    n = G.shape[0]
    G_ii2 = np.diag(G) ** 2
    P = G_ii2 / np.sum(G_ii2)

    k = rank
    c = columns
    if columns is None:
        c = min(rank * 1.2, n)

    S = np.zeros((n, c), dtype=np.float64)
    D = np.zeros((c, c), dtype=np.float64)

    #create random draws
    P_cum = np.cumsum(P)
    I_uniform = np.random.random(c)
    I_t = np.digitize(I_uniform, P_cum)
    for t, i_t in enumerate(I_t):
        p_it = P[i_t]
        D[t, t] = 1.0 / np.sqrt(c * p_it)
        S[i_t, t] = 1.0

    C = np.dot(G, np.dot(S, D))
    W = np.dot(np.dot(D, S.T), C)
    W_k = low_rank_approximation(W, k)
    W_kp = np.linalg.pinv(W, rcond=1e-15)
    return C, W_kp


def low_rank_approximation(M, rank):
    U, s, V = np.linalg.svd(M, full_matrices=True)
    s[rank:] = 0.0
    s = np.diag(s)
    M_dash = np.dot(np.dot(U, s), V)
    return M_dash


def woodbury_inverse(a, U, V):
    """assuming the matrix a is the identity times a scalar a"""
    a_inv = float(1.0 / a)
    term_1 = np.eye(U.shape[1]) + np.dot(V, U) * a_inv
    term_1_inv = svd_inverse(term_1)
    term_2 = reduce(np.dot, [U, term_1_inv, V]) * a_inv * a_inv
    result = np.eye(term_2.shape[0]) * a_inv - term_2
    return result


def svd_inverse(M):
    M_p = np.linalg.pinv(M, rcond=1e-15)
    return M_p

def multiple_matrix_multiply(m1,m2):
    """multiplies 2 sets of matrices, m1[0]*m2[0], m1[1]*m2[1] etc."""
    a = m1[:, np.newaxis,:,:]
    b = np.swapaxes(m2,1,2)[:,:,np.newaxis,:] 
    result = np.swapaxes(np.sum(a*b,axis=3),1,2)
    return result


def normalise(x):
    assert(np.sum(x) != 0)
    total = np.sum(x)
    result = x/total 
    return result


def multidot(x,y):
    #assert(x.ndim >= 2)
    #assert(y.ndim >= 2)
    """Computes the dot products of an array of vectors.
    A 1D array is treated as a single vector"""
    product = x*y
    result = np.sum(product,axis=-1)
    return result

def hermitian_eigs(G):
    eigvals, eigvecs = np.linalg.eigh(G, overwrite_a=False)
    #order from big to small, PCA style
    #lp_eigvals = cvxopt.matrix(np.zeros(G.shape[0]))
    #lp_eigvecs = cvxopt.matrix(np.copy(G))
    #cvxopt.lapack.syev(lp_eigvecs, lp_eigvals, jobz = 'V', uplo = 'L')
    #eigvecs = np.array(lp_eigvecs)
    #eigvals = np.array(lp_eigvals)[:,0]
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:,::-1].transpose() #eigh uses fortran nonsense
    return eigvals, eigvecs

class NDGrid(object):
    """really want a list of bounds to a list of points"""
    def __init__(self, bounds, shape):
        assert type(shape) is tuple
        assert bounds.ndim == len(shape)
        self.bounds = bounds
        self.shape = shape
        np.mgrid[-1:1:5j]
        dim = len(shape)
        slices = []
        for i in xrange(dim):
            minp = bounds[i, 0]
            maxp = bounds[i, 1]
            size = shape[i]
            slice = np.s_[minp:maxp:size*1j]
            slices.append(slice)
        slices = tuple(slices)
        grid = np.mgrid[slices]
        self.grid = np.rollaxis(grid, axis=0, start=grid.ndim)
        self.list = self.grid.reshape(-1, self.grid.shape[-1])
