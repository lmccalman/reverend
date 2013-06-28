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

###################################################################
# Filtering Visualiser -- Lorenz Attractor
###################################################################

#makes life a bit easier
import sys;
sys.path.append("../") #might need to change to backslash on windows

#3rd party imports
import numpy as np
import matplotlib.pylab as pl
import matplotlib.cm as cm
import mayavi.mlab as ml

#local imports
from reverend import distrib

X = np.load('lzX.npy')
Y = np.load('lzY.npy')
X = X[:-1]
Y = Y[:-1]
X_s = np.load('lzX_s.npy')
Y_s = np.load('lzY_s.npy')
PDF = np.load('lzPDF.npy')

imsize_w = PDF.shape[1]
imsize_h = PDF.shape[2]
X_s = X_s.reshape((imsize_w, imsize_h, 2))
wmin = np.amin(X_s[:,:,0])
wmax = np.amax(X_s[:,:,0])
hmin = np.amin(X_s[:,:,1])
hmax = np.amax(X_s[:,:,1])

# ml.contour3d(PDF)
# ml.show()

fig = pl.figure()
for i, frame in enumerate(PDF):
    print i
    fig.clf()
    ax = fig.add_subplot(111)
    ax.imshow(frame.T, extent=(wmin,wmax,hmin,hmax),cmap=cm.hot)
    ax.plot(X[:,0], X[:,1])
    pl.savefig("lz%04d.png" % i)

