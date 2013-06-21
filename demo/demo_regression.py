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
# Regression Demo -- Motorcycle Dataset
###################################################################

#makes life a bit easier
import sys;
sys.path.append("../reverend") #might need to change to backslash on windows

#3rd party imports
import numpy as np
import matplotlib.pylab as pl

#local imports
from reverend import distrib
from reverend import preimage
from reverend import kbrcpp

#evaluation image size
xssize = 50
yssize = 50

#some training parameters for kernel width
sigma_x_min = 0.02
sigma_x = 0.2
sigma_x_max = 0.5
sigma_y_min = 0.05
sigma_y = 0.3
sigma_y_max = 0.5

#Some other settings
wall_time = 20.0
folds = 5

def main():
    X = np.load('motorcycle_X.npy')
    Y = np.load('motorcycle_Y.npy')
            
    # Make sure we shuffle for the benefit of cross-validation
    random_indices = np.random.permutation(X.shape[0])
    X = X[random_indices]
    Y = Y[random_indices]

    #whiten and rescale inputs
    X_mean, X_sd = distrib.scale_factors(X)
    Y_mean, Y_sd = distrib.scale_factors(Y)
    X = distrib.scale(X, X_mean, X_sd)
    Y = distrib.scale(Y, Y_mean, Y_sd)

    # simple prior
    U = X

    # We just want to plot the result, not evaluate it
    xsmin = np.amin(X)
    xsmax = np.amax(X)
    ysmin = np.amin(Y)
    ysmax = np.amax(Y)
    Y_s = np.linspace(ysmin, ysmax, yssize)[:, np.newaxis]
    X_s = np.linspace(xsmin, xsmax, xssize)[:, np.newaxis]

    #construct settings and data files for kbrcpp
    filename_config = 'motorcycle_vis.ini'
    prefix = 'mc'  # will automatically construct all filenames
    settings = kbrcpp.Settings(prefix)
    #training parameters
    settings.sigma_x = sigma_x
    settings.sigma_y = sigma_y
    settings.sigma_x_min = sigma_x_min
    settings.sigma_y_min = sigma_y_min
    settings.sigma_x_max = sigma_x_max
    settings.sigma_y_max = sigma_y_max
    settings.wall_time = wall_time
    settings.folds = folds
    kbrcpp.write_config_file(settings, filename_config)
    kbrcpp.write_data_files(settings, U=U, X=X, Y=Y, X_s=X_s, Y_s=Y_s,)

    #now we're ready to invoke kbrpp
    kbrcpp.run(filename_config, '../cpp/kbrcpp')

    #read in the weights we've just calculated
    W = np.load(settings.filename_weights)

    #And plot...
    pdf = preimage.posterior_embedding_image(W, X, X_s, sigma_x)
    fig = pl.figure()
    axes = fig.add_subplot(111)
    axes.set_title('PDF estimate')
    axes.imshow(pdf.T, origin='lower', extent=(ysmin, ysmax, xsmin, xsmax))
    axes.scatter(Y, X, c='y')
    pl.show()

if __name__ == "__main__":
    main()
