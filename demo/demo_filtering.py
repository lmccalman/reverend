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
# Filtering Demo -- Lorenz Attractor
###################################################################

#makes life a bit easier
import sys;
sys.path.append("../") #might need to change to backslash on windows

#3rd party imports
import numpy as np
import matplotlib.pylab as pl

#local imports
from reverend import distrib
from reverend import preimage
from reverend import kbrcpp

#evaluation image size
xssize = 800
yssize = 800

# how much data to use
step_size = 10
training_size = 500

#some training parameters for kernel width
sigma_x_min = 0.05
sigma_x = 0.155
sigma_x_max = 0.2
sigma_y_min = 0.05
sigma_y = 0.0804
sigma_y_max = 0.1

#for preimage
preimage_reg = 1e-6
preimage_reg_min = 1e-10
preimage_reg_max = 1e1
normed_weights = True

#Some other settings
walltime = 1200.0
preimage_walltime = 120.0
folds = 2

def main():
    data = np.load('lorenz.npy')
    #2D only at the moment
    all_X = data[::step_size,1:]
    all_Y = data[::step_size,1:] 
    #add noise
    all_Y = all_Y + np.random.normal(loc=0.0,scale=0.1,size=all_Y.shape)
    
    #create training and testing data
    X = all_X[0:training_size]
    X_s = all_X[training_size:]
    Y = all_Y[0:training_size]
    Y_s = all_Y[training_size:]
    
    #whiten and rescale inputs
    X_mean, X_sd = distrib.scale_factors(X)
    Y_mean, Y_sd = distrib.scale_factors(Y)
    X = distrib.scale(X, X_mean, X_sd)
    X_s = distrib.scale(X_s, X_mean, X_sd)
    Y = distrib.scale(Y, Y_mean, Y_sd)
    Y_s = distrib.scale(Y_s, X_mean, X_sd)

    # import matplotlib.pyplot as pl
    # pl.figure()
    # pl.plot(Y[:,0], Y[:,1])
    # pl.show()
    # sys.exit()

    # simple prior
    U = X

    # We just want to plot the result, not evaluate it
    xsmin = np.amin(X, axis=0)
    xsmax = np.amax(X, axis=0)
    ysmin = np.amin(Y, axis=0)
    ysmax = np.amax(Y, axis=0)
    # Y_s = np.linspace(ysmin, ysmax, yssize)[:, np.newaxis]
    # X_s = np.linspace(xsmin, xsmax, xssize)[:, np.newaxis]

    #construct settings and data files for kbrcpp
    filename_config = 'lorenz_filter.ini'
    prefix = 'lz'  # will automatically construct all filenames
    settings = kbrcpp.Settings(prefix)
    #training parameters
    settings.sigma_x = sigma_x
    settings.sigma_y = sigma_y
    settings.sigma_x_min = sigma_x_min
    settings.sigma_y_min = sigma_y_min
    settings.sigma_x_max = sigma_x_max
    settings.sigma_y_max = sigma_y_max
    settings.preimage_reg = preimage_reg
    settings.preimage_reg_min = preimage_reg_min
    settings.preimage_reg_max = preimage_reg_max
    settings.normed_weights = normed_weights
    settings.walltime = walltime
    settings.preimage_walltime = preimage_walltime
    settings.folds = folds
    kbrcpp.write_config_file(settings, filename_config)
    kbrcpp.write_data_files(settings, U=U, X=X, Y=Y, X_s=X_s, Y_s=Y_s,)

    #now we're ready to invoke the filter
    kbrcpp.run(filename_config, '../cpp/kbrfilter')

    #read in the weights we've just calculated
    W = np.load(settings.filename_weights)
    pdf = preimage.posterior_embedding_image(W, X, X_s, sigma_x)
    P = None
    if normed_weights is False:
        P = np.load(settings.filename_preimage)
        pdf2 = preimage.posterior_embedding_image(P, X, X_s, sigma_x)

    sys.exit()
    #And plot...
    fig = pl.figure()
    if normed_weights is False:
        axes = fig.add_subplot(121)
    else:
        axes = fig.add_subplot(111)
    axes.set_title('raw estimate')
    axes.imshow(pdf.T, origin='lower', extent=(ysmin, ysmax, xsmin, xsmax))
    axes.scatter(Y, X, c='y')
    if normed_weights is False:
        axes = fig.add_subplot(122)
        axes.set_title('Preimage estimate')
        axes.imshow(pdf2.T, origin='lower', extent=(ysmin, ysmax, xsmin, xsmax))
        axes.scatter(Y, X, c='y')
    pl.show()

if __name__ == "__main__":
    main()