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
# Filtering Demo -- Lorenz Dataset
###################################################################

#makes life a bit easier
import sys;
sys.path.append("../") #might need to change to backslash on windows
kbrcpp_directory = "../cpp"

#3rd party imports
import numpy as np
import matplotlib.pylab as pl
import matplotlib.cm as cm

#local imports
from reverend import distrib
from reverend import kbrcpp

#evaluation image size
xssize = (200, 200)

# how much data to use
step_size = 5
training_size = 500
testing_size = 5

#construct settings and data files for kbrcpp
filename_config = 'lorenz_filter.ini'
prefix = 'lz'  # will automatically construct all filenames
settings = kbrcpp.Settings(prefix)
    
#some training parameters for kernel width
settings.cost_function = 'logp'  # {'logp', 'hilbert', 'joint'}
settings.sigma_x_min = 0.05
settings.sigma_x = 0.494
settings.sigma_x_max = 0.8
settings.sigma_y_min = 0.05
settings.sigma_y = 0.197
settings.sigma_y_max = 0.8

#for preimage
settings.preimage_reg = 1e-6
settings.preimage_reg_min = 1e-10
settings.preimage_reg_max = 1e1
settings.normed_weights = True

#Some other settings
settings.inference_type = 'filter'  # {'filter', 'regress'}
settings.walltime = 12.0
settings.preimage_walltime = 12.0
settings.folds = 2
settings.observation_period = 1
settings.cumulative_estimate = False
settings.quantile_estimate = False
settings.quantile = 0.5

def main():
    data = np.load('lorenz.npy')
    #2D only at the moment
    all_X = data[::step_size,1:]
    all_Y = data[::step_size,1:] 
    #add noise
    all_Y = all_Y + np.random.normal(loc=0.0,scale=0.1,size=all_Y.shape)
    #create training and testing data
    X = all_X[0:training_size]
    Y = all_Y[0:training_size]
    Y_s = all_Y[training_size:training_size+testing_size]

    #whiten and rescale inputs
    X_mean, X_sd = distrib.scale_factors(X)
    Y_mean, Y_sd = distrib.scale_factors(Y)
    X = distrib.scale(X, X_mean, X_sd)
    Y = distrib.scale(Y, Y_mean, Y_sd)
    Y_s = distrib.scale(Y_s, X_mean, X_sd)

    # simple prior
    U = X

    # We just want to plot the result, not evaluate it
    xsmin = np.amin(X, axis=0) - 3*settings.sigma_x
    xsmax = np.amax(X, axis=0) + 3*settings.sigma_x
    fakeX_s = np.mgrid[xsmin[0]:xsmax[0]:xssize[0]*1j,
            xsmin[1]:xsmax[1]:xssize[1]*1j]
    fakeX_s = np.rollaxis(fakeX_s, 0, 3).reshape((-1,2))
    #Needed so we don't screw up the file writing
    X_s = np.zeros((xssize[0]*xssize[1], 2))
    X_s[:] = fakeX_s

    #parameters
    kbrcpp.write_config_file(settings, filename_config)
    kbrcpp.write_data_files(settings, U=U, X=X, Y=Y, X_s=X_s, Y_s=Y_s,)

    #now we're ready to invoke the regressor
    kbrcpp.run(filename_config, kbrcpp_directory)

    #read in the weights we've just calculated
    W = np.load(settings.filename_weights)
    
    WP = None
    if settings.normed_weights is False:
        WP = np.load(settings.filename_preimage)
    
    embedding = np.load(settings.filename_embedding)
    embedding = embedding.reshape((testing_size,xssize[0],xssize[1]))
    pdf = np.load(settings.filename_posterior)
    pdf = pdf.reshape((testing_size,xssize[0],xssize[1]))

    # our filter lops off the last training point to get deltas
    X = X[:-1]

    
    fig = pl.figure()
    for i, frame in enumerate(pdf):
        print i
        fig.clf()
        ax = fig.add_subplot(111)
        ax.imshow(frame.T, extent=(xsmin[0],xsmax[0],xsmin[1],xsmax[1]),cmap=cm.hot)
        ax.plot(X[:,0], X[:,1])
        pl.savefig("lz%04d.png" % i)

if __name__ == "__main__":
    main()
