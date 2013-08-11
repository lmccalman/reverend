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
xssize = 100
yssize = 100

#construct settings and data files for kbrcpp
filename_config = 'motorcycle_regressor.ini'
prefix = 'mc'  # will automatically construct all filenames
settings = kbrcpp.Settings(prefix)
# 'logp_norm'
# 'logp_preimage'
# 'logp_joint'
# 'pinball_norm'
# 'pinball_direct'
# 'pinball_joint'
# 'hilbert'
settings.cost_function = 'logp_joint'

settings.sigma_x_min = 0.1
settings.sigma_x = 0.2
settings.sigma_x_max = 0.5

settings.sigma_y_min = 0.1
settings.sigma_y = 0.17
settings.sigma_y_max = 0.5

settings.epsilon_min_min = 1e-10
settings.epsilon_min = 1e-5
settings.epsilon_min_max = 1e-4

settings.delta_min_min = 1e-3
settings.delta_min = 1e-1
settings.delta_min_max = 1e1

#for preimage
settings.preimage_reg = 1e2
settings.preimage_reg_min = 1e-1
settings.preimage_reg_max = 1e2
#Some other settings
settings.inference_type = 'regress'
settings.cumulative_estimate = True
settings.cumulative_mean_map = True
settings.quantile_estimate = True
settings.quantile = 0.5
settings.walltime = 120.0
settings.preimage_walltime = 12.0
settings.folds = 20
settings.observation_period = 1


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
    xsmin = np.amin(X) - 1.0
    xsmax = np.amax(X) + 1.0
    ysmin = np.amin(Y)
    ysmax = np.amax(Y)
    Y_s = np.linspace(ysmin, ysmax, yssize)[:, np.newaxis]
    X_s = np.linspace(xsmin, xsmax, xssize)[:, np.newaxis]

    #parameters
    kbrcpp.write_config_file(settings, filename_config)
    kbrcpp.write_data_files(settings, U=U, X=X, Y=Y, X_s=X_s, Y_s=Y_s,)

    #now we're ready to invoke the regressor
    kbrcpp.run(filename_config, kbrcpp_directory)

    #read in the weights we've just calculated
    #W = np.load(settings.filename_weights)
    #PW = np.load(settings.filename_preimage)
    E = np.load(settings.filename_embedding)
    pdf = np.load(settings.filename_posterior)
    cdf = None
    if settings.cumulative_estimate:
        cdf = np.load(settings.filename_cumulative)
    quantile = None
    if settings.quantile_estimate:
        quantile = np.load(settings.filename_quantile)

    #And plot...
    fig = pl.figure()
    axes = fig.add_subplot(121)
    axes.set_title('Posterior Embedding')
    axes.imshow(E.T, origin='lower', 
                extent=(ysmin, ysmax, xsmin, xsmax),cmap=cm.jet, aspect='auto')
    axes.scatter(Y, X, c='y')
    axes.set_xlim(ysmin, ysmax)
    axes.set_ylim(xsmin, xsmax)
    axes = fig.add_subplot(122)
    axes.set_title('PDF estimate')
    axes.imshow(np.exp(pdf).T, origin='lower', 
            extent=(ysmin, ysmax, xsmin, xsmax), cmap=cm.jet, aspect='auto')
    axes.scatter(Y, X, c='y')
    axes.set_xlim(ysmin, ysmax)
    axes.set_ylim(xsmin, xsmax)
    
    if settings.cumulative_estimate:
        fig = pl.figure()
        axes = fig.add_subplot(121)
        axes.set_title('CDF Estimate')
        axes.imshow(cdf.T, origin='lower', 
                extent=(ysmin, ysmax, xsmin, xsmax),cmap=cm.jet, aspect='auto')
        axes.scatter(Y, X, c='y')
        axes.set_xlim(ysmin, ysmax)
        axes.set_ylim(xsmin, xsmax)
        axes = fig.add_subplot(122)
        axes.set_title('Quantile Estimate')
        axes.imshow(np.exp(pdf).T, origin='lower', 
                extent=(ysmin, ysmax, xsmin, xsmax),cmap=cm.jet, aspect='auto')
        axes.scatter(Y, X, c='y')
        axes.set_xlim(ysmin, ysmax)
        axes.set_ylim(xsmin, xsmax)
        axes.plot(Y_s[:,0], quantile[:,0], 'b-',linewidth=2.0)
    pl.show()

if __name__ == "__main__":
    main()
