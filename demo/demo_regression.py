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
#some training parameters for kernel width
settings.cost_function = 'jointpinball'  # {'logp', 'hilbert', 'jointlogp', 'pinball', 'jointpinball'}
settings.sigma_x_min = 0.002
settings.sigma_x = 0.236
settings.sigma_x_max = 0.3
settings.sigma_y_min = 0.03
settings.sigma_y = 0.08
settings.sigma_y_max = 0.9
#for preimage
settings.preimage_reg = 1e-6
settings.preimage_reg_min = 1e-10
settings.preimage_reg_max = 1e1
settings.normed_weights = False
#Some other settings
settings.inference_type = 'regress'
settings.cumulative_estimate = True
settings.quantile_estimate = True
settings.quantile = 0.9
settings.walltime = 120.0
settings.preimage_walltime = 5.0
settings.folds = 5
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
                extent=(ysmin, ysmax, xsmin, xsmax),cmap=cm.hot, aspect='auto')
    axes.scatter(Y, X, c='y')
    axes.set_xlim(ysmin, ysmax)
    axes.set_ylim(xsmin, xsmax)
    axes = fig.add_subplot(122)
    axes.set_title('PDF estimate')
    axes.imshow(pdf.T, origin='lower', 
            extent=(ysmin, ysmax, xsmin, xsmax), cmap=cm.hot, aspect='auto')
    axes.scatter(Y, X, c='y')
    axes.set_xlim(ysmin, ysmax)
    axes.set_ylim(xsmin, xsmax)
    
    if settings.cumulative_estimate:
        fig = pl.figure()
        axes = fig.add_subplot(121)
        axes.set_title('CDF Estimate')
        axes.imshow(cdf.T, origin='lower', 
                extent=(ysmin, ysmax, xsmin, xsmax),cmap=cm.hot, aspect='auto')
        axes.scatter(Y, X, c='y')
        axes.set_xlim(ysmin, ysmax)
        axes.set_ylim(xsmin, xsmax)
        axes = fig.add_subplot(122)
        axes.set_title('Quantile Estimate')
        axes.imshow(pdf.T, origin='lower', 
                extent=(ysmin, ysmax, xsmin, xsmax),cmap=cm.hot, aspect='auto')
        axes.scatter(Y, X, c='y')
        axes.set_xlim(ysmin, ysmax)
        axes.set_ylim(xsmin, xsmax)
        axes.plot(Y_s[:,0], quantile[:,0], 'b-',linewidth=2.0)
    pl.show()

if __name__ == "__main__":
    main()
