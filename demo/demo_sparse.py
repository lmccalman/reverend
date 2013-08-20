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
import sys
sys.path.append("../") #might need to change to backslash on windows
kbrcpp_directory = "../cpp"

#3rd party imports
import numpy as np
import matplotlib.pylab as pl
import matplotlib.cm as cm

#local imports
from reverend import distrib
from reverend import kbrcpp

#construct settings and data files for kbrcpp
filename_config = 'sparse_motorcycle.ini'
prefix = 'smc'  # will automatically construct all filenames
settings = kbrcpp.SparseSettings(prefix)
#some training parameters for kernel width
settings.method = 'both'  # {sparse, lowrank, both}
settings.sigma_x_min = 0.1
settings.sigma_x = 0.3
settings.sigma_x_max = 0.3
settings.sigma_y_min = 0.005
settings.sigma_y = 0.4
settings.sigma_y_max = 0.4

settings.epsilon_min_min = 1e-10
settings.epsilon_min = 1e-5
settings.epsilon_min_max = 1e-4

settings.delta_min_min = 1e-3
settings.delta_min = 1e-1
settings.delta_min_max = 1e1

settings.low_rank_scale_min = 1.0
settings.low_rank_scale = 1.0
settings.low_rank_scale_max = 5.0
settings.low_rank_weight_min = 0.01   # 100% sparse solution
settings.low_rank_weight = 0.5   #this MUST be between 0 and 1
settings.low_rank_weight_max = 0.99  # 100% low rank solution
#Training settings
settings.walltime = 10.0
settings.folds = 5


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
    
    # We just want to plot the result, not evaluate it
    yssize = 100
    xssize = 100
    xsmin = np.amin(X) - 1.0
    xsmax = np.amax(X) + 1.0
    ysmin = np.amin(Y)
    ysmax = np.amax(Y)
    Y_s = np.linspace(ysmin, ysmax, yssize)[:, np.newaxis]
    X_s = np.linspace(xsmin, xsmax, xssize)[:, np.newaxis]
    
    # simple prior
    U = X
    
    #parameters
    kbrcpp.write_config_file(settings, filename_config)
    kbrcpp.write_data_files(settings, U=U, X=X, Y=Y, X_s=X_s, Y_s=Y_s,)

    #now we're ready to invoke the regressor
    kbrcpp.run_sparse(filename_config, kbrcpp_directory)

    #read in the weights we've just calculated
    pdf = np.load(settings.filename_embedding)
    fig = pl.figure()
    axes = fig.add_subplot(111)
    axes.imshow(pdf.T, origin='lower', 
            extent=(ysmin, ysmax, xsmin, xsmax),cmap=cm.hot, aspect='auto')
    axes.scatter(Y, X, c='y')
    axes.set_xlim(ysmin, ysmax)
    axes.set_ylim(xsmin, xsmax)
    pl.show()

if __name__ == "__main__":
    main()
