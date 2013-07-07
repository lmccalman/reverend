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
training_size = 5000
testing_size = 5000
    
#construct settings and data files for kbrcpp
filename_config = 'sparse_regressor.ini'
prefix = 'smc'  # will automatically construct all filenames
settings = kbrcpp.Settings(prefix)
#some training parameters for kernel width
settings.cost_function = 'logp'  # {'logp', 'hilbert', 'jointlogp', 'pinball', 'jointpinball'}
settings.sigma_x_min = 0.1
settings.sigma_x = 0.236
settings.sigma_x_max = 0.3
settings.sigma_y_min = 0.08
settings.sigma_y = 0.11
settings.sigma_y_max = 0.9
#for preimage
settings.preimage_reg = 1e-6
settings.preimage_reg_min = 1e-10
settings.preimage_reg_max = 1e1
settings.normed_weights = True
#Some other settings
settings.inference_type = 'regress'
settings.cumulative_estimate = True
settings.quantile_estimate = True
settings.quantile = 0.9
settings.walltime = 30.0
settings.preimage_walltime = 30.0
settings.folds = 8
settings.observation_period = 1


def main():
    data = np.load('housing.npy')
    all_X = data[:, 2:]
    all_Y = data[:, 0:2]
    # Make sure we shuffle for the benefit of cross-validation
    random_indices = np.random.permutation(all_X.shape[0])
    all_X = all_X[random_indices]
    all_Y = all_Y[random_indices]
    #create training and testing data
    X = all_X[0:training_size]
    Y = all_Y[0:training_size]
    Y_s = all_Y[training_size:training_size+testing_size]
    X_s = all_X[training_size:training_size+testing_size]

    #whiten and rescale inputs
    X_mean, X_sd = distrib.scale_factors(X)
    Y_mean, Y_sd = distrib.scale_factors(Y)
    X = distrib.scale(X, X_mean, X_sd)
    Y = distrib.scale(Y, Y_mean, Y_sd)
    Y_s = distrib.scale(Y_s, Y_mean, Y_sd)
    X_s = distrib.scale(X_s, X_mean, X_sd)

    # simple prior
    U = X

    #parameters
    kbrcpp.write_config_file(settings, filename_config)
    kbrcpp.write_data_files(settings, U=U, X=X, Y=Y, X_s=X_s, Y_s=Y_s,)

    #now we're ready to invoke the regressor
    kbrcpp.run_sparse(filename_config, kbrcpp_directory)

    #read in the weights we've just calculated
    E = np.load(settings.filename_embedding)
    pdf = np.load(settings.filename_posterior)

    #And plot...
    # fig = pl.figure()
    # axes = fig.add_subplot(121)
    # axes.set_title('Posterior Embedding')
    # axes.imshow(E.T, origin='lower',
                # extent=(ysmin, ysmax, xsmin, xsmax),cmap=cm.hot, aspect='auto')
    # axes.scatter(Y, X, c='y')
    # axes.set_xlim(ysmin, ysmax)
    # axes.set_ylim(xsmin, xsmax)
    # axes = fig.add_subplot(122)
    # axes.set_title('PDF estimate')
    # axes.imshow(pdf.T, origin='lower', 
            # extent=(ysmin, ysmax, xsmin, xsmax), cmap=cm.hot, aspect='auto')
    # axes.scatter(Y, X, c='y')
    # axes.set_xlim(ysmin, ysmax)
    # axes.set_ylim(xsmin, xsmax)
    # pl.show()

if __name__ == "__main__":
    main()
