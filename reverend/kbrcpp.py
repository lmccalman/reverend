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

import ConfigParser
import numpy as np
import subprocess
import os.path


class Settings(object):
    def __init__(self, prefix=None):
        self.prefix = prefix
        if prefix is not None:
            self.filename_X = prefix + 'X.npy'
            self.filename_Y = prefix + 'Y.npy'
            self.filename_X_b = prefix + 'X_b.npy'
            self.filename_Y_b = prefix + 'Y_b.npy'
            self.filename_Xr = prefix + 'Xr.npy'
            self.filename_Yr = prefix + 'Yr.npy'
            self.filename_X_s = prefix + 'X_s.npy'
            self.filename_Y_s = prefix + 'Y_s.npy'
            self.filename_U = prefix + 'U.npy'
            self.filename_weights = prefix + 'W.npy'
            self.filename_preimage = prefix + 'PW.npy'
            self.filename_embedding = prefix + 'E.npy'
            self.filename_posterior = prefix + 'P.npy'
            self.filename_cumulative = prefix + 'C.npy'
            self.filename_sgd = prefix + 'SGD.npy'

def write_config_file(settings, filename):
    config = ConfigParser.RawConfigParser()
    config.add_section('Algorithm')
    config.add_section('Input')
    config.set('Input', 'filename_x', settings.filename_X)
    config.set('Input', 'filename_y', settings.filename_Y)
    config.set('Input', 'filename_xr', settings.filename_Xr)
    config.set('Input', 'filename_yr', settings.filename_Yr)
    config.set('Input', 'filename_xs', settings.filename_X_s)
    config.set('Input', 'filename_ys', settings.filename_Y_s)
    config.set('Input', 'filename_xb', settings.filename_X_b)
    config.set('Input', 'filename_yb', settings.filename_Y_b)
    config.set('Input', 'filename_u', settings.filename_U)
    config.add_section('Kernel')
    config.set('Kernel', 'sigma_x', settings.sigma_x)
    config.set('Kernel', 'sigma_y', settings.sigma_y)
    config.set('Kernel', 'sigma_x_min', settings.sigma_x_min)
    config.set('Kernel', 'sigma_y_min', settings.sigma_y_min)
    config.set('Kernel', 'sigma_x_max', settings.sigma_x_max)
    config.set('Kernel', 'sigma_y_max', settings.sigma_y_max)
    config.add_section('Output')
    config.set('Output', 'filename_weights', settings.filename_weights)
    config.set('Output', 'filename_embedding', settings.filename_embedding)
    config.set('Output', 'filename_sgd', settings.filename_sgd)
    config.add_section('Training')
    config.set('Training', 'walltime', settings.walltime)
    config.set('Training', 'folds', settings.folds)
    config.set('Algorithm', 'epsilon_min', settings.epsilon_min)
    config.set('Algorithm', 'delta_min', settings.delta_min)
    config.set('Algorithm', 'epsilon_min_min', settings.epsilon_min_min)
    config.set('Algorithm', 'delta_min_min', settings.delta_min_min)
    config.set('Algorithm', 'epsilon_min_max', settings.epsilon_min_max)
    config.set('Algorithm', 'delta_min_max', settings.delta_min_max)
    config.set('Algorithm', 'normed_weights', int(settings.normed_weights))

    settings.filename_quantile = settings.prefix + 'Q_{}.npy'.format(
        settings.quantile)
    config.set('Algorithm', 'observation_period',
               settings.observation_period)
    config.set('Algorithm', 'inference_type', settings.inference_type)
    config.set('Algorithm', 'cumulative_estimate',
               int(settings.cumulative_estimate))
    config.set('Algorithm', 'cumulative_mean_map',
            int(settings.cumulative_mean_map))
    config.set('Algorithm', 'quantile_estimate',
               int(settings.quantile_estimate))
    config.set('Algorithm', 'pinball_loss',
            int(settings.pinball_loss))
    config.set('Algorithm', 'direct_cumulative',
            int(settings.direct_cumulative))
    config.set('Algorithm', 'quantile', settings.quantile)
    config.set('Output', 'filename_preimage', settings.filename_preimage)
    config.set('Output', 'filename_posterior', settings.filename_posterior)
    config.set('Output', 'filename_cumulative',
               settings.filename_cumulative)
    config.set('Output', 'filename_quantile', settings.filename_quantile)
    config.set('Training', 'preimage_walltime', settings.preimage_walltime)
    config.add_section('Preimage')
    config.set('Preimage', 'preimage_reg', settings.preimage_reg)
    config.set('Preimage', 'preimage_reg_min', settings.preimage_reg_min)
    config.set('Preimage', 'preimage_reg_max', settings.preimage_reg_max)
    config.add_section('Scaling')
    config.set('Scaling', 'scaling_strategy', settings.scaling_strategy)
    config.set('Scaling', 'data_fraction', settings.data_fraction)
    config.set('Scaling', 'sgd_iterations', settings.sgd_iterations)
    config.set('Scaling', 'sgd_learn_rate', settings.sgd_learn_rate)
    config.set('Scaling', 'sgd_batch_size', settings.sgd_batch_size)
    with open(filename, 'w') as configfile:
        config.write(configfile)


def write_data_files(settings,U, X, Y, X_s, Y_s, X_b, Y_b):
    np.save(settings.filename_U, U)
    np.save(settings.filename_X, X)
    np.save(settings.filename_Y, Y)
    np.save(settings.filename_X_s, X_s)
    np.save(settings.filename_Y_s, Y_s)
    np.save(settings.filename_X_b, X_b)
    np.save(settings.filename_Y_b, Y_b)


def run(filename_config, directory):
    path = os.path.join(os.path.abspath(directory), 'kbrcpp')
    proc = subprocess.Popen([path, filename_config],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    for line in iter(proc.stdout.readline, ""):
        print line,

