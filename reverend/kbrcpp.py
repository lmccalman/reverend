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
            self.filename_X_s = prefix + 'X_s.npy'
            self.filename_Y_s = prefix + 'Y_s.npy'
            self.filename_U = prefix + 'U.npy'
            self.filename_weights = prefix + 'W.npy'
            self.filename_preimage = prefix + 'PW.npy'
            self.filename_embedding = prefix + 'E.npy'
            self.filename_posterior = prefix + 'P.npy'
            self.filename_cumulative = prefix + 'C.npy'


class SparseSettings(object):
    def __init__(self, prefix=None):
        self.prefix = prefix
        if prefix is not None:
            self.filename_X = prefix + 'X.npy'
            self.filename_Y = prefix + 'Y.npy'
            self.filename_X_s = prefix + 'X_s.npy'
            self.filename_Y_s = prefix + 'Y_s.npy'
            self.filename_U = prefix + 'U.npy'
            self.filename_weights = prefix + 'W.npy'
            self.filename_embedding = prefix + 'E.npy'


def write_config_file(settings, filename):
    config = ConfigParser.RawConfigParser()
    config.add_section('Algorithm')
    config.add_section('Input')
    config.set('Input', 'filename_x', settings.filename_X)
    config.set('Input', 'filename_y', settings.filename_Y)
    config.set('Input', 'filename_xs', settings.filename_X_s)
    config.set('Input', 'filename_ys', settings.filename_Y_s)
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
    config.add_section('Training')
    config.set('Training', 'walltime', settings.walltime)
    config.set('Training', 'folds', settings.folds)
    config.set('Algorithm', 'epsilon_min', settings.epsilon_min)
    config.set('Algorithm', 'delta_min', settings.delta_min)

    if isinstance(settings, Settings):
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
        config.set('Algorithm', 'quantile', settings.quantile)
        config.set('Output', 'filename_preimage', settings.filename_preimage)
        config.set('Output', 'filename_posterior', settings.filename_posterior)
        config.set('Output', 'filename_cumulative',
                   settings.filename_cumulative)
        config.set('Output', 'filename_quantile', settings.filename_quantile)
        config.set('Training', 'preimage_walltime', settings.preimage_walltime)
        config.set('Training', 'cost_function', settings.cost_function)
        config.add_section('Preimage')
        config.set('Preimage', 'preimage_reg', settings.preimage_reg)
        config.set('Preimage', 'preimage_reg_min', settings.preimage_reg_min)
        config.set('Preimage', 'preimage_reg_max', settings.preimage_reg_max)
    else:
        config.set('Kernel', 'low_rank_scale', settings.low_rank_scale)
        config.set('Kernel', 'low_rank_weight', settings.low_rank_weight)
        config.set('Kernel', 'low_rank_scale_min', settings.low_rank_scale_min)
        config.set('Kernel', 'low_rank_weight_min', settings.low_rank_weight_min)
        config.set('Kernel', 'low_rank_scale_max', settings.low_rank_scale_max)
        config.set('Kernel', 'low_rank_weight_max', settings.low_rank_weight_max)
        config.set('Algorithm', 'method', settings.method)
    with open(filename, 'w') as configfile:
        config.write(configfile)


def write_data_files(settings,U, X, Y, X_s, Y_s):
    np.save(settings.filename_U, U)
    np.save(settings.filename_X, X)
    np.save(settings.filename_Y, Y)
    np.save(settings.filename_X_s, X_s)
    np.save(settings.filename_Y_s, Y_s)


def run(filename_config, directory):
    path = os.path.join(os.path.abspath(directory), 'kbrcpp')
    proc = subprocess.Popen([path, filename_config],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    for line in iter(proc.stdout.readline, ""):
        print line,

def run_sparse(filename_config, directory):
    path = os.path.join(os.path.abspath(directory), 'skbrcpp')
    proc = subprocess.Popen([path, filename_config],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    for line in iter(proc.stdout.readline, ""):
        print line,
