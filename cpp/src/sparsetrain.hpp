// Reverend -- Practical Bayesian Inference with Kernel Embeddings
// Copyright (C) 2013 Lachlan McCalman
// lachlan@mccalman.info

// Reverend is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Reverend is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Reverend.  If not, see <http://www.gnu.org/licenses/>.
#pragma once
#include "train.hpp"
#include "sparsecostfuncs.hpp"
#include "sparsecrossval.hpp"

//Epic training function
template <class A, class K>
void sparseTrainSettings(const TrainingData& data, SparseSettings& settings)
{
  uint folds = settings.folds;
  double wallTime = settings.walltime;
  std::vector<double> thetaMin(6);
  std::vector<double> thetaMax(6);
  std::vector<double> theta0(6);
  theta0[0] = settings.sigma_x;
  theta0[1] = settings.sigma_y;
  theta0[2] = log(settings.epsilon_min);
  theta0[3] = log(settings.delta_min);
  theta0[4] = settings.low_rank_scale;
  theta0[5] = settings.low_rank_weight;
  thetaMin[0] = settings.sigma_x_min;
  thetaMin[1] = settings.sigma_y_min;
  thetaMin[2] = log(settings.epsilon_min_min);
  thetaMin[3] = log(settings.delta_min_min);
  thetaMin[4] = settings.low_rank_scale_min;
  thetaMin[5] = settings.low_rank_weight_min;
  thetaMax[0] = settings.sigma_x_max;
  thetaMax[1] = settings.sigma_y_max;
  thetaMax[2] = log(settings.epsilon_min_max);
  thetaMax[3] = log(settings.delta_min_max);
  thetaMax[4] = settings.low_rank_scale_max;
  thetaMax[5] = settings.low_rank_weight_max;
  std::vector<double> thetaBest(2);
  SparseKFoldCVCost< SparseLogPCost<A,K> > costfunc(folds, data, settings);
  thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
  settings.sigma_x = thetaBest[0];
  settings.sigma_y = thetaBest[1];
  settings.epsilon_min = exp(thetaBest[2]);
  settings.delta_min = exp(thetaBest[3]);
  settings.low_rank_scale = thetaBest[4];
  settings.low_rank_weight = thetaBest[5];
}
