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
#include <iostream>
#include <nlopt.hpp>
#include "crossval.hpp"
#include "costfuncs.hpp"

double costWrapper(const std::vector<double>&x, std::vector<double>&grad, void* costClass)
{ 
  NloptCost* ptr = reinterpret_cast<NloptCost*>(costClass); 
  double result = (*ptr)(x, grad);
  return result;
}
  
std::vector<double> globalOptimum(NloptCost& costFunction, const std::vector<double>& thetaMin,
    const std::vector<double>& thetaMax, const std::vector<double>& theta0, double wallTime)
{
  uint n = theta0.size();
  nlopt::opt opt(nlopt::G_MLSL_LDS, n);
  nlopt::opt localopt(nlopt::LN_COBYLA, n);
  opt.set_local_optimizer(localopt);
  opt.set_min_objective(costWrapper, &costFunction);
  opt.set_lower_bounds(thetaMin);
  opt.set_upper_bounds(thetaMax);
  opt.set_maxtime(wallTime);
  double minf;
  std::vector<double> x = theta0;
  nlopt::result result = opt.optimize(x, minf);
  std::cout << "Global optimisation complete." << std::endl;
  std::cout << "Best Result:" << std::endl;
  std::cout << "[ "; 
  for (uint i=0;i<x.size();i++)
  {
    std::cout << std::setw(10) << x[i] << " ";
  }
  std::cout << " ] cost:" << minf << std::endl << std::endl;
  return x;
}

//Epic training function
template <class A, class K>
void trainSettings(const TrainingData& data, Settings& settings)
{
  uint folds = settings.folds;
  double wallTime = settings.walltime;
  if (settings.cost_function == std::string("jointlogp") 
      && (!settings.normed_weights))
  {
    std::vector<double> thetaMin(3);
    std::vector<double> thetaMax(3);
    std::vector<double> theta0(3);
    theta0[0] = settings.sigma_x;
    theta0[1] = settings.sigma_y;
    theta0[2] = log(settings.preimage_reg);
    thetaMin[0] = settings.sigma_x_min;
    thetaMin[1] = settings.sigma_y_min;
    thetaMin[2] = log(settings.preimage_reg_min);
    thetaMax[0] = settings.sigma_x_max;
    thetaMax[1] = settings.sigma_y_max;
    thetaMax[2] = log(settings.preimage_reg_max);
    std::vector<double> thetaBest(3);
    KFoldCVCost< JointLogPCost<A,K> > costfunc(folds,
        data, settings);
    thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
    settings.sigma_x = thetaBest[0];
    settings.sigma_y = thetaBest[1];
    settings.preimage_reg = thetaBest[2];
  }
  else if (settings.cost_function == std::string("jointpinball") 
      && (!settings.normed_weights))
  {
    std::vector<double> thetaMin(3);
    std::vector<double> thetaMax(3);
    std::vector<double> theta0(3);
    theta0[0] = settings.sigma_x;
    theta0[1] = settings.sigma_y;
    theta0[2] = log(settings.preimage_reg);
    thetaMin[0] = settings.sigma_x_min;
    thetaMin[1] = settings.sigma_y_min;
    thetaMin[2] = log(settings.preimage_reg_min);
    thetaMax[0] = settings.sigma_x_max;
    thetaMax[1] = settings.sigma_y_max;
    thetaMax[2] = log(settings.preimage_reg_max);
    std::vector<double> thetaBest(3);
    KFoldCVCost< JointPinballCost<A,K> > costfunc(folds,
        data, settings);
    thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
    settings.sigma_x = thetaBest[0];
    settings.sigma_y = thetaBest[1];
    settings.preimage_reg = thetaBest[2];
  }
  else
  {
    std::vector<double> thetaMin(2);
    std::vector<double> thetaMax(2);
    std::vector<double> theta0(2);
    theta0[0] = settings.sigma_x;
    theta0[1] = settings.sigma_y;
    thetaMin[0] = settings.sigma_x_min;
    thetaMin[1] = settings.sigma_y_min;
    thetaMax[0] = settings.sigma_x_max;
    thetaMax[1] = settings.sigma_y_max;
    std::vector<double> thetaBest(2);
    if (settings.cost_function == std::string("hilbert"))
    {
      KFoldCVCost< HilbertCost<A,K> > costfunc(
          folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
      settings.sigma_x = thetaBest[0];
      settings.sigma_y = thetaBest[1];
    }
    else if (settings.cost_function == std::string("pinball"))
    {
      KFoldCVCost< PinballCost<A,K> > costfunc(
          folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
      settings.sigma_x = thetaBest[0];
      settings.sigma_y = thetaBest[1];
    }
    else
    {
      KFoldCVCost< LogPCost<A,K> > costfunc(folds,
          data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
      settings.sigma_x = thetaBest[0];
      settings.sigma_y = thetaBest[1];
    }
    if (!settings.normed_weights)
    { 
      KFoldCVCost<PreimageCost<K> > pmcostfunc(folds, data, settings);
      std::vector<double> thetaPMin(1);
      std::vector<double> thetaPMax(1);
      std::vector<double> thetaP0(1);
      thetaP0[0] = log(settings.preimage_reg);
      thetaPMin[0] = log(settings.preimage_reg_min);
      thetaPMax[0] = log(settings.preimage_reg_max);
      double preimageWalltime = settings.preimage_walltime;
      auto thetaPBest = globalOptimum(pmcostfunc, thetaPMin, thetaPMax, thetaP0,
          preimageWalltime);
      settings.preimage_reg = thetaPBest[0];
    }
  }
}
