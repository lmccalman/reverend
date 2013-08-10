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
  localopt.set_ftol_rel(1e-5);
  localopt.set_ftol_abs(1e-4);

  opt.set_local_optimizer(localopt);
  opt.set_min_objective(costWrapper, &costFunction);
  opt.set_lower_bounds(thetaMin);
  opt.set_upper_bounds(thetaMax);
  opt.set_maxtime(wallTime);
  double minf = 0.0;
  std::vector<double> x(theta0.size());
  x = theta0;
  std::vector<double> grad(x.size());
  // nlopt::result result = opt.optimize(x, minf);
  double firstcost = costFunction(x, grad);
  opt.optimize(x, minf);
  
  std::cout << "Approximate Solution Found." << std::endl;
  std::cout << "[ "; 
  for (uint i=0;i<x.size();i++)
  {
    std::cout << std::setw(10) << x[i] << " ";
  }
  std::cout << " ] cost:" << minf << std::endl << std::endl;
  std::cout << "Refining..." << std::endl;
  minf = 0.0;
  nlopt::opt refopt(nlopt::LN_COBYLA, n);
  refopt.set_ftol_rel(1e-15);
  refopt.set_ftol_abs(1e-7);
  refopt.set_min_objective(costWrapper, &costFunction);
  refopt.set_lower_bounds(thetaMin);
  refopt.set_upper_bounds(thetaMax);
  refopt.set_maxtime(wallTime/2.0);
  refopt.optimize(x, minf);
  std::cout << "Final Estimate" << std::endl;
  std::cout << "[ "; 
  for (uint i=0;i<x.size();i++)
  {
    std::cout << std::setw(10) << x[i] << " ";
  }
  std::cout << " ] cost:" << minf << std::endl << std::endl;
  return x;
}

// std::vector<double> globalOptimum(NloptCost& costFunction, const std::vector<double>& thetaMin,
    // const std::vector<double>& thetaMax, const std::vector<double>& theta0, double wallTime)
// {
  // uint n = theta0.size();
  // uint xevals = 50; 
  // uint yevals = 50;
  // double bestx = 0.0;
  // double besty = 0.0;
  // double bestCost = -1e200;
  // double dx = (thetaMax[0] - thetaMin[0])/double(xevals);
  // double dy = (thetaMax[1] - thetaMin[1])/double(yevals);
  // std::vector<double> x = theta0;
  // std::vector<double> grad(n);
  // Eigen::MatrixXd results(xevals*yevals,3);
  // uint counter = 0;
  // for (uint i=0; i<xevals; i++) 
  // {
    // for (uint j=0; j<yevals; j++)
    // {
      // x[0] = thetaMin[0] + i*dx;
      // x[1] = thetaMin[1] + j*dy;
      // results(counter, 0) = x[0];
      // results(counter, 1) = x[1];
      // double cost = costFunction(x, grad);
      // results(counter, 2) = cost;
      // counter++;
      // if (cost < bestCost)
      // {
        // bestCost = cost;
        // bestx = x[0];
        // besty = x[1];
      // }
    // }
    // std::cout << counter/double(xevals*yevals)*100 << "% complete" << std::endl; 
  // }
  // x[0] = bestx;
  // x[1] = besty;
  // writeCSV(results, "costLandscape.csv");
  // return x;
// }

//Epic training function
template <class A, class K>
void trainSettings(const TrainingData& data, Settings& settings)
{
  uint folds = settings.folds;
  double wallTime = settings.walltime;
  double preimageWalltime = settings.preimage_walltime;
  std::string cf = settings.cost_function;
  bool jointMethod = ((cf == "logp_joint") || (cf == "pinball_joint"));
  if (jointMethod)
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
    if (cf == "logp_joint")
    {
      settings.normed_weights = false;
      KFoldCVCost< JointLogPCost<A,K> > costfunc(folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
    }
    else // must be pinball joint
    {
      settings.normed_weights = false;
      KFoldCVCost< JointPinballCost<A,K> > costfunc(folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
    }
    settings.sigma_x = thetaBest[0];
    settings.sigma_y = thetaBest[1];
    settings.preimage_reg = exp(thetaBest[2]);
  }
  else // not a joint method
  {
    std::vector<double> thetaMin(2);
    std::vector<double> thetaMax(2);
    std::vector<double> theta0(2);
    std::vector<double> thetaBest = theta0;
    std::vector<double> thetaPMin(1);
    std::vector<double> thetaPMax(1);
    std::vector<double> thetaP0(1);
    std::vector<double> thetaPBest = thetaP0;
    theta0[0] = settings.sigma_x;
    theta0[1] = settings.sigma_y;
    thetaMin[0] = settings.sigma_x_min;
    thetaMin[1] = settings.sigma_y_min;
    thetaMax[0] = settings.sigma_x_max;
    thetaMax[1] = settings.sigma_y_max;
    thetaP0[0] = log(settings.preimage_reg);
    thetaPMin[0] = log(settings.preimage_reg_min);
    thetaPMax[0] = log(settings.preimage_reg_max);
    if (cf == "logp_norm")
    {
      settings.normed_weights = true;
      KFoldCVCost< LogPCost<A,K> > costfunc(folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
    }
    else if (cf == "logp_preimage")
    {
      settings.normed_weights = false;
      KFoldCVCost< LogPCost<A,K> > costfunc(folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
      KFoldCVCost< PreimageCost<K> > pmcostfunc(folds, data, settings);
      thetaPBest = globalOptimum(pmcostfunc, thetaPMin, thetaPMax, thetaP0,
                                 preimageWalltime);
    }
    else if (cf == "pinball_norm")
    {
      settings.normed_weights = true;
      KFoldCVCost< PinballCost<A,K> > costfunc(folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
    }
    else if (cf == "pinball_direct")
    {
      settings.normed_weights = false;
      KFoldCVCost< PinballCost<A,K> > costfunc(folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
    }
    else if (cf == "hilbert")
    {
      settings.normed_weights = false;
      KFoldCVCost< HilbertCost<A,K> > costfunc(folds, data, settings);
      thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
    }
    else
    {
      std::cout << "ERROR -- invalid cost function" << std::endl;
    }
    settings.sigma_x = thetaBest[0];
    settings.sigma_y = thetaBest[1];
    settings.preimage_reg = exp(thetaPBest[0]);
  }
}


