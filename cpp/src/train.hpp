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

//This must be inherited by anything that we want to optimize with NLOpt
struct NloptCost
{
  public:
    virtual double operator()(const std::vector<double>&x, std::vector<double>&grad) = 0;
};

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
  std::cout << "Best result: " << x[0] << "," << x[1] << std::endl;
  return x;
}
