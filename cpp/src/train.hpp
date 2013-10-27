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
#include <iterator>
#include <algorithm>
#include <nlopt.hpp>
#include "crossval.hpp"
#include "costfuncs.hpp"
#include <cmath>

double costWrapper(const std::vector<double>&x, std::vector<double>&grad, void* costClass)
{ 
  NloptCost* ptr = reinterpret_cast<NloptCost*>(costClass); 
  std::vector<double> xDash = x;
  for (uint i=0; i<x.size();i++)
  {
    xDash[i] = exp(x[i]);
  }
  double result = (*ptr)(xDash, grad);
  return result;
}

double nonLogCostWrapper(const std::vector<double>&x, std::vector<double>&grad, void* costClass)
{ 
  NloptCost* ptr = reinterpret_cast<NloptCost*>(costClass); 
  double result = (*ptr)(x, grad);
  return result;
}

std::vector<double> SGD(NloptCost& costFunction, const std::vector<double>& thetaMin,
    const std::vector<double>& thetaMax, const std::vector<double>& theta0,
    const Settings& settings)
{
  uint iterations = settings.sgd_iterations; 
  double initialLearnRate = settings.sgd_learn_rate;
  double finalLearnRate = initialLearnRate * 0.01;
  
  std::vector<double> grad(theta0.size());
  std::vector<double> x = theta0;
  std::vector<double> bestx = theta0;
  double bestcost = 1e100;
  Eigen::VectorXd allCosts(iterations);
  for (uint iter=0; iter < iterations; iter++) 
  {
    double cost = costFunction(x, grad);
    allCosts(iter) = cost; 
    double modgrad = 0.0;
    
    //update learn rate 
    double frac = iter / double(iterations);
    double learnRate = initialLearnRate*(1.0 - frac) + finalLearnRate * frac;
    
    //update x 
    for (uint i=0; i< x.size(); i++)
    {
      x[i] = x[i] - learnRate*grad[i];
      if (x[i] < thetaMin[i])
        x[i] = thetaMin[i];
      if (x[i] > thetaMax[i])
        x[i] = thetaMax[i];
      modgrad += grad[i]*grad[i];
    }
    modgrad = sqrt(modgrad);
    std::cout << "cost: " << cost << " grad: " << modgrad << std::endl;
    if (cost < bestcost)
    {
      bestcost = cost;
      bestx = x;
    }
  }
  writeNPY(allCosts, settings.filename_sgd);
  std::cout << "Final Estimate" << std::endl;
  std::cout << "[ "; 
  for (uint i=0;i<bestx.size();i++)
  {
    std::cout << bestx[i] << " ";
  }
  std::cout << " ] cost:" << bestcost << std::endl << std::endl;
  return bestx;
}


std::vector<double> localOptimum(NloptCost& costFunction, const std::vector<double>& thetaMin,
    const std::vector<double>& thetaMax, const std::vector<double>& theta0)
{
  uint n = theta0.size();
  nlopt::opt opt(nlopt::LN_COBYLA, n);
  // nlopt::opt opt(nlopt::LN_SBPLX, n);
  // nlopt::opt opt(nlopt::LD_LBFGS, n);
  opt.set_ftol_abs(1e-4);
  opt.set_ftol_rel(1e-4);
  // opt.set_xtol_rel(1e-5);
  // opt.set_xtol_abs(1e-5);
  opt.set_maxtime(3600);
  
  std::cout << "Optimizer Initialized..." << std::endl; 
  opt.set_min_objective(nonLogCostWrapper, &costFunction);
  opt.set_lower_bounds(thetaMin);
  opt.set_upper_bounds(thetaMax);
  double minf = 0.0;
  std::vector<double> x(theta0.size());
  x = theta0;
  std::vector<double> grad(x.size());
  
  try {opt.optimize(x, minf);}
  catch(nlopt::roundoff_limited a)
  {
    std::cout << "WARNING: optimisation round-off limited" << std::endl; 
  }
  
  std::cout << "Final Estimate" << std::endl;
  std::cout << "[ "; 
  for (uint i=0;i<x.size();i++)
  {
    std::cout << std::setw(10) << x[i] << " ";
  }
  std::cout << " ] cost:" << minf << std::endl << std::endl;
  return x;
}

void writeOptimalParams(const std::vector<double>& theta)
{
  std::ofstream file("optimal_params.txt");
  if (file.is_open())
  {
    for (uint i=0; i<theta.size();i++)
    {
      file << theta[i] << ",";
    }
    file << std::endl;
  }
}
  
std::vector<double> globalOptimum(NloptCost& costFunction, const std::vector<double>& thetaMin,
    const std::vector<double>& thetaMax, const std::vector<double>& theta0, double wallTime)
{
  std::cout << "Optimizer Initialized..." << std::endl; 
  std::cout << "thetaMin:" << std::endl;
  for (double e : thetaMin)
    std::cout << e << " ";
  std::cout << std::endl << "theta0:" << std::endl;
  for (double e : theta0)
    std::cout << e << " ";
  std::cout << std::endl << "thetaMax:" << std::endl;
  for (double e : thetaMax)
    std::cout << e << " ";
  std::cout << std::endl;
  
  uint n = theta0.size();
  // nlopt::opt opt(nlopt::G_MLSL_LDS, n);
  // nlopt::opt localopt(nlopt::LN_COBYLA, n);
  // localopt.set_ftol_rel(1e-5);
  // localopt.set_ftol_abs(1e-5);
  // localopt.set_xtol_rel(1e-5);
  nlopt::opt opt(nlopt::GN_CRS2_LM, n);
  opt.set_ftol_rel(1e-5);
  opt.set_ftol_abs(1e-5);
  opt.set_xtol_rel(1e-5);

  // opt.set_local_optimizer(localopt);
  opt.set_min_objective(costWrapper, &costFunction);
  
  std::vector<double> logThetaMin(n);
  std::vector<double> logThetaMax(n);
  std::vector<double> logTheta0(n);
  for (uint i=0; i<n; i++)
  {
    logThetaMin[i] = log(thetaMin[i]);
    logThetaMax[i] = log(thetaMax[i]);
    logTheta0[i] = log(theta0[i]);
  }

  opt.set_lower_bounds(logThetaMin);
  opt.set_upper_bounds(logThetaMax);
  opt.set_maxtime(wallTime);
  double minf = 0.0;
  std::vector<double> x(theta0.size());
  x = logTheta0;
  std::vector<double> grad(x.size());
  // nlopt::result result = opt.optimize(x, minf);
  // double firstcost = costFunction(x, grad);
  
  opt.optimize(x, minf);
  for (uint i=0; i<n; i++)
  {
    x[i] = exp(x[i]);
    if (x[i] < thetaMin[i])
      x[i] = thetaMin[i];
    if (x[i] > thetaMax[i])
      x[i] = thetaMax[i];
  }
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
  refopt.set_ftol_rel(1e-8);
  refopt.set_ftol_abs(1e-7);
  refopt.set_min_objective(nonLogCostWrapper, &costFunction);
  refopt.set_lower_bounds(thetaMin);
  refopt.set_upper_bounds(thetaMax);
  refopt.set_maxtime(wallTime/2.0);
  try {refopt.optimize(x, minf);}
  catch(nlopt::roundoff_limited a)
  {}
  std::cout << "Final Estimate" << std::endl;
  std::cout << "[ "; 
  for (uint i=0;i<x.size();i++)
  {
    std::cout << std::setw(10) << x[i] << " ";
  }
  std::cout << " ] cost:" << minf << std::endl << std::endl;
  
  writeOptimalParams(x);
  return x;
}

struct TrainingVectors
{
  std::vector<double> thetaMin;
  std::vector<double> thetaMax;
  std::vector<double> theta0;
};


TrainingVectors trainingVectors(uint dx, uint dy, const Settings& settings)
{
  uint totalParams;
  bool normedWeights = settings.normed_weights;
  if (normedWeights)
  {
    totalParams = dx + dy + 2;
  }
  else
  {
    totalParams = dx + dy + 3;
  }
  std::vector<double> thetaMin(totalParams);
  std::vector<double> thetaMax(totalParams);
  std::vector<double> theta0(totalParams);
  for (uint i=0; i<dx; i++)
  { 
    theta0[i] = settings.sigma_x(i);
    thetaMin[i] = settings.sigma_x_min(i);
    thetaMax[i] = settings.sigma_x_max(i);
  }
  for (uint i=0; i<dy; i++)
  { 
    theta0[i+dx] = settings.sigma_y(i);
    thetaMin[i+dx] = settings.sigma_y_min(i);
    thetaMax[i+dx] = settings.sigma_y_max(i);
  }
  // theta0[dx+dy] = log(settings.epsilon_min);
  // theta0[dx+dy+1] = log(settings.delta_min);
  // thetaMin[dx+dy] = log(settings.epsilon_min_min);
  // thetaMin[dx+dy+1] = log(settings.delta_min_min);
  // thetaMax[dx+dy] = log(settings.epsilon_min_max);
  // thetaMax[dx+dy+1] = log(settings.delta_min_max);
  theta0[dx+dy] = settings.epsilon_min;
  theta0[dx+dy+1] = settings.delta_min;
  thetaMin[dx+dy] = settings.epsilon_min_min;
  thetaMin[dx+dy+1] =settings.delta_min_min;
  thetaMax[dx+dy] = settings.epsilon_min_max;
  thetaMax[dx+dy+1] = settings.delta_min_max;

  if (!normedWeights) 
  {
    theta0[dx+dy+2] = settings.preimage_reg;
    thetaMin[dx+dy+2] = settings.preimage_reg_min;
    thetaMax[dx+dy+2] = settings.preimage_reg_max;
  }
  TrainingVectors v;
  v.thetaMin = thetaMin;
  v.thetaMax = thetaMax;
  v.theta0 = theta0;
  return v;
}

Settings newSettings(std::vector<double> thetaBest, uint dx, uint dy, const Settings& oldsettings)
{
  uint normedWeights = oldsettings.normed_weights;
  Settings news = oldsettings;
  for (uint i=0; i<dx; i++)
  { 
    news.sigma_x(i) = thetaBest[i];
  }
  for (uint i=0; i<dy; i++)
  { 
    news.sigma_y(i) = thetaBest[i+dx];
  }
  news.epsilon_min = thetaBest[dx+dy];
  news.delta_min = thetaBest[dx+dy+1];
  if (!normedWeights)
  {
    news.preimage_reg = thetaBest[dx+dy+2];
  }
  return news;
}


template <class A, class K>
void trainSettings(const TrainingData& data, Settings& settings)
{
  uint folds = settings.folds;
  double wallTime = settings.walltime;
  double preimageWalltime = settings.preimage_walltime;
  bool normedWeights = settings.normed_weights;
  uint dx = data.x.cols();
  uint dy = data.y.cols();
  TrainingVectors  v = trainingVectors(dx, dy, settings);
  std::vector<double> thetaBest = v.theta0;
  if (settings.pinball_loss)
  {
    KFoldCVCost< PinballCost<A,K> > costfunc(folds, data, settings);
    thetaBest = globalOptimum(costfunc, v.thetaMin, v.thetaMax, v.theta0, wallTime);
  } 
  else
  {
    KFoldCVCost< LogPCost<A,K> > costfunc(folds, data, settings);
    thetaBest = globalOptimum(costfunc, v.thetaMin, v.thetaMax, v.theta0, wallTime);
  }
  Settings result = newSettings(thetaBest,dx,dy,settings);
  settings = result; 
}
  
template <class A, class K>
void passthroughTrainSettings(const TrainingData& data, const TestingData& test, Settings& settings)
{
  double wallTime = settings.walltime;
  bool normedWeights = true;
  uint dx = data.x.cols();
  uint dy = data.y.cols();
  TrainingVectors  v = trainingVectors(dx, dy, settings);
  std::vector<double> thetaBest = v.theta0;
  PassthroughCost< LogPCost<A,K> > costfunc(data, test, settings);
  thetaBest = globalOptimum(costfunc, v.thetaMin, v.thetaMax, v.theta0, wallTime);
  Settings result = newSettings(thetaBest,dx,dy,settings);
  settings = result; 
}


