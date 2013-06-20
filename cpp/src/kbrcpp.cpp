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
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "io.hpp"
#include "config.hpp"
#include "crossval.hpp"
#include "train.hpp"
#include "costfuncs.hpp"

int main(int argc, char** argv)
{

  std::cout << "kbrcpp initialised." << std::endl;
  srand(time(NULL));
  auto settings = getSettings(argv[1]);

  
  auto x = readNPY(settings.filename_x);
  auto y = readNPY(settings.filename_y);
  auto xs = readNPY(settings.filename_xs);
  auto ys = readNPY(settings.filename_ys);
  uint n = x.rows();
  uint s = ys.rows();

  Eigen::MatrixXd u = x;
  uint m = n;
  if (settings.filename_u != "")
  {
    u = readNPY(settings.filename_u);
    m = u.rows();
  }

  //for the moment keep the weights constant
  Eigen::VectorXd lambda = Eigen::VectorXd::Ones(m);
  lambda = lambda / double(m);

  Eigen::MatrixXd weights(s,n);
  TrainingData data(u, lambda, x, y);
 
  //lets try some training 
  uint folds = settings.folds;
  KFoldCVCost<MyCost> costfunc(folds, data);
  std::vector<double> thetaMin(2);
  std::vector<double> thetaMax(2);
  std::vector<double> theta0(2);
  theta0[0] = settings.sigma_x;
  theta0[1] = settings.sigma_y;
  thetaMin[0] = settings.sigma_x_min;
  thetaMin[1] = settings.sigma_y_min;
  thetaMax[0] = settings.sigma_x_max;
  thetaMax[1] = settings.sigma_y_max;
  double wallTime = settings.wall_time;
  auto thetaBest = globalOptimum(costfunc, thetaMin, thetaMax, theta0, wallTime);
  
  //Initialise from params
  double sigma_x = thetaBest[0];
  double sigma_y = thetaBest[1];
  Kernel kx = boost::bind(rbfKernel, _1, _2, sigma_x);
  Kernel ky = boost::bind(rbfKernel, _1, _2, sigma_y);

  Regressor r(n, m);
  r(data, kx, ky, ys, weights);
  
  //write out the results 
  writeNPY(weights, settings.filename_weights);
  std::cout << "kbrcpp inference complete."<< std::endl;
  
  return 0;
}

