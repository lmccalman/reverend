#pragma once
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
#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Core>

template <class K>
double logKernelMixture(const Eigen::VectorXd& point,
    const Eigen::MatrixXd& means,
    const Eigen::VectorXd& coeffs,
    const K& kx,
    bool soft)
{
  assert(point.size() == means.cols());
  assert(means.rows() == coeffs.size());
  uint numberOfMeans = means.rows();
  double logScaleFactor = -log(kx.volume());
  //find the min exp coeff
  double maxPower = -1e200; //ie infinity
  for (uint i=0; i<numberOfMeans; i++)
  {
    double expCoeff = kx.logk(point, means.row(i).transpose());
    maxPower = std::max(maxPower, expCoeff);
  }
  //now compute everything
  double sumAdjProb = 0.0; 
  for (uint i=0; i<numberOfMeans; i++)
  {
    double alpha = coeffs[i];
    double expCoeff = kx.logk(point, means.row(i).transpose());
    double adjExpCoeff = expCoeff - maxPower;
    double adjProbs = alpha*exp(adjExpCoeff);
    sumAdjProb += adjProbs;
  }
  double result;
  if (soft)
  {
    result = log(std::max(sumAdjProb, 1e-200)) + maxPower + logScaleFactor;
  }
  else
  {
    result =  log(sumAdjProb) + maxPower + logScaleFactor;
    if (!(result == result))
    {
      result = -1*std::numeric_limits<double>::infinity();
    }
  }
  return result;
}
