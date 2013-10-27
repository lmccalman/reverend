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


double multivariateDiagonalGaussian(const Eigen::VectorXd& point,
                                     const Eigen::VectorXd& mu,
                                     const Eigen::VectorXd& sigma)
{
  int d = point.rows();
  Eigen::VectorXd invsig = sigma.array().inverse();
  double exponent = -0.5*(point - mu).cwiseQuotient(sigma).squaredNorm();
  double val = exp(exponent);
  double coeff = pow(2.0*M_PI, -0.5*d) * pow(sigma.prod(), -0.5);
  double result = coeff*val;
  return result;
}

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
    if (alpha < 0)
    {
      // std::cout << alpha <<  "ALPHA < 0 in mixture coeff" << std::endl;
      std::cout << ".";
    }
    double expCoeff = kx.logk(point, means.row(i).transpose());
    double adjExpCoeff = expCoeff - maxPower;
    double adjProbs = alpha*exp(adjExpCoeff);
    sumAdjProb += adjProbs;
  }
  double result;
  if (soft)
  {
    result =  log(sumAdjProb) + maxPower + logScaleFactor;
    if (!(result == result))
    {
      // std::cout << point(0) << " " << point(1) << ":";
      // result = -1e10;
      result = -1*std::numeric_limits<double>::infinity();
    }
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
