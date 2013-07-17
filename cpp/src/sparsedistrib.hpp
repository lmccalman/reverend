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
    const Kernel<K>& kx)
{
  assert(point.size() == means.cols());
  assert(means.rows() == coeffs.size());
  uint numberOfMeans = means.rows();
  double logScaleFactor = -log(kx.volume(kx.width(), means.cols()));
  //find the min exp coeff
  double maxPower = 0.0; // ie infinity;
  for (uint i=0; i<numberOfMeans; i++)
  {
    double val = kx(point, means.row(i).transpose());
    double expCoeff = log(std::max(val,1e-20));
    maxPower = std::max(maxPower, expCoeff);
  }
  //now compute everything
  double sumAdjProb = 0.0; 
  for (uint i=0; i<numberOfMeans; i++)
  {
    double alpha = coeffs[i];
    double val = kx(point, means.row(i).transpose());
    double expCoeff = log(std::max(val,1e-200));
    double adjExpCoeff = expCoeff - maxPower;
    double adjProbs = alpha*exp(adjExpCoeff);
    sumAdjProb += adjProbs;
  }
  // this means that if my sumAdjProb is zero or negative, things don't
  // actually break I just get a very low result
  double result =  log(std::max(sumAdjProb,1e-200)) + maxPower + logScaleFactor;
  return result;
}

template <class K>
double multiLogKernelMixture(const Eigen::VectorXd& point,
    const Eigen::MatrixXd& means,
    const Eigen::VectorXd& coeffs,
    const Kernel<K>& kx,
    double lowRankScale,
    double lowRankWeight)
{
  Kernel<Q1CompactKernel> kx_lr(means);
  assert(point.size() == means.cols());
  assert(means.rows() == coeffs.size());
  uint numberOfMeans = means.rows();
  double sigma = kx.width();
  uint dim = means.cols();
  //find the min exp coeff
  double maxPower = 0.0; // ie infinity;
  for (uint i=0; i<numberOfMeans; i++)
  {
    double val1 = (1.0 - lowRankWeight) * kx(point, means.row(i).transpose()) / kx.volume(sigma, dim);  
    double val2 = lowRankWeight * kx_lr(point, means.row(i).transpose(), sigma*lowRankScale) / kx_lr.volume(sigma*lowRankScale, dim);
    double val = val1 + val2;
    double expCoeff = log(val);
    maxPower = std::max(maxPower, expCoeff);
  }
  //now compute everything
  double sumAdjProb = 0.0; 
  for (uint i=0; i<numberOfMeans; i++)
  {
    double alpha = coeffs[i];
    double val1 = (1.0 - lowRankWeight) * kx(point, means.row(i).transpose()) / kx.volume(sigma, dim);  
    double val2 = lowRankWeight * kx_lr(point, means.row(i).transpose(), sigma*lowRankScale) / kx_lr.volume(sigma*lowRankScale, dim);
    double val = val1 + val2;
    double expCoeff = log(val);
    double adjExpCoeff = expCoeff - maxPower;
    double adjProbs = alpha*exp(adjExpCoeff);
    sumAdjProb += adjProbs;
  }
  // this means that if my sumAdjProb is zero or negative, things don't
  // actually break I just get a very low result
  double result =  log(sumAdjProb) + maxPower;
  if (!(result == result))
  {
    result = -1*std::numeric_limits<double>::infinity();
  }
  return result;
}