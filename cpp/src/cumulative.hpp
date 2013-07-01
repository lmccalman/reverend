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
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include "data.hpp"
#include "kernel.hpp"

template <class K>
class Cumulative
{
  public:
    Cumulative(const Eigen::VectorXd& coeffs, const Eigen::MatrixXd& X,
               const K& kx, const Settings& settings)
      : coeffs_(coeffs), X_(X), settings_(settings), indicator_(coeffs.rows()),
      kx_(kx)
    {
      if (!settings.normed_weights)
      {
        double sigma_x = kx.width();
        Eigen::VectorXd zeroVal = X_.colwise().minCoeff();
        Eigen::VectorXd oneVal = X_.colwise().maxCoeff();
        zeroVal -= 5*sigma_x*Eigen::VectorXd::Ones(zeroVal.rows());
        oneVal += 5*sigma_x*Eigen::VectorXd::Ones(oneVal.rows());
        
        kx.embedIndicator(zeroVal, indicator_);
        double zeroResult = kx.innerProduct(indicator_, coeffs);
        kx.embedIndicator(oneVal, indicator_);
        double oneResult = kx.innerProduct(indicator_, coeffs);
        scale_ = 1.0 / (oneResult - zeroResult);
        offset_ = zeroResult;
      }
    }

    double operator()(const Eigen::VectorXd& x)
    {
      if (settings_.normed_weights)
      {
        return fromNormedWeights(x);
      }
      else
      {
        return fromWeights(x);
      }
    }

  protected:
    double fromNormedWeights(const Eigen::VectorXd& x);
    double fromWeights(const Eigen::VectorXd& x);
    
    const Eigen::VectorXd& coeffs_;
    const Eigen::MatrixXd& X_;
    const Settings& settings_;
    Eigen::VectorXd indicator_;
    const K& kx_;
    double scale_;
    double offset_;
};

template <class K>
double Cumulative<K>::fromNormedWeights(const Eigen::VectorXd& x)
{
  assert( coeffs_.sum() == 1.0);
  double result = 0.0;
  uint n = X_.rows();
  for (int i=0; i<n; i++)
  {
    result += coeffs_(i) * kx_.cumulative(x, X_.row(i));
  }
  return result;
}

template <class K>
double Cumulative<K>::fromWeights(const Eigen::VectorXd& x)
{
  kx_.embedIndicator(x, indicator_);
  double rawResult = kx_.innerProduct(indicator_, coeffs_);
  double scaledResult = (rawResult - offset_) * scale_;
  return scaledResult;
}

template <class K>
void computeCumulates(const TrainingData& trainingData, const TestingData& testingData,
    const Eigen::MatrixXd& weights, const K& kx, const Settings& settings,
    Eigen::MatrixXd& cumulates)
{
  uint testPoints = weights.rows(); 
  uint evalPoints = testingData.xs.rows();

#pragma omp parallel for
  for (int i=0;i<testPoints;i++)  
  {
    Cumulative<K> cumulator(weights.row(i), trainingData.x, kx, settings);
    for (int j=0;j<evalPoints;j++)
    {
      cumulates(i,j) = cumulator(testingData.xs.row(j));
    }
  }
}
