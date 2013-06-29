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
#include <boost/math/special_functions/erf.hpp>
#include <Eigen/Core>
#include "kernel.hpp"
#include "data.hpp"
#include "rkhs.hpp"

void scaleFactors(const Eigen::VectorXd& weights, 
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& g_xx,
    double sigma_x, 
    double& scale,
    double& offset);

class Cumulative
{
  public:
    Cumulative(const Eigen::VectorXd& coeffs, const Eigen::MatrixXd& X,
               const Eigen::MatrixXd& g_xx, double sigma_x , const Settings& settings)
      : coeffs_(coeffs), X_(X), g_xx_(g_xx), settings_(settings), sigma_x_(sigma_x),
        indicator_(coeffs.rows())
    {
      if (!settings.normed_weights)
      {
        Eigen::VectorXd zeroVal = X_.colwise().minCoeff();
        Eigen::VectorXd oneVal = X_.colwise().maxCoeff();
        zeroVal -= 5*sigma_x*Eigen::VectorXd::Ones(zeroVal.rows());
        oneVal += 5*sigma_x*Eigen::VectorXd::Ones(oneVal.rows());
        optimalIndicatorEmbedding(zeroVal, X, sigma_x, indicator_);
        double zeroResult = hilbertDotProduct(indicator_, coeffs, g_xx);
        optimalIndicatorEmbedding(oneVal, X, sigma_x, indicator_);
        double oneResult = hilbertDotProduct(indicator_, coeffs, g_xx);
        scale_ = 1.0 / (oneResult - zeroResult);
        offset_ = zeroResult;
        std::cout << "zeroVal: " << zeroVal.sum() << " oneVal: " << oneVal.sum() << std::endl;
        std::cout << "zero: " << zeroResult << " one: " << oneResult << std::endl;
        std::cout << "scale: " << scale_ << " offset: " << offset_ << std::endl;
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
    const Eigen::MatrixXd& g_xx_;
    const Settings& settings_;
    Eigen::VectorXd indicator_;
    double sigma_x_;
    double scale_;
    double offset_;
};

double Cumulative::fromNormedWeights(const Eigen::VectorXd& x)
{
  assert( coeffs_.sum() == 1.0);
  double result = 0.0;
  uint n = X_.rows();
  uint dx = X_.cols();
  double denom = 1.0 / (sigma_x_*std::sqrt(2.0));
  for (int b=0; b<n; b++)
  {
    double dim_result = 1.0;
    for (int d=0; d<dx; d++)
    {
      double p = x(d);
      double m = X_(b,d);
      dim_result *= coeffs_(b) * 0.5 * (1.0 + boost::math::erf( (p - m)*denom ));
    }
    result += dim_result;
  }
  return result;
}

double Cumulative::fromWeights(const Eigen::VectorXd& x)
{
  optimalIndicatorEmbedding(x, X_, sigma_x_, indicator_); 
  double rawResult = hilbertDotProduct(coeffs_, indicator_, g_xx_);
  double scaledResult = (rawResult - offset_) * scale_;
  return scaledResult;
}

void scaleFactors(const Eigen::VectorXd& weights, 
                  const Eigen::MatrixXd& X,
                  const Eigen::MatrixXd& g_xx,
                  double sigma_x,
                  double& scale,
                  double& offset)
{
}

