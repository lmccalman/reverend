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

double hilbertDotProduct(const Eigen::VectorXd& h1,
    const Eigen::VectorXd& h2,
    const Eigen::MatrixXd& G_xx)
{
  auto val = h1.transpose() * G_xx * h2;
  return val(0,0);
}

void optimalIndicatorEmbedding(const Eigen::VectorXd& x, 
    const Eigen::MatrixXd& X,
    double sigma_x,
    Eigen::VectorXd& weights)
{
  uint n = X.rows();
  uint dx = X.cols();
  double a = std::sqrt(2.0 * M_PI) * sigma_x;
  double denom = 1.0 / (sigma_x*std::sqrt(2.0));
  for (int i=0; i<n; i++)
  {
    double dim_result = 1.0;
    for (int d=0; d<dx; d++)
    {
      double p = x(d);
      double m = X(i,d);
      dim_result *= a * 0.5 * (1.0 + std::erf((p - m)*denom));
    }
    weights[i] = dim_result;
  }
}

void computeGramMatrix(const Eigen::MatrixXd& x, 
    const Eigen::MatrixXd& x_dash, 
    Kernel k,
    Eigen::MatrixXd& g)
{
  uint n = x.rows();
  uint m = x_dash.rows();
  for(uint i=0; i<n;i++)
  {
    for(uint j=0;j<m;j++)
    {
      g(i,j) = k(x.row(i), x_dash.row(j));
    }
  }
}

void computeKernelVector(const Eigen::MatrixXd& x, 
    const Eigen::VectorXd& x_dash, 
    Kernel k,
    Eigen::VectorXd& g)
{
  uint n = x.rows();
  for(uint i=0; i<n;i++)
  {
    g(i) = k(x.row(i), x_dash);
  }
}
