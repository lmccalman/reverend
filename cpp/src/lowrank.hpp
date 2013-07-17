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
#define EIGEN_DONT_PARALLELIZE
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include "kernel.hpp"

typedef Eigen::SparseMatrix<double> SparseMatrix;

template <class K>
void nystromApproximation(const Eigen::MatrixXd& X, const K& kx,
                          uint rank, uint columns,
                          Eigen::MatrixXd& C,
                          Eigen::MatrixXd& W_k,
                          Eigen::MatrixXd& W_plus)
{
  uint n = X.rows();
  Eigen::MatrixXd W(columns, columns);
  std::vector<uint> I(columns);
  Eigen::VectorXd gCol(n);
  for (int t=0; t<columns; t++)
  {
    uint i = rand() % n;
    for (int j=0; j<n; j++)
    {
      C(j,t) = kx(X.row(j),X.row(i));
    }
    I.push_back(i);
  }
  for (int i=0;i<columns;i++)
  {
    for (int j=0;j<columns;j++)
    {
      W(i,j) = kx(X.row(I[i]),X.row(I[j]));
    }
  } 
  // compute best rank-k approximation of W
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd newDiags = svd.singularValues();
  for (int i=rank;i<columns;i++)
  {
    newDiags(i) = 0;
  }
  Eigen::VectorXd invDiags = newDiags;
  double tolerance = columns * newDiags.maxCoeff() * std::numeric_limits<double>::epsilon();
  for (int i=0; i<columns;i++)
  {
    if (newDiags(i) > tolerance)
    {
      invDiags(i) = 1.0 / newDiags(i);
    }
    else
    {
      invDiags(i) = 0.0;
    }
  }
  W_k = svd.matrixU() * newDiags.asDiagonal() * svd.matrixV().transpose();
  W_plus = svd.matrixU() * invDiags.asDiagonal() * svd.matrixV().transpose();
}
  
  
template <class K>
void simpleNystromApproximation(const Eigen::MatrixXd& X, const K& kx,
    uint columns,
    Eigen::MatrixXd& C,
    Eigen::MatrixXd& W_k)
{
  uint n = X.rows();
  std::vector<uint> I(columns);
  Eigen::VectorXd gCol(n);
  for (int t=0; t<columns; t++)
  {
    uint i = rand() % n;
    for (int j=0; j<n; j++)
    {
      C(j,t) = kx(X.row(j),X.row(i));
    }
    I.push_back(i);
  }
  for (int i=0;i<columns;i++)
  {
    for (int j=0;j<columns;j++)
    {
      W_k(i,j) = kx(X.row(I[i]),X.row(I[j]));
    }
  } 
}
