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
#include "matvec.hpp"

typedef Eigen::SparseMatrix<double> SparseMatrix;

template <class K>
void nystromApproximation(const Eigen::MatrixXd& X, const Kernel<K>& kx,
                          uint rank, uint columns,
                          double sigma,
                          Eigen::MatrixXd& C,
                          Eigen::MatrixXd& W_k)
{
  std::cout << "Computing low-rank update..." << std::endl;
  uint n = X.rows();
  Eigen::MatrixXd W(columns, columns);
  double scaleFactor = 1.0 / sqrt(columns/double(n));
  std::vector<uint> I(columns);
  Eigen::VectorXd gCol(n);
  for (int t=0; t<columns; t++)
  {
    uint i = rand() % n;
    assert (i < n);
    for (int j=0; j<n; j++)
    {
      C(j,t) = kx(X.row(j),X.row(i)) * scaleFactor;
    }
    I.push_back(i);
  }
  double scaleFactor2 = 1.0 / (columns / double(n));
  for (int i : I)
  {
    for (int j : I)
    {
      W(i,j) = kx(X.row(i),X.row(j), sigma) * scaleFactor2;
    }
  } 
  // compute best rank-k approximation of W
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  W_k = svd.matrixU().block(0, 0, columns, rank) *
        svd.singularValues().head(rank).asDiagonal() *
        svd.matrixV().block(0,0,columns,rank).transpose();
}

template<class K>
void lowRankGramUpdate(const Eigen::MatrixXd& X, const Kernel<K>& kx,
    double sigma, uint rank, uint columns, 
    const SparseMatrix& A,
    Eigen::VectorXd& x) 
{
  uint r = rank;
  uint n = X.rows();
  Eigen::MatrixXd C(n,columns);
  Eigen::MatrixXd W(r,r);
  nystromApproximation(X, kx, rank, columns,sigma, C, W);
  SparseCholeskySolver<Eigen::MatrixXd> chol_c;
  Eigen::MatrixXd L(n,r);
  chol_c.solve(A, C, L);
  Eigen::VectorXd M(n);
  Eigen::MatrixXd A_dash =  W + C.transpose() * L;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(A_dash);
  M = solver.solve(C.transpose()* x);
  x = x - L * C * M;
  std::cout << "Low rank update complete." << std::endl;
}




