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
void nystromApproximation(const Eigen::MatrixXd& X, const Kernel<K>& kx,
                          uint rank, uint columns,
                          double sigma,
                          Eigen::MatrixXd& C,
                          Eigen::MatrixXd& W_k,
                          Eigen::MatrixXd& W_plus)
{
  uint n = X.rows();
  Eigen::MatrixXd W(columns, columns);
  double scaleFactor = 1.0 / sqrt(columns/double(n));
  std::vector<uint> I(columns);
  Eigen::VectorXd gCol(n);
  for (int t=0; t<columns; t++)
  {
    uint i = rand() % n;
    for (int j=0; j<n; j++)
    {
      C(j,t) = kx(X.row(j),X.row(i)) * scaleFactor;
    }
    I.push_back(i);
  }
  double scaleFactor2 = 1.0 / (columns / double(n));
  for (int i=0;i<columns;i++)
  {
    for (int j=0;j<columns;j++)
    {
      W(i,j) = kx(X.row(I[i]),X.row(I[j]), sigma) * scaleFactor2;
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

// template<class K, class T>
// void lowRankGramUpdate(const Eigen::MatrixXd& X, const Kernel<K>& kx,
    // double sigma, uint rank, uint columns, 
    // const SparseMatrix& G,
    // double jitter,
    // const T& diag_a,
    // SparseCholeskySolver<Eigen::MatrixXd>& chol_c,
    // Eigen::VectorXd& x) 
// {
  // // std::cout << "Computing low-rank update..." << std::endl;
  // uint r = rank;
  // uint n = X.rows();
  // SparseMatrix A(n,n);
  // setJitter(G,jitter,n,A);
  // A = diag_a * A;

  // Eigen::MatrixXd C(n,columns);
  // Eigen::MatrixXd W(columns,columns);
  // nystromApproximation(X, kx, rank, columns,sigma, C, W);
  // Eigen::MatrixXd U = diag_a * C;
  // Eigen::MatrixXd L(n,columns);
  // chol_c.solve(A, U, L);
  // Eigen::VectorXd M(n);
  // Eigen::MatrixXd A_dash =  W + C.transpose() * L;
  // Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(A_dash);
  // M = solver.solve(C.transpose()* x);
  // x = x - (L * M);
  // // std::cout << "Low rank update complete." << std::endl;
// }




