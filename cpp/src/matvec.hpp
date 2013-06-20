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
#include <Eigen/Cholesky>

template <class T>
class VerifiedCholeskySolver
{
  public:
    VerifiedCholeskySolver(uint A_n, uint bRows, uint bCols)
      : cholSolver_(A_n), bDash_(bRows, bCols), xDashDown_(bRows, bCols), aJit_(A_n, A_n)
    { 
      jitter_ = 1e-7;
    };

    void solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, T& x);
  
  private:
    Eigen::LLT<Eigen::MatrixXd> cholSolver_;
    Eigen::MatrixXd bDash_;
    Eigen::MatrixXd xDashDown_;
    Eigen::MatrixXd aJit_;
    double jitter_;
};

template <class T>
void VerifiedCholeskySolver<T>::solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, T& x)
{
  uint n = A.rows();
  double maxJitter = 1.0e2;
  double precision = 1e-8;
  aJit_ = A;
  // start with the jitter from last time
  aJit_ += Eigen::MatrixXd::Identity(n,n)*jitter_;
  cholSolver_.compute(aJit_);
  x = cholSolver_.solve(b);
  bDash_ = aJit_ * x;
  bool solved = (bDash_).isApprox(b, precision);
  //if we solved, the jitter may be too big
  bool smallestFound = false;
  if (solved)
  {
    while (!smallestFound)
    {
      //decrease the jitter
      aJit_ -= Eigen::MatrixXd::Identity(n,n)*jitter_;
      //compute Cholesky decomposition
      cholSolver_.compute(aJit_);
      xDashDown_ = cholSolver_.solve(b);
      bDash_ = aJit_ * xDashDown_;
      smallestFound = !((bDash_).isApprox(b, precision));
      if (!smallestFound)
      {
        x = xDashDown_;
        jitter_ /= 2.0;
      }
    }
  }
  //if we didn't solve, the jitter is too small
  else
  {
    while ((jitter_ < maxJitter) && (!solved))
    {
      //increase the jitter
      aJit_ += Eigen::MatrixXd::Identity(n,n)*jitter_;
      jitter_ *= 2.0;
      //compute Cholesky decomposition
      cholSolver_.compute(aJit_);
      x = cholSolver_.solve(b);
      bDash_ = aJit_ * x;
      solved = (bDash_).isApprox(b, precision);
    }
    if (!solved)
    {
      std::cout << "WARNING: max jitter reached" << std::endl;
    }
  }
}
