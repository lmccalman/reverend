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
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/CholmodSupport>

typedef Eigen::SparseMatrix<double,0> SparseMatrix;


template <class T>
class VerifiedCholeskySolver
{
  public:
    VerifiedCholeskySolver(uint A_n, uint bRows, uint bCols, double minJitter)
      : cholSolver_(A_n), bDash_(bRows, bCols), xDashDown_(bRows, bCols), aJit_(A_n, A_n)
    { 
      jitter_ = minJitter;
      minJitter_ = minJitter;
    };

    void solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, T& x);
    double jitter(){return jitter_;};

  
  private:
    Eigen::LLT<Eigen::MatrixXd> cholSolver_;
    Eigen::MatrixXd bDash_;
    Eigen::MatrixXd xDashDown_;
    Eigen::MatrixXd aJit_;
    double jitter_;
    double minJitter_;
};

template <class T>
void VerifiedCholeskySolver<T>::solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, T& x)
{
  uint n = A.rows();
  double maxJitter = 1.0e10;
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
    while (!smallestFound && (jitter_ > minJitter_))
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
      jitter_ /=2.0;
    }
  }
  if (jitter_ <= minJitter_)
  {
    // std::cout << "WARNING: min jitter reached" << std::endl;
  }
}

template <class T>
class SparseCholeskySolver
{
  public:
    SparseCholeskySolver(double minJitter):minJitter_(minJitter){};
    void solve(const SparseMatrix& A, const T& b, T& x);
    double jitter(){return jitter_;}

  private:
    Eigen::CholmodSupernodalLLT<SparseMatrix> cholSolver_;
    T bDash_;
    T xDashDown_;
    double jitter_ = 1e-1;
    double minJitter_;
};

void setJitter(const SparseMatrix& A, double jitter, int n, SparseMatrix& aJit)
{
  std::vector< Eigen::Triplet<double> > coeffs;
  for(uint i=0; i<n;i++)
  {
    coeffs.push_back(Eigen::Triplet<double>(i,i,jitter));
  }
  aJit.setFromTriplets(coeffs.begin(), coeffs.end()); 
  aJit += A;
}

template <class T>
void SparseCholeskySolver<T>::solve(const SparseMatrix& A, const T& b, T& x)
{
  cholmod_common& config = cholSolver_.cholmod();
  config.print = 1;
  config.print_function = NULL;
  config.try_catch = false;

  uint n = A.rows();
  SparseMatrix aJit_(n,n);
  double maxJitter = 1.0e20;
  double precision = 1e-8;
  // start with the jitter from last time
  setJitter(A, jitter_, n, aJit_);
  cholSolver_.compute(aJit_);
  x = cholSolver_.solve(b);
  bDash_ = aJit_ * x;
  bool solved = (bDash_).isApprox(b, precision);
  //if we solved, the jitter may be too big
  bool smallestFound = false;
  if (solved)
  {
    while (!smallestFound && (jitter_ > minJitter_))
    {
      //decrease the jitter
      setJitter(A, jitter_/2.0, n, aJit_);
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
      // std::cout << "increasing jitter to " << jitter_ << std::endl;
      jitter_ *= 2.0;
      setJitter(A, jitter_, n, aJit_);
      //compute Cholesky decomposition
      cholSolver_.compute(aJit_);
      x = cholSolver_.solve(b);
      bDash_ = aJit_ * x;
      solved = (bDash_).isApprox(b, precision);
    }
    if (!solved)
    {
      jitter_ /=2.0;
    }
  }
  
  if (jitter_ <= minJitter_)
  {
    // std::cout << "WARNING: min jitter reached" << std::endl;
  }
}

class LowRankCholeskySolver
{
  public:
    LowRankCholeskySolver(double minJitter):minJitter_(minJitter){};
    void solve(const Eigen::VectorXd& diag,
               const Eigen::MatrixXd& C, 
               const Eigen::MatrixXd& W, 
               const Eigen::MatrixXd& W_plus, 
               const Eigen::VectorXd& b,
               Eigen::VectorXd& x);
    double jitter(){return jitter_;}

  private:
    Eigen::VectorXd bDash_;
    Eigen::VectorXd xDashDown_;
    double jitter_ = 1e-1;
    double minJitter_;
};

void woodburySolve(const Eigen::VectorXd& diag,
                   const Eigen::MatrixXd& C,
                   const Eigen::MatrixXd& W,
                   double jitter,
                   const Eigen::VectorXd& b,
                   Eigen::VectorXd& x)
{
  Eigen::MatrixXd A = W + (1.0/jitter) * C.transpose() * diag.asDiagonal() * C; 
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> alphaSolver(A);
  Eigen::VectorXd alpha = alphaSolver.solve( C.transpose() * b);
  x = (1.0/jitter) * b + (1.0/(jitter*jitter)) * diag.asDiagonal() * C * alpha;
}

void LowRankCholeskySolver::solve(const Eigen::VectorXd& diag,
                                  const Eigen::MatrixXd& C, 
                                  const Eigen::MatrixXd& W, 
                                  const Eigen::MatrixXd& W_plus,
                                  const Eigen::VectorXd& b,
                                  Eigen::VectorXd& x)
{
  uint n = C.rows();
  double maxJitter = 1.0e20;
  double precision = 1e-4;
  // start with the jitter from last time
  woodburySolve(diag, C, W, jitter_, b, x);
  bDash_ = diag.asDiagonal() * (C * (W_plus * (C.transpose() * x))) + (jitter_ * x);
  bool solved = (bDash_).isApprox(b, precision);
  //if we solved, the jitter may be too big
  bool smallestFound = false;
  if (solved)
  {
    while (!smallestFound && (jitter_ > minJitter_))
    {
      //decrease the jitter
      woodburySolve(diag, C, W, jitter_/2.0, b, xDashDown_);
      bDash_ = diag.asDiagonal() * (C * (W_plus * (C.transpose() * xDashDown_))) + (jitter_/2.0 * xDashDown_);
      //compute Cholesky decomposition
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
      // std::cout << "increasing jitter to " << jitter_ << std::endl;
      jitter_ *= 2.0;
      woodburySolve(diag, C, W, jitter_, b, x);
      bDash_ = diag.asDiagonal() * (C * (W_plus * (C.transpose() * x))) + (jitter_ * x);
      solved = (bDash_).isApprox(b, precision);
    }
    if (!solved)
    {
      std::cout << "WARNING: max jitter reached" << std::endl;
      jitter_ /=2.0;
    }
  }
  
  if (jitter_ <= minJitter_)
  {
    // std::cout << "WARNING: min jitter reached" << std::endl;
  }
}
