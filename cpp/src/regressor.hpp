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
#include "kernel.hpp"
#include "data.hpp"
#include "io.hpp"
#include "matvec.hpp"

class Regressor
{
  public:
    Regressor(uint trainingLength, uint priorLength, const Settings& settings);
    void operator()(const TrainingData& data, 
               const Kernel& kx,
               const Kernel& ky, 
               const Eigen::MatrixXd& ys,
               Eigen::MatrixXd& weights); 

  private:

    //Settings
    const Settings& settings_;
 
    //Useful numbers
    uint n_; // number of training points
    uint m_; // number of prior points
    uint dim_x_;
    uint dim_y_;

    //stuff I'm going to compute
    Eigen::MatrixXd g_xx_;
    Eigen::MatrixXd g_yy_;
    Eigen::MatrixXd g_xu_;
    Eigen::VectorXd mu_pi_;
    Eigen::VectorXd beta_;
    Eigen::MatrixXd beta_g_yy_;
    Eigen::MatrixXd beta_diag_;
    Eigen::MatrixXd r_xy_;
    VerifiedCholeskySolver<Eigen::VectorXd> chol_g_xx_;
    VerifiedCholeskySolver<Eigen::MatrixXd> chol_beta_g_yy_;
    Eigen::VectorXd w_;

};

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

Regressor::Regressor(uint trainLength, uint testLength, const Settings& settings)
  : g_xx_(trainLength,trainLength),
  g_xu_(trainLength,testLength),
  g_yy_(trainLength,trainLength),
  beta_g_yy_(trainLength,trainLength),
  beta_(trainLength),
  mu_pi_(trainLength),
  beta_diag_(trainLength, trainLength),
  r_xy_(trainLength,trainLength),
  chol_g_xx_(trainLength,trainLength,1),
  chol_beta_g_yy_(trainLength,trainLength,trainLength),
  w_(trainLength),
  settings_(settings){}

void Regressor::operator()(const TrainingData& data, 
                          const Kernel& kx,
                          const Kernel& ky, 
                          const Eigen::MatrixXd& ys,
                          Eigen::MatrixXd& weights)
{
  const Eigen::MatrixXd& u = data.u;
  const Eigen::VectorXd& lam = data.lambda;
  const Eigen::MatrixXd& x = data.x;
  const Eigen::MatrixXd& y = data.y;

  //compute gram matrices
  computeGramMatrix(x, x, kx, g_xx_);
  computeGramMatrix(x, u, kx, g_xu_);
  computeGramMatrix(y, y, ky, g_yy_);
  
  //compute prior embedding
  mu_pi_ = g_xu_ * lam;
  //get jitchol of gram matrix
  chol_g_xx_.solve(g_xx_, mu_pi_, beta_);
  
  if (settings_.normed_weights)
  {
    beta_ = beta_.cwiseMax(0.0);
    beta_ = beta_ / beta_.sum();
    beta_diag_ = beta_.asDiagonal();
    beta_g_yy_ = beta_diag_ * g_yy_;
    chol_beta_g_yy_.solve(beta_g_yy_, beta_diag_, r_xy_);
  }
  else
  {
    double scaleFactor = beta_.cwiseAbs().maxCoeff();
    beta_ /= scaleFactor;
    beta_ = beta_.cwiseAbs2();
    beta_diag_ = beta_.asDiagonal();
    Eigen::MatrixXd b = g_yy_ * beta_diag_;
    Eigen::MatrixXd A = b * g_yy_;
    chol_beta_g_yy_.solve(A, b, r_xy_);
  }
  //now infer
  auto s = weights.rows();
  for (uint i=0; i<s; i++)
  {
    auto yi = ys.row(i);
    computeKernelVector(y, yi, ky, w_);
    w_ = r_xy_ * w_;
    if (settings_.normed_weights)
    {
      w_ = w_.cwiseMax(0.0);
      w_ = w_ / w_.sum();
    }
    weights.row(i) = w_;
  }
}

