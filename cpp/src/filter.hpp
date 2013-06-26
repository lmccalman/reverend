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

struct FilterSettings
{
  bool positiveNormed;
};

class Filter
{
  public:
    Filter(uint trainingLength, uint priorLength, const FilterSettings& settings);
    void operator()(const TrainingData& data, 
               const Kernel& kx,
               const Kernel& ky, 
               const Eigen::MatrixXd& ys,
               Eigen::MatrixXd& weights); 

  private:

    //Settings
    const FilterSettings& settings_;
 
    //Useful numbers
    uint n_; // number of training points
    uint m_; // number of prior points
    uint dim_x_;
    uint dim_y_;

    //stuff I'm going to compute
    Eigen::MatrixXd g_xx_;
    Eigen::MatrixXd g_xxtp1_;
    Eigen::MatrixXd g_yy_;
    Eigen::MatrixXd g_xu_;
    Eigen::VectorXd mu_pi_;
    Eigen::VectorXd beta_;
    Eigen::VectorXd beta0_;
    Eigen::MatrixXd beta_g_yy_;
    Eigen::MatrixXd beta_diag_;
    Eigen::MatrixXd r_xy_;
    VerifiedCholeskySolver<Eigen::VectorXd> chol_g_xx_;
    VerifiedCholeskySolver<Eigen::MatrixXd> chol_beta_g_yy_;
    Eigen::VectorXd w_;

};


Filter::Filter(uint trainLength, uint testLength, const FilterSettings& settings)
  : g_xx_(trainLength-1,trainLength-1),
  g_xxtp1_(trainLength-1,trainLength-1),
  g_xu_(trainLength-1,testLength),
  g_yy_(trainLength-1,trainLength-1),
  beta_g_yy_(trainLength-1,trainLength-1),
  beta_(trainLength-1),
  beta0_(trainLength-1),
  mu_pi_(trainLength-1),
  beta_diag_(trainLength-1, trainLength-1),
  r_xy_(trainLength-1,trainLength-1),
  chol_g_xx_(trainLength-1,trainLength-1,1),
  chol_beta_g_yy_(trainLength-1,trainLength-1,trainLength-1),
  w_(trainLength-1),
  settings_(settings){}

void Filter::operator()(const TrainingData& data, 
                          const Kernel& kx,
                          const Kernel& ky, 
                          const Eigen::MatrixXd& ys,
                          Eigen::MatrixXd& weights)
{
  const Eigen::MatrixXd& u = data.u;
  const Eigen::VectorXd& lam = data.lambda;
  uint n = data.x.rows();
  dim_x_ = data.x.cols();
  dim_y_ = data.y.cols();
  Eigen::MatrixXd x = data.x.block(1,0,n-1,dx);
  Eigen::MatrixXd y = data.y.block(1,0,n-1,dy);
  Eigen::MatrixXd xtm1 = data.x.block(0,0,n-1,dx);

  //compute gram matrices
  computeGramMatrix(x, xtm1, kx, g_xxtp1_);
  computeGramMatrix(x, x, kx, g_xxtp1_);
  computeGramMatrix(x, u, kx, g_xu_);
  computeGramMatrix(y, y, ky, g_yy_);
  
  //compute initial prior embedding
  //mu_pi_ = (G_y + delta)^-1 k_y(y1)
  Eigen::VectorXd y0 = data.y.block(0,0,1,dy);
  Eigen::VectorXd mu_dash(n-1);
  computeKernelVector(y, y0, ky, mu_dash);
  chol_g_xx_.solve(g_yy_, mu_dash, mu_pi_);
  mu_pi_ = mu_pi_ * (n-1);
  
  //compute the inverse of g_xx_
  chol_g_xx_.solve(g_xx_, Eigen::MatrixXd::Identity(n,n), g_xx_inv_);
  //compute the prediction operator
  predictOp_ = g_xx_inv_ * g_xxtp1_ * g_xx_inv * g_xx_;
  // now do the inference (which does depend on 
  auto s = weights.rows();
  for (uint i=0; i<s; i++)
  {
    beta_ = predictOp_ * mu_pi_;
    if (isPositiveNormed_)
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
    //actual inference part 
    auto yi = ys.row(i);
    computeKernelVector(y, yi, ky, w_);
    w_ = r_xy_ * w_;
    if (isPositiveNormed_)
    {
      w_ = w_.cwiseMax(0.0);
      w_ = w_ / w_.sum();
    }
    weights.row(i) = w_;
    //ensure the current posterior is the next prior
    mu_pi_ = w_;
  }
}

