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
#include "data.hpp"
#include "matvec.hpp"
#include "kernel.hpp"

template <class K>
class Regressor
{
  public:
    Regressor(uint trainingLength, uint priorLength, const Settings& settings);
    void operator()(const TrainingData& data, 
               const Kernel<K>& kx,
               const Kernel<K>& ky, 
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
    Eigen::VectorXd mu_pi_;
    Eigen::VectorXd beta_;
    Eigen::MatrixXd beta_g_yy_;
    Eigen::MatrixXd beta_diag_;
    Eigen::MatrixXd r_xy_;
    VerifiedCholeskySolver<Eigen::VectorXd> chol_g_xx_;
    VerifiedCholeskySolver<Eigen::MatrixXd> chol_beta_g_yy_;
    Eigen::VectorXd w_;

};

template <class K>
Regressor<K>::Regressor(uint trainLength, uint testLength, const Settings& settings)
  : settings_(settings),
    mu_pi_(trainLength),
    beta_(trainLength),
    beta_g_yy_(trainLength,trainLength),
    beta_diag_(trainLength, trainLength),
    r_xy_(trainLength,trainLength),
    chol_g_xx_(trainLength,trainLength,1),
    chol_beta_g_yy_(trainLength,trainLength,trainLength),
    w_(trainLength){}

template <class K>
void Regressor<K>::operator()(const TrainingData& data, 
                          const Kernel<K>& kx,
                          const Kernel<K>& ky, 
                          const Eigen::MatrixXd& ys,
                          Eigen::MatrixXd& weights)
{
  //compute prior embedding
  kx.embed(data.u, data.lambda, mu_pi_);
  //get jitchol of gram matrix
  chol_g_xx_.solve(kx.gramMatrix(), mu_pi_, beta_);
  
  if (settings_.normed_weights)
  {
    beta_ = beta_.cwiseMax(0.0);
    beta_ = beta_ / beta_.sum();
    beta_diag_ = beta_.asDiagonal();
    beta_g_yy_ = beta_diag_ * ky.gramMatrix();
    chol_beta_g_yy_.solve(beta_g_yy_, beta_diag_, r_xy_);
  }
  else
  {
    double scaleFactor = beta_.cwiseAbs().maxCoeff();
    beta_ /= scaleFactor;
    beta_ = beta_.cwiseAbs2();
    beta_diag_ = beta_.asDiagonal();
    Eigen::MatrixXd b = ky.gramMatrix() * beta_diag_;
    Eigen::MatrixXd A = b * ky.gramMatrix();
    chol_beta_g_yy_.solve(A, b, r_xy_);
  }
  //now infer
  auto s = weights.rows();
  for (uint i=0; i<s; i++)
  {
    auto yi = ys.row(i);
    ky.embed(yi, w_);
    w_ = r_xy_ * w_;
    if (settings_.normed_weights)
    {
      w_ = w_.cwiseMax(0.0);
      w_ = w_ / w_.sum();
    }
    weights.row(i) = w_;
  }
}

