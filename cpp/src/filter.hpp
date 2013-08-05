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
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "data.hpp"
#include "matvec.hpp"
#include "kernel.hpp"


template <class K>
class Filter
{
  public:
    Filter(uint trainingLength, uint priorLength, const Settings& settings);
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
    Eigen::MatrixXd g_xxtp1_;
    Eigen::VectorXd mu_pi_;
    Eigen::VectorXd beta_;
    Eigen::VectorXd beta_0_;
    Eigen::MatrixXd beta_g_yy_;
    Eigen::MatrixXd beta_diag_;
    Eigen::MatrixXd r_xy_;
    VerifiedCholeskySolver<Eigen::VectorXd> chol_g_yy_;
    VerifiedCholeskySolver<Eigen::VectorXd> chol_beta_0_;
    VerifiedCholeskySolver<Eigen::VectorXd> chol_beta_;
    VerifiedCholeskySolver<Eigen::MatrixXd> chol_beta_g_yy_;
    Eigen::VectorXd w_;

};


template <class K>
Filter<K>::Filter(uint trainLength, uint testLength, const Settings& settings)
  : settings_(settings),
    g_xxtp1_(trainLength,trainLength),
    mu_pi_(trainLength),
    beta_(trainLength),
    beta_0_(trainLength),
    beta_g_yy_(trainLength,trainLength),
    beta_diag_(trainLength, trainLength),
    r_xy_(trainLength,trainLength),
    chol_g_yy_(trainLength,trainLength,1, settings.delta_min),
    chol_beta_0_(trainLength,trainLength,1, settings.epsilon_min),
    chol_beta_(trainLength,trainLength,1,settings.epsilon_min),
    chol_beta_g_yy_(trainLength,trainLength,trainLength, settings.delta_min),
    w_(trainLength){}

template <class K>
void Filter<K>::operator()(const TrainingData& data, 
                          const Kernel<K>& kx,
                          const Kernel<K>& ky, 
                          const Eigen::MatrixXd& ys,
                          Eigen::MatrixXd& weights)
{
  uint n = data.x.rows();
  dim_x_ = data.x.cols();
  dim_y_ = data.y.cols();

  //compute transition matrix
  for (int i=0; i<n;i++)
  {
    for (int j=0; j<n; j++)
    {
      g_xxtp1_(i,j) = kx(data.x.row(i), data.xtp1.row(j));
    }
  }
  
  //compute initial embedding from kbr
  Eigen::VectorXd y0(dim_y_); 
  y0 = data.y.block(0,0,1,dim_y_).transpose();
  Eigen::VectorXd mu_dash(n);
  ky.embed(y0, mu_dash);
  chol_g_yy_.solve(ky.gramMatrix(), mu_dash, mu_pi_);
  weights.row(0) = mu_pi_;

  auto s = weights.rows();
  for (uint i=1; i<s; i++)
  {
    mu_pi_ = mu_pi_.cwiseMax(0.0);
    mu_pi_ = mu_pi_ / mu_pi_.sum();
    mu_pi_ = kx.gramMatrix() * mu_pi_; 
    chol_beta_0_.solve(kx.gramMatrix(), mu_pi_, beta_0_);
    beta_0_ = g_xxtp1_ * beta_0_;
    chol_beta_.solve(kx.gramMatrix(), beta_0_, beta_);
    if (settings_.normed_weights)
    {
      beta_ = beta_.cwiseMax(0.0);
      beta_ = beta_ / beta_.sum();
    }
    
    if (i%settings_.observation_period == 0) 
    {
      if (settings_.normed_weights)
      {
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
      //actual inference part 
      auto yi = ys.row(i);
      ky.embed(yi, w_);
      w_ = r_xy_ * w_;
      if (settings_.normed_weights)
      {
        w_ = w_.cwiseMax(0.0);
        w_ = w_ / w_.sum();
      }
      else
      {
        if (w_.norm() > 0.0)
        {
          w_ = w_ / w_.norm();
        }
      }
      if (!(w_.norm() > 0))
      {
        w_ = Eigen::VectorXd::Ones(w_.size()) / double(w_.size());
        std::cout << "WARNING: DIVERGED" << std::endl;
      }
      weights.row(i) = w_;
      //ensure the current posterior is the next prior
      mu_pi_ = w_;
    }
    else
    {
      weights.row(i) = beta_;
      mu_pi_ = beta_;
    }
  }
}

