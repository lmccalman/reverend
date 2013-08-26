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
#include <Eigen/IterativeLinearSolvers>
#include "data.hpp"
#include "matvec.hpp"
#include "kernel.hpp"
#include "lowrank.hpp"

template <class K>
class LowRankRegressor
{
  public:
    LowRankRegressor(uint trainingLength, uint priorLength, const Settings& settings);
    void operator()(const TrainingData& data, 
               const Kernel<K>& kx,
               const Kernel<K>& ky, 
               const Eigen::MatrixXd& ys,
               double epsilonMin,
               double deltaMin,
               Eigen::MatrixXd& weights); 
  private:

    //Settings
    const Settings& settings_;
 
    //Useful numbers
    uint n_; // number of training points
    uint m_; // number of prior points
    uint dim_x_;
    uint dim_y_;

    VerifiedCholeskySolver<Eigen::VectorXd> chol_g_xx_;
    VerifiedCholeskySolver<Eigen::VectorXd> chol_beta_g_yy_;
    //stuff I'm going to compute
    Eigen::VectorXd mu_pi_;
    Eigen::VectorXd beta_;
    Eigen::VectorXd embed_y_;
    Eigen::VectorXd w_;
};

template <class K>
LowRankRegressor<K>::LowRankRegressor(uint trainLength,
    uint testLength, const Settings& settings)
  : n_(trainLength),
    chol_g_xx_(int(trainLength*settings.data_fraction),
               int(trainLength*settings.data_fraction),1),
    chol_beta_g_yy_(int(trainLength*settings.data_fraction),
                    int(trainLength*settings.data_fraction),1),
    mu_pi_(trainLength),
    beta_(trainLength),
    embed_y_(trainLength),
    settings_(settings),
    w_(trainLength){}

template <class K>
void LowRankRegressor<K>::operator()(const TrainingData& data, 
                          const Kernel<K>& kx,
                          const Kernel<K>& ky, 
                          const Eigen::MatrixXd& ys,
                          double epsilonMin,
                          double deltaMin,
                          Eigen::MatrixXd& weights)
{
  const Eigen::MatrixXd& x = data.x;
  const Eigen::MatrixXd& y = data.y;
  //low rank stuff 
  int columns = (n_*settings_.data_fraction);
  // int rank = int(columns*0.9);
  Eigen::MatrixXd C(n_, columns);
  Eigen::MatrixXd W(columns, columns);

  //compute prior embedding
  kx.embed(data.u, data.lambda, mu_pi_);
  double invreg = 1.0/epsilonMin; 
  simpleNystromApproximation(data.x, kx, columns, C, W);
  Eigen::VectorXd x1(columns);
  chol_g_xx_.solve(W + invreg*C.transpose()*C, C.transpose()*mu_pi_, 1e-10, x1);
  
  beta_ = invreg*mu_pi_ - invreg*invreg*C*x1;
  beta_ = beta_.cwiseMax(0.0);
  beta_ = beta_ / beta_.sum();
  auto s = weights.rows();
  Eigen::VectorXd x2(columns);
  double invdelta = 1.0/deltaMin;
  for (uint i=0; i<s; i++)
  {
    auto yi = ys.row(i);
    ky.embed(yi, embed_y_);
    embed_y_ = beta_.asDiagonal() * embed_y_;
    chol_beta_g_yy_.solve(W + invreg*C.transpose()*(beta_.asDiagonal()*C),
                          C.transpose()*embed_y_,1e-10, x2);
    w_ = invdelta*embed_y_ - invdelta * invdelta * beta_.asDiagonal()*(C*x2);
    w_ = w_.cwiseMax(0.0);
    if (w_.sum() > 0.0)
    {
      w_ = w_ / w_.sum();
    }
    weights.row(i) = w_;
  }
}
