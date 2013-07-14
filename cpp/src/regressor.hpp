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
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <Eigen/IterativeLinearSolvers>
#include "data.hpp"
#include "matvec.hpp"
#include "kernel.hpp"
#include "lowrank.hpp"

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
    Eigen::VectorXd embed_y_;
    SparseCholeskySolver<Eigen::VectorXd> chol_g_xx_;
    SparseCholeskySolver<Eigen::VectorXd > chol_R_xy_;
    LowRankCholeskySolver chol_lr_g_xx_;
    LowRankCholeskySolver chol_lr_w_;
    Eigen::VectorXd w_;
};

template <class K>
Regressor<K>::Regressor(uint trainLength, uint testLength, const Settings& settings)
  : beta_(trainLength),
    mu_pi_(trainLength),
    embed_y_(trainLength),
    w_(trainLength),
    n_(trainLength),
    settings_(settings){}

template <class K>
void Regressor<K>::operator()(const TrainingData& data, 
                          const Kernel<K>& kx,
                          const Kernel<K>& ky, 
                          const Eigen::MatrixXd& ys,
                          Eigen::MatrixXd& weights)
{
  const Eigen::MatrixXd& x = data.x;
  const Eigen::MatrixXd& y = data.y;

  //compute prior embedding
  // std::cout << "Embedding prior..." << std::endl;
  kx.embed(data.u, data.lambda, mu_pi_);
 
  //Sparse section 
  // chol_g_xx_.solve(kx.gramMatrix(), mu_pi_, beta_);
  //end sparse section
  
  // add low rank update with bigger kernel.
  int columns = n_*0.1;
  int rank = int(columns*0.9);
  Eigen::MatrixXd C(n_, columns);
  Eigen::MatrixXd W(columns, columns);
  Eigen::MatrixXd W_plus(columns, columns);
  double sigma_lrx = kx.width();
  double sigma_lry = ky.width();
  Kernel<Q1CompactKernel> kx_lr(data.x);
  Kernel<Q1CompactKernel> ky_lr(data.y);
  nystromApproximation(data.x,kx_lr, rank, columns, sigma_lrx, C, W, W_plus);
  chol_lr_g_xx_.solve(Eigen::VectorXd::Ones(n_),C, W, W_plus, mu_pi_, beta_);
  //END LOW RANK SECTION
  
  if (settings_.normed_weights)
  {
    beta_ = beta_.cwiseMax(0.0);
    beta_ = beta_ / beta_.sum();
  }
  
  std::vector< Eigen::Triplet<double> > coeffs;
  Eigen::SparseMatrix<double, 0> beta_diag_(n_,n_);
  for(uint j=0;j<n_;j++)
  {
    coeffs.push_back(Eigen::Triplet<double>(j,j,beta_(j)));
  }
  beta_diag_.setFromTriplets(coeffs.begin(), coeffs.end()); 
  
  auto s = weights.rows();
  for (uint i=0; i<s; i++)
  {
    auto yi = ys.row(i);
    ky.embed(yi, embed_y_);
    embed_y_ = beta_.asDiagonal() * embed_y_;

    //sparse section
    // chol_R_xy_.solve(beta_diag_ * ky.gramMatrix(), embed_y_, w_);
    //end sparse section
    
    //low rank section 
    nystromApproximation(data.y,ky_lr, rank, columns, sigma_lry, C, W, W_plus);
    chol_lr_w_.solve(beta_, C, W, W_plus, embed_y_, w_);
    //end low rank section
    
    if (settings_.normed_weights)
    {
      w_ = w_.cwiseMax(0.0);
      if (w_.sum() > 0.0)
      {
        w_ = w_ / w_.sum();
      }
    }
    weights.row(i) = w_;
  }
}

