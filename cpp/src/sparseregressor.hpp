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
class SparseRegressor
{
  public:
    SparseRegressor(uint trainingLength, uint priorLength, const SparseSettings& settings);
    void operator()(const TrainingData& data, 
               const Kernel<K, Eigen::SparseMatrix<double> >& kx,
               const Kernel<K, Eigen::SparseMatrix<double> >& ky, 
               const Eigen::MatrixXd& ys,
               double lowRankScale,
               double lowRankWeight,
               Eigen::MatrixXd& weights); 

  private:

    //Settings
    const SparseSettings& settings_;
 
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
    VerifiedCholeskySolver<Eigen::VectorXd> chol_g_xx_lr_;
    VerifiedCholeskySolver<Eigen::VectorXd> chol_ur_lr_;
    SparseCholeskySolver<Eigen::MatrixXd> chol_u_;
    SparseCholeskySolver<Eigen::MatrixXd> chol_ur_;
    SparseCholeskySolver<Eigen::SparseMatrix<double> > chol_R_xy_;
    SparseCholeskySolver<Eigen::VectorXd > chol_w_;
    VerifiedCholeskySolver<Eigen::VectorXd > chol_n_;
    VerifiedCholeskySolver<Eigen::VectorXd > chol_nr_;
    
    Eigen::VectorXd w_;
};

template <class K>
SparseRegressor<K>::SparseRegressor(uint trainLength, uint testLength, const SparseSettings& settings)
  : n_(trainLength),
    mu_pi_(trainLength),
    beta_(trainLength),
    embed_y_(trainLength),
    chol_g_xx_(settings.epsilon_min),
    chol_g_xx_lr_(int(trainLength*0.1), int(trainLength*0.1),1, settings.epsilon_min),
    chol_ur_lr_(int(trainLength*0.1), int(trainLength*0.1),1, settings.delta_min ),
    chol_u_(settings.epsilon_min),
    chol_ur_(settings.delta_min),
    chol_R_xy_(settings.delta_min),
    chol_w_(settings.delta_min),
    chol_n_(int(trainLength*0.1), int(trainLength*0.1),1, settings.epsilon_min),
    chol_nr_(int(trainLength*0.1), int(trainLength*0.1),1, settings.delta_min),
    settings_(settings),
    w_(trainLength){}

template <class K>
void SparseRegressor<K>::operator()(const TrainingData& data, 
                          const Kernel<K, Eigen::SparseMatrix<double> >& kx,
                          const Kernel<K, Eigen::SparseMatrix<double> >& ky, 
                          const Eigen::MatrixXd& ys,
                          double lowRankScale,
                          double lowRankWeight,
                          Eigen::MatrixXd& weights)
{
  const Eigen::MatrixXd& x = data.x;
  const Eigen::MatrixXd& y = data.y;
  //low rank stuff 
  int columns = n_*0.2;
  int rank = int(columns*0.9);
  Eigen::MatrixXd C(n_, columns);
  Eigen::MatrixXd W(columns, columns);
  double sigma_lrx = kx.width() * lowRankScale;
  double sigma_lry = ky.width() * lowRankScale;
  Kernel<K, Eigen::SparseMatrix<double> > kx_lr(data.x, sigma_lrx);
  Kernel<K, Eigen::SparseMatrix<double> > ky_lr(data.y, sigma_lry);

  std::cout << "kx fill:" << kx.gramMatrix().nonZeros() / double(n_*n_) * 100 << std::endl;
  std::cout << "ky fill:" << ky.gramMatrix().nonZeros() / double(n_*n_) * 100 << std::endl;

  //compute prior embedding
  kx.embed(data.u, data.lambda, mu_pi_);
  
  if (settings_.method == "sparse")
  {
    chol_g_xx_.solve(kx.gramMatrix(), mu_pi_, beta_);
  }
  else if (settings_.method == "lowrank")
  {
    simpleNystromApproximation(data.x, kx_lr, columns, C, W);
    VerifiedCholeskySolver<Eigen::MatrixXd> quickchol(columns,columns,n_, settings_.epsilon_min);
    Eigen::MatrixXd T(columns, n_);
    quickchol.solve(W, C.transpose(), T);
    chol_g_xx_lr_.solve(C * T, mu_pi_, beta_);
  }
  else
  {
    Eigen::VectorXd L(n_);
    chol_g_xx_.solve( kx.gramMatrix(), mu_pi_, L);
    L *= 1.0 / (1.0 - lowRankWeight);
    // add low rank update with bigger kernel.
    Eigen::MatrixXd M(n_, columns);  
    simpleNystromApproximation(data.x, kx_lr, columns, C, W);
    chol_u_.solve(kx.gramMatrix(), C , M);
    M *= lowRankWeight / (1.0 - lowRankWeight);
    Eigen::VectorXd N(n_);
    chol_n_.solve(W + C.transpose() * M, C.transpose()*L, N);
    beta_ = L - M*N;
  }
  
  beta_ = beta_.cwiseMax(0.0);
  beta_ = beta_ / beta_.sum();
  
  std::vector< Eigen::Triplet<double> > coeffs;
  Eigen::SparseMatrix<double> beta_diag_(n_,n_);
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
    if (settings_.method == "sparse")
    {
      chol_w_.solve(beta_diag_ * ky.gramMatrix(), embed_y_, w_);
    }
    else if (settings_.method == "lowrank")
    {
      VerifiedCholeskySolver<Eigen::MatrixXd> quickchol(columns,columns,n_, settings_.delta_min);
      Eigen::MatrixXd T(columns, n_);
      quickchol.solve(W, C.transpose(), T);
      chol_g_xx_lr_.solve(beta_.asDiagonal()*C * T, embed_y_, w_);
    } 
    else
    {
      Eigen::VectorXd L(n_);
      Eigen::MatrixXd M(n_, columns);  
      Eigen::VectorXd N(n_);
      chol_w_.solve(beta_diag_ * ky.gramMatrix(), embed_y_, L);
      L *= 1.0 / (1.0 - lowRankWeight);
      //low rank section 
      simpleNystromApproximation(data.x, kx_lr, columns, C, W);
      chol_ur_.solve(beta_diag_ * ky.gramMatrix(), beta_.asDiagonal() * C , M);
      M *= lowRankWeight / (1.0 - lowRankWeight);
      chol_nr_.solve(W + C.transpose() * M, C.transpose()*L, N);
      w_ = L - M*N;
    }
    
    w_ = w_.cwiseMax(0.0);
    if (w_.sum() > 0.0)
    {
      w_ = w_ / w_.sum();
    }
    weights.row(i) = w_;
  }
}

