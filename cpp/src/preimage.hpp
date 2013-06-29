#pragma once
#include <iostream>
#include <Eigen/Dense>
#include "eiquadprog.hpp"
#include "data.hpp"
#include "kernel.hpp"
#include "distrib.hpp"

template <class K>
void positiveNormedCoeffs(const Eigen::VectorXd& embedding,
      const Kernel<K>& kx, uint k, double regulariser, Eigen::VectorXd& mixtureCoeffs)
{
  uint n = kx.gramMatrix().rows();
  uint p = 1;
  uint m = n;

  Eigen::MatrixXd G(n,n);
  G = kx.gramMatrix() + regulariser*Eigen::MatrixXd::Identity(n, n); 

  Eigen::VectorXd g0(n);
  g0 = -1.0 * pow(2.0, k/2.0) * embedding.transpose() * kx.gramMatrix();
  Eigen::MatrixXd CE(n,p);
  CE = Eigen::MatrixXd::Ones(n,p);
  Eigen::VectorXd ce0(p);
  ce0 = -1.0 * Eigen::VectorXd::Ones(p);
  Eigen::MatrixXd CI(n, n);
  CI = Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd ci0(m);
  ci0 = Eigen::VectorXd::Zero(m);
  Eigen::VectorXd x(n);
  double cost = solve_quadprog(G, g0,  CE, ce0,  CI, ci0, mixtureCoeffs);
}

template <class K>
void computeNormedWeights(const Eigen::MatrixXd& weights,
   const Kernel<K>& kx, uint dimension, const Settings& settings,
   Eigen::MatrixXd& preimageWeights)
{
  uint n = weights.cols();
  uint s = preimageWeights.rows();
  Eigen::VectorXd coeff_i(n);
  Eigen::MatrixXd regularisedGxx(n,n);
  for (int i=0; i<s; i++)
  {
    coeff_i = Eigen::VectorXd::Ones(n) * (1.0/double(n));
    positiveNormedCoeffs(weights.row(i), kx, dimension, settings.preimage_reg, coeff_i);
    preimageWeights.row(i) = coeff_i;
  }
}

void computePosterior(const Eigen::MatrixXd& xs,
    const Eigen::MatrixXd& x, const Eigen::MatrixXd& weights,
    const Settings& settings, Eigen::MatrixXd& posterior)
{
  for (int i=0; i<xs.rows();i++)
  {
    for (int j=0; j<posterior.cols();j++)
    {
      posterior(i,j) = std::exp(logGaussianMixture(xs.row(i),
                        x, weights.row(j), settings.sigma_x));
    }
  }
}



