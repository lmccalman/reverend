#pragma once
#include <iostream>
#include <Eigen/Dense>

#include "eiquadprog.hpp"

void positiveNormedCoeffs(const Eigen::VectorXd& embedding,
    const Eigen::MatrixXd& g_xx, uint k, double regulariser, Eigen::VectorXd& mixtureCoeffs)
{
  uint n = g_xx.rows();
  uint p = 1;
  uint m = n;

  Eigen::MatrixXd G(n,n);
  G = g_xx + regulariser*Eigen::MatrixXd::Identity(n, n); 

  Eigen::VectorXd g0(n);
  g0 = -1.0 * pow(2.0, k/2.0) * embedding.transpose() * g_xx;
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

