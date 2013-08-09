#pragma once
#include <iostream>
#include <Eigen/Dense>
#include "data.hpp"
#include "kernel.hpp"
#include "distrib.hpp"

template <class K>
Eigen::MatrixXd AMatrix(const Eigen::MatrixXd& means,
                        const Kernel<K>& kx, double preimageSigma)
{
  uint n = means.rows();
  double inva =  kx.volume();
  double sigma = kx.width();
  double sigmaDash = sqrt(2*sigma*sigma + preimageSigma*preimageSigma);
  Eigen::MatrixXd A(n,n);
  for (uint i=0; i<n;i++)
  {
    for (uint j=0; j<n;j++)
    {
      A(i,j) = inva * multivariateSymmetricGaussian(means.row(i), means.row(j),sigmaDash);
    }
  }
  return A;
}

template <class K>
Eigen::MatrixXd BMatrix(const Eigen::MatrixXd& means,
    const Kernel<K>& kx, double preimageSigma)
{
  uint n = means.rows();
  double inva =  kx.volume();
  double sigma = kx.width();
  double sigmaDash = sqrt(sigma*sigma + preimageSigma*preimageSigma);
  Eigen::MatrixXd B(n,n);
  for (uint i=0; i<n;i++)
  {
    for (uint j=0; j<n;j++)
    {
      B(i,j) = inva * multivariateSymmetricGaussian(means.row(i), means.row(j),sigmaDash);
    }
  }
  return B;
}

// void positiveNormedCoeffs(const Eigen::VectorXd& embedding,
                          // const Eigen::MatrixXd& A,
                          // const Eigen::MatrixXd& B,  
                          // double regulariser, Eigen::VectorXd& mixtureCoeffs)
// {
  // uint n = A.rows();
  // uint p = 1;
  // uint m = n;

  // Eigen::MatrixXd G(n,n);
  // Eigen::VectorXd g0(n);
  // G = A + regulariser*Eigen::MatrixXd::Identity(n, n); 
  // // G = A;
  // g0 = -1.0 * embedding.transpose() * B;
  // Eigen::MatrixXd CE(n,p);
  // CE = Eigen::MatrixXd::Ones(n,p);
  // Eigen::VectorXd ce0(p);
  // ce0 = -1.0 * Eigen::VectorXd::Ones(p);
  // Eigen::MatrixXd CI(n, n);
  // CI = Eigen::MatrixXd::Identity(n, n);
  // Eigen::VectorXd ci0(m);
  // ci0 = Eigen::VectorXd::Zero(m);
  // Eigen::VectorXd x(n);
  // mixtureCoeffs = embedding.cwiseMax(0.0);
  // mixtureCoeffs = mixtureCoeffs / mixtureCoeffs.sum();
  // solve_quadprog(G, g0,  CE, ce0,  CI, ci0, mixtureCoeffs);
  // bool resultOK = true;
  // for (int i=0;i<n; i++)
  // {
    // if (mixtureCoeffs(i) < 0.0)
    // {
      // resultOK = false;
      // break;
    // }
  // }
  // if (!resultOK)
  // {
    // mixtureCoeffs = mixtureCoeffs.cwiseMax(0.0);
    // mixtureCoeffs = mixtureCoeffs / mixtureCoeffs.sum();
  // }
// }

struct PreimageData
{
  PreimageData(Eigen::MatrixXd& inG, Eigen::VectorXd& ing0)
    : G(inG), g0(ing0){}
  Eigen::MatrixXd& G;
  Eigen::VectorXd& g0;
};

double preimageCostFn(const std::vector<double> &x, std::vector<double>&grad, void* data)
{
  PreimageData* d = reinterpret_cast<PreimageData*>(data);
  uint n = x.size();
  Eigen::VectorXd vx(n);
  Eigen::VectorXd vgrad(n);
  for (int i=0;i<n;i++)
  {
    vx(i) = x[i];
  }
  Eigen::MatrixXd cost = 0.5 * vx.transpose() * d->G * vx + d->g0.transpose() * vx;
  if (!grad.empty())
  {
    vgrad = d->G * vx + d->g0;
    for (int i=0;i<n;i++)
    {
      grad[i] = vgrad(i);
    }
  }
  return cost(0,0);
}

double preimageConstraint(const std::vector<double> &x, std::vector<double>&grad, void* data)
{
  uint n = x.size();
  double total = 0.0;
  for (int i=0;i<n;i++)
  {
    total += x[i];
  }
  double result = total - 1.0;
  if (!grad.empty())
  {
    for (int i=0;i<n;i++)
    {
      grad[i] = 1.0;
    }
  }
  return result;
}

void positiveNormedCoeffs(const Eigen::VectorXd& embedding,
                                const Eigen::MatrixXd& A,
                                const Eigen::MatrixXd& B,  
    double regulariser, Eigen::VectorXd& mixtureCoeffs)
{
  uint n = A.rows();
  Eigen::MatrixXd G(n,n);
  Eigen::VectorXd g0(n);
  G = A + regulariser*Eigen::MatrixXd::Identity(n, n); 
  // G = A;
  g0 = -1.0 * embedding.transpose() * B;
  PreimageData data(G,g0);
  
  mixtureCoeffs = embedding.cwiseMax(0.0);
  mixtureCoeffs = mixtureCoeffs / mixtureCoeffs.sum();
  
  std::vector<double> thetaMin(n); 
  std::vector<double> x(n); 
  for (int i=0;i<n;i++) 
  {
    thetaMin[i] = 0.0;
    x[i] = mixtureCoeffs(i);
  }
  
  nlopt::opt opt(nlopt::LD_MMA, n);
  opt.set_lower_bounds(thetaMin);
  opt.set_min_objective(preimageCostFn, &data);
  opt.set_maxtime(10.0); // if it's longer than this we're prolly stuffed anyway
  // opt.add_equality_constraint(preimageConstraint, NULL, 1e-6);
  opt.set_xtol_rel(1e-4);
  double minf;
  nlopt::result result = opt.optimize(x, minf);
  double total = 0.0;
  for (int i=0;i<n; i++)
  {
    mixtureCoeffs(i) = x[i];
  }
  mixtureCoeffs = mixtureCoeffs.cwiseMax(0.0);
  mixtureCoeffs = mixtureCoeffs / mixtureCoeffs.sum();
}


template <class K>
void computeNormedWeights( const Eigen::MatrixXd& X,
                            const Eigen::MatrixXd& weights,
                           const Kernel<K>& kx, const Settings& settings,
                           Eigen::MatrixXd& preimageWeights)
{
  uint n = weights.cols();
  uint s = preimageWeights.rows();
  Eigen::VectorXd coeff_i(n);
  Eigen::MatrixXd regularisedGxx(n,n);
  Eigen::MatrixXd A = AMatrix(X, kx, kx.width());
  Eigen::MatrixXd B = BMatrix(X, kx, kx.width());
  for (int i=0; i<s; i++)
  {
    coeff_i = Eigen::VectorXd::Ones(n) * (1.0/double(n));
    positiveNormedCoeffs(weights.row(i),A, B, settings.preimage_reg, coeff_i);
    preimageWeights.row(i) = coeff_i;
  }
}

template <class K>
void computeEmbedding(const TrainingData& trainingData, const TestingData& testingData,
    const Eigen::MatrixXd& weights, const K& kx, Eigen::MatrixXd& embedding)
{
  uint s = weights.rows(); 
  uint p = testingData.xs.rows();
  uint n = trainingData.x.rows();
  #pragma omp parallel for
  for (int i=0;i<s;i++)  
  {
    for (int j=0;j<p;j++)
    {
      Eigen::VectorXd w_i = weights.row(i);
      Eigen::VectorXd testpoint = testingData.xs.row(j);
      double result = 0.0;
      for (int k=0;k<n;k++)
      {
        result += w_i(k) * kx(trainingData.x.row(k), testpoint); 
      } 
      embedding(i,j) = result;
    }
  }
}
  
template <class K>
void computeLogPosterior(const TrainingData& trainingData, const TestingData& testingData,
    const Eigen::MatrixXd& weights, const K& kx, Eigen::MatrixXd& embedding)
{
  uint s = weights.rows(); 
  uint p = testingData.xs.rows();
  uint n = trainingData.x.rows();
  #pragma omp parallel for
  for (int i=0;i<s;i++)  
  {
    for (int j=0;j<p;j++)
    {
      embedding(i,j) = logKernelMixture(testingData.xs.row(j),
                                        trainingData.x,
                                        weights.row(i),
                                        kx, true);
    }
  }
}
