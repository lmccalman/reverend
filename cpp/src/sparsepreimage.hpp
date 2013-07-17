#pragma once
#include <iostream>
#include <Eigen/Dense>
#include "eiquadprog.hpp"
#include "data.hpp"
#include "kernel.hpp"
#include "distrib.hpp"

template <class K>
void computeSparseEmbedding(const TrainingData& trainingData, const TestingData& testingData,
    const Eigen::MatrixXd& weights, const K& kx, Eigen::MatrixXd& embedding, 
    double lowRankScale, double lowRankWeight)
{
  uint s = weights.rows(); 
  uint p = testingData.xs.rows();
  uint n = trainingData.x.rows();
  double sigma = kx.width();
  Kernel<Q1CompactKernel, Eigen::SparseMatrix<double> > kx_lr(trainingData.x, sigma*lowRankScale);
  uint dim = trainingData.x.cols();
  for (int i=0;i<s;i++)  
  {
    for (int j=0;j<p;j++)
    {
      Eigen::VectorXd w_i = weights.row(i);
      Eigen::VectorXd testpoint = testingData.xs.row(j);
      double result = 0.0;
      for (int k=0;k<n;k++)
      {
        result += (1.0 - lowRankWeight) * w_i(k) 
                  * kx(trainingData.x.row(k), testpoint) 
                   / kx.volume(); 
        result += lowRankWeight  * w_i(k) 
                  * kx_lr(trainingData.x.row(k), testpoint)
                  / kx_lr.volume(); 
      } 
      embedding(i,j) = result;
    }
  }
}



