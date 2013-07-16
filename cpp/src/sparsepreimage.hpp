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
  Kernel<Q1CompactKernel> kx_lr(trainingData.x);
  uint dim = trainingData.x.cols();
  double sigma = kx.width();
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
                   / kx.volume(sigma, dim); 
        result += lowRankWeight  * w_i(k) 
                  * kx_lr(trainingData.x.row(k), testpoint, sigma*lowRankScale)
                  / kx_lr.volume(sigma*lowRankScale, dim); 
      } 
      embedding(i,j) = result;
    }
  }
}



