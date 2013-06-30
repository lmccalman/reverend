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
#include <iomanip>
#include <algorithm>
#include <vector>

//This must be inherited by anything that we want to optimize with NLOpt
struct NloptCost
{
  public:
    virtual double operator()(const std::vector<double>&x, std::vector<double>&grad) = 0;
};

void ithFoldTestIndices(uint k, uint i, uint n, uint& lowIndex, uint& highIndex)
//Indices are INCLUSIVE
{
  assert(i < k);
  uint smallFoldSize = n/k;
  uint bigFoldSize = smallFoldSize + 1;
  uint bigFolds = n%k;
  if (i < bigFolds)
  {
    lowIndex = i*bigFoldSize;
    highIndex = (i+1)*bigFoldSize - 1;
  }
  else
  {
    uint idash = i - bigFolds;
    lowIndex = bigFolds*bigFoldSize + idash*smallFoldSize;
    highIndex = bigFolds*bigFoldSize + (idash+1)*smallFoldSize - 1;
  }
}

void ithFoldSizes(uint k, uint i, uint n, uint& trainSize, uint& testSize)
{
  uint lowIndex;
  uint highIndex;
  ithFoldTestIndices(k, i, n, lowIndex, highIndex);
  testSize = highIndex - lowIndex + 1;
  trainSize = n - testSize;
}

void fillSubset(uint k, uint ithFold, const Eigen::MatrixXd& fullMat,
                Eigen::MatrixXd& training, Eigen::MatrixXd& testing)
{
  assert(training.rows() + testing.rows() == fullMat.rows());
  assert(fullMat.cols() == training.cols());
  assert(fullMat.cols() == testing.cols());
  uint n = fullMat.rows();
  uint lowIndex;
  uint highIndex;
  ithFoldTestIndices(k, ithFold, n, lowIndex, highIndex);
  uint testSize = highIndex - lowIndex + 1;
  assert (testing.rows() == testSize);
  uint preSplitSize = lowIndex;
  uint postSplitSize = n - preSplitSize - testSize;
  assert(preSplitSize + testSize + postSplitSize == n);
  assert(preSplitSize + postSplitSize == training.rows());
  //testing matrix
  testing = fullMat.block(lowIndex,0,testSize,fullMat.cols());
  //training matrix
  if (ithFold > 0)
  {
    training.block(0,0, preSplitSize, fullMat.cols()) 
      = fullMat.block(0,0, preSplitSize, fullMat.cols());
  }
  if (ithFold < k-1)
  {
    uint postSplitStart = lowIndex;  
    training.block(postSplitStart,0, postSplitSize, fullMat.cols())
      = fullMat.block(highIndex+1,0, postSplitSize, fullMat.cols());
  }
}

void kFoldData(uint k, const TrainingData& allData, std::vector<TrainingData>& foldTraining,
    std::vector<TestingData>& foldTesting)
{
  uint n = allData.x.rows();
  uint m = allData.u.rows();
  assert(n == m);
  assert(n == allData.y.rows());
  uint dx = allData.x.cols();
  uint dy = allData.y.cols();
  uint du = allData.u.cols();
  assert(dx == du);
  for (int i=0; i<k; i++)
  {
    uint trainSize;
    uint testSize;
    ithFoldSizes(k,i,n,trainSize,testSize);
    Eigen::MatrixXd x(trainSize, dx);
    Eigen::MatrixXd xs(testSize, dx);
    Eigen::MatrixXd y(trainSize, dy);
    Eigen::MatrixXd ys(testSize, dy);
    Eigen::MatrixXd u(trainSize, du);
    Eigen::MatrixXd us(testSize, du);
    fillSubset(k, i, allData.x, x, xs);
    fillSubset(k, i, allData.y, y, ys);
    fillSubset(k, i, allData.u, u, us);
    Eigen::VectorXd lambda = Eigen::VectorXd::Ones(trainSize);
    lambda = lambda / double(trainSize);
    if (allData.xtp1.rows() > 0)
    {
      Eigen::MatrixXd xtp1(trainSize, dx);
      Eigen::MatrixXd xtp1s(testSize, dx);
      fillSubset(k, i, allData.xtp1, xtp1,xtp1s);
      foldTraining.push_back(TrainingData(u, lambda, x, y, xtp1)); 
    }
    else
    {
      foldTraining.push_back(TrainingData(u, lambda, x, y)); 
    }
    foldTesting.push_back(TestingData(xs, ys)); 
  }
}

//A K-fold cross validating cost function for my nlopt wrapper
template <class T>
struct KFoldCVCost : NloptCost
{
  public:
    KFoldCVCost(uint k, const TrainingData& data, const Settings& settings)
    {
      n_ = data.x.rows();
      k_ = k;
      //initialise k-fold data
      kFoldData(k, data, trainingFolds_, testingFolds_);
      //initialise cost functions
      for (uint i=0;i<k;i++)
      {
        rawCostFunctions_.push_back(T(trainingFolds_[i], testingFolds_[i], settings)); 
      } 

    };
    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double totalCost = 0.0;
      #pragma omp parallel for reduction(+:totalCost)
      for (uint i=0;i<k_;i++)
      {
        totalCost += rawCostFunctions_[i](x, grad);
      }
      totalCost = totalCost / double(n_);
      std::cout << "[ "; 
      for (uint i=0;i<x.size();i++)
      {
        std::cout << std::setw(10) << x[i] << " ";
      }
      std::cout << " ] cost:" << totalCost << std::endl;
      return totalCost;
    };
  protected:
    std::vector<T> rawCostFunctions_;
    std::vector<TrainingData> trainingFolds_;
    std::vector<TestingData> testingFolds_;
    uint n_;
    uint k_;
};
