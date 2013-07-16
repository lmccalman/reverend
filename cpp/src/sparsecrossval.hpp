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
#include "crossval.hpp"

//A K-fold cross validating cost function for my nlopt wrapper
template <class T>
struct SparseKFoldCVCost : NloptCost
{
  public:
    KFoldCVCost(uint k, const TrainingData& data, const Settings& settings)
      : data_(data), settings_(settings), k_(k), n_(data_.x.rows()){}
    
    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      //initialise k-fold data
      kFoldData(k_, data_, trainingFolds_, testingFolds_);
      T* costfn; 
      double totalCost = 0.0;
      //initialise cost functions
      for (uint i=0;i<k_;i++)
      {
        costfn = new T(trainingFolds_[i], testingFolds_[i], settings_); 
        totalCost += (*costfn)(x, grad);
        delete costfn;
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
    std::vector<TrainingData> trainingFolds_;
    std::vector<TestingData> testingFolds_;
    const TrainingData& data_;
    const Settings& settings_;
    uint n_;
    uint k_;
};
