#pragma once
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
#define _USE_MATH_DEFINES
#include <cmath>
#include "regressor.hpp"
#include "filter.hpp"
#include "preimage.hpp"
#include "distrib.hpp"
#include "cumulative.hpp"


//This is a 'Raw' cost function, which must be wrapped in some way to become an
//NloptCost for use with the optimizer. Usually this would involve a k-fold or
//LOO cross validator
struct Cost
{
  public:
    Cost(const TrainingData& train, const TestingData& test)
      : trainingData_(train), testingData_(test){}; 
    virtual double operator()(const std::vector<double>&x, std::vector<double>&grad) = 0;
  protected:
    const TrainingData& trainingData_;
    const TestingData& testingData_;
};

//This is my particular 'Raw' cost function
template <class T, class K>
class LogPCost:Cost
{
  public:
   LogPCost(const TrainingData& train, const TestingData& test, const Settings& settings)
      : Cost(train, test), 
      algo_(train.x.rows(), train.u.rows(), settings),
      weights_(test.ys.rows(), train.x.rows()),
      kx_(train.x, 1.0e-10), ky_(train.y, 1.0e-10)
  {
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      double lowRankScale = x[2];
      double lowRankWeight = x[3];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, lowRankScale,
          lowRankWeight, weights_);
      uint testPoints = testingData_.xs.rows();
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        totalCost += multiLogKernelMixture(testingData_.xs.row(i),
                                            trainingData_.x,
                                            weights_.row(i),
                                            kx_,
                                            lowRankScale,
                                            lowRankWeight);
      }
      totalCost *= -1; // minimize this maximizes probability
      return totalCost;
    };
  
  private: 
    Kernel<K> kx_;
    Kernel<K> ky_;
    T algo_;
    Eigen::MatrixXd weights_;
};
