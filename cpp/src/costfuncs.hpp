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
      kx_(train.x, 1.0), ky_(train.y, 1.0)
  {
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, weights_);
      uint testPoints = testingData_.xs.rows();
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        totalCost += logGaussianMixture(testingData_.xs.row(i),
                                        trainingData_.x,
                                        weights_.row(i),
                                        sigma_x);
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


template <class T, class K>
class JointCost:Cost
{
  public:
    JointCost(const TrainingData& train, const TestingData& test, const Settings& settings)
      : Cost(train, test), 
      algo_(train.x.rows(), train.u.rows(), settings),
      weights_(test.ys.rows(), train.x.rows()),
      posWeights_(train.x.rows()),
      kx_(train.x, 1.0), ky_(train.y, 1.0)

  {
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      double preimage_reg = exp(x[2]);
      algo_(trainingData_, kx_, ky_, testingData_.ys, weights_);
      uint testPoints = testingData_.xs.rows();
      double totalCost = 0.0;
      double dim = trainingData_.x.cols();
      for (int i=0;i<testPoints;i++)
      {
        positiveNormedCoeffs(weights_.row(i), kx_, dim, preimage_reg, posWeights_);
        totalCost += logGaussianMixture(testingData_.xs.row(i),
            trainingData_.x,
            weights_.row(i),
            sigma_x);
      }
      totalCost *= -1; // minimize this maximizes probability
      return totalCost;

    };

  private: 
    T algo_;
    Eigen::MatrixXd weights_;
    Eigen::VectorXd posWeights_;
    Kernel<K> kx_;
    Kernel<K> ky_;
};


template <class T, class K>
struct HilbertCost:Cost
{
  public:
    HilbertCost(const TrainingData& train, const TestingData& test, const Settings& settings)
      : Cost(train, test), 
      algo_(train.x.rows(), train.u.rows(), settings),
      weights_(test.ys.rows(), train.x.rows()),
      kx_(train.x, 1.0), ky_(train.y, 1.0)
  {
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, weights_);
      uint testPoints = testingData_.xs.rows();
      uint n = trainingData_.x.rows();
      Eigen::VectorXd pointEmbedding(n);
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        kx_.embed(testingData_.xs.row(i), pointEmbedding);
        totalCost += kx_.innerProduct(weights_.row(i), weights_.row(i))
              -2 * kx_.innerProduct(weights_.row(i), pointEmbedding);
      }
      return totalCost;
    };

  private: 
    T algo_;
    Eigen::MatrixXd weights_;
    Kernel<K> kx_;
    Kernel<K> ky_;
};


template <class K>
class PreimageCost:Cost
{
  public:
    PreimageCost(const TrainingData& train, const TestingData& test, const Settings& settings)
      : Cost(train, test), 
      regressor_(train.x.rows(), train.u.rows(), settings),
      weights_(test.ys.rows(), train.x.rows()),
      preimageWeights_(test.ys.rows(), train.x.rows()),
      kx_(train.x, settings.sigma_x), ky_(train.y, settings.sigma_y)
      {
        regressor_(trainingData_, kx_, ky_, testingData_.ys, weights_);
      }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = kx_.width();
      uint n = trainingData_.x.rows();
      double preimage_reg = exp(x[0]);
      uint dim = trainingData_.x.cols();
      Eigen::VectorXd coeff_i(n);
      uint testPoints = testingData_.xs.rows();
      for (int i=0; i<testPoints; i++)
      {
        coeff_i = Eigen::VectorXd::Ones(n) * (1.0/double(n));
        positiveNormedCoeffs(weights_.row(i), kx_, dim, preimage_reg, coeff_i);
        preimageWeights_.row(i) = coeff_i;
      }
      
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        totalCost += logGaussianMixture(testingData_.xs.row(i),
            trainingData_.x,
            preimageWeights_.row(i),
            sigma_x);
      }
      totalCost *= -1; // minimize this maximizes probability
      return totalCost;
    };

  private: 
    Regressor<K> regressor_;
    Eigen::MatrixXd weights_;
    Eigen::MatrixXd preimageWeights_;
    Kernel<K> kx_;
    Kernel<K> ky_;
};
