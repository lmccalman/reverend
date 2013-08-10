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

double pinballLoss(double z, double y, double tau)
{
  double result = 0.0;
  if (y >= z)
  {
    result = (y-z)*tau;
  }
  else
  {
    result= (z-y)*(1.0 - tau);
  }
  return result;
}

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
      kx_(train.x, 1.0), ky_(train.y, 1.0),
      algo_(train.x.rows(), train.u.rows(), settings),
      weights_(test.ys.rows(), train.x.rows())
  {
    if (settings.normed_weights)
    {
      std::cout << "Initializing LogP cost with Normed weights..." << std::endl;
    }
    else
    {
      std::cout << "Initializing LogP cost with Un-normed weights..." << std::endl;
    }
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      double epsilon_min = exp(x[2]);
      double delta_min = exp(x[3]);
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, epsilon_min, delta_min, weights_);
      uint testPoints = testingData_.xs.rows();
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        totalCost += logKernelMixture(testingData_.xs.row(i),
                                      trainingData_.x,
                                      weights_.row(i),
                                      kx_, true);
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
class JointLogPCost:Cost
{
  public:
    JointLogPCost(const TrainingData& train, const TestingData& test, const Settings& settings)
      : Cost(train, test), 
      algo_(train.x.rows(), train.u.rows(), settings),
      weights_(test.ys.rows(), train.x.rows()),
      posWeights_(train.x.rows()),
      kx_(train.x, 1.0), ky_(train.y, 1.0)

  {
      std::cout << "Initializing Joint LogP cost..." << std::endl;
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      double epsilon_min = exp(x[2]);
      double delta_min = exp(x[3]);
      double preimage_reg = exp(x[4]);
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, epsilon_min, delta_min, weights_);
      uint testPoints = testingData_.xs.rows();
      Eigen::MatrixXd A = AMatrix(trainingData_.x, kx_, sigma_x);
      Eigen::MatrixXd B = BMatrix(trainingData_.x, kx_, sigma_x);
      double totalCost = 0.0;
      double dim = trainingData_.x.cols();
      for (int i=0;i<testPoints;i++)
      {
        positiveNormedCoeffs(weights_.row(i),A,B, preimage_reg, posWeights_);
        totalCost += logKernelMixture(testingData_.xs.row(i),
            trainingData_.x, posWeights_, kx_, true);
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
    std::cout << "Initializing Hilbert cost..." << std::endl;
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      double epsilon_min = exp(x[2]);
      double delta_min = exp(x[3]);
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, epsilon_min, delta_min, weights_);
      uint testPoints = testingData_.xs.rows();
      uint n = trainingData_.x.rows();
      Eigen::VectorXd pointEmbedding(n);
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        kx_.embed(testingData_.xs.row(i), pointEmbedding);
        totalCost += (weights_.row(i).transpose() * kx_.gramMatrix()  * weights_.row(i)
              -2 * weights_.row(i).transpose() * kx_.gramMatrix() * pointEmbedding)(0,0);
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
      kx_(train.x, settings.sigma_x),
      ky_(train.y, settings.sigma_y),
      weights_(test.ys.rows(), train.x.rows()),
      posWeights_(train.x.rows()),
      regressor_(train.x.rows(), train.u.rows(), settings),
      settings_(settings)
      {
        std::cout << "Initializing Preimage cost..." << std::endl;
        regressor_(trainingData_, kx_, ky_, testingData_.ys, settings.epsilon_min,
           settings.delta_min, weights_);
      }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = kx_.width();
      uint n = trainingData_.x.rows();
      double preimage_reg = exp(x[0]);
      uint dim = trainingData_.x.cols();
      Eigen::VectorXd coeff_i(n);
      uint testPoints = testingData_.xs.rows();
      double totalCost = 0.0;
      Eigen::MatrixXd A = AMatrix(trainingData_.x, kx_, sigma_x);
      Eigen::MatrixXd B = BMatrix(trainingData_.x, kx_, sigma_x);
      for (int i=0; i<testPoints; i++)
      {
        positiveNormedCoeffs(weights_.row(i),A, B, preimage_reg, posWeights_);
        totalCost += logKernelMixture(testingData_.xs.row(i),
            trainingData_.x, posWeights_, kx_, true);
      }
      return -1*totalCost;
    };

  private: 
    Kernel<K> kx_;
    Kernel<K> ky_;
    Eigen::MatrixXd weights_;
    Eigen::VectorXd posWeights_;
    Regressor<K> regressor_;
    const Settings& settings_;
};

template <class T, class K>
class PinballCost:Cost
{
  public:
    PinballCost(const TrainingData& train, const TestingData& test,
                const Settings& settings)
      : Cost(train, test), 
        kx_(train.x, 1.0), ky_(train.y, 1.0),
        algo_(train.x.rows(), train.u.rows(), settings),
        weights_(test.ys.rows(), train.x.rows()),
        settings_(settings)
  {
    if (settings.normed_weights)
    {
      std::cout << "Initializing Pinball cost with Normed quantiles..." << std::endl;
    }
    else
    {
      std::cout << "Initializing Pinball cost with Direct quantiles..." << std::endl;
    }
  };

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      double epsilon_min = exp(x[2]);
      double delta_min = exp(x[3]);
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, epsilon_min, delta_min, weights_);
      uint testPoints = testingData_.xs.rows();
      double tau = settings_.quantile;
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        Quantile<Kernel<K> > q(weights_.row(i), trainingData_.x,
                               kx_, settings_.cumulative_mean_map);
        double z = q(tau);
        double y = testingData_.xs(i,0);
        totalCost += pinballLoss(z,y,tau);
      }
      return totalCost;
    };

  private: 
    Kernel<K> kx_;
    Kernel<K> ky_;
    T algo_;
    Eigen::MatrixXd weights_;
    const Settings& settings_;
};

template <class T, class K>
class JointPinballCost:Cost
{
  public:
    JointPinballCost(const TrainingData& train, const TestingData& test,
                     const Settings& settings)
      : Cost(train, test), 
        algo_(train.x.rows(), train.u.rows(), settings),
        weights_(test.ys.rows(), train.x.rows()),
        posWeights_(train.x.rows()),
        kx_(train.x, 1.0), ky_(train.y, 1.0),
        settings_(settings)
  {
    std::cout << "Initializing Joint Pinball cost with Normed quantiles..." << std::endl;
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      double epsilon_min = exp(x[2]);
      double delta_min = exp(x[3]);
      double preimage_reg = exp(x[4]);
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      uint dim = trainingData_.x.cols();
      algo_(trainingData_, kx_, ky_, testingData_.ys, epsilon_min, delta_min, weights_);
      uint testPoints = testingData_.xs.rows();
      Eigen::MatrixXd A = AMatrix(trainingData_.x, kx_, sigma_x);
      Eigen::MatrixXd B = BMatrix(trainingData_.x, kx_, sigma_x);
      double tau = settings_.quantile;
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        positiveNormedCoeffs(weights_.row(i), A, B, preimage_reg, posWeights_);
        Quantile<Kernel<K> > q(posWeights_, trainingData_.x, kx_, settings_.cumulative_mean_map);
        double z = q(tau);
        double y = testingData_.xs(i,0);
        totalCost += pinballLoss(z,y,tau);
      }
      return totalCost;
    };

  private: 
    T algo_;
    Eigen::MatrixXd weights_;
    Eigen::VectorXd posWeights_;
    Kernel<K> kx_;
    Kernel<K> ky_;
    const Settings& settings_;
};
