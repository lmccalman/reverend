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
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      double preimage_reg = x[2];
      algo_(trainingData_, kx_, ky_, testingData_.ys, weights_);
      uint testPoints = testingData_.xs.rows();
      double totalCost = 0.0;
      double dim = trainingData_.x.cols();
      for (int i=0;i<testPoints;i++)
      {
        positiveNormedCoeffs(weights_.row(i), kx_, dim, preimage_reg, posWeights_);
        totalCost += logKernelMixture(testingData_.xs.row(i),
            trainingData_.x,
            weights_.row(i),
            kx_, true);
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
      preimageWeights_(test.ys.rows(), train.x.rows()),
      regressor_(train.x.rows(), train.u.rows(), settings),
      settings_(settings)
      {
        //compute the values
        regressor_(trainingData_, kx_, ky_, testingData_.ys, weights_);
      }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = kx_.width();
      uint n = trainingData_.x.rows();
      double preimage_reg = x[0];
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
      double tau = settings_.quantile;
      for (int i=0;i<testPoints;i++)
      {
        if (settings_.cost_function == "pinball")
        {
          Quantile<Kernel<K> > q(preimageWeights_.row(i), trainingData_.x,
              kx_, true);
          double z = q(tau);
          double y = testingData_.xs(i,0);
          totalCost += pinballLoss(z,y,tau);
        }
        else
        {
          //note the minus to minimise negative probability
          totalCost -= logKernelMixture(testingData_.xs.row(i),
              trainingData_.x,
              preimageWeights_.row(i),
              kx_, false);
        }
      }
      return totalCost;
    };

  private: 
    Kernel<K> kx_;
    Kernel<K> ky_;
    Eigen::MatrixXd weights_;
    Eigen::MatrixXd preimageWeights_;
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
        settings_(settings){};

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, weights_);
      uint testPoints = testingData_.xs.rows();
      double tau = settings_.quantile;
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        Quantile<Kernel<K> > q(weights_.row(i), trainingData_.x,
                               kx_, settings_.normed_weights);
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
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      double preimage_reg = x[2];
      uint dim = trainingData_.x.cols();
      algo_(trainingData_, kx_, ky_, testingData_.ys, weights_);
      uint testPoints = testingData_.xs.rows();
      double tau = settings_.quantile;
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        if (settings_.normed_weights)
        {
          std::cout << "WARNING--  NORMED WEIGHTS WITH JOINT PINBALL IS EQUIVALENT TO PINBALL" << std::endl;
          posWeights_ = weights_;
        }
        else
        {
          positiveNormedCoeffs(weights_.row(i), kx_, 
                              dim, preimage_reg, posWeights_);
        }
        Quantile<Kernel<K> > q(posWeights_, trainingData_.x, kx_, true);
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
    bool normedWeights_;
};
