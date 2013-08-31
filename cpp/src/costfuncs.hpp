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
#include <omp.h>
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

template <class T, class K>
class LogPCost:Cost
{
  public:
    LogPCost(const TrainingData& train, const TestingData& test, const Settings& settings)
      : Cost(train, test), 
      algo_(train.x.rows(), train.u.rows(), settings),
      weights_(test.ys.rows(), train.x.rows()),
      posWeights_(train.x.rows()),
      kx_(train.x, settings.sigma_x), ky_(train.y, settings.sigma_y),
      settings_(settings)
    {
      std::cout << "Initializing LogP cost..." << std::endl;
    }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      uint dx = trainingData_.x.cols();
      uint dy = trainingData_.y.cols();
      Eigen::VectorXd sigma_x(dx);
      for (int i=0; i<dx;i++)
      {
        sigma_x(i) = x[i];
      }
      Eigen::VectorXd sigma_y(dy);
      for (int i=0; i<dy;i++)
      {
        sigma_y(i) = x[dx+i];
      }
      // double epsilon_min = exp(x[dx+dy]);
      // double delta_min = exp(x[dx+dy+1]);
      double epsilon_min = x[dx+dy];
      double delta_min = x[dx+dy+1];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, epsilon_min, delta_min, weights_);
      uint testPoints = testingData_.xs.rows();
      double totalCost = 0.0;
      if (!omp_in_parallel())
      {
        #pragma omp parallel for reduction(+:totalCost)
        for (int i=0;i<testPoints;i++)
        {
          Eigen::VectorXd localWeights(trainingData_.x.rows());
          if (settings_.normed_weights)
          {
            localWeights = weights_.row(i);
          }
          else
          {
            Eigen::MatrixXd A = AMatrix(trainingData_.x, kx_);
            Eigen::MatrixXd B = BMatrix(trainingData_.x, kx_);
            double preimage_reg = exp(x[dx+dy+2]);
            positiveNormedCoeffs(weights_.row(i),A,B, preimage_reg, localWeights);
          }
          
          totalCost += logKernelMixture(testingData_.xs.row(i),
              trainingData_.x, localWeights, kx_, true);
        }
      }
      else
      {
        for (int i=0;i<testPoints;i++)
        {
          if (settings_.normed_weights)
          {
            posWeights_ = weights_.row(i);
          }
          else
          {
            Eigen::MatrixXd A = AMatrix(trainingData_.x, kx_);
            Eigen::MatrixXd B = BMatrix(trainingData_.x, kx_);
            double preimage_reg = exp(x[dx+dy+2]);
            positiveNormedCoeffs(weights_.row(i),A,B, preimage_reg, posWeights_);
          }

          totalCost += logKernelMixture(testingData_.xs.row(i),
              trainingData_.x, posWeights_, kx_, true);
        }
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
    const Settings& settings_;
};

template <class T, class K>
class PinballCost:Cost
{
  public:
    PinballCost(const TrainingData& train, const TestingData& test, const Settings& settings)
      : Cost(train, test), 
      algo_(train.x.rows(), train.u.rows(), settings),
      weights_(test.ys.rows(), train.x.rows()),
      posWeights_(train.x.rows()),
      kx_(train.x, settings.sigma_x), ky_(train.y, settings.sigma_y),
      settings_(settings)
  {
    std::cout << "Initializing Pinball cost..." << std::endl;
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      uint dx = trainingData_.x.cols();
      uint dy = trainingData_.y.cols();
      Eigen::VectorXd sigma_x(dx);
      for (int i=0; i<dx;i++)
      {
        sigma_x(i) = x[i];
      }
      Eigen::VectorXd sigma_y(dy);
      for (int i=0; i<dy;i++)
      {
        sigma_y(i) = x[dx+i];
      }
      // double epsilon_min = exp(x[dx+dy]);
      // double delta_min = exp(x[dx+dy+1]);
      double epsilon_min = x[dx+dy];
      double delta_min = x[dx+dy+1];
      kx_.setWidth(sigma_x);
      ky_.setWidth(sigma_y);
      algo_(trainingData_, kx_, ky_, testingData_.ys, epsilon_min, delta_min, weights_);
      uint testPoints = testingData_.xs.rows();
      double tau = settings_.quantile;
      double totalCost = 0.0;
      for (int i=0;i<testPoints;i++)
      {
        if (settings_.normed_weights)
        {
          posWeights_ = weights_.row(i);
        }
        else
        {
          Eigen::MatrixXd A = AMatrix(trainingData_.x, kx_);
          Eigen::MatrixXd B = BMatrix(trainingData_.x, kx_);
          double preimage_reg = exp(x[dx+dy+2]);
          positiveNormedCoeffs(weights_.row(i),A,B, preimage_reg, posWeights_);
        }
        double z;
        if (settings_.direct_cumulative)
        {
          Quantile<Kernel<K> > q(weights_.row(i), trainingData_.x, kx_, settings_.cumulative_mean_map);
          z = q(tau);
        }
        else
        {
          Quantile<Kernel<K> > q(posWeights_, trainingData_.x, kx_, settings_.cumulative_mean_map);
          z = q(tau);
        }
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
