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
#include <math.h>
#include "kbr.hpp"


double logGaussianMixture(const Eigen::VectorXd& point,
    const Eigen::MatrixXd& means,
    const Eigen::VectorXd& coeffs,
    double sigma);

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
struct MyCost:Cost
{
  public:
    MyCost(const TrainingData& train, const TestingData& test)
      : Cost(train, test), 
      regressor_(train.x.rows(), train.u.rows()),
      weights_(test.ys.rows(), train.x.rows())
  {
  }; 

    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      double sigma_x = x[0];
      double sigma_y = x[1];
      Kernel kx = boost::bind(rbfKernel, _1, _2, sigma_x);
      Kernel ky = boost::bind(rbfKernel, _1, _2, sigma_y);
      regressor_(trainingData_, kx, ky, testingData_.ys, weights_);
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
    Regressor regressor_;
    Eigen::MatrixXd weights_;
};


double logGaussianMixture(const Eigen::VectorXd& point,
                          const Eigen::MatrixXd& means,
                          const Eigen::VectorXd& coeffs,
                          double sigma)
{
  assert(point.size() == means.cols());
  assert(means.rows() == coeffs.size());
  uint numberOfMeans = means.rows();
  double sigma2 = sigma * sigma;
  double logScaleFactor = -1.0*log(sigma*sqrt(2.0*M_PI));
  
  //find the min exp coeff
  double maxPower = -1e200 // ie infinity;
  for (uint i=0; i<numberOfMeans; i++)
  {
    double deltaNormSquared = (point - means.row(i)).squaredNorm();
    double expCoeff = -0.5 * deltaNormSquared / sigma2;
    maxPower = std::max(maxPower, expCoeff);
  }
  //now compute everything
  double sumAdjProb = 0.0; 
  for (uint i=0; i<numberOfMeans; i++)
  {
    double alpha = coeffs[i];
    double deltaNormSquared = (point - means.row(i)).squaredNorm();
    double expCoeff = -0.5 * deltaNormSquared / sigma2;
    double adjExpCoeff = expCoeff - maxPower;
    double adjProbs = alpha*exp(adjExpCoeff);
    sumAdjProb += adjProbs;
  }
  // this means that if my sumAdjProb is zero or negative, things don't
  // actually break I just get a very low result
  double result =  log(std::max(sumAdjProb,1e-200)) + maxPower + logScaleFactor;
  return result;
}
