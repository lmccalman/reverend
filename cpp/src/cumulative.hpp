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
#include <cmath>
#include <iostream>
#include <boost/math/tools/roots.hpp>
#include <Eigen/Core>
#include "data.hpp"
#include "kernel.hpp"

template <class K>
class Cumulative
{
  public:
    Cumulative(const Eigen::VectorXd& coeffs,
               const Eigen::MatrixXd& X,
               const K& kx,
               bool meanMap)
      : kx_(kx),
        coeffs_(coeffs), 
        X_(X),
        meanMap_(meanMap)
  {
    coeffs_ = coeffs_ / double(coeffs_.sum());
  }

    double operator()(const Eigen::VectorXd& x)
    {
      if (meanMap_)
      {
        return fromMeanMappedWeights(x);
      }
      else
      {
        return fromWeights(x);
      }
    }
    
    double operator()(double x)
    {
      Eigen::VectorXd v(1); 
      v(0) = x;
      if (meanMap_)
      {
        return fromMeanMappedWeights(v);
      }
      else
      {
        return fromWeights(v);
      }
    }

  protected:
    double fromMeanMappedWeights(const Eigen::VectorXd& x) const;
    double fromWeights(const Eigen::VectorXd& x) const;
    const K& kx_;
    Eigen::VectorXd coeffs_;
    const Eigen::MatrixXd& X_;
    bool meanMap_;
    double scale_;
    double offset_;
};

template <class K>
double Cumulative<K>::fromMeanMappedWeights(const Eigen::VectorXd& x) const
{
  double result = 0.0;
  uint n = X_.rows();
  for (int i=0; i<n; i++)
  {
    const Eigen::VectorXd mu = X_.row(i);
    double val = kx_.cumulative(x, mu);
    result += coeffs_(i) * val;
  }
  return result;
}

template <class K>
double Cumulative<K>::fromWeights(const Eigen::VectorXd& x) const
{
  double result = 0.0;
  uint n = X_.rows();
  uint dim = x.size();
  for (int i=0; i<n; i++)
  {
    const Eigen::VectorXd mu = X_.row(i);
    bool smaller = true;
    for (uint j=0; j<dim;j++)
    {
      smaller = smaller && (mu(j) < x(j));
    }
    if (smaller)
    {
      result += coeffs_(i);
    }
  }
  return result;
}


  template <class K>
void computeCumulates(const TrainingData& trainingData, const TestingData& testingData,
    const Eigen::MatrixXd& weights, const K& kx, bool normedWeights,
    Eigen::MatrixXd& cumulates)
{
  uint testPoints = weights.rows(); 
  uint evalPoints = testingData.xs.rows();
  #pragma omp parallel for
  for (int i=0;i<testPoints;i++)  
  {
    Eigen::VectorXd coeffs = weights.row(i);
    Cumulative<K> cumulator(coeffs, trainingData.x, kx, normedWeights);
    for (int j=0; j<evalPoints; j++)
    {
      Eigen::VectorXd point = testingData.xs.row(j);
      double result = cumulator(point);
      cumulates(i,j) = result;
    }
  }
}

template <class K>
class QuantileWrapper
{
  public:
    QuantileWrapper(Cumulative<K>& c, double tau)
      : c_(c), tau_(tau){}
    double operator()(double x) const
    {
      return c_(x) - tau_;
    }
  protected:
    Cumulative<K>& c_;
    double tau_;
};

template <class K>
class Quantile
{
  public:
    Quantile(const Eigen::VectorXd& coeffs, const Eigen::MatrixXd& X,
        const K& kx, bool meanMap):
      c_(coeffs, X, kx, meanMap)
  {
    xmin_ = X.minCoeff() - kx.halfSupport();
    xmax_ = X.maxCoeff() + kx.halfSupport();
  }

    double operator()(double tau) 
    {
      QuantileWrapper<K> F(c_,tau);
      boost::math::tools::eps_tolerance<double> tol(11); 
      long long unsigned int maxIterations = 1000;
      double result;
      double fmin = F(xmin_);
      double fmax = F(xmax_);
      try
      {
        auto boundingPair = boost::math::tools::toms748_solve(F, xmin_, xmax_,
           fmin, fmax, tol, maxIterations);
        result = 0.5*(boundingPair.first + boundingPair.second);
      }
      catch (...)
      {
        if (fmax > 0)
        {
          result = 1e3;
        }
        else
        {
          result = -1e3;
        }
      }
      return result;
    }

  protected:
    Cumulative<K> c_;
    double xmin_;
    double xmax_;
};
  
template <class K>
void computeQuantiles(const TrainingData& trainingData, const TestingData& testingData,
    const Eigen::MatrixXd& weights, const K& kx, double quantile, bool meanMap, 
    Eigen::VectorXd& quantiles)
{
  uint testPoints = weights.rows(); 
  #pragma omp parallel for
  for (int i=0;i<testPoints;i++)  
  {
    Eigen::VectorXd coeffs = weights.row(i);
    Quantile<K> qEstimator(coeffs, trainingData.x, kx, meanMap);
    quantiles(i) = qEstimator(quantile);
  }
}


