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
#define EIGEN_DONT_PARALLELIZE
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <boost/function.hpp>
  
template <class K>
void computeGramMatrix(const K& k, 
                       const Eigen::VectorXd& width,
                       const Eigen::MatrixXd& X,
                       Eigen::MatrixXd& g);


template <class T, class M = Eigen::MatrixXd>
class Kernel
{
  public:
    Kernel(const Eigen::MatrixXd& X, const Eigen::VectorXd& width):
      isGramUpdated_(false), X_(X), g_xx_(X.rows(), X.rows()), width_(width)
    {
      volume_ = k_.volume(width_, X_.cols());
    }
    
    const M& gramMatrix() const
    {
      if (isGramUpdated_ == false)
      {
        isGramUpdated_ = true;
        computeGramMatrix(k_, width_, X_, g_xx_);
      }
      return g_xx_;
    }
    
    Eigen::VectorXd width() const {return width_;}
    
    double volume() const {return volume_;}
    
    void setWidth(const Eigen::VectorXd& w)
    {
      isGramUpdated_ = false;
      width_ = w;
      volume_ = k_.volume(width_, X_.cols());
    }

    void embed(const Eigen::MatrixXd& u,
               const Eigen::VectorXd& lam,
               Eigen::VectorXd& alpha) const
    {
      uint n = X_.rows();
      uint m = u.rows();
      for(uint i=0; i<n;i++)
      {
        double c = 0.0;
        for(uint j=0;j<m;j++)
        {
          c += lam(j)*k_(X_.row(i), u.row(j), width_);
        }
        alpha(i) = c;
      }
    }
    
    void embed(const Eigen::VectorXd& x,
               Eigen::VectorXd& alpha) const
    {
      uint n = X_.rows();
      for(uint i=0; i<n;i++)
      {
        alpha(i) = k_(X_.row(i), x, width_);
      }
    }

    double innerProduct(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
    {
      double result = x1.transpose() * this->gramMatrix() * x2;
      return result;
    }
    
    double halfSupport() const
    {
      return k_.halfSupport(width_);
    }
    
    double operator()(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
    {
      return k_(x1,x2, width_);
    }
    
    double logk(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
    {
      return k_.logk(x1,x2, width_);
    }
    
    void embedIndicator(const Eigen::VectorXd& cutoff, Eigen::VectorXd& weights) const
    {
      return k_.embedIndicator(cutoff, X_, width_, weights);
    }
    
    double cumulative(const Eigen::VectorXd& cutoff, const Eigen::VectorXd& centre) const
    {
      return k_.cumulative(cutoff, centre, width_); 
    }
      
  protected:
    mutable bool isGramUpdated_ = false;
    T k_;
    const Eigen::MatrixXd& X_;
    mutable M g_xx_;
    Eigen::VectorXd width_;
    double volume_ = 0.0;
};

template <class K>
void computeGramMatrix(const K& k, const Eigen::VectorXd& width, const Eigen::MatrixXd& X, Eigen::MatrixXd& g)
{
  uint n = X.rows();
  for(uint i=0; i<n;i++)
  {
    for(uint j=0;j<n;j++)
    {
      g(i,j) = k(X.row(i), X.row(j), width);
    }
  }
}

class RBFKernel
{
  public:
    RBFKernel(){}
    double operator()(const Eigen::VectorXd& x,
          const Eigen::VectorXd& x_dash,
          const Eigen::VectorXd& sigma) const
    {
      return exp(-0.5*(x - x_dash).cwiseQuotient(sigma).squaredNorm());
    }
    
    double logk(const Eigen::VectorXd& x,
        const Eigen::VectorXd& x_dash,
        const Eigen::VectorXd& sigma) const
    {
      return -0.5*(x - x_dash).cwiseQuotient(sigma).squaredNorm();
    }
    
    
    double halfSupport(const Eigen::VectorXd& width) const
    {
      return 5.0*width.maxCoeff(); 
    }

    double volume(const Eigen::VectorXd& sigma, uint dimension) const
    {
      return pow(2.0*M_PI, dimension*0.5) * sigma.prod();
    }

    void embedIndicator(const Eigen::VectorXd& cutoff,
        const Eigen::MatrixXd& X, const Eigen::VectorXd& sigma_x, Eigen::VectorXd& weights) const
    {
      uint n = X.rows();
      uint dx = X.cols();
      double a = volume(sigma_x,dx);
      for (uint i=0; i<n; i++)
      {
        double dim_result = 1.0;
        for (uint d=0; d<dx; d++)
        {
          double denom = 1.0 / (sigma_x(d)*std::sqrt(2.0));
          double p = cutoff(d);
          double m = X(i,d);
          dim_result *= a * 0.5 * (1.0 + std::erf((p - m)*denom));
        }
        weights[i] = dim_result;
      }
    }
    double cumulative(const Eigen::VectorXd& cutoff,
                      const Eigen::VectorXd& centre,
                      const Eigen::VectorXd& sigma_x) const
    {
      uint dx = cutoff.size(); 
      double dim_result = 1.0;
      for (uint d=0;d<dx; d++)
      {
        double denom = 1.0 / (sigma_x(d)*std::sqrt(2.0));
        double p = cutoff(d);
        double m = centre(d);
        dim_result *= 0.5 * (1.0 + std::erf( (p - m)*denom ));
      }
      return dim_result;
    }
};
