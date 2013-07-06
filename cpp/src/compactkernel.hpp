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
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <boost/function.hpp>
#include "kernel.hpp"

typedef Eigen::SparseMatrix<double> SMatrix;

template <class T>
class CompactKernel
{
  public:
    CompactKernel(const Eigen::MatrixXd& X, double width):
      X_(X), g_xx_(X.rows(), X.rows())
    {
      setWidth(width); 
    }

    const SMatrix& gramMatrix() const {return g_xx_;}
    double width() const {return width_;}
    
    void setWidth(double w)
    {
      width_ = w; //recompute gram matrix
      std::vector< Eigen::Triplet<double> > coeffs;
      uint n = X_.rows();
      for(uint i=0; i<n;i++)
      {
        for(uint j=0;j<n;j++)
        {
          double r = (X_.row(i) - X_.row(j)).norm();
          if (r < w)
          {
            double val = k_(X_.row(i),X_.row(j), w);
            coeffs.push_back(Eigen::Triplet<double>(i,j,val));
          }
        }
      }
      g_xx_.setFromTriplets(coeffs.begin(), coeffs.end()); 
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
      return (double)(x1.transpose() * g_xx_ * x2);
    }

    double halfSupport() const
    {
      return k_.halfSupport(width_);
    }

    double operator()(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
    {
      return k_(x1,x2, width_);
    }

    void embedIndicator(const Eigen::VectorXd& cutoff, Eigen::VectorXd& weights) const
    {
      return k_.embedIndicator(cutoff, X_, width_, weights);
    };

    double cumulative(const Eigen::VectorXd& cutoff, const Eigen::VectorXd& centre) const
    {
      return k_.cumulative(cutoff, centre, width_); 
    }

  protected:
    T k_;
    const Eigen::MatrixXd& X_;
    double width_ = 1.0;
    SMatrix g_xx_;
};


template <int D>
class Q1CompactKernel
{
  public:
    Q1CompactKernel(){}
    double operator()(const Eigen::VectorXd& x,
        const Eigen::VectorXd& x_dash,
        double sigma) const
    {
      double j = D/2.0 + 2;
      double r = (x-x_dash).norm() / sigma / 4.0;
      double result = 0;
      if (r < 1.0)
      {
        result = pow((1-r),(j+1)) * ((j+1)*r + 1);
      }
      return result;
    }
    double halfSupport(double width) const
    {
      return width; 
    }

    void embedIndicator(const Eigen::VectorXd& cutoff,
        const Eigen::MatrixXd& X, double sigma_x, Eigen::VectorXd& weights) const
    {
      std::cout << "COMPACT KERNEL SUPPORT FOR CUMULATIVE NOT IMPLEMENTED" << std::endl;
    }
    
    double cumulative(const Eigen::VectorXd& cutoff,
        const Eigen::VectorXd& centre,
        double sigma_x) const
    {
      std::cout << "COMPACT KERNEL SUPPORT FOR CUMULATIVE NOT IMPLEMENTED" << std::endl;
      return 0.0;
    }



};
