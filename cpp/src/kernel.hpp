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
#include <Eigen/Core>
#include <boost/function.hpp>

// class Kernel
// {
  // public:
    // Kernel(const Eigen::MatrixXd& X):
      // g_xx_(X.rows(),X.rows())
    // {
      // uint n = X.rows();
      // for (int r=0;r<n;r++)
      // {
        // for (int c=0;c<n;c++)
        // {
          // g_xx_(r,c) = this(X.row(r),X.row(c));
        // }
      // }
    // }
    // const Eigen::MatrixXd& gramMatrix(){return g_xx_;}
    // double width(){return width_;}
    // virtual double approximateHalfSupport() = 0;
    // virtual operator()(const Eigen::Vector& x1, const Eigen::Vector& x2) = 0;

  // protected:
    // Eigen::MatrixXd g_xx_;
// }



typedef boost::function<double (const Eigen::VectorXd& x,
                                const Eigen::VectorXd& x_dash)> Kernel;

double rbfKernel(const Eigen::VectorXd& x,
                 const Eigen::VectorXd& x_dash,
                 double sigma)
{
  return exp(-0.5 * (x - x_dash).squaredNorm() / (sigma*sigma));
}




