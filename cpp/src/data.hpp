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

#include <Eigen/Core>

struct TrainingData
{
  public:
    TrainingData();
    TrainingData(const Eigen::MatrixXd& in_u,
                 const Eigen::VectorXd& in_lambda,
                 const Eigen::MatrixXd& in_x,
                 const Eigen::MatrixXd& in_y) :
      u(in_u), lambda(in_lambda), x(in_x), y(in_y) {};
    Eigen::MatrixXd u;
    Eigen::VectorXd lambda;
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
};

struct TestingData
{
  public:
    TestingData();
    TestingData(const Eigen::MatrixXd& in_xs,
        const Eigen::MatrixXd& in_ys) : xs(in_xs), ys(in_ys) {};
    Eigen::MatrixXd xs;
    Eigen::MatrixXd ys;
};
