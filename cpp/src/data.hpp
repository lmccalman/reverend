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

struct Settings
{
  std::string filename_x;
  std::string filename_y;
  std::string filename_ys;
  std::string filename_xs;
  std::string filename_u;
  std::string filename_weights;
  std::string filename_preimage;
  std::string filename_posterior;
  std::string filename_embedding;
  std::string filename_cumulative;
  std::string filename_quantile;
  std::string inference_type;
  Eigen::VectorXd sigma_x;
  Eigen::VectorXd sigma_x_min;
  Eigen::VectorXd sigma_x_max;
  Eigen::VectorXd sigma_y;
  Eigen::VectorXd sigma_y_min;
  Eigen::VectorXd sigma_y_max;
  double epsilon_min;
  double delta_min;
  double epsilon_min_min;
  double delta_min_min;
  double epsilon_min_max;
  double delta_min_max;
  double walltime;
  double preimage_reg;
  double preimage_reg_min;
  double preimage_reg_max;
  double preimage_walltime;
  double quantile;
  uint folds;
  uint observation_period;
  bool cumulative_estimate;
  bool cumulative_mean_map;
  bool quantile_estimate;
  bool normed_weights;
  bool pinball_loss;
  bool direct_cumulative;
  double rank_fraction;
};

struct TrainingData
{
  public:
    TrainingData();
    TrainingData(const Eigen::MatrixXd& in_u,
                 const Eigen::VectorXd& in_lambda,
                 const Eigen::MatrixXd& in_x,
                 const Eigen::MatrixXd& in_y) :
      u(in_u), lambda(in_lambda), x(in_x), y(in_y) {};
    TrainingData(const Eigen::MatrixXd& in_u,
        const Eigen::VectorXd& in_lambda,
        const Eigen::MatrixXd& in_x,
        const Eigen::MatrixXd& in_y,
        const Eigen::MatrixXd& in_xtp1) :
      u(in_u), lambda(in_lambda), x(in_x), y(in_y),
      xtp1(in_xtp1) {};
    
    Eigen::MatrixXd u;
    Eigen::VectorXd lambda;
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
    Eigen::MatrixXd xtp1;
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
