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
#include <iomanip>
#include <algorithm>
#include <vector>

void printCost(const std::vector<double>& x, double cost)
{
  std::cout << "[ "; 
  for (uint i=0;i<x.size();i++)
  {
    std::cout << std::setw(10) << x[i] << " ";
  }
  std::cout << " ] cost:" << cost << std::endl;
}

void vectorToParams(const std::vector<double>& x, 
                    Eigen::MatrixXd& X, Eigen::MatrixXd& Y,
                    Eigen::VectorXd& sigma_x, Eigen::VectorXd& sigma_y,
                    double& epsilon_min)
{
  uint size = X.rows();
  uint dx = X.cols();
  uint dy = Y.cols();
  uint c = 0;
  for (uint i=0; i<size; i++)
  {
    for (uint j=0; j<dx; j++)
    {
      X(i,j) = x[c];
      c++;
    }
  }
  for (uint i=0; i<size; i++)
  {
    for (uint j=0; j<dy; j++)
    {
      Y(i,j) = x[c];
      c++;
    }
  }
  //sigma_x
  for (int i=0; i<dx; i++)
  {
    sigma_x(i) = x[c];
    c++;
  }
  //sigma_y
  for (int i=0; i<dy; i++)
  {
    sigma_y(i) = x[c];
    c++;
  }
  epsilon_min = x[c];
  c++;
}

void paramsToVector(const Eigen::MatrixXd& x,
                    const Eigen::MatrixXd& y,
                    const Eigen::MatrixXd& fullX,
                    const Eigen::MatrixXd& fullY,
                    const Settings& settings,
                    std::vector<double>& thetaMin,
                    std::vector<double>& theta0,
                    std::vector<double>& thetaMax)
{
  Eigen::VectorXd minX = fullX.colwise().minCoeff();
  Eigen::VectorXd maxX = fullX.colwise().maxCoeff();
  Eigen::VectorXd minY = fullY.colwise().minCoeff();
  Eigen::VectorXd maxY = fullY.colwise().maxCoeff();
  uint setSize = x.rows();
  uint dx = x.cols();
  uint dy = y.cols();
  uint c = 0; 
  //x points
  for (int i=0;i<setSize;i++)
  {
    for (int j=0; j<dx; j++)
    {
      theta0[c] = x(i,j);
      thetaMin[c] = minX[j];
      thetaMax[c] = maxX[j];
      c++;
    }
  }
  //y points
  for (int i=0;i<setSize;i++)
  {
    for (int j=0; j<dy; j++)
    {
      theta0[c] = y(i,j);
      thetaMin[c] = minY[j];
      thetaMax[c] = maxY[j];
      c++;
    }
  }
  //sigma_x
  for (int i=0; i<dx; i++)
  {
    theta0[c] = settings.sigma_x(i);
    thetaMin[c] = settings.sigma_x_min(i);
    thetaMax[c] = settings.sigma_x_max(i);
    c++;
  }
  for (int i=0; i<dy; i++)
  {
    theta0[c] = settings.sigma_y(i);
    thetaMin[c] = settings.sigma_y_min(i);
    thetaMax[c] = settings.sigma_y_max(i);
    c++;
  }
  theta0[c] = settings.epsilon_min;
  thetaMin[c] = settings.epsilon_min_min;
  thetaMax[c] = settings.epsilon_min_max;
  c++;
}

template <class K>
struct ReducedSetCost : NloptCost
{
  public:
    ReducedSetCost(const TrainingData& data, const Settings& settings)
      : setSize_( int(settings.data_fraction*data.x.rows()) ), 
        data_(data),
      r_( int(settings.data_fraction*data.x.rows()) , data.x.rows(), settings),
      lweights_( int(settings.data_fraction*data.x.rows()) ),
      settings_(settings)
      {
        currentBestCost_ = 1e8;
      };
    
    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      uint n = data_.x.rows();
      uint dx = data_.x.cols();
      uint dy = data_.y.cols();
      Eigen::MatrixXd X(setSize_, dx); 
      Eigen::MatrixXd Y(setSize_, dy); 
      Eigen::VectorXd sigma_x(dx);
      Eigen::VectorXd sigma_y(dy);
      double epsilon_min;
      vectorToParams(x, X, Y, sigma_x, sigma_y, epsilon_min);
      
      const TrainingData minidata(data_.u, data_.lambda, X, Y);
      Kernel<K> kx(X, sigma_x);
      Eigen::MatrixXd J(setSize_, dx+dy);
      J.leftCols(dx) = X;
      J.rightCols(dy) = Y;
      Eigen::VectorXd sigma_j(dx+dy);
      sigma_j.head(dx) = sigma_x;
      sigma_j.tail(dy) = sigma_y;
      Kernel<K> kj(J, sigma_j);
      //get weights of beta
      r_.likelihood(minidata, kx, epsilon_min, lweights_);
      lweights_ = lweights_.cwiseMax(0.0);
      lweights_ = lweights_ / lweights_.sum();
      // evaluate the JOINT
      double totalCost = 0;
      Eigen::RowVectorXd j(dx+dy);
      //don't re-do the ones we initialized with (overfit)
      for (uint i=setSize_; i<n; i++)
      {
        j.head(dx) = data_.x.row(i);
        j.tail(dy) = data_.y.row(i);
        totalCost += logKernelMixture(j, J, lweights_, kj, true);
      }
      totalCost *= -1.0;
      if (int(totalCost) < currentBestCost_)
      {
        currentBestCost_ = int(totalCost);
        std::cout << currentBestCost_ << std::endl;
      }
      return totalCost;
    };
  
  protected:
    uint setSize_;
    const TrainingData& data_;
    Regressor<K> r_;
    Eigen::VectorXd lweights_;
    const Settings& settings_;
    int currentBestCost_;
};

template <class K>
void findReducedSet(const TrainingData& fulldata, Settings& settings,
                    TrainingData& trainData, TestingData& testData)
{
  uint n = fulldata.x.rows();
  uint dx = fulldata.x.cols();
  uint dy = fulldata.y.cols();
  uint setSize = int(settings.data_fraction*n);
  //initialize the thetas 
  std::vector<double> theta0((setSize * dx) + (setSize * dy) + dx + dy + 1);
  std::vector<double> thetaMin((setSize * dx) + (setSize * dy) + dx + dy + 1);
  std::vector<double> thetaMax((setSize * dx) + (setSize * dy) + dx + dy + 1);
  //get subsets
  Eigen::MatrixXd X = fulldata.x.topRows(setSize);
  Eigen::MatrixXd Y = fulldata.y.topRows(setSize);
  Eigen::MatrixXd X_s = fulldata.x.bottomRows(n-setSize);
  Eigen::MatrixXd Y_s = fulldata.y.bottomRows(n-setSize);
  //fill the thetas 
  paramsToVector(X, Y, fulldata.x, fulldata.y, settings,
                 thetaMin, theta0, thetaMax);
  
  //optimize
  ReducedSetCost<K> costfunc(fulldata, settings); 
  std::vector<double> thetaBest = localOptimum(costfunc, thetaMin, thetaMax, theta0);

  Eigen::MatrixXd bestX(setSize, dx); 
  Eigen::MatrixXd bestY(setSize, dy); 
  Eigen::VectorXd sigma_x(dx);
  Eigen::VectorXd sigma_y(dy);
  double epsilon_min;
  vectorToParams(thetaBest, bestX, bestY,
                 sigma_x, sigma_y, epsilon_min);
  settings.sigma_x = sigma_x;
  settings.sigma_y = sigma_y;
  settings.epsilon_min = epsilon_min;
  TrainingData result(fulldata.u, fulldata.lambda, bestX, bestY);
  trainData = result;
  testData = TestingData(X_s, Y_s);
}

void randomReducedSet(const TrainingData& fulldata, Settings& settings,
    TrainingData& trainData, TestingData& testData)
{
  uint n = fulldata.x.rows();
  uint dx = fulldata.x.cols();
  uint dy = fulldata.y.cols();
  uint setSize = int(settings.data_fraction*n);
  Eigen::MatrixXd X = fulldata.x.topRows(setSize);
  Eigen::MatrixXd Y = fulldata.y.topRows(setSize);
  Eigen::MatrixXd X_s = fulldata.x.bottomRows(n-setSize);
  Eigen::MatrixXd Y_s = fulldata.y.bottomRows(n-setSize);
  TrainingData result(fulldata.u, fulldata.lambda, X, Y);
  TestingData t(X_s, Y_s);  
  trainData = result;
  testData = t; 
}
