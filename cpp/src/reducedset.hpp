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
#include <limits>

std::vector<uint> randomIndices(uint size, uint nmax)
{
  uint counter = 0;
  std::vector<uint> allnums(nmax);
  std::vector<uint> result(size);
  for (uint i=0; i<nmax; i++)
  {
    allnums[i] = i;
  }
  std::random_shuffle(allnums.begin(),allnums.end());
  for (uint i=0; i<size; i++)
  {
    result[i] = allnums[i];
  }
  return result;
}

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
                    double& epsilon_min, double& delta_min)
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
  delta_min = x[c];
  c++;
}

void paramsToVector(const Eigen::MatrixXd& x,
                    const Eigen::MatrixXd& y,
                    const Settings& settings,
                    std::vector<double>& thetaMin,
                    std::vector<double>& theta0,
                    std::vector<double>& thetaMax)
{
  uint setSize = x.rows();
  uint dx = x.cols();
  uint dy = y.cols();
  uint c = 0; 
  //x points
  for (uint i=0;i<setSize;i++)
  {
    for (uint j=0; j<dx; j++)
    {
      theta0[c] = x(i,j);
      thetaMin[c] = x(i,j) - 5*settings.sigma_x(j);
      thetaMax[c] = x(i,j) + 5*settings.sigma_x(j);
      c++;
    }
  }
  //y points
  for (int i=0;i<setSize;i++)
  {
    for (int j=0; j<dy; j++)
    {
      theta0[c] = y(i,j);
      thetaMin[c] = y(i,j) - 5*settings.sigma_y(j);
      thetaMax[c] = y(i,j) + 5*settings.sigma_y(j);
      c++;
    }
  }
  // sigma_x
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
  theta0[c] = settings.delta_min;
  thetaMin[c] = settings.delta_min_min;
  thetaMax[c] = settings.delta_min_max;
  c++;
}

template <class K>
struct ReducedSetCost
{
  public:
    ReducedSetCost(const TrainingData& trainData, const TestingData& testData, 
                   const Settings& settings)
      : trainData_(trainData), testData_(testData), 
      r_(trainData.x.rows() , trainData.u.rows(), settings),
      lweights_(trainData.x.rows()), settings_(settings)
      {};

    double operator()(const std::vector<double>&x, const std::vector<uint>& indices)
    {
      uint n = trainData_.x.rows();
      uint dx = trainData_.x.cols();
      uint dy = trainData_.y.cols();
      Eigen::MatrixXd X(n, dx); 
      Eigen::MatrixXd Y(n, dy); 
      Eigen::VectorXd sigma_x(dx);
      Eigen::VectorXd sigma_y(dy);
      double epsilon_min;
      double delta_min;
      vectorToParams(x, X, Y, sigma_x, sigma_y, epsilon_min, delta_min);
      // std::cout << sigma_x << ":" << sigma_y << ":" << epsilon_min << ":" << delta_min << std::endl;
      const TrainingData minidata(trainData_.u, trainData_.lambda, X, Y);
      Kernel<K> kx(X, sigma_x);
      Kernel<K> ky(Y, sigma_y);
      Eigen::MatrixXd Rxy = r_.RMatrix(minidata, kx, ky, epsilon_min, delta_min); 
      double totalCost = 0;
      uint batchSize = indices.size();
      for (uint p=0; p<batchSize; p++)
      {
        uint i = indices[p];
        auto yi = testData_.ys.row(i);
        ky.embed(yi, lweights_);
        lweights_ = Rxy * lweights_;
        lweights_ = lweights_.cwiseMax(0.0);
        lweights_ = lweights_ / lweights_.sum();
        totalCost += logKernelMixture(testData_.xs.row(i),
            minidata.x, lweights_, kx, true);
      }
      return totalCost;
    };
    
  
  protected:
    uint setSize_;
    const TrainingData& trainData_;
    const TestingData& testData_;
    Regressor<K> r_;
    Eigen::VectorXd lweights_;
    const Settings& settings_;
};

template <class K>
struct SGDReducedSetCost : NloptCost
{
  public:
    SGDReducedSetCost(const TrainingData& trainData,
                      const TestingData& testData, 
                      const Settings& settings)
      : trainData_(trainData),
      testData_(testData),
      settings_(settings) {};
    
    double operator()(const std::vector<double>&x, std::vector<double>&grad)
    {
      
      // std::cout << "in cost fn:" << std::endl;
      // for (uint i=0;i<x.size();i++)
      // {
        // std::cout << x[i] << ",";
      // }
      // std::cout << std::endl;


      uint testN = testData_.xs.rows();
      double eps = sqrt(std::numeric_limits<double>::epsilon());
      bool stochastic = false;
      
      std::vector< std::vector<uint> > indexList;
      uint counter = 0;
      while (counter < testN)
      {
        std::vector<uint> ids;
        for (uint i=0; i<100; i++)
        {
          ids.push_back(counter);
          counter++;
          if (counter >= testN)
            break;
        }
        indexList.push_back(ids);
      }
      double totalCost = 0.0; 
      #pragma omp parallel for reduction(+:totalCost)
      for (uint i=0;i<indexList.size();i++)
      {
        ReducedSetCost<K> rscost(trainData_, testData_, settings_);
        totalCost += rscost(x, indexList[i]);
      }
      totalCost /= double(testN);
      totalCost *= -1.0;
      
      
      // uint params = x.size(); 
      // std::vector<double> xdash = x;
      // for (uint i=0;i<params;i++)
      // {
        // double h = eps * std::min(fabs(x[i]),1.0);
        // if (h == 0.0) 
        // {
          // h = eps;
        // }
        // xdash[i] += h;
        // double cdash = rscost(xdash, indices);
        // double delta = cdash - c0;
        // double gpos = (cdash - c0)/h ;
        // grad[i] = gpos;
        // xdash[i] -= h;
      // }
      uint print = x.size() - (trainData_.x.cols() + trainData_.y.cols() + 2);
      std::cout << "[ ";
      for (uint i=print;i<x.size();i++)
      {
        std::cout << std::setw(10) << x[i] << " ";
      }
      std::cout << " ] reduced set cost:" << totalCost << std::endl;
      return totalCost;
    }


  private:
    const TrainingData& trainData_;
    const TestingData& testData_;
    const Settings& settings_;
};

template <class K>
void findReducedSet(TrainingData& trainData, const TestingData& testData, Settings& settings)
{
  uint n = trainData.x.rows();
  uint dx = trainData.x.cols();
  uint dy = trainData.y.cols(); 
  
  // std::cout << "pretrain:" << std::endl;
  // std::cout << settings.sigma_x << std::endl;
  // std::cout << settings.sigma_y << std::endl;
  // std::cout << settings.epsilon_min << std::endl;
  // std::cout << settings.delta_min << std::endl;
  //initialize the thetas 
  std::vector<double> theta0((n * dx) + (n * dy) + dx + dy + 2);
  std::vector<double> thetaMin((n * dx) + (n * dy) + dx + dy + 2);
  std::vector<double> thetaMax((n * dx) + (n * dy) + dx + dy + 2);
  //fill the thetas
  paramsToVector(trainData.x, trainData.y, settings,
                 thetaMin, theta0, thetaMax);
  // std::cout << "params:" << std::endl;
  // for (uint i=0; i< theta0.size(); i++)
  // {
    // std::cout << theta0[i] << ",";
  // }
  // std::cout << std::endl;
  //optimize
  SGDReducedSetCost<K> costfunc(trainData, testData, settings); 
  std::vector<double> thetaBest = localOptimum(costfunc, thetaMin, thetaMax, theta0);
  // std::vector<double> thetaBest = SGD(costfunc, thetaMin, thetaMax, theta0, settings);

  Eigen::MatrixXd bestX(n, dx); 
  Eigen::MatrixXd bestY(n, dy); 
  Eigen::VectorXd sigma_x(dx);
  Eigen::VectorXd sigma_y(dy);
  double epsilon_min;
  double delta_min;
  vectorToParams(thetaBest, bestX, bestY,
                 sigma_x, sigma_y, epsilon_min, delta_min);
  settings.sigma_x = sigma_x;
  settings.sigma_y = sigma_y;
  settings.epsilon_min = epsilon_min;
  settings.delta_min = delta_min;
  TrainingData result(trainData.u, trainData.lambda, bestX, bestY);
  trainData = result;
}









