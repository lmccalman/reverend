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
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define EIGEN_DONT_PARALLELIZE
#include "io.hpp"
#include "train.hpp"
#include "preimage.hpp"
#include "cumulative.hpp"
#include "lowrankregressor.hpp"
#include "reducedset.hpp"


void trainReducedSet(const TrainingData& fullData, 
    TrainingData& trainData, Settings& settings)
{
  double frac = settings.data_fraction;
  Eigen::MatrixXd X_b = readNPY(settings.filename_xb);
  Eigen::MatrixXd Y_b = readNPY(settings.filename_yb);
  TestingData testData(fullData.x, fullData.y);
  TrainingData newTrain(fullData.u, fullData.lambda, X_b, Y_b);
  
  
  passthroughTrainSettings<Regressor<RBFKernel>, RBFKernel>(
      newTrain, testData, settings);
  
  std::cout << "reduced set size: " << newTrain.x.rows() << std::endl;
  
  if (settings.scaling_strategy == "optimal")
  {
    findReducedSet<RBFKernel>(newTrain, testData, settings);
  }
  trainData = newTrain;
  //write out reduced set
  writeNPY(trainData.x, settings.filename_xr);
  writeNPY(trainData.y, settings.filename_yr);
}

void trainAlgorithm(const TrainingData& fulldata,
                    TrainingData& data,
                    Settings& settings)
{
  if (settings.data_fraction < 1.0)
  {
    if (settings.scaling_strategy == "random" ||
        settings.scaling_strategy == "optimal")
    {
      trainReducedSet(fulldata, data, settings);
    }
    else
    {
      data = fulldata;
      if (settings.scaling_strategy == "lowrank")
        trainSettings<LowRankRegressor<RBFKernel>, RBFKernel>(data, settings);
      else
        trainSettings<Regressor<RBFKernel>, RBFKernel>(data, settings);
    }
  }
  else
  {
    data = fulldata;
    trainSettings<Regressor<RBFKernel>, RBFKernel>(data, settings);
  }
}


int main(int argc, char** argv)
{
  
  std::cout << "kbrcpp initialised." << std::endl;
  srand(time(NULL));
  auto settings = getSettings(argv[1]);
  
  TrainingData fullData = readTrainingData(settings); 
  TestingData testData = readTestingData(settings);
  TrainingData trainData;
  std::cout << "Training..." << std::endl;
  trainAlgorithm(fullData, trainData, settings);
  //useful sizes for preallocation
  uint n = trainData.x.rows();
  uint m = trainData.u.rows();
  uint s = testData.ys.rows();
  Eigen::MatrixXd weights(s,n);
  
  //Create kernels and algorithm 
  std::cout << "Inferring..." << std::endl;
  Kernel<RBFKernel> kx(trainData.x, settings.sigma_x);
  Kernel<RBFKernel> ky(trainData.y, settings.sigma_y);
  if (settings.inference_type == std::string("filter"))
  {
    Filter<RBFKernel> f(n, m, settings);
    f(trainData, kx, ky, testData.ys, 
      settings.epsilon_min, settings.delta_min, weights);
  }
  else
  {
    if (settings.scaling_strategy == "lowrank")
    {
      LowRankRegressor<RBFKernel> r(n, m, settings);
      r(trainData, kx, ky, testData.ys,
        settings.epsilon_min, settings.delta_min, weights);
    }
    else
    {
      Regressor<RBFKernel> r(n, m, settings);
      r(trainData, kx, ky, testData.ys,
        settings.epsilon_min, settings.delta_min, weights);
    }
  }
  //write out the results 
  writeNPY(weights, settings.filename_weights);

  //Normalise weights
  Eigen::MatrixXd preimageWeights(s,n);
  if (settings.normed_weights)
  {
    preimageWeights = weights;
  }
  else
  {
    std::cout << "Normalizing Weights..." << std::endl;
    computeNormedWeights(trainData.x, weights, kx, settings, preimageWeights);
  }

  //---------Post Processing------//

  //Cumulative Estimates
  bool meanMap = settings.cumulative_mean_map;
  if (settings.cumulative_estimate)
  {
    std::cout << "Estimating Cumulative..." << std::endl;
    Eigen::VectorXd cumulates(testData.ys.rows());
    if (settings.direct_cumulative)
    {
      computeCumulates(trainData,testData, weights, kx, meanMap, cumulates);
    }
    else 
    {
      computeCumulates(trainData,testData, preimageWeights, kx, meanMap, cumulates);
    }
    writeNPY(cumulates, settings.filename_cumulative);
  }
  
  //Quantile Estimates 
  if (settings.quantile_estimate)
  {
    std::cout << "Estimating Quantile..." << std::endl;
    Eigen::VectorXd quantiles(testData.ys.rows());
    double tau = settings.quantile;
    if (settings.direct_cumulative)
    {
      computeQuantiles(trainData,testData,weights,kx,tau,meanMap,quantiles);
    }
    else
    {
      computeQuantiles(trainData,testData,preimageWeights,kx,tau,meanMap,quantiles);
    }
    writeNPY(quantiles, settings.filename_quantile);
  }
  
  //Evaluate the embedding and posterior
  // Eigen::MatrixXd embedding(testData.ys.rows(), testData.xs.rows());
  // Eigen::MatrixXd posterior(testData.ys.rows(), testData.xs.rows());
  Eigen::VectorXd embedding(testData.ys.rows());
  Eigen::VectorXd posterior(testData.ys.rows());
  std::cout << "Evaluating Embedding..." << std::endl;
  computeEmbedding(trainData,testData,weights,kx, embedding);
  std::cout << "Evaluating Log Posterior..." << std::endl;
  computeLogPosterior(trainData,testData,preimageWeights,kx, posterior);
  writeNPY(embedding, settings.filename_embedding);
  writeNPY(posterior, settings.filename_posterior);
  std::cout << "kbrcpp task complete." << std::endl;

  return 0;
}

