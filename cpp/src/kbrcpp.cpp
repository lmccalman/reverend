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

int main(int argc, char** argv)
{
  
  std::cout << "kbrcpp initialised." << std::endl;
  srand(time(NULL));
  auto settings = getSettings(argv[1]);

  TrainingData trainData = readTrainingData(settings); 
  TestingData testData = readTestingData(settings);
  //useful sizes for preallocation
  uint n = trainData.x.rows();
  uint m = trainData.u.rows();
  uint s = testData.ys.rows();
  Eigen::MatrixXd weights(s,n);

  std::cout << "Training..." << std::endl;
  //how about some training  
  if (settings.inference_type == std::string("filter"))
  {
    trainSettings<Filter<RBFKernel>, RBFKernel>(trainData, settings);
  }
  else
  {
    trainSettings<Regressor<RBFKernel>, RBFKernel>(trainData, settings);
  }

  //Create kernels and algorithm 
  std::cout << "Inferring..." << std::endl;
  Kernel<RBFKernel> kx(trainData.x, settings.sigma_x);
  Kernel<RBFKernel> ky(trainData.y, settings.sigma_y);
  if (settings.inference_type == std::string("filter"))
  {
    Filter<RBFKernel> f(n, m, settings);
    f(trainData, kx, ky, testData.ys, weights);
  }
  else
  {
    Regressor<RBFKernel> r(n, m, settings);
    r(trainData, kx, ky, testData.ys, weights);
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
    Eigen::MatrixXd cumulates(testData.ys.rows(), testData.xs.rows());
    if (settings.cost_function == "pinball_direct")
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
    if (settings.cost_function == "pinball_direct")
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
  Eigen::MatrixXd embedding(testData.ys.rows(), testData.xs.rows());
  Eigen::MatrixXd posterior(testData.ys.rows(), testData.xs.rows());
  std::cout << "Evaluating Embedding..." << std::endl;
  computeEmbedding(trainData,testData,weights,kx, embedding);
  std::cout << "Evaluating Log Posterior..." << std::endl;
  computeLogPosterior(trainData,testData,preimageWeights,kx, posterior);
  writeNPY(embedding, settings.filename_embedding);
  writeNPY(posterior, settings.filename_posterior);
  std::cout << "kbrcpp task complete." << std::endl;

  return 0;
}

