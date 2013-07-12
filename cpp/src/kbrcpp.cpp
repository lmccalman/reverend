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
  // trainSettings<Regressor<Q1CompactKernel>, Q1CompactKernel>(trainData, settings);

  //Create kernels and algorithm 
  std::cout << "Inferring..." << std::endl;
  Kernel<Q1CompactKernel> kx(trainData.x, settings.sigma_x);
  Kernel<Q1CompactKernel> ky(trainData.y, settings.sigma_y);
  Regressor<Q1CompactKernel> r(n, m, settings);
  r(trainData, kx, ky, testData.ys, weights);
  //write out the results 
  writeNPY(weights, settings.filename_weights);
  
  // //evaluate the raw posterior 
  // Eigen::MatrixXd embedding(testData.ys.rows(), testData.xs.rows());
  // std::cout << "Evaluating embedded posterior..." << std::endl;
  // computeEmbedding(trainData, testData, weights, kx, embedding);
  // writeNPY(embedding, settings.filename_embedding);

  // //Normalise and compute posterior
  // Eigen::MatrixXd preimageWeights(s,n);
  // Eigen::MatrixXd posterior(testData.ys.rows(), testData.xs.rows());
  // if (!settings.normed_weights && settings.preimage_estimate)
  // {
    // std::cout << "Computing normed weights..." << std::endl;
    // computeNormedWeights(weights, kx, trainData.x.cols(),
                         // settings, preimageWeights);
    // writeNPY(preimageWeights, settings.filename_preimage);
    
    // std::cout << "Evaluating posterior..." << std::endl;
    // computePosterior(trainData, testData, preimageWeights, settings.sigma_x,
        // posterior);
    // //and write the posterior
    // writeNPY(posterior, settings.filename_posterior);
  // }
  
  // //Just compute posterior if normed weights 
  // if (settings.normed_weights && settings.preimage_estimate)
  // {
    // std::cout << "Evaluating posterior..." << std::endl;
    // computePosterior(trainData, testData, weights, settings.sigma_x,
                          // posterior);
    // //and write the posterior
    // writeNPY(posterior, settings.filename_posterior);
  // }

  // //compute cumulative estimates
  // if (settings.cumulative_estimate)
  // {
    // std::cout << "Estimates Cumulative distributions..." << std::endl;
    // Eigen::MatrixXd cumulates(testData.ys.rows(), testData.xs.rows());
    // if (settings.preimage_estimate)
    // {
      // computeCumulates(trainData, testData, preimageWeights, kx, settings, cumulates);
    // }
    // else
    // {
      // computeCumulates(trainData, testData, weights, kx, settings, cumulates);
    // }
    // writeNPY(cumulates, settings.filename_cumulative);
  // }
  
  // //compute quantile estimates
  // if (settings.quantile_estimate && (trainData.x.cols() == 1))
  // {
    // std::cout << "Estimating " << settings.quantile
      // << " quantile..." << std::endl;
    // Eigen::VectorXd quantiles(testData.ys.rows());
    // if (settings.preimage_estimate)
    // {
      // computeQuantiles(trainData, testData, preimageWeights, kx, settings, quantiles);
    // }
    // else
    // {
      // computeQuantiles(trainData, testData, weights, kx, settings, quantiles);
    // }
    // writeNPY(quantiles, settings.filename_quantile);
  // }
  
  std::cout << "kbrcpp task complete." << std::endl;

  return 0;
}

