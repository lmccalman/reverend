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
#define EIGEN_DONT_PARALLELIZE
#include "io.hpp"
#include "sparsetrain.hpp"
#include "sparsepreimage.hpp"
#include "sparseregressor.hpp"

int main(int argc, char** argv)
{
  
  std::cout << "skbrcpp initialised." << std::endl;
  srand(time(NULL));
  SparseSettings settings = getSparseSettings(argv[1]);

  TrainingData trainData = readTrainingData(settings); 
  TestingData testData = readTestingData(settings);
  //useful sizes for preallocation
  uint n = trainData.x.rows();
  uint m = trainData.u.rows();
  uint s = testData.ys.rows();
  Eigen::MatrixXd weights(s,n);

  std::cout << "Training..." << std::endl;
  //how about some training  
  sparseTrainSettings<SparseRegressor<Q1CompactKernel>, Q1CompactKernel>(trainData, settings);
  //Create kernels and algorithm 
  std::cout << "Inferring..." << std::endl;
  Kernel<Q1CompactKernel, SparseMatrix> kx(trainData.x, settings.sigma_x);
  Kernel<Q1CompactKernel, SparseMatrix> ky(trainData.y, settings.sigma_y);
  SparseRegressor<Q1CompactKernel> r(n, m, settings);
  r(trainData, kx, ky, testData.ys, settings.low_rank_scale,
     settings.low_rank_weight, settings.epsilon_min, settings.delta_min, weights);
  //write out the results 
  writeNPY(weights, settings.filename_weights);
  
  // //evaluate the raw posterior 
  Eigen::MatrixXd embedding(testData.ys.rows(), testData.xs.rows());
  std::cout << "Evaluating embedded posterior..." << std::endl;
  computeSparseEmbedding(trainData, testData, weights, kx,
                         embedding, settings.low_rank_scale,
                         settings.low_rank_weight);
  writeNPY(embedding, settings.filename_embedding);
  std::cout << "skbrcpp task complete." << std::endl;
  return 0;
}

