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

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "io.hpp"
#include "train.hpp"
#include "preimage.hpp"
#include "compactkernel.hpp"

int main(int argc, char** argv)
{
  
  std::cout << "skbrcpp initialised." << std::endl;
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
  trainSettings<Regressor<CompactKernel<Q1CompactKernel<2> > >,
                CompactKernel<Q1CompactKernel<2> > >(trainData, settings);

  //Create kernels and algorithm 
  std::cout << "Inferring..." << std::endl;
  CompactKernel<Q1CompactKernel<2> > kx(trainData.x, settings.sigma_x);
  CompactKernel<Q1CompactKernel<2> > ky(trainData.y, settings.sigma_y);
  
  Regressor<CompactKernel<Q1CompactKernel<2> > > r(n, m, settings);
  r(trainData, kx, ky, testData.ys, weights);
 
  //write out the results 
  writeNPY(weights, settings.filename_weights);
  
  //evaluate the raw posterior 
  Eigen::MatrixXd embedding(testData.ys.rows(), testData.xs.rows());
  std::cout << "Evaluating embedded posterior..." << std::endl;
  computeEmbedding(trainData, testData, weights, kx, embedding);
  writeNPY(embedding, settings.filename_embedding);

  // //Normalise and compute posterior
  Eigen::MatrixXd posterior(testData.ys.rows(), testData.xs.rows());
  if (!settings.normed_weights)
  {
    std::cout << "Computing normed weights..." << std::endl;
    Eigen::MatrixXd preimageWeights(s,n);
    computeNormedWeights(weights, kx, trainData.x.cols(),
                         settings, preimageWeights);
    writeNPY(preimageWeights, settings.filename_preimage);
    
    std::cout << "Evaluating posterior..." << std::endl;
    computePosterior(trainData, testData, preimageWeights, settings.sigma_x,
        posterior);
  }
  else
  {
    std::cout << "Evaluating posterior..." << std::endl;
    computePosterior(trainData, testData, weights, settings.sigma_x,
                          posterior);
  }
  //and write the posterior
  writeNPY(posterior, settings.filename_posterior);

  std::cout << "skbrcpp task complete." << std::endl;

  return 0;
}

