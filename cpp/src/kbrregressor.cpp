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

int main(int argc, char** argv)
{

  std::cout << "kbrregressor initialised." << std::endl;
  srand(time(NULL));
  auto settings = getSettings(argv[1]);

  TrainingData trainData = readTrainingData(settings); 
  TestingData testData = readTestingData(settings);
  //useful sizes for preallocation
  uint n = trainData.x.rows();
  uint m = trainData.u.rows();
  uint s = testData.ys.rows();
  Eigen::MatrixXd weights(s,n);

  //how about some training  
  trainSettings(trainData, settings);

  //Create kernels and regressor
  Kernel<RBFKernel> kx(trainData.x, settings.sigma_x);
  Kernel<RBFKernel> ky(trainData.y, settings.sigma_y);
  Regressor<RBFKernel> r(n, m, settings);
 
  //Kernel bayes rule
  r(trainData, kx, ky, testData.ys, weights);

  //write out the results 
  writeNPY(weights, settings.filename_weights);

  //Normalise and compute posterior
  Eigen::MatrixXd posterior(testData.xs.rows(), s);
  if (!settings.normed_weights)
  {
    Eigen::MatrixXd preimageWeights(s,n);
    computeNormedWeights(weights, kx, trainData.x.cols(),
                         settings, preimageWeights);
    writeNPY(preimageWeights, settings.filename_preimage);
    computePosterior(testData.xs, trainData.x, preimageWeights, settings, posterior);
  }
  else
  {
    computePosterior(testData.xs, trainData.x, weights, settings, posterior);
  }
  //and write the posterior
  writeNPY(posterior, settings.filename_posterior);
  return 0;
}

