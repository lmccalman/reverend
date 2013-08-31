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

#include <fstream>
#include <string>
#include <Eigen/Core>
#include <boost/bind.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "cnpy.h"
#include "data.hpp"
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

Eigen::VectorXd stringToVector(const std::string& s)
{
  std::vector<std::string> elems;
  split(s, ',', elems);
  uint n = elems.size();
  Eigen::VectorXd result(elems.size());
  for (uint i=0; i<n; i++)
  {
    result(i) = atof(elems[i].c_str());
  }
  return result;
}

Settings getSettings(const std::string& filename)
{
  Settings s;
  boost::property_tree::ptree pt;
  boost::property_tree::ini_parser::read_ini(filename, pt);
  s.filename_u = pt.get<std::string>("Input.filename_u"); 
  s.filename_x = pt.get<std::string>("Input.filename_x"); 
  s.filename_y = pt.get<std::string>("Input.filename_y"); 
  s.filename_xr = pt.get<std::string>("Input.filename_xr"); 
  s.filename_yr = pt.get<std::string>("Input.filename_yr"); 
  s.filename_xs = pt.get<std::string>("Input.filename_xs"); 
  s.filename_ys = pt.get<std::string>("Input.filename_ys"); 
  s.filename_weights = pt.get<std::string>("Output.filename_weights"); 
  s.filename_preimage = pt.get<std::string>("Output.filename_preimage"); 
  s.filename_posterior = pt.get<std::string>("Output.filename_posterior"); 
  s.filename_embedding = pt.get<std::string>("Output.filename_embedding"); 
  s.filename_cumulative = pt.get<std::string>("Output.filename_cumulative"); 
  s.filename_quantile = pt.get<std::string>("Output.filename_quantile"); 
  s.filename_sgd = pt.get<std::string>("Output.filename_sgd"); 
  s.sigma_x = stringToVector(pt.get<std::string>("Kernel.sigma_x")); 
  s.sigma_x_min =stringToVector( pt.get<std::string>("Kernel.sigma_x_min")); 
  s.sigma_x_max =stringToVector( pt.get<std::string>("Kernel.sigma_x_max")); 
  s.sigma_y = stringToVector( pt.get<std::string>("Kernel.sigma_y"));
  s.sigma_y_min =stringToVector( pt.get<std::string>("Kernel.sigma_y_min"));
  s.sigma_y_max =stringToVector( pt.get<std::string>("Kernel.sigma_y_max"));
  s.epsilon_min = pt.get<double>("Algorithm.epsilon_min");
  s.delta_min = pt.get<double>("Algorithm.delta_min");
  s.epsilon_min_min = pt.get<double>("Algorithm.epsilon_min_min");
  s.delta_min_min = pt.get<double>("Algorithm.delta_min_min");
  s.epsilon_min_max = pt.get<double>("Algorithm.epsilon_min_max");
  s.delta_min_max = pt.get<double>("Algorithm.delta_min_max");
  s.walltime = pt.get<double>("Training.walltime");
  s.preimage_walltime = pt.get<double>("Training.preimage_walltime");
  s.folds = pt.get<uint>("Training.folds");
  s.preimage_reg = pt.get<double>("Preimage.preimage_reg");
  s.preimage_reg_min = pt.get<double>("Preimage.preimage_reg_min");
  s.preimage_reg_max = pt.get<double>("Preimage.preimage_reg_max");
  s.inference_type = pt.get<std::string>("Algorithm.inference_type"); 
  s.observation_period = pt.get<uint>("Algorithm.observation_period");
  s.cumulative_estimate = pt.get<bool>("Algorithm.cumulative_estimate");
  s.cumulative_mean_map = pt.get<bool>("Algorithm.cumulative_mean_map");
  s.quantile_estimate = pt.get<bool>("Algorithm.quantile_estimate");
  s.normed_weights = pt.get<bool>("Algorithm.normed_weights");
  s.quantile = pt.get<double>("Algorithm.quantile");
  s.pinball_loss = pt.get<bool>("Algorithm.pinball_loss");
  s.direct_cumulative = pt.get<bool>("Algorithm.direct_cumulative");
  s.sgd_iterations = pt.get<uint>("Scaling.sgd_iterations");
  s.sgd_learn_rate = pt.get<double>("Scaling.sgd_learn_rate");
  s.sgd_batch_size = pt.get<double>("Scaling.sgd_batch_size");
  s.scaling_strategy = pt.get<std::string>("Scaling.scaling_strategy");
  s.data_fraction = pt.get<double>("Scaling.data_fraction");
  s.data_fraction = pt.get<double>("Scaling.data_fraction");
  return s;
}

Eigen::MatrixXd readCSV(const std::string& filename)
{
  Eigen::MatrixXd input;
  std::ifstream inFile(filename.c_str());
  if (!inFile.is_open())
  {
    std::cout << "ERROR: input file " << filename << " won't open" << std::endl;
    return input;
  }

  std::vector<std::string> elements;
  std::string line;
  //Read first line to get the number of columns
  getline(inFile,line);
  boost::tokenizer< boost::escaped_list_separator<char> > tok(line);
  elements.assign(tok.begin(), tok.end());
  uint columns = elements.size();
  uint rows = 1;
  while (getline(inFile,line))
  {
    tok = boost::tokenizer< boost::escaped_list_separator<char> >(line);
    elements.assign(tok.begin(), tok.end());
    if (elements.size() < columns)
    {
      continue;
    }
    else
    {
      rows++;
    }
  }
  //Now initialise the Matrix and read it again
  input = Eigen::MatrixXd(rows, columns);
  inFile.clear();
  inFile.seekg(0, std::ios::beg);
  uint i = 0;
  while (getline(inFile,line))
  {
    tok = boost::tokenizer< boost::escaped_list_separator<char> >(line);
    elements.assign(tok.begin(), tok.end());
    for (uint j=0;j<elements.size();j++)
    {
      input(i,j) = boost::lexical_cast<double>(elements[j]);
    }
    i++;
  }
  return input;
}

Eigen::MatrixXd readNPY(const std::string& filename)
{
  cnpy::NpyArray arr = cnpy::npy_load(filename);
  double* loaded_data = reinterpret_cast<double*>(arr.data);
  assert(arr.word_size == sizeof(double));
  assert(arr.shape.size() == 2);
  uint rows = arr.shape[0];
  uint cols = arr.shape[1];
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> mat(rows, cols);
  uint counter = 0;
  for (uint i=0;i<rows;i++)
  {
    for (uint j=0;j<cols;j++)
    {
      mat(i,j) = loaded_data[counter];
      counter++;
    }
  }
  return mat;
}

TrainingData readTrainingData(const Settings& settings)
{
  auto x = readNPY(settings.filename_x);
  auto y = readNPY(settings.filename_y);
  uint dim_x = x.cols();
  uint dim_y = y.cols();
  uint n = x.rows();
  Eigen::MatrixXd u = x;
  uint m = x.rows();
  if (settings.filename_u != "")
  {
    u = readNPY(settings.filename_u);
    m = u.rows();
  }
  //for the moment keep the weights constant
  Eigen::VectorXd lambda = Eigen::VectorXd::Ones(m);
  lambda = lambda / double(m);

  if (settings.inference_type == std::string("filter"))
  {
    Eigen::MatrixXd su = x.block(0,0,n-1,dim_x);
    Eigen::VectorXd slambda = Eigen::VectorXd::Ones(n-1);
    slambda = slambda / double(n-1);
    Eigen::MatrixXd sx = x.block(0,0,n-1,dim_x);
    Eigen::MatrixXd sy = y.block(0,0,n-1,dim_y);
    Eigen::MatrixXd xtp1 = x.block(1,0,n-1,dim_x);
    return TrainingData(su, slambda, sx, sy, xtp1);
  }
  else
  {
    return TrainingData(u, lambda, x, y);
  }
}


TestingData readTestingData(const Settings& settings)
{
  auto xs = readNPY(settings.filename_xs);
  auto ys = readNPY(settings.filename_ys);
  return TestingData(xs,ys);
}

void writeNPY(const Eigen::MatrixXd& matrix, const std::string& filename)
{
  uint rows = matrix.rows(); 
  uint cols = matrix.cols();
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> rowMat(rows, cols);
  rowMat = matrix;
  const uint shape[] = {rows,cols};
  const double* data = rowMat.data();
  cnpy::npy_save(filename, data, shape, 2, "w");
}



void writeCSV(const Eigen::MatrixXd& matrix, const std::string& filename)
{
  std::ofstream file(filename);
  if (file.is_open())
  {
    file << matrix;
  }
}


