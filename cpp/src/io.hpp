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
#include <Eigen/Core>
#include <boost/bind.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

#include "cnpy.h"


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
  Eigen::MatrixXd mat(rows, cols);
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

void writeNPY(const Eigen::MatrixXd& matrix, const std::string& filename)
{
  const unsigned int shape[] = {matrix.rows(), matrix.cols()};
  const double* data = matrix.data();
  cnpy::npy_save(filename, data, shape, 2, "w");
  // //load it into a new array
  // cnpy::NpyArray arr = cnpy::npy_load(filename);
  // double* loaded_data = reinterpret_cast<double*>(arr.data);
  // //make sure the loaded data matches the saved data
  // assert(arr.word_size == sizeof(double));
  // assert(arr.shape.size() == 2 
         // && arr.shape[0] == matrix.rows()
         // && arr.shape[1] == matrix.cols());
  // for(uint i = 0; i < matrix.size();i++) assert(data[i] == loaded_data[i]);
}



void writeCSV(const Eigen::MatrixXd& matrix, const std::string& filename)
{
  std::ofstream file(filename);
  if (file.is_open())
  {
    file << matrix;
  }
}


