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
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "data.hpp"

struct Settings
{
  std::string filename_x;
  std::string filename_y;
  std::string filename_ys;
  std::string filename_xs;
  std::string filename_u;
  std::string filename_weights;
  std::string filename_preimage;
  double sigma_x;
  double sigma_x_min;
  double sigma_x_max;
  double sigma_y;
  double sigma_y_min;
  double sigma_y_max;
  double walltime;
  double preimage_reg;
  double preimage_reg_min;
  double preimage_reg_max;
  double preimage_walltime;
  uint folds;
  bool normed_weights;
};

Settings getSettings(const std::string& filename)
{
  Settings s;
  boost::property_tree::ptree pt;
  boost::property_tree::ini_parser::read_ini(filename, pt);
  s.filename_u = pt.get<std::string>("Input.filename_u"); 
  s.filename_x = pt.get<std::string>("Input.filename_x"); 
  s.filename_y = pt.get<std::string>("Input.filename_y"); 
  s.filename_xs = pt.get<std::string>("Input.filename_xs"); 
  s.filename_ys = pt.get<std::string>("Input.filename_ys"); 
  s.filename_weights = pt.get<std::string>("Output.filename_weights"); 
  s.filename_preimage = pt.get<std::string>("Output.filename_preimage"); 
  s.sigma_x = pt.get<double>("Kernel.sigma_x"); 
  s.sigma_x_min = pt.get<double>("Kernel.sigma_x_min"); 
  s.sigma_x_max = pt.get<double>("Kernel.sigma_x_max"); 
  s.sigma_y = pt.get<double>("Kernel.sigma_y");
  s.sigma_y_min = pt.get<double>("Kernel.sigma_y_min");
  s.sigma_y_max = pt.get<double>("Kernel.sigma_y_max");
  s.walltime = pt.get<double>("Training.walltime");
  s.preimage_walltime = pt.get<double>("Training.preimage_walltime");
  s.folds = pt.get<uint>("Training.folds");
  s.preimage_reg = pt.get<double>("Preimage.preimage_reg");
  s.preimage_reg_min = pt.get<double>("Preimage.preimage_reg_min");
  s.preimage_reg_max = pt.get<double>("Preimage.preimage_reg_max");
  s.normed_weights = pt.get<bool>("Preimage.normed_weights");
  return s;
}
