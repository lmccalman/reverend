# Reverend -- Practical Bayesian Inference with Kernel Embeddings
# Copyright (C) 2013 Lachlan McCalman
# lachlan@mccalman.info

# Reverend is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Reverend is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Reverend.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 2.8)
project(reverend) 
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(REVEREND_SRC_DIRECTORY ${PROJECT_SOURCE_DIR}/src)
set(REVEREND_EXECUTABLE kbrcpp)

set(CNPY_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../cnpy)
set(CNPY_LIB_DIR ${PROJECT_SOURCE_DIR}/../cnpy)

set(QUADPROG_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../quadprog)

find_library(CNPY_LIBRARY NAMES "cnpy" PATHS ${CNPY_LIB_DIR})
find_library(NLOPT_LIBRARY NAMES "nlopt") 

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11")
set(CMAKE_BUILD_TYPE release)

set(CMAKE_VERBOSE_MAKEFILE on)

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(Eigen3)
find_package(SuiteSparse)
find_package(LAPACK)

include_directories( ${EIGEN3_INCLUDE_DIR}
                     ${Boost_INCLUDE_DIR}
                     ${CNPY_INCLUDE_DIR}
                     ${QUADPROG_INCLUDE_DIR})

add_executable(${REVEREND_EXECUTABLE} ${REVEREND_SRC_DIRECTORY}/${REVEREND_EXECUTABLE}.cpp)

add_custom_command(TARGET ${REVEREND_EXECUTABLE} POST_BUILD 
                    COMMAND ctags -R 
                    WORKING_DIRECTORY ${REVEREND_SRC_DIRECTORY} 
                    COMMENT "Creating tags" VERBATIM)

target_link_libraries(${REVEREND_EXECUTABLE} 
                      ${SUITESPARSE_LIBRARIES}
                      ${LAPACK_LIBRARIES}
                      # cholmod
                      # ccolamd
                      # camd
                      # colamd
                      # amd
                      # suitesparseconfig
                      # lapack
                      # blas
                      ${CNPY_LIBRARY}
                      ${NLOPT_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY} )
