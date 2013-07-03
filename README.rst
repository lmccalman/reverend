reverend
========

Kernel Bayes' Rule implementation and extensions as part of my PhD.


Requirements
------------

For the C++ Executables:
* Boost 1.53 (may work with earlier versions)
* Eigen3
* Nlopt (included in this source tree)

For the Python scripts:
* Python2
* Numpy/Scipy
* Matplotlib

Building
--------

First you need to build cnpy. In the directory cnpy type:

$ cmake .
$ make

Now, go into the cpp directory and build the executable in-source:

$ cmake .
$ make

CMake will complain if the required libraries aren't installed.

Demo
----
Just run

$python demo_regression.py

In the regression folder! If all is well, you will be rewarded with pretty
pictures.

Bugs
----
It is almost certain that this code will break on a Windows machine. I'll fix
this fairly soon.

