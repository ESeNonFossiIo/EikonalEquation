/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */

#include <fstream>
#include <iostream>
#include <laplace_problem.h>

using namespace dealii;
using namespace deal2lkit;

int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  deallog.depth_console (2);

  std::string parameters = argc > 1 ? argv[1] : "parameters.prm";

  LaplaceProblem laplace_problem;
  ParameterAcceptor::initialize(
    "parameters.prm",
    "used_parameters.prm" );
  laplace_problem.run ();

  return 0;
}
