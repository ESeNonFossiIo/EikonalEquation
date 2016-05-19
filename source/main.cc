#include <fstream>
#include <iostream>
#include <eikonal_equation.h>

using namespace dealii;
using namespace deal2lkit;

int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  deallog.depth_console (2);

  std::string parameters = argc > 1 ? argv[1] : "parameters.prm";

  EikonalEquation eikonal_equation;
  ParameterAcceptor::initialize(
    "parameters.prm",
    "used_parameters.prm" );
  eikonal_equation.run ();

  return 0;
}
