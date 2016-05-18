
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/mapping_manifold.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>

#include <laplace_problem.h>

LaplaceProblem::LaplaceProblem ()
  :
  pfe("ParsedFiniteElement", "FE_Q(1)"),
  pq("ParsedQuadrature", "gauss", 2, 2),
  dof_handler (triangulation),
  forcing_term("Forcing term", 1, "0"),
  exact_solution("Exact solution", 1, "0"),
  pout("ParsedDataOut"),
  eh("ErrorHandler")
{}

void LaplaceProblem::declare_parameters(ParameterHandler &prm)
{
  this->add_parameter(  prm, &manifold_id,
                        "Use Manifold IDs",
                        "none",
                        Patterns::Selection("none|everywhere|boundary"),
                        "Available options: \n"
                        " none:       No manifold is used\n"
                        " boundary:   Manifold ids are used on the boundary\n"
                        " everywhere: Manifold ids are used in all the domain\n");

  this->add_parameter(  prm, &mapping_string,
                        "Mapping",
                        "MappingQ",
                        Patterns::Selection("MappingQ|MappingManifold"),
                        "Available options: \n"
                        " MappingQ:        Use MappingQ\n"
                        " MappingManifold: Use MappingManifold\n");

  this->add_parameter(prm, &polynomial_degree,
                      "Mapping - Polynomial degree", "1",
                      Patterns::Integer(0),
                      "Mapping - Polynomial degree");

  this->add_parameter(prm, &use_mapping_q_on_all_cells,
                      "Mapping - all cells", "false",
                      Patterns::Bool(),
                      "Mapping - Use Mapping q on all cells");

  this->add_parameter(prm, &n_cells,
                      "n cells", "10",
                      Patterns::Integer(0),
                      "The number n_cells of elements for this initial triangulation");

  this->add_parameter(prm, &refinements,
                      "Refinements", "2",
                      Patterns::Integer(0),
                      "Number of refiniments");

  this->add_parameter(prm, &n_cycles,
                      "Number of cycles", "6",
                      Patterns::Integer(0),
                      "Number of refiniments");
}

void LaplaceProblem::make_grid ()
{
  const Point<2> center (0,0);
  const double inner_radius = 0.5,
               outer_radius = 1.0;

  GridGenerator::hyper_shell (triangulation,
                              center, inner_radius, outer_radius,
                              n_cells);

  static const SphericalManifold<2> boundary_description(center);

  if ( manifold_id ==  "everywhere" )
    triangulation.set_all_manifold_ids(0);
  else if (manifold_id == "boundary")
    triangulation.set_all_manifold_ids_on_boundary(0);

  if (manifold_id != "none")
    triangulation.set_manifold(0, boundary_description);

  triangulation.refine_global(refinements);

  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;
}

void LaplaceProblem::setup_system ()
{
  fe = pfe();
  dof_handler.distribute_dofs (*fe);
  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
  constraints.clear ();
  // DoFTools::make_hanging_node_constraints (dof_handler,
  //  constraints);
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<2>(),
                                            constraints);
  constraints.close ();
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);

  //Mapping:
  if (mapping_string == "MappingQ")
    mapping = SP(new MappingQ<2>(polynomial_degree,use_mapping_q_on_all_cells));
  else
    mapping = SP(new MappingManifold<2>());
}

void LaplaceProblem::assemble_system ()
{
  FEValues<2> fe_values (*mapping, *fe, pq,
                         update_values | update_gradients | update_JxW_values |  update_quadrature_points);

  const unsigned int   dofs_per_cell = fe->dofs_per_cell;
  const unsigned int   n_q_points    = pq.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  // const Coefficient<dim> coefficient;
  std::vector<double>    forcing_term_values (n_q_points);

  typename DoFHandler<2>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit (cell);
      forcing_term.value_list(fe_values.get_quadrature_points(),
                              forcing_term_values);
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_grad(i,q_index)      *
                                   fe_values.shape_grad(j,q_index) *
                                   fe_values.JxW(q_index));
            cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                            forcing_term_values[q_index] *
                            fe_values.JxW(q_index));
          }
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }
}


void LaplaceProblem::solve ()
{
//  SolverControl           solver_control (system_matrix.m(), 1e-12);
//  SolverCG<>              solver (solver_control);

//  PreconditionJacobi<SparseMatrix<double> > preconditioner;
//  preconditioner.initialize (system_matrix, 1.4);

//  LinearOperator<> A  =
//    linear_operator<>( system_matrix );

//  LinearOperator<> A_inv =
//    inverse_operator<>( A, solver, preconditioner);

//  solution = A_inv * system_rhs;

  SparseDirectUMFPACK a_inverse;
  a_inverse.initialize(system_matrix);
  a_inverse.vmult(solution, system_rhs);

  constraints.distribute (solution);
}


void LaplaceProblem::output_results (unsigned int cycle) const
{
  pout.prepare_data_output (dof_handler, Utilities::int_to_string(cycle, 2));
  pout.add_data_vector (solution, "solution");
  pout.write_data_and_clear(*mapping);

  eh.error_from_exact( *mapping,
                       dof_handler, solution, exact_solution);
}

void LaplaceProblem::run ()
{
  make_grid ();
  for (unsigned int i=0; i<n_cycles; ++i)
    {
      if (i>0)
        triangulation.refine_global(1);

      setup_system ();
      assemble_system ();
      solve ();
      output_results (i);
    }
  eh.output_table(std::cout);
}
