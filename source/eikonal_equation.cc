
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
#include <deal.II/numerics/error_estimator.h>

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

#include <eikonal_equation.h>

EikonalEquation::EikonalEquation ()
  :
  pfe("ParsedFiniteElement", "FE_Q(1)"),
  pgg("ParsedGridGenerator"),
  pdbc("ParsedDirichletBCs"),
  pq("ParsedQuadrature", "gauss", 2, 2),
  dof_handler (triangulation),
  forcing_term("Forcing term", 1, "0"),
  exact_solution("Exact solution", 1, "0"),
  pout("ParsedDataOut"),
  eh("ErrorHandler"),
  norm(0.0)
{}

void EikonalEquation::declare_parameters(ParameterHandler &prm)
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

  this->add_parameter(prm, &epsilon,
                      "Epsilon", "1e-5",
                      Patterns::Double(),
                      "Wight for the Laplacian stabilization");
  this->add_parameter(prm, &epsilon_decrement,
                      "Epsilon Decrement", "1e-3",
                      Patterns::Double(0.0),
                      "Wight for the Laplacian stabilization");

  this->add_parameter(prm, &alpha,
                      "Alpha", "0.5",
                      Patterns::Double(0.0, 1.0),
                      "Wight for the Laplacian stabilization");

  this->add_parameter(prm, &n_cycles,
                      "Number of cycles", "6",
                      Patterns::Integer(0),
                      "Number of refiniments");


  this->add_parameter(prm, &norm_tolerance,
                      "Norm Tolerance", "1e-3",
                      Patterns::Double(0.0),
                      "Wight for the Laplacian stabilization");
}

void EikonalEquation::make_grid ()
{
  pgg.create(triangulation);
  triangulation.refine_global(refinements);


  // const Point<2> center (0,0);
  // const double inner_radius = 0.5,
  //              outer_radius = 1.0;
  //
  // GridGenerator::hyper_shell (triangulation,
  //                             center, inner_radius, outer_radius,
  //                             n_cells);
  //
  // static const SphericalManifold<2> boundary_description(center);



  // if ( manifold_id ==  "everywhere" )
  //   triangulation.set_all_manifold_ids(0);
  // else if (manifold_id == "boundary")
  //   triangulation.set_all_manifold_ids_on_boundary(0);
  //
  // if (manifold_id != "none")
  //   triangulation.set_manifold(0, boundary_description);



  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;
}

void EikonalEquation::setup_system ()
{
  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
   constraints);
  pdbc.interpolate_boundary_values (  dof_handler,
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

void EikonalEquation::assemble_system ()
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

  std::cout << "===============" << std::endl << std::flush;
  Vector<double> local_norm (triangulation.n_active_cells());


  VectorTools::integrate_difference (*mapping, dof_handler,
                                     old_solution,
                                     ZeroFunction< 2, double >(triangulation.n_active_cells()),
                                     local_norm,
                                     pq,
                                     VectorTools::H1_seminorm );
  norm = local_norm.l2_norm();
  // double tmp = 0;
  // norm = 0;
  // for (; cell!=endc; ++cell)
  //   {
  //     fe_values.reinit (cell);
  //     for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
  //
  //       {
  //         tmp = 0;
  //         for (unsigned int i=0; i<dofs_per_cell; ++i)
  //           {
  //             tmp += fe_values.shape_grad(i,q_index) * fe_values.shape_grad(i,q_index)* fe_values.JxW(q_index);
  //           }
  //         norm += tmp;
  //       }
  //   }
  // norm = std::sqrt(norm);
  std::cout << "Norm = " << norm << std::endl << std::flush;

  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit (cell);
      forcing_term.value_list(fe_values.get_quadrature_points(),
                              forcing_term_values);

      std::vector< Tensor< 1, 2, double >> old_solution_values(n_q_points);
      fe_values.get_function_gradients(old_solution, old_solution_values);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          // std::cout << old_solution[q_index] << std::endl << std::flush;
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  // for (unsigned int k=0; k<dofs_per_cell; ++k)
                  // {
                  if (norm == 0)
                    {
                      cell_matrix(i,j) += epsilon*(
                                            fe_values.shape_grad(i,q_index) *
                                            fe_values.shape_grad(j,q_index) *
                                            fe_values.JxW(q_index)
                                          );

                    }
                  else
                    {
                      cell_matrix(i,j) += (
                                            fe_values.shape_grad( j,q_index) *
                                            old_solution_values[q_index]/norm *
                                            // fe_values.shape_grad( k,q_index) *
                                            (fe_values.shape_value(i,q_index)
                                             + fe_values.shape_grad(i,q_index) *
                                             old_solution_values[q_index]/norm
                                            )*
                                            fe_values.JxW(q_index)
                                          );
                      cell_matrix(i,j) += epsilon*(
                                            fe_values.shape_grad(i,q_index) *
                                            fe_values.shape_grad(j,q_index) *
                                            fe_values.JxW(q_index)
                                          );
                    }
                  // cell_matrix(i,j) += (//std::abs(
                  //                       (
                  //                       //     fe_values.shape_grad( j,q_index) *
                  //                       //     old_solution_values[q_index]/norm *
                  //                       //     // fe_values.shape_grad( k,q_index) *
                  //                       //     fe_values.shape_value(i,q_index) *
                  //                         fe_values.shape_grad( j, q_index) ) *
                  //                       // fe_values.shape_grad( k,q_index) *
                  //                       fe_values.shape_grad( i, q_index) *
                  //                       fe_values.JxW(q_index)
                  //                     );
                  // }
                }

              if (norm==0)
                cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                                forcing_term_values[q_index] *
                                fe_values.JxW(q_index));
              else                cell_rhs(i) += ((fe_values.shape_value(i,q_index)+ fe_values.shape_grad(i,q_index) *
                                                     old_solution_values[q_index]/norm
                                                    ) *
                                                    forcing_term_values[q_index] *
                                                    fe_values.JxW(q_index));
              // else
              //   cell_rhs(i) += fe_values.shape_grad(i,q_index) *
              //                  old_solution_values[q_index]/norm  *
              //                  fe_values.JxW(q_index);


            }
        }
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }
  epsilon = epsilon - epsilon_decrement;
  if(epsilon <= 0)
  {
    epsilon = epsilon_decrement;
  }
  std::cout << "epsilon = " << epsilon << std::endl << std::flush;

}


void EikonalEquation::solve ()
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

  auto tmp = old_solution;
  tmp *= alpha;
  solution *= (1-alpha);
  solution += tmp;
  constraints.distribute (solution);

}


void EikonalEquation::output_results (unsigned int cycle) const
{
  pout.prepare_data_output (dof_handler, Utilities::int_to_string(cycle, 2));
  pout.add_data_vector (solution, "solution");
  pout.write_data_and_clear(*mapping);

  eh.error_from_exact( *mapping,
                       dof_handler, solution, exact_solution);
}

void EikonalEquation::run ()
{
  make_grid ();
  unsigned int i=0;
  do
    {
      fe = pfe();
      dof_handler.distribute_dofs (*fe);
      if (i>0)
        {
          SolutionTransfer<2, Vector<double> > soltrans(dof_handler);
          soltrans.prepare_for_coarsening_and_refinement(solution);

          Vector<float> estimated_error_per_cell (triangulation.n_active_cells());



          KellyErrorEstimator<2,2>::estimate (*mapping,
                                              dof_handler,
                                              QGauss <1> (fe->degree + 1),
                                              typename FunctionMap<2>::type(),
                                              solution,
                                              estimated_error_per_cell,
                                              ComponentMask(),
                                              0,
                                              0,
                                              triangulation.locally_owned_subdomain());


          pgr.mark_cells(estimated_error_per_cell, triangulation);

          triangulation.prepare_coarsening_and_refinement();
          soltrans.prepare_for_coarsening_and_refinement (old_solution);


          triangulation.execute_coarsening_and_refinement ();

          // triangulation.refine_global();
          dof_handler.distribute_dofs (*fe);
          old_solution.reinit(dof_handler.n_dofs());

          soltrans.interpolate(solution, old_solution);
        }
      else
        {
          dof_handler.distribute_dofs (*fe);
          old_solution.reinit(dof_handler.n_dofs());
        }
      setup_system ();
      assemble_system ();
      solve ();
      output_results (i);
      i++;
    }
  while (std::abs(norm-1)>norm_tolerance );
  eh.output_table(std::cout);
}
