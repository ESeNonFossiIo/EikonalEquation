#ifndef _eikonal_equation_h
#define _eikonal_equation_h

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal2lkit/parameter_acceptor.h>

#include <deal2lkit/parsed_grid_refinement.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_quadrature.h>

#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_grid_generator.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>

#include <deal2lkit/parsed_data_out.h>
#include <deal2lkit/error_handler.h>

using namespace dealii;
using namespace deal2lkit;

class EikonalEquation  : public ParameterAcceptor
{
public:
  EikonalEquation ();

  void run ();

  /** Declare parameters for this class to function properly. */
  virtual void declare_parameters(ParameterHandler &prm);

private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void output_results (unsigned int cycle) const;

  Triangulation<2>            triangulation;

  // Finite Element:
  ParsedFiniteElement<2>      pfe;
  FiniteElement<2>            *fe;
  ParsedGridGenerator<2>      pgg;
  ParsedDirichletBCs<2>       pdbc;
  ParsedGridRefinement        pgr;

  // Quadrature:
  ParsedQuadrature<2>         pq;

  DoFHandler<2>               dof_handler;
  std::shared_ptr<Mapping<2 > >  mapping;

  SparsityPattern             sparsity_pattern;
  SparseMatrix<double>        system_matrix;

  Vector<double>              solution;
  Vector<double>              old_solution;
  Vector<double>              system_rhs;

  ConstraintMatrix            constraints;

  // Forcing Term:
  ParsedFunction<2>           forcing_term;
  ParsedFunction<2>           exact_solution;

  // Problems parameters:
  /**
   * The number n_cells of elements for this initial triangulation.
   */
  unsigned int               n_cells;

  /**
   * Number of refinements.
   */
  unsigned int                refinements;

  /**
   * Number of cycles.
   */
  unsigned int                n_cycles;

  /**
   * Define whether and where use manifold ids.
   */
  std::string             manifold_id;

  // Mapping:
  /**
   * Type of mapping
   */
  std::string             mapping_string;

  /**
   * polynomial_degree
   */
  unsigned int polynomial_degree;

  /**
   * use_mapping_q_on_all_cells
   */
  bool use_mapping_q_on_all_cells;

  double epsilon;
  double epsilon_decrement;
  double alpha;
  double norm_tolerance;
  double norm;
  // Output:
  mutable ParsedDataOut<2>    pout;
  mutable ErrorHandler<1>     eh;
};

#endif
