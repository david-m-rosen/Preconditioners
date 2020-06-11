/** This class provides functionality for computing an incomplete LDL^T
 * factorization of a symmetric indefinite matrix using the SYM-ILDL library.
 *
 * The interface it provides is based upon the ones used by the Eigen library's
 * built-in matrix factorization types.
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "lilc_matrix.h" // SYM-ILDL matrix type

#include "SymILDLSupport_types.h"

namespace SymILDLSupport {

/// Enum class that sets the type of pivoting to use during factorization
enum class PivotType { Rook, BunchKaufman };

/// Enum class that sets the fill-reducing ordering to apply during
/// factorization
enum class Ordering { AMD, RCM, None };

/// Enum class that determines the type of equilibration (scaling) to apply to
/// the matrix before factorization
enum class Equilibration { Bunch, None };

/// This lightweight struct contains a simplified set of configuration options
/// for the SYM-ILDL library, as it is used in the SymILDLSupport
struct SymILDLOpts {

  /** Parameter controlling the maximum fill-in for the incomplete
   * lower-triangular factor L: each column of L is guanteed to have at most
   * max_fill_factor * (nnz(A) / dim(A)) nonzero elements. */
  double max_fill_factor = 3.0;

  /** Drop tolerance for elements of the incomplete lower-triangular factor L:
   * any elements l in L_k (the kth column of L) satisfying
   * |l| <= drop_tol * |L_k|_1
   * will be set to 0. */
  double drop_tol = 1e-3;

  /** This parameter controls the aggressiveness of the Bunch-Kaufman pivoting
   * procedure.  When BK_pivot_tol = 1, full Bunch-Kaufman pivoting is used;
   * if BK_pivot_tol = 0, partial pivoting is turned off, and the first non-zero
   * pivot under the diagonal will be used.  Intermediate values continuously
   * vary the aggressiveness of the pivoting: wither values closer to 0 favoring
   * locality in pivoting (pivots closer do the diagonal are used), and values
   * closer to 1 increasing the stability of the selected pivots.
   *
   * This parameter is useful for trading off preservation of the *structure* of
   * the incomplete factor L vs. controlling the magnitudes of its elements */
  double BK_pivot_tol = 1.0;

  /** This parameter determines the type of pivoting strategy to use during
   * factorization */
  PivotType pivot_type = PivotType::Rook;

  /** This parameter determines the fill-reducing variable reordering strategy
   * to use when factoring the matrix */
  Ordering order = Ordering::AMD;

  /** This parameter determines the equilibration (scaling) strategy to apply
   * when factoring the matrix */
  Equilibration equilibration = Equilibration::Bunch;

  /** A Boolean value indicating whether the block-diagonal matrix D should be
   * modified to enforce positive-definiteness of the factorization */
  bool pos_def_mod = false;
};

class ILDLFactorization {
private:
  /// Data members

  /** Structure containing options for the SYM-ILDL library */
  SymILDLOpts opts_;

  /// FACTORIZATION ELEMENTS: Elements of the factorization of PSASP = LDL'

  /** Lower-triangular factor */
  lilc_matrix<Scalar> L_;

  /** Block-diagonal matrix D */
  block_diag_matrix<Scalar> D_;

  /** Fill-reducing permutation P */
  lilc_matrix<Scalar>::idx_vector_type perm_;

  /** Working space for linear algebra operations */
  std::vector<Scalar> tmp_;
  std::vector<Scalar> x_;

  // Boolean value indicating whether the object contains a valid cached
  // factorization
  bool initialized_ = false;

public:
  /// Constructors

  /** Construct an empty ILDLFactorization object */
  ILDLFactorization(const SymILDLOpts &options = SymILDLOpts());

  /** Construct an ILDLFactorization object containing a factorization
   * of the passed matrix A */
  ILDLFactorization(const SparseMatrix &A,
                    const SymILDLOpts &options = SymILDLOpts());

  /// Mutators

  /** Set the options for the incomplete LDLT factorization.  Note that calling
   * this function will release any cached factorizations currently held
   * by the ILDLFactorization object */
  void setOptions(const SymILDLOpts &options);

  /** Compute an incomplete LDL^T factorization of the matrix A. */
  void compute(const SparseMatrix &A);

  /** Frees any cached factorizations currently held by the
   * ILDLFactorization object */
  void clear();

  /** Solve the linear system Ax = b, and return the solution x */
  Vector solve(const Vector &b) const;
};

} // namespace SymILDLSupport
