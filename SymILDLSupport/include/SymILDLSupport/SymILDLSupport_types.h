/** A set of typedefs describing the basic types that will be used throughout
 * the SymILDL library.
 *
 *  Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace SymILDLSupport {

/// Some useful typedefs for the SymILDL library

/// Linear algebra types
typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

/** We use row-major storage order to take advantage of fast (sparse-matrix)
 * (dense-vector) multiplications when OpenMP is available (cf. the Eigen
 * documentation page on "Eigen and Multithreading") */
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;

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
   * max_fill_factor*nnz(A) nonzero elements. */
  double max_fill_factor = 3.0;

  /** Drop tolerance for elements of the incomplete lower-triangular factor L:
   * any elements l in the kth column of L |l| <= drop_tol * |L_k|_1, where L_k
   * is the kth column, will be set to 0. */
  double drop_tol = 1e-3;

  /** This parameter controls the aggressiveness of the Bunch-Kaufman pivoting
   * procedure.  When BK_pivot_tol >= 1, full Bunch-Kaufman pivoting is used;
   * when BK_pivot_tol is 0, the pivoting procedure is faster, chooses poorer
   * pivots. Values between 0 and 1 vary the aggressiveness of Bunch-Kaufman
   * pivoting. */
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
};

} // namespace SymILDLSupport
