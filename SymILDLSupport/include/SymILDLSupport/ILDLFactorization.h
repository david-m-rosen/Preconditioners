/** This class provides functionality for computing an incomplete LDL^T
 * factorization of a symmetric indefinite matrix using the SYM-ILDL library.
 *
 * The interface it provides is based upon the ones used by the Eigen library's
 * built-in matrix factorization types.
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include <unordered_map>

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
};

/** This lightweight class computes an incomplete LDL^T factorization of the
 * form:
 *
 * P'SASP ~ LDL'
 *
 * where:
 *
 * - S is an [optional] diagonal scaling matrix used to equilibrate A
 * - P is an [optional] fill-reducing row and column permutation for A
 * - L is a lower-triangular factor
 * - D is a block-diagonal matrix with blocks of size <= 2
 */
class ILDLFactorization {
private:
  /// Data members

  /** Dimension of the matrix stored in this factorization */
  size_t dim_;

  /** Structure containing options for the SYM-ILDL library */
  SymILDLOpts opts_;

  /// FACTORIZATION ELEMENTS: Elements of the factorization of PSASP = LDL'

  /** Permutation P */
  PermutationVector P_;

  /** Diagonal scaling matrix S */
  Vector S_;

  /** Lower-triangular factor */
  SparseMatrix L_;

  /// We store an eigendecomposition of the block-diagonal matrix D

  Vector Lambda_;

  /** These vectors keep track of the starting (upper-left) index of each of the
   * blocks on D's diagonal, and the dimension (1 or 2) of that block */
  std::vector<int> block_start_idxs_;
  std::vector<int> block_sizes_;

  /** This map associates to each 2x2 block Di the orthogonal matrix Qi such
   * that Di = Qi * Lambda_i * Qi' */
  std::unordered_map<int, Matrix2d> Q_;

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

  /** Approximate the solution of Ax = b using the incomplete factorization */
  // Vector solve(const Vector &b) const;

  /// Accessors

  /** Return the dimension of the matrix stored in this factorization */
  const size_t dim() const { return dim_; }

  /** Return fill-reducing permutation ordering used in the factorization */
  const PermutationVector &P() const { return P_; }

  /** Return the equilibration (scaling) matrix S */
  const Vector &S() const { return S_; }

  /** Return the lower-triangular factor L */
  const SparseMatrix &L() const { return L_; }

  /** Return the block-diagonal matrix D.  If pos_def_mod = true, the returned
   * matrix is modified to ensure that it is positive-definite */
  SparseMatrix D(bool pos_def_mod = false) const;

  /** Compute the matrix-vector product D*x.  If pos_def_mod is 'true'; the
   * product is computed with a positive-definite modification with D*/
  Vector Dproduct(const Vector &x, bool pos_def_mod = false) const;

  /** Solve the linear system Dx = b.  If pos_def_mod is 'true', the system is
   * solved with D replaced by its positive-definite modification */
  Vector Dsolve(const Vector &b, bool pos_def_mode = false) const;

  /** Solve the linear system (D+)^{1/2} * x = b, where D+ is the
   * positive-definite modification of the block-diagonal matrix D, and M^(1/2)
   * denotes the symmetric square root of the positive-definite matrix M. */
  Vector sqrtDsolve(const Vector &b) const;

  /** Return the total number of blocks in the block-diagonal matrix D */
  size_t num_blocks() const { return block_sizes_.size(); }

  /** Return the number of 2x2 blocks in the block-diagonal matrix D */
  size_t num_2x2_blocks() const { return Q_.size(); }
};

} // namespace SymILDLSupport
