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

class ILDLFactorization {
private:
  /// Data members

  /** Structure containing options for the SYM-ILDL library */
  SymILDLOpts opts_;

  /** Local copy of A; this matrix will be modified in-place as the
   * factorization is performed */
  lilc_matrix<Scalar> A_;

  /** Block-diagonal matrix D */
  block_diag_matrix<Scalar> D_;

  /** Fill-reducing permutation */
  lilc_matrix<Scalar>::idx_vector_type perm_;

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
