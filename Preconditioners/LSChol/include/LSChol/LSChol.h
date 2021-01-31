#pragma once


#include "Preconditioners/Types.h"

namespace Preconditioners {

class LSCholesky {
private:
  /// Data members

  /// FACTORIZATION ELEMENTS

  /** Upper-triangular factor */
  SparseMatrix R_;

  // Boolean value indicating whether the object contains a valid cached
  // factorization
  bool initialized_ = false;

public:
  /// Constructors

  /** Construct an empty LSCholesky object */
  LSCholesky();

  /** Construct an LSCholesky object containing a factorization
   * of the passed matrix A */
  LSCholesky(const SparseMatrix &A);

  /// Mutators

  /** Compute the "Q-less" QR factorization of the matrix matrix A. */
  void compute(const SparseMatrix &A);

  /** Frees any cached factorizations currently held by the
   * LSCholesky object */
  void clear();

  /// Accessors

  /** Return the upper-triangular factor R */
  const SparseMatrix &R() const { return R_; }

  /// Linear-algebraic operations

};

} // namespace Preconditioners
