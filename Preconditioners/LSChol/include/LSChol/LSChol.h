#pragma once

#include "Preconditioners/Types.h"

#include "suitesparse/SuiteSparse_config.h"
#include "suitesparse/cholmod_core.h"

namespace Preconditioners {

class LSChol {
private:
  /// Data members

  /// FACTORIZATION ELEMENTS

  /// Cholmod configuration struct
  cholmod_common common;

  /// Cholmod representation of sparse matrix A
  cholmod_sparse *A_;

  // Boolean value indicating whether the object contains a valid cached
  // factorization
  bool initialized_ = false;

  /// Helper function: initialize Cholmod
  void init();

public:
  /// Constructors

  /** Construct an empty LSChol object */
  LSChol() {}

  /** Construct an LSChol object containing a factorization
   * of the passed matrix A */
  LSChol(const SparseMatrix &A);

  /// Mutators

  /** Compute the "Q-less" QR factorization of the matrix matrix A. */
  void compute(const SparseMatrix &A);

  /** Frees any cached factorizations currently held by the
   * LSChol object */
  void clear();

  ~LSChol();
};

} // namespace Preconditioners
