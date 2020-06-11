#include <exception>
#include <iostream>

#include "SymILDLSupport/ILDLFactorization.h"
#include "SymILDLSupport/SymILDLUtils.h"

namespace SymILDLSupport {

/// Constructors

// Basic constructor: just set the options
ILDLFactorization::ILDLFactorization(const SymILDLOpts &options) {
  setOptions(options);
}

// Advanced constructor: set the options, and then call compute() function to
// factor the passed matrix A
ILDLFactorization::ILDLFactorization(const SparseMatrix &A,
                                     const SymILDLOpts &options) {
  setOptions(options);
  compute(A);
}

void ILDLFactorization::setOptions(const SymILDLOpts &options) {
  /// Release any currently-held (cached) factorizations
  clear();

  /// Input checking

  if (options.max_fill_factor <= 0)
    throw std::invalid_argument("Maximum fill-factor must be a positive value");

  if (options.drop_tol < 0 || options.drop_tol > 1)
    throw std::invalid_argument(
        "Drop tolerance must be a value in the range [0,1]");

  if (options.BK_pivot_tol < 0 || options.BK_pivot_tol > 1)
    throw std::invalid_argument(
        "Bunch-Kaufman pivoting tolerance must be a value in the range [0,1]");

  // Save the passed options
  opts_ = options;
}

void ILDLFactorization::compute(const SparseMatrix &A) {

  // If we already have a cached factorization stored ...
  if (initialized_) {
    // Release it
    clear();
  }

  /// Argument checking
  if (A.rows() != A.cols())
    throw std::invalid_argument("Argument A must be a symmetric matrix!");

  /// Construct representation of A

  // Construct CSR representation of
  std::vector<int> row_ptr, col_idx;
  std::vector<Scalar> val;
  toCSR(A, row_ptr, col_idx, val);

  // Construct SYM-ILDL representation of passed matrix A.  Note that although
  // SYM-ILDL expects compressed COLUMN storage arguments, here we take
  // advantage of the fact that the CSR representation of A's UPPER TRIANGLE
  // actually coincides with the CSC representation of A's LOWER TRIANGLE :-)
  A_.load(row_ptr, col_idx, val);

  /// Equilibrate A using a diagonal scaling matrix S, if requested.
  // This will overwrite A_ with SAS, and save the diagonal scaling matrix as
  // A_.S
  if (opts_.equilibration == Equilibration::Bunch)
    A_.sym_equil();

  /// Compute fill-reducing reordering of A_, if requested
  switch (opts_.order) {
  case Ordering::AMD:
    A_.sym_amd(perm_);
    break;
  case Ordering::RCM:
    A_.sym_rcm(perm_);
    break;
  case Ordering::None:
    // Set perm to be the identity permutation
    perm_.resize(A.rows());
    for (int k = 0; k < A.rows(); ++k)
      perm_[k] = k;
    break;
  }

  // Apply this permutation to A_, if one was requested
  if (opts_.order != Ordering::None)
    A_.sym_perm(perm_);

  /// Compute in-place LDL factorization of A_ = P*S*A*S*P
  A_.ildl_inplace(D_, perm_, opts_.max_fill_factor, opts_.drop_tol,
                  opts_.BK_pivot_tol,
                  (opts_.pivot_type == PivotType::Rook
                       ? lilc_matrix<Scalar>::pivot_type::ROOK
                       : lilc_matrix<Scalar>::pivot_type::BKP));

  // Record the fact that we now have a valid cached factorization
  initialized_ = true;
}

Vector ILDLFactorization::solve(const Vector &b) const {

  if (!initialized_)
    throw std::logic_error(
        "You must compute() a factorization before solving linear systems!");
}

void ILDLFactorization::clear() {

  // If we have a cached factorization ...
  if (initialized_) {
    // Release the memory associated with this factorization
    A_.list.clear();
    A_.row_first.clear();
    A_.col_first.clear();
    A_.S.main_diag.clear();
    A_.S.off_diag.clear();
  }

  // Record the fact that we no longer have a valid cached factorization
  initialized_ = false;
}

} // namespace SymILDLSupport
