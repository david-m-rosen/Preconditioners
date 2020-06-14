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
  lilc_matrix<double> Alilc;

  Alilc.load(row_ptr, col_idx, val);

  /// Equilibrate A using a diagonal scaling matrix S, if requested.
  // This will overwrite Alilc with SAS, and save the diagonal scaling matrix as
  // Alilc.S
  if (opts_.equilibration == Equilibration::Bunch)
    Alilc.sym_equil();

  // Store the (diagonal) scaling matrix S
  S_.main_diag.resize(A.rows());
  for (int k = 0; k < S_.main_diag.size(); ++k)
    S_.main_diag[k] = Alilc.S.main_diag[k];

  /// Compute fill-reducing reordering of A, if requested
  switch (opts_.order) {
  case Ordering::AMD:
    Alilc.sym_amd(perm_);
    break;
  case Ordering::RCM:
    Alilc.sym_rcm(perm_);
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
    Alilc.sym_perm(perm_);

  /// Compute incomplete LDL factorization of P*S*A*S*P
  Alilc.ildl(L_, D_, perm_, opts_.max_fill_factor, opts_.drop_tol,
             opts_.BK_pivot_tol,
             (opts_.pivot_type == PivotType::Rook
                  ? lilc_matrix<Scalar>::pivot_type::ROOK
                  : lilc_matrix<Scalar>::pivot_type::BKP));

  /// Preallocate working space for linear algebra operations

  tmp_.resize(A.rows());
  x_.resize(A.rows());

  // Record the fact that we now have a valid cached factorization
  initialized_ = true;
}

Vector ILDLFactorization::solve(const Vector &b) const {

  if (!initialized_)
    throw std::logic_error(
        "You must compute() a factorization before solving linear systems!");

  // Recall that since P'SASP ~ LDL', then A^-1 ~ S P L^{-T} D^-1 L^-1 P^T S

  // Get non-const references to working variables
  std::vector<Scalar> &tmp = const_cast<std::vector<Scalar> &>(tmp_);
  std::vector<Scalar> &x = const_cast<std::vector<Scalar> &>(x_);
  lilc_matrix<Scalar> &L = const_cast<lilc_matrix<Scalar> &>(L_);
  block_diag_matrix<Scalar> &S = const_cast<block_diag_matrix<Scalar> &>(S_);
  block_diag_matrix<Scalar> &D = const_cast<block_diag_matrix<Scalar> &>(D_);

  /// STEP 1: Scale and permute right-hand side vector
  for (int k = 0; k < b.size(); ++k)
    tmp[k] = S[perm_[k]] * b(perm_[k]);

  /// STEP 2:  SOLVE LDL'y = rhs

  L.backsolve(tmp, x);

  if (opts_.pos_def_mod)
    D.pos_def_solve(x, tmp);
  else
    D.solve(x, tmp);

  L.forwardsolve(tmp, x);

  /// STEP 3:  Scale and permute solution
  Vector X(b.size());
  for (int k = 0; k < b.size(); ++k)
    X(perm_[k]) = S[perm_[k]] * x[k];

  return X;
}

void ILDLFactorization::clear() {

  // If we have a cached factorization ...
  if (initialized_) {
    // Release the memory associated with this factorization
    L_.list.clear();
    L_.row_first.clear();
    L_.col_first.clear();
    S_.main_diag.clear();
    S_.off_diag.clear();
    tmp_.clear();
    x_.clear();
  }

  // Record the fact that we no longer have a valid cached factorization
  initialized_ = false;
}

} // namespace SymILDLSupport
