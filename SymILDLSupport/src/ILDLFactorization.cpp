#include <exception>
#include <iostream>

#include "Eigen/Eigenvalues"
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

  // Dimension of A
  dim_ = A.rows();

  /// Preallocate storage to hold the incomplete factorization of A, computed
  /// using SYM-ILDL

  lilc_matrix<Scalar> L;
  block_diag_matrix<Scalar> D;
  std::vector<int> perm(dim_);

  /// Construct representation of A

  // Construct CSR representation of
  std::vector<int> row_ptr, col_idx;
  std::vector<Scalar> val;
  toCSR(A, row_ptr, col_idx, val);

  // Construct SYM-ILDL representation of passed matrix A.  Note that although
  // SYM-ILDL expects compressed COLUMN storage arguments, here we take
  // advantage of the fact that the CSR representation of A's UPPER TRIANGLE
  // actually coincides with the CSC representation of A's LOWER TRIANGLE :-)
  L.load(row_ptr, col_idx, val);

  /// Equilibrate A using a diagonal scaling matrix S, if requested.
  // This will overwrite A_ with SAS, and save the diagonal scaling matrix as
  // A_.S
  if (opts_.equilibration == Equilibration::Bunch)
    L.sym_equil();

  /// Record scaling matrix S
  S_.resize(dim_);
  for (int k = 0; k < dim_; ++k)
    S_(k) = L.S.main_diag[k];

  /// Compute fill-reducing reordering of A, if requested
  switch (opts_.order) {
  case Ordering::AMD:
    L.sym_amd(perm);
    break;
  case Ordering::RCM:
    L.sym_rcm(perm);
    break;
  case Ordering::None:
    // Set perm to be the identity permutation
    perm.resize(dim_);
    for (int k = 0; k < dim_; ++k)
      perm[k] = k;
    break;
  }

  // Apply this permutation to A_, if one was requested
  if (opts_.order != Ordering::None)
    L.sym_perm(perm);

  /// Compute in-place LDL factorization of P*S*A*S*P
  L.ildl_inplace(D, perm, opts_.max_fill_factor, opts_.drop_tol,
                 opts_.BK_pivot_tol,
                 (opts_.pivot_type == PivotType::Rook
                      ? lilc_matrix<Scalar>::pivot_type::ROOK
                      : lilc_matrix<Scalar>::pivot_type::BKP));

  /// Record the final permutation in P
  P_.resize(dim_);
  for (int k = 0; k < dim_; ++k)
    P_(k) = perm[k];

  /// Construct lower-triangular Eigen matrix L_ from L
  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve(L.nnz());

  // From the lilc_matrix documentation: A(m_idx[k][j], k) = m_x[k][j]
  for (int k = 0; k < L.n_cols(); ++k)
    for (int j = 0; j < L.m_idx[k].size(); ++j)
      triplets.emplace_back(L.m_idx[k][j], k, L.m_x[k][j]);

  L_.resize(dim_, dim_);
  L_.setFromTriplets(triplets.begin(), triplets.end());

  L.save("L.txt");
  D.save("D.txt");

  /// Construct and record eigendecomposition for block diagonal matrix D

  // Get the number of 1- and 2-d blocks in D
  size_t num_2d_blocks = D.off_diag.size();
  size_t num_1d_blocks = dim_ - 2 * num_2d_blocks;
  size_t num_blocks = num_1d_blocks + num_2d_blocks;

  // Preallocate storage for this computation
  Lambda_.resize(dim_);
  block_start_idxs_.resize(num_blocks);
  block_sizes_.resize(num_blocks);

  // 2x2 matrix we will use to store any 2x2 blocks of D
  Matrix2d Di;
  // Eigensolver for computing an eigendecomposition of the 2x2 blocks of D
  Eigen::SelfAdjointEigenSolver<Matrix2d> eig;

  int idx = 0; // Starting (upper-left) index of the current block
  for (size_t i = 0; i < num_blocks; ++i) {
    // Record the starting index of this block
    block_start_idxs_[i] = idx;

    if (D.block_size(i) > 1) {
      // This is a 2x2 block
      block_sizes_[i] = 2;

      // Extract 2x2 block from D

      // Extract diagonal elements
      Di(0, 0) = D.main_diag[idx];
      Di(1, 1) = D.main_diag[idx + 1];
      // Extract off-diagonal elements
      Di(0, 1) = D.off_diag.at(idx);
      Di(1, 0) = D.off_diag.at(idx);

      // Compute eigendecomposition of Di
      eig.compute(Di);

      // Record eigenvalues of this block
      Lambda_.segment<2>(idx) = eig.eigenvalues();

      // Record eigenvectors of this block
      Q_[i] = eig.eigenvectors();

      // Increment index
      idx += 2;
    } else {
      /// This is a 1x1 block
      block_sizes_[i] = 1;

      // Record eigenvalue
      Lambda_(idx) = D.main_diag[idx];

      // Increment index
      ++idx;
    }
  }

  // Record the fact that we now have a valid cached factorization
  initialized_ = true;
}

void ILDLFactorization::clear() {

  // If we have a cached factorization ...
  if (initialized_) {
    // Release the memory associated with this factorization
    block_start_idxs_.clear();
    block_sizes_.clear();
    Q_.clear();
  }

  // Record the fact that we no longer have a valid cached factorization
  initialized_ = false;
}

// Vector ILDLFactorization::solve(const Vector &b) const {

//  if (!initialized_)
//    throw std::logic_error(
//        "You must compute() a factorization before solving linear
//        systems!");

//  // Recall that since P'SASP ~ LDL', then A^-1 ~ S P L^{-T} D^-1 L^-1 P^T S

//  // Get non-const references to working variables
//  std::vector<Scalar> &tmp = const_cast<std::vector<Scalar> &>(tmp_);
//  std::vector<Scalar> &x = const_cast<std::vector<Scalar> &>(x_);
//  lilc_matrix<Scalar> &L = const_cast<lilc_matrix<Scalar> &>(L_);
//  block_diag_matrix<Scalar> &D = const_cast<block_diag_matrix<Scalar>
//  &>(D_);

//  /// STEP 1: Scale and permute right-hand side vector
//  for (int k = 0; k < b.size(); ++k)
//    tmp[k] = L.S[perm_[k]] * b(perm_[k]);

//  /// STEP 2:  SOLVE LDL'y = rhs

//  L.backsolve(tmp, x);

//  if (opts_.pos_def_mod)
//    D.pos_def_solve(x, tmp);
//  else
//    D.solve(x, tmp);

//  L.forwardsolve(tmp, x);

//  /// STEP 3:  Scale and permute solution
//  Vector X(b.size());
//  for (int k = 0; k < b.size(); ++k)
//    X(perm_[k]) = L.S[perm_[k]] * x[k];

//  return X;
//}

SparseMatrix ILDLFactorization::D(bool pos_def_mod) const {
  // We rebuild D from its eigendecomposition according to whether we are
  // enforcing positive-definiteness

  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve(dim_ + 2 * num_2x2_blocks());

  // Preallocate working variables
  int idx; // Starting index of current block

  Matrix2d Di; // Working space for reconstructing 2x2 blocks

  // Iterate over the blocks of D
  for (size_t i = 0; i < num_blocks(); ++i) {
    idx = block_start_idxs_[i];
    if (block_sizes_[i] == 1) {
      triplets.emplace_back(idx, idx,
                            pos_def_mod ? fabs(Lambda_(idx)) : Lambda_(idx));
    } else {
      // Reconstruct the 2x2 block here
      const Matrix2d &Qi = Q_.at(i);

      if (pos_def_mod)
        Di = Qi * Lambda_.segment<2>(idx).cwiseAbs().asDiagonal() *
             Qi.transpose();
      else
        Di = Qi * Lambda_.segment<2>(idx).asDiagonal() * Qi.transpose();

      for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
          triplets.emplace_back(idx + r, idx + c, Di(r, c));
    }
  }

  /// Reconstruct and return D
  SparseMatrix D(dim_, dim_);
  D.setFromTriplets(triplets.begin(), triplets.end());

  return D;
}

} // namespace SymILDLSupport
