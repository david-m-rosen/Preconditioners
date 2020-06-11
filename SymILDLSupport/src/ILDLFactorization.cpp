#include "PardisoSupport/SymmetricFactorization.h"
#include "PardisoSupport/PardisoUtils.h"

#include <exception>
#include <iostream>
#include <limits>

namespace PardisoSupport {

/// Constructors

// Basic constructor: just set the options
SymmetricFactorization::SymmetricFactorization(const PardisoOpts &options) {
  setOptions(options);

  // Suppress printing Pardiso licensing info
  putenv(const_cast<char *>(pardiso_lic_message_string_.c_str()));
}

// Advanced constructor: set the options, and then call compute() function to
// factor the passed matrix A
SymmetricFactorization::SymmetricFactorization(const SparseMatrix &A,
                                               const PardisoOpts &options) {
  setOptions(options);
  compute(A);

  // Suppress printing Pardiso licensing info
  putenv(const_cast<char *>(pardiso_lic_message_string_.c_str()));
}

void SymmetricFactorization::setOptions(const PardisoOpts &options) {
  /// Release any currently-held (cached) factorizations
  clear();

  /// Input checking
  if (options.max_iter_refine_steps < 0)
    throw std::invalid_argument("Maximum number of iterative refinement steps "
                                "must be a nonnegative integer");

  /// Check parameters for multi-recursive iterative linear solver (if used)
  if (options.mode == LinearAlgebraMode::Iterative) {
    if (options.max_SQMR_iters < 0)
      throw std::invalid_argument(
          "Maximum number of SQMR iterations for multi-recursive iterative "
          "solver must be a nonnegative integer");

    if ((options.rtol <= 0) || (options.rtol >= 1))
      throw std::invalid_argument("Target relative reduction in residual norm "
                                  "must be in the range (0,1)");

    if (options.grid_fact_dim <= 0)
      throw std::invalid_argument(
          "Grid size at which to revert to direct matrix factorization must be "
          "a positive integer");

    if (options.max_grid_levels <= 0)
      throw std::invalid_argument(
          "Maximum number of grid levels must be a positive integer");

    if (options.incomplete_fact_drop_tol < 0)
      throw std::invalid_argument(
          "Drop tolerance for the incomplete factorization must be a "
          "nonnegative real value");

    if (options.inv_fact_bound <= 0)
      throw std::invalid_argument("Bound on the norm of the inverse factor "
                                  "must be a positive real value");

    if (options.max_stag_steps < 0)
      throw std::invalid_argument(
          "Maximum admissible number of stagnation steps in Krylov subspace "
          "solver must be a nonnegative integer");
  } // Iterative mode parameter checking

  // Save the passed options
  opts_ = options;

  // Set a few additional Pardiso control parameters based upon these options
  mtype_ = -2;
  solver_ = (opts_.mode == LinearAlgebraMode::Iterative ? 1 : 0);
  msglvl_ = (opts_.verbose ? 1 : 0);
}

void SymmetricFactorization::compute(const SparseMatrix &A, int nrhs) {

  // If we already have a cached factorization stored ...
  if (initialized_) {
    // Release it
    clear();
  }

  // Argument checking
  if (A.rows() != A.cols())
    throw std::invalid_argument("Argument A must be a symmetric matrix !");

  if (nrhs <= 0)
    throw std::invalid_argument("The column dimension of the right-hand side "
                                "matrix B must be a positive integer");

  // Record dimension of A
  n_ = A.rows();

  // Record dimension of right-hand side vectors
  nrhs_ = nrhs;

  /// Construct CSR representation of A
  toCSR(A, ia_, ja_, va_);

  /// Initialize Pardiso

  // Get the number of processors to use from the OMP_NUM_THREADS environment
  // flag
  char *var = getenv("OMP_NUM_THREADS");
  numprocs_ = (var == NULL ? 0 : atoi(var));
  if (numprocs_ <= 0)
    throw std::runtime_error(
        "Environment variable OMP_NUM_THREADS must be set to a positive "
        "integer value to specify the number of threads the PARDISO library "
        "should use");

  // Initialize iparm[2-1] and iparm[4-1] through iparm[64-1]
  // with default values when calling pardiso_init
  iparm_[1 - 1] = 0;

  pardisoinit(pt_, &mtype_, &solver_, iparm_, dparm_, &error_);

  if (error_ != 0) {
    if (error_ == -10)
      throw std::runtime_error("Pardiso init: No Pardiso license file found!");
    else if (error_ == -11)
      throw std::runtime_error("Pardiso init: Pardiso license is expired!");
    else if (error_ == -12)
      throw std::runtime_error("Pardiso init: Wrong username or hostname!");
  }

  /// Fill in the rest of the Pardiso options

  /// iparm params

  iparm_[2 - 1] = 2;         // Use nested dissection ordering from METIS v. 4.1
  iparm_[3 - 1] = numprocs_; // NB:  THIS MUST BE SET EXPLICITLY!
  // Set maximum number of iterative refinement steps to perform
  iparm_[8 - 1] = opts_.max_iter_refine_steps;
  // Use variable scaling
  iparm_[11 - 1] = 1;
  // Use advanced symmetric weighted matchings (enables higher accuracy solves
  // for highly indefinite symmetric matrices)
  iparm_[13 - 1] = 2;
  // Use multi-recursive iterative solver?
  iparm_[32 - 1] = solver_;

  /// dparm params (only used with multi-recursive iterative solver)

  dparm_[1 - 1] =
      opts_.max_SQMR_iters;   // Maximum number of Krylov-subspace iterations
  dparm_[2 - 1] = opts_.rtol; // Target relative reduction in residual
  // Dimension at which to switch to direct sparse factorization
  dparm_[3 - 1] = opts_.grid_fact_dim;
  dparm_[4 - 1] = opts_.max_grid_levels; // Maximum number of grid levels
  // Drop tolerance for incomplete factor
  dparm_[5 - 1] = opts_.incomplete_fact_drop_tol;
  // Drop tolerance for Schur complement
  dparm_[6 - 1] = opts_.schur_comp_drop_tol;
  // Maximum fill-in per column in the incomplete factor
  dparm_[7 - 1] = opts_.max_fill_per_column;
  // Bound on the condition number of the inverse of the incomplete factor
  dparm_[8 - 1] = opts_.inv_fact_bound;
  // Maximum number of stagnation steps in the Krylov subspace method
  dparm_[9 - 1] = opts_.max_stag_steps;

  /// Perform analysis and numerical factorization
  int phase = 12;

  pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase, &n_, va_, ia_, ja_, NULL,
          &nrhs_, iparm_, &msglvl_, NULL, NULL, &error_, dparm_);

  // Record the fact that we now have a valid cached factorization
  initialized_ = true;
}

Vector SymmetricFactorization::solve(const Vector &b) const {

  if (!initialized_)
    throw std::logic_error(
        "You must compute() a factorization before solving linear systems!");

  if (b.size() != n_)
    throw std::invalid_argument("A and b must have the same row dimension!");

  if (nrhs_ != 1)
    throw std::invalid_argument(
        "Factorization of A was not constructed for vector right-hand sides!");

  // Allocate output vector x
  Vector x(n_);

  // Solve linear system using Pardiso

  int phase = 33; // Solve and iterative refinement
  pardiso(const_cast<void **>(pt_), const_cast<int *>(&maxfct_),
          const_cast<int *>(&mnum_), const_cast<int *>(&mtype_), &phase,
          const_cast<int *>(&n_), const_cast<double *>(va_),
          const_cast<int *>(ia_), const_cast<int *>(ja_), NULL,
          const_cast<int *>(&nrhs_), const_cast<int *>(iparm_),
          const_cast<int *>(&msglvl_), const_cast<double *>(b.data()), x.data(),
          const_cast<int *>(&error_), const_cast<double *>(dparm_));

  return x;
}

Matrix SymmetricFactorization::solve(const Matrix &B) const {
  if (!initialized_)
    throw std::logic_error(
        "You must compute() a factorization before solving linear systems!");

  if (B.rows() != n_)
    throw std::invalid_argument("A and B must have the same row dimension!");

  if (nrhs_ != B.cols())
    throw std::invalid_argument(
        "Right-hand side vector B has incorrect column dimension!");

  // Allocate output matrix X
  Matrix X(n_, nrhs_);

  // Solve linear system using Pardiso

  int phase = 33; // Solve and iterative refinement
  pardiso(const_cast<void **>(pt_), const_cast<int *>(&maxfct_),
          const_cast<int *>(&mnum_), const_cast<int *>(&mtype_), &phase,
          const_cast<int *>(&n_), const_cast<double *>(va_),
          const_cast<int *>(ia_), const_cast<int *>(ja_), NULL,
          const_cast<int *>(&nrhs_), const_cast<int *>(iparm_),
          const_cast<int *>(&msglvl_), const_cast<double *>(B.data()), X.data(),
          const_cast<int *>(&error_), const_cast<double *>(dparm_));

  return X;
}

Inertia SymmetricFactorization::inertia() const {

  if (!initialized_)
    throw std::logic_error(
        "You must compute() a factorization of A before querying its inertia!");

  size_t pos = iparm_[22 - 1];
  size_t neg = iparm_[23 - 1];
  size_t zero = n_ - pos - neg;

  return Inertia(pos, neg, zero);
}

void SymmetricFactorization::clear() {

  // If we have a cached factorization ...
  if (initialized_) {

    // Release it
    delete[] ia_;
    delete[] ja_;
    delete[] va_;

    // Release Pardiso internal memory
    int phase = -1;

    pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase, &n_, NULL, NULL, NULL, NULL,
            &nrhs_, iparm_, &msglvl_, NULL, NULL, &error_, dparm_);
  }

  // Record the fact that we no longer have a valid cached factorization
  initialized_ = false;
}

int SymmetricFactorization::check_error() const { return error_; }

SymmetricFactorization::~SymmetricFactorization() {
  // Release all internal memory
  clear();
}

} // namespace PardisoSupport
