/** This class provides functionality for factoring and solving symmetric
 * indefinite linear systems, using the functionality provided by the Parallel
 * Sparse Direct (Pardiso) library.
 *
 * The interface it provides is based upon the ones used by the Eigen library's
 * matrix factorizations.
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "SYMILDLSupport/SYMILDLSupport_types.h"

namespace SYMILDLSupport {

class ILDLFactorization {
private:
  /// Data members

  // Options structure for the Pardiso library
  PardisoOpts opts_;

  // Pointers to store the row, column, and value arrays used to represent the
  // symmetric matrix to be factored in the compressed sparse row (CSR) format
  // required by the Pardiso library
  int *ia_ = nullptr;
  int *ja_ = nullptr;
  double *va_ = nullptr;

  // Working space and control parameters for Pardiso
  void *pt_[64];     // Working space for Pardiso library
  int iparm_[64];    // Control parameters for direct factorization methods
  double dparm_[64]; // Additional control parameters for iterative solver
  int mtype_;  // Pardiso's internal representation for the type of matrix to be
               // factored
  int solver_; // Pardiso's internal representation for the solver to be used
               // (direct vs. multi-recursive iterative)
  int msglvl_; // Message level for Pardiso solver
  int numprocs_;   // Number of parallel processes to use (set from environment
                   // variable OMP_NUM_THREADS
  int maxfct_ = 1; // Number of matrix factorizations to cache (=== 1)
  int mnum_ = 1;   // Ordinal index of current matrix to factor (=== 1)
  int n_;          // Dimension of matrix to be factored
  int nrhs_;       // Columns of B in AX = B

  int error_ = 0; // Error flag from Pardiso solver

  // Boolean value indicating whether the object contains a valid cached
  // factorization
  bool initialized_ = false;

  std::string pardiso_lic_message_string_ = "PARDISOLICMESSAGE=1";

public:
  /// Constructors

  /** Construct an empty SymmetricFactorization object */
  SymmetricFactorization(const PardisoOpts &options = PardisoOpts());

  /** Construct a SymmetriFactorization object containing a factorization
   * of the passed matrix A */
  SymmetricFactorization(const SparseMatrix &A,
                         const PardisoOpts &options = PardisoOpts());

  /// Mutators

  /** Set the options for the symmetric factorization.  Note that calling
   * this function will release any cached factorizations currently held
   * by the SymmetricFactorization object */
  void setOptions(const PardisoOpts &options);

  /** Compute a symmetric factorization of the matrix A.  Here nrhs is a
   * parameter indicating the column dimension of the right-hand side matrix B
   * in the linear systems AX = B that this factorization will be used to
   * solve.
   */
  void compute(const SparseMatrix &A, int nrhs = 1);

  /** Frees any cached factorizations currently held by the
   * SymmetricFactorization object */
  void clear();

  /** Solve the linear system Ax = b, and return the solution x */
  Vector solve(const Vector &b) const;

  /** Solve the linear system AX = B, and return the solution X */
  Matrix solve(const Matrix &B) const;

  /** Returns a three-element tuple (pos, neg, zero) containing the number of
   * positive, negative, and zero eigenvalues of the matrix A) */
  Inertia inertia() const;

  /** Returns the value of the Pardiso error status flag returned by the last
   * operation */
  int check_error() const;

  /// Destructor
  ~SymmetricFactorization();
};

} // namespace SYMILDLSupport
