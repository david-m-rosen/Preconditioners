#include "PardisoSupport/PardisoUtils.h"

namespace PardisoSupport {

/** Given a SYMMETRIC, ROW-MAJOR sparse Eigen matrix A, this function constructs
 * and returns three arrays ia, ja, and va describing the matrix A in the
 * compressed sparse row (CSR) format required by the Pardiso library.  Note
 * that since A is assumed to be symmetric, only the UPPER TRIANGLE of A is
 * referenced.
 *
 * The user is responsible for deleting the arrays ia, ja, and va
 * returned by this function.
 */

void toCSR(const SparseMatrix &A, int *&ia, int *&ja, double *&va, double eps) {

  // Construct a sparse matrix by adding the smallest possible double value to
  // A's diagonal -- this is to ensure that EVERY DIAGONAL ELEMENT of A
  // (whether or not it is 0) is explicitly represented in the CSR arrays
  // returned by this function, as required by Pardiso

  // Construct a version of A obtained by adding eps to its diagonal
  SparseMatrix Id(A.rows(), A.rows());
  Id.setIdentity();

  SparseMatrix Aplus_UT = (A + eps * Id).triangularView<Eigen::Upper>();

  ia = new int[A.rows() + 1];
  ja = new int[A.nonZeros()];
  va = new double[A.nonZeros()];

  size_t idx = 0;
  for (size_t r = 0; r < Aplus_UT.outerSize(); ++r) {
    // Store starting index for the for the current (rth) row.
    // NB:  Pardiso uses 1-based indexing!
    ia[r] = idx + 1;

    for (SparseMatrix::InnerIterator it(Aplus_UT, r); it; ++it) {
      // Store the column of the current value
      // NB:  Pardiso uses 1-based indexing!
      ja[idx] = it.col() + 1;

      va[idx] = it.value();

      // Check whether the current value is a diagonal element
      if (it.row() == it.col()) {
        // If so, we need to correct for the diagonal perturbation that we
        // added earlier
        if (it.value() == eps) {
          // Store an EXACT zero here
          va[idx] = 0.0;
        } else {
          va[idx] = it.value() - eps;
        }
      } else {
        va[idx] = it.value();
      }

      ++idx;
    }
  }

  // Don't forget the last element of ia!
  ia[A.rows()] = idx + 1;
}

} // namespace PardisoSupport
