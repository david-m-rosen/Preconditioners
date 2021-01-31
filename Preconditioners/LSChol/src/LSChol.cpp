#include "LSChol/LSChol.h"

namespace Preconditioners {

void LSChol::init() { cholmod_l_start(&common); }

void LSChol::compute(const SparseMatrix &A) {

  // Dimesions of A
  size_t m = A.rows();
  size_t n = A.cols();
  size_t nnz = A.nonZeros();

  /// Construct cholmod_sparse version of A to pass into SPQR

  // Construct cholmod_triplet representation of A
  cholmod_triplet *T =
      cholmod_l_allocate_triplet(m, n, nnz, 0, CHOLMOD_REAL, &common);

  size_t idx = 0;
  for (size_t k = 0; k < A.outerSize(); ++k)
    for (SparseMatrix::InnerIterator iter(A, k); iter; ++iter) {
      // Record idx-th element
      ((SuiteSparse_long *)T->i)[idx] = iter.row();
      ((SuiteSparse_long *)T->j)[idx] = iter.col();
      ((double *)T->x)[idx] = iter.value();
      ++idx;
    }
  // Set total number of elements in T
  T->nnz = A.nonZeros();

  // Construct cholmod_sparse matrix from this triplet representation
  A_ = cholmod_l_triplet_to_sparse(T, nnz, &common);
}

LSChol::~LSChol() { cholmod_l_finish(&common); }

} // namespace Preconditioners
