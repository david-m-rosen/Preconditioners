/** This file provides several convenient utility functions for working with the
 * Pardiso linear algebra library.
 *
 * Copyright (C) 2019 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "PardisoSupport/PardisoSupport_types.h"

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

void toCSR(const SparseMatrix &A, int *&ia, int *&ja, double *&va,
           double eps = std::numeric_limits<Scalar>::min());

} // namespace PardisoSupport
