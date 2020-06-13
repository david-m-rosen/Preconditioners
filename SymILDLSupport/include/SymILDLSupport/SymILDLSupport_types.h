/** A set of typedefs describing the basic types that will be used throughout
 * the SymILDL library.
 *
 *  Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace SymILDLSupport {

/// Some useful typedefs for the SymILDL library

/// Linear algebra types
typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

/// Typedef for a (fixed-size) 2x2 matrix
typedef Eigen::Matrix2d Matrix2d;

/** We use row-major storage order to take advantage of fast (sparse-matrix)
 * (dense-vector) multiplications when OpenMP is available (cf. the Eigen
 * documentation page on "Eigen and Multithreading") */
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;

typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;

typedef Eigen::PermutationMatrix<Eigen::Dynamic>::IndicesType PermutationVector;

} // namespace SymILDLSupport
