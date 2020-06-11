#include "SymILDLSupport/ILDLFactorization.h"
#include "SymILDLSupport/SymILDLUtils.h"

#include <Eigen/Eigenvalues>
#include <complex>

#include "gtest/gtest.h"

using namespace SymILDLSupport;
using namespace std;

class ILDLFactorizationTest : public testing::Test {
protected:
  /// Test configuration

  double rel_tol = 1e-6; // Relative tolerance for linear solutions

  /// Test data

  // Coefficient matrix
  SparseMatrix A;

  // Test vector x
  Vector xtest;

  // Pardiso options struct
  SymILDLOpts opts;

  ILDLFactorization Afact;

  void SetUp() override {
    /// Set the upper triangle of A to be:
    ///
    /// A = 1  2  0  3
    ///       -5  0  0
    ///           2  0
    ///              7
    ///
    ///
    ///

    SparseMatrix AUT(4, 4);
    AUT.resize(4, 4);

    AUT.insert(0, 0) = 1;
    AUT.insert(0, 1) = 2;
    AUT.insert(0, 3) = 3;

    AUT.insert(1, 1) = -5;

    AUT.insert(2, 2) = 2;

    AUT.insert(3, 3) = 7;

    A = AUT.selfadjointView<Eigen::Upper>();

    // Randomly sample test vector x
    xtest = Vector::Random(A.rows());
  }
};

TEST_F(ILDLFactorizationTest, toCSR) {

  // Construct CSR representation of A

  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<Scalar> val;

  toCSR(A, row_ptr, col_idx, val);

  /// Verify that these vectors are what they should be

  // Check row_ptr
  std::vector<int> row_ptr_true = {0, 3, 4, 5, 6};
  for (size_t i = 0; i < row_ptr_true.size(); ++i) {
    EXPECT_EQ(row_ptr_true[i], row_ptr[i]);
  }

  // Check col_idx
  std::vector<int> col_idx_true = {0, 1, 3, 1, 2, 3};
  for (size_t i = 0; i < col_idx_true.size(); ++i) {
    EXPECT_EQ(col_idx_true[i], col_idx[i]);
  }

  // Check val
  std::vector<double> val_true = {1, 2, 3, -5, 2, 7};
  for (size_t i = 0; i < val_true.size(); ++i) {
    EXPECT_FLOAT_EQ(val_true[i], val[i]);
  }
}

TEST_F(ILDLFactorizationTest, ExactFactorization) {

  // Setting max-fill to a be huge and drop tol = 0 results in an exact LDL
  // factorization
  opts.max_fill_factor = 1e3;
  opts.drop_tol = 0;
  opts.pos_def_mod = false;

  // Set factorization options
  Afact.setOptions(opts);

  // Compute factorization
  Afact.compute(A);

  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(A.rows(), A.rows());
  Eigen::MatrixXd Ainv(A.rows(), A.rows());

  // Compute inverse of A by solving A*Ainv = Id column-by-column
  for (int k = 0; k < A.rows(); ++k)
    Ainv.col(k) = Afact.solve(Id.col(k));

  Eigen::MatrixXd Ainv_gt = Eigen::MatrixXd(A).inverse();

  EXPECT_LT((Ainv_gt - Ainv).norm(), rel_tol * Ainv_gt.norm());
}

TEST_F(ILDLFactorizationTest, PositiveDefiniteFactorization) {

  // Setting max-fill to a be huge and drop tol = 0 results in an exact LDL
  // factorization
  opts.max_fill_factor = 1e3;
  opts.drop_tol = 0;

  // Modify the block-diagonal matrix D to ensure that LDL is
  // positive-definitite
  opts.pos_def_mod = true;

  // Set factorization options
  Afact.setOptions(opts);

  // Compute factorization
  Afact.compute(A);

  Eigen::MatrixXd Adense = Eigen::MatrixXd(A);
  Eigen::MatrixXd PA(A.rows(), A.rows());

  // Compute the preconditioned matrix PA by solving PA = Afact^-1 * A
  // column-by-column
  for (int k = 0; k < A.rows(); ++k)
    PA.col(k) = Afact.solve(Adense.col(k));

  // Compute eigenvalues of PA: these should be +/- 1
  Eigen::EigenSolver<Eigen::MatrixXd> eigs(PA);

  /// Ensure that each eigenvalue is either +/- 1
  for (int k = 0; k < PA.rows(); ++k)
    EXPECT_TRUE((abs(eigs.eigenvalues()(k) - 1.0) < rel_tol) ||
                (abs(eigs.eigenvalues()(k) + 1.0) < rel_tol));
}
