#include "LSChol/LSChol.h"
#include "gtest/gtest.h"

using namespace Preconditioners;
using namespace std;

class LSCholTest : public testing::Test {
protected:
  /// Test configuration

  double rel_tol = 1e-6; // Relative error tolerance
  double eps = 1e-6;     // Absolute error tolerance

  /// Test data

  SparseMatrix A;
  LSChol Afact;

  void SetUp() override {
    A.resize(3, 2);

    A.insert(0, 0) = 2;
    A.insert(0, 1) = 8;

    A.insert(1, 0) = 5;
    A.insert(1, 1) = 10;

    A.insert(2, 0) = 10;
    A.insert(2, 1) = 7;
  }
};

/// Basic test: check computation of upper-triangular factor
TEST_F(LSCholTest, compute) {

  /// Perform factorization
  Afact.compute(A);

  /// Extract triangular factor
  const SparseMatrix &R = Afact.R();

  /// Check output
  EXPECT_EQ(R.rows(), 2);
  EXPECT_EQ(R.cols(), 2);
  EXPECT_EQ(Afact.rank(), 2);

  /// If

  /// Compute the product A*R^-1
  Matrix Id = Matrix::Identity(R.rows(), R.rows());
  Matrix Rinv = R.triangularView<Eigen::Upper>().solve(Id);
  Matrix ARinv = A * Rinv;

  /// Compt
}
