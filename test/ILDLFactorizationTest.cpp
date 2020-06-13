#include "SymILDLSupport/ILDLFactorization.h"
#include "SymILDLSupport/SymILDLUtils.h"

#include "solver.h"

#include <Eigen/Eigenvalues>
#include <complex>

#include "gtest/gtest.h"

using namespace SymILDLSupport;
using namespace std;

typedef Eigen::MatrixXd Matrix;

class ILDLFactorizationTest : public testing::Test {
protected:
  /// Test configuration

  double rel_tol = 1e-6; // Relative error tolerance
  double eps = 1e-6;     // Absolute error tolerance

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
    ///           0  4
    ///              7

    SparseMatrix AUT(4, 4);
    AUT.resize(4, 4);

    AUT.insert(0, 0) = 1;
    AUT.insert(0, 1) = 2;
    AUT.insert(0, 3) = 3;

    AUT.insert(1, 1) = -5;

    AUT.insert(2, 3) = 4;

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
  std::vector<int> col_idx_true = {0, 1, 3, 1, 3, 3};
  for (size_t i = 0; i < col_idx_true.size(); ++i) {
    EXPECT_EQ(col_idx_true[i], col_idx[i]);
  }

  // Check val
  std::vector<double> val_true = {1, 2, 3, -5, 4, 7};
  for (size_t i = 0; i < val_true.size(); ++i) {
    EXPECT_FLOAT_EQ(val_true[i], val[i]);
  }
}

/// Compute an *exact* LDL factorization, and verify that the elements P, S, L,
/// and D are computed correctly
TEST_F(ILDLFactorizationTest, ExactFactorizationElements) {
  // Setting max-fill to a be huge and drop tol = 0 results in an exact LDL
  // factorization
  opts.equilibration = Equilibration::Bunch;
  opts.order = Ordering::AMD;
  opts.pivot_type = PivotType::BunchKaufman;
  opts.max_fill_factor = 1e3;
  opts.BK_pivot_tol = 0;
  opts.drop_tol = 0;

  /// Compute factorization using SYM-ILDL's built-in solver

  // Construct CSR representation of A

  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<Scalar> val;
  toCSR(A, row_ptr, col_idx, val);

  symildl::solver<Scalar> solver;

  // Turn off messaging
  solver.msg_lvl = 0;

  // Load in initial matrix
  solver.load(row_ptr, col_idx, val);

  // Set reordering scheme
  switch (opts.order) {
  case Ordering::AMD:
    solver.reorder_type = symildl::reordering_type::AMD;
    break;
  case Ordering::RCM:
    solver.reorder_type = symildl::reordering_type::RCM;
    break;
  case Ordering::None:
    solver.reorder_type = symildl::reordering_type::NONE;
    break;
  }

  // Set equilibration scheme
  solver.equil_type = (opts.equilibration == Equilibration::Bunch
                           ? symildl::equilibration_type::BUNCH
                           : symildl::equilibration_type::NONE);

  // Set pivoting type
  solver.piv_type = (opts.pivot_type == PivotType::Rook
                         ? lilc_matrix<Scalar>::pivot_type::ROOK
                         : lilc_matrix<Scalar>::pivot_type::BKP);

  solver.has_rhs = false;
  solver.perform_inplace = true;
  solver.solve(opts.max_fill_factor, opts.drop_tol, opts.BK_pivot_tol);

  /// Compute factorization using ILDLFactorization

  // Set factorization options
  Afact.setOptions(opts);

  // Compute factorization
  Afact.compute(A);

  // Extract elements of this factorization
  SparseMatrix D = Afact.D();
  const SparseMatrix &L = Afact.L();

  /// Ensure that the elements P, S, L, and D of the factorizations computed by
  /// SYM-ILDL and ILDLFactorization coincide

  /// Ensure that the permutations P agree
  EXPECT_EQ(Afact.P().size(), solver.perm.size());
  for (int k = 0; k < Afact.P().size(); ++k)
    EXPECT_EQ(Afact.P()(k), solver.perm[k]);

  /// Ensure that the scaling matrices agree
  EXPECT_EQ(Afact.S().size(), solver.A.S.main_diag.size());
  for (int k = 0; k < Afact.S().size(); ++k)
    EXPECT_FLOAT_EQ(Afact.S()(k), solver.A.S.main_diag[k]);

  /// Ensure that the lower-triangular factors agree
  EXPECT_EQ(Afact.L().nonZeros(), solver.A.nnz());
  for (int k = 0; k < Afact.L().outerSize(); ++k)
    for (SparseMatrix::InnerIterator it(Afact.L(), k); it; ++it)
      EXPECT_FLOAT_EQ(it.value(), solver.A.coeff(it.row(), it.col()));

  /// Ensure that the block-diagonal matrices D agree

  // Extract lower triangle from D
  SparseMatrix DLT = D.triangularView<Eigen::Lower>();
  EXPECT_EQ(DLT.nonZeros(), solver.D.nnz());
  for (int k = 0; k < DLT.outerSize(); ++k)
    for (SparseMatrix::InnerIterator it(DLT, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      if (i == j) {
        // This is a diagonal element
        EXPECT_LT(fabs(it.value() - solver.D.main_diag.at(i)), eps);
      } else {
        // This is the off-diagonal element *below* the element D(j,j)
        EXPECT_LT(fabs(it.value() - solver.D.off_diag.at(j)), eps);
      }
    }

  /// Save the matrices constructed by the SYM-ILDL solver
  solver.L.save("L.txt");
  solver.D.save("D.txt");
}

// TEST_F(ILDLFactorizationTest, ExactFactorization) {

//  // Setting max-fill to a be huge and drop tol = 0 results in an exact LDL
//  // factorization
//  opts.max_fill_factor = 1e3;
//  opts.drop_tol = 0;
//  opts.pos_def_mod = false;

//  // Set factorization options
//  Afact.setOptions(opts);

//  // Compute factorization
//  Afact.compute(A);

//  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(A.rows(), A.rows());
//  Eigen::MatrixXd Ainv(A.rows(), A.rows());

//  // Compute inverse of A by solving A*Ainv = Id column-by-column
//  for (int k = 0; k < A.rows(); ++k)
//    Ainv.col(k) = Afact.solve(Id.col(k));

//  Eigen::MatrixXd Ainv_gt = Eigen::MatrixXd(A).inverse();

//  EXPECT_LT((Ainv_gt - Ainv).norm(), rel_tol * Ainv_gt.norm());
//}

// TEST_F(ILDLFactorizationTest, PositiveDefiniteFactorization) {

//  // Setting max-fill to a be huge and drop tol = 0 results in an exact LDL
//  // factorization
//  opts.max_fill_factor = 1e3;
//  opts.drop_tol = 0;

//  // Modify the block-diagonal matrix D to ensure that LDL is
//  // positive-definitite
//  opts.pos_def_mod = true;

//  // Set factorization options
//  Afact.setOptions(opts);

//  // Compute factorization
//  Afact.compute(A);

//  Eigen::MatrixXd Adense = Eigen::MatrixXd(A);
//  Eigen::MatrixXd PA(A.rows(), A.rows());

//  // Compute the preconditioned matrix PA by solving PA = Afact^-1 * A
//  // column-by-column
//  for (int k = 0; k < A.rows(); ++k)
//    PA.col(k) = Afact.solve(Adense.col(k));

//  // Compute eigenvalues of PA: these should be +/- 1
//  Eigen::EigenSolver<Eigen::MatrixXd> eigs(PA);

//  /// Ensure that each eigenvalue is either +/- 1
//  for (int k = 0; k < PA.rows(); ++k)
//    EXPECT_TRUE((abs(eigs.eigenvalues()(k) - 1.0) < rel_tol) ||
//                (abs(eigs.eigenvalues()(k) + 1.0) < rel_tol));
//}
