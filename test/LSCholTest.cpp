#include "LSChol/LSChol.h"

using namespace Preconditioners;
using namespace std;
int main() {

  Matrix Adense(3, 2);
  Adense << 2., 8., 5., 10., 10., 7.;

  SparseMatrix A = Adense.sparseView();

  LSChol Afact;

  Afact.compute(A);
}
