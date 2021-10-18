#include <iostream>
#include <cassert>
#include "mpi.h"
#include "Scalapack.hpp"


#define myassert(x) do {                              \
if (!(x)) {                                           \
  printf("Assertion failed: %s, file %s, line %d\n"   \
         , #x, __FILE__, __LINE__);                   \
  MPI_Abort(MPI_COMM_WORLD, 1);                       \
  }                                                   \
} while (0)

int my_rank;

void test_Matrix() {
  const size_t N = 3, M = 5;
  Scalapack::GMatrix g(3, 5);

  if (my_rank == 0) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        g.Set(i, j, i*5+j);
      }
    }
    std::cerr << g;
  }
  g.BcastFrom(0);
  Scalapack::LMatrix lm(g, 2, 2);

  myassert(lm.SUB_ROWS == 2);
  myassert(lm.SUB_COLS == 3);

  for (int pi = 0; pi < Scalapack::NPROW; pi++) {
    for (int pj = 0; pj < Scalapack::NPCOL; pj++) {
      if (my_rank == pi*Scalapack::NPCOL+pj) {
        std::cerr << "(pi, pj): " << pi << ", " << pj << std::endl;
        std::cerr << lm;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

void test_PDGESV() {

}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  Scalapack::Initialize({2, 2});

  test_Matrix();

  MPI_Finalize();
  return 0;
}