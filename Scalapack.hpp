//
// Created by Yohsuke Murase on 2021/10/17.
//
#include <iostream>
#include <vector>
#include <array>
#include "mpi.h"

#ifndef CPP_SCALAPACK_HPP
#define CPP_SCALAPACK_HPP



extern "C" {
  void sl_init_(int *icontext, int *nprow, int *npcolumn);
  // SL_INIT initializes an NPROW x NPCOL process grid using a row-major ordering
  // of the processes. This routine retrieves a default system context which will
  // include all available processes. (out) ictxt, (in) nprow, npcolumn

  void blacs_gridinfo_(int *icontext, int *nprow, int *npcolumn, int *myrow,
                       int *mycolumn);
  // (in) icontext: BLACS context
  // (out) nprow, npcolumn: the numbers of rows and columns in this process grid
  // (out) myrow, mycolumn: the process grid row- and column-index

  void blacs_exit_(int *cont);
  // (in) continue: if 0, all the resources are released. If nonzero, MPI
  // resources are not released.

  void blacs_gridexit_(int *icontext);
  // (in) icontext: BLACS context

  void descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *irsrc,
                 int *icsrc, int *icontext, int *lld, int *info);
  // (out) descriptor for the global matrix. `desc` must be an array of int of
  //   length 9. int[9]
  // (in) m, n: rows and columns of the matrix (in) mb, nb: row,
  //   column block sizes
  // (in) irsrc, icsrc: the process row (column) over which the
  // first row of the global matrix is distributed.
  // (in) icontext: BLACS context
  // (in) lld: leading dimension of the local array
  // (out) info: 0 => completed successfully

  void dgesd2d_(int *icontext, int *m, int *n, double *A, int *lda, int *r_dest,
                int *c_dest);
  // Takes a general rectangular matrix and sends it to the destination process.
  // (in) icontext: BLACS context
  // (in) m,n: matrix sizes
  // (in) A: matrix
  // (in) lda: leading dimension (m)
  // (in) r_dest, c_dest: the process corrdinate of the process to send the
  // message to

  void dgerv2d_(int *icontext, int *m, int *n, double *A, int *lda, int *row_src,
                int *col_src);
  // Receives a message from the process into the general rectangular matrix.
  // (in) icontext: BLACS context
  // (in) m,n,lda: sizes of the matrix
  // (out) A: matrix
  // (in) row_src, col_src: the process coordinate of the source of the message

  void pdgesv_(int *n, int *nrhs, double *A, int *ia, int *ja, int desc_a[9],
               int *ipvt, double *B, int *ib, int *jb, int desc_b[9], int *info);
  // These subroutines solve the following systems of equations for multiple
  // right-hand sides: AX = B
  // (in) n: order of the submatrix = the number of rows of B
  // (in) nrhs: the number of columns of B
  // (in/out) A: the local part of the global general matrix A.
  // (in) ia, ja: the row and the column indices of the
  //   global matrix A, identifying the first row and column of the submatrix A.
  // (in) desc_a: descriptor of A matrix
  // (out) ipvt: the local part of the global vector ipvt, containing the pivot
  // indices.
  // (in/out) B: the local part of the global general matrix B,
  //   containing the right-hand sides of the system.
  // (in) ib, jb: the row and the column indices of the global matrix B,
  //   identifying the first row and column of the submatrix B.
  // (in) desc_b: descriptor of B matrix (out) info: error code
}


class Scalapack {
  public:
  static int ICTXT, NPROW, NPCOL, MYROW, MYCOL;
  static void Initialize(const std::array<int,2>& proc_grid_size) {
    NPROW = proc_grid_size[0];
    NPCOL = proc_grid_size[1];
    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    if (num_proc != NPROW * NPCOL) {
      std::cerr << "Error: invalid number of procs" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    sl_init_(&ICTXT, &NPROW, &NPCOL);
    blacs_gridinfo_(&ICTXT, &NPROW, &NPCOL, &MYROW, &MYCOL);
  }

  // global matrix
  class GMatrix {
    public:
    GMatrix(size_t N, size_t M) : N(N), M(M) {
      A.resize(N * M, 0.0);
    }
    size_t N, M;
    std::vector<double> A;
    double At(size_t I, size_t J) const { return A[I*M+J]; }
    void Set(size_t I, size_t J, double val) { A[I*M+J] = val; }
    double* Data() { return A.data(); }
    size_t Size() { return A.size(); }
    friend std::ostream& operator<<(std::ostream& os, const GMatrix& gm) {
      for (size_t i = 0; i < gm.N; i++) {
        for (size_t j = 0; j < gm.M; j++) {
          os << gm.At(i, j) << ' ';
        }
        os << "\n";
      }
      return os;
    }
    void BcastFrom(int root_rank) {
      int my_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
      std::array<uint64_t,2> sizes = {N, M};
      MPI_Bcast(sizes.data(), 2, MPI_UINT64_T, root_rank, MPI_COMM_WORLD);
      if (my_rank != root_rank) { A.resize(N*M); }
      MPI_Bcast(A.data(), N*M, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
    }
  };

  // local matrix for scalapack
  class LMatrix {
    public:
    LMatrix(int N, int M, int NB, int MB) : N(N), M(M), NB(NB), MB(MB) {  // matrix N x M with block NB x MB
      SUB_ROWS = (N / (NB * NPROW)) * NB + std::min(N % (NB * NPROW), NB);
      SUB_COLS = (M / (MB * NPCOL)) * MB + std::min(M % (MB * NPCOL), MB);
      int RSRC = 0, CSRC = 0, INFO;
      descinit_(DESC, &N, &M, &NB, &MB, &RSRC, &CSRC, &ICTXT, &SUB_ROWS, &INFO);
      assert(INFO == 0);
      SUB.resize(SUB_ROWS * SUB_COLS, 0.0);
    }
    LMatrix(const GMatrix& gm, int NB, int MB) : N(gm.N), M(gm.M), NB(NB), MB(MB) {
      SUB_ROWS = (N / (NB * NPROW)) * NB + std::min(N % (NB * NPROW), NB);
      SUB_COLS = (M / (MB * NPCOL)) * MB + std::min(M % (MB * NPCOL), MB);
      int RSRC = 0, CSRC = 0, INFO;
      descinit_(DESC, &N, &M, &NB, &MB, &RSRC, &CSRC, &ICTXT, &SUB_ROWS, &INFO);
      assert(INFO == 0);
      SUB.resize(SUB_ROWS * SUB_COLS, 0.0);
      for (int i = 0; i < SUB_ROWS; i++) {
        for (int j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < N && J < M) {
            Set(i, j, gm.At(I, J));
          }
        }
      }
    };
    int N, M;  // size of the global matrix
    int NB, MB; // block sizes
    int SUB_ROWS, SUB_COLS;  // size of the local matrix
    int DESC[9];
    std::vector<double> SUB;

    // convert submatrix index (i,j) at process (p_row, p_col) into global coordinate (I,J)
    std::array<size_t,2> ToGlobalCoordinate(size_t i, size_t j, int p_row = MYROW, int p_col = MYCOL) const {
      // block coordinate (bi, bj)
      size_t bi = i / NB;
      size_t bj = j / MB;
      // local coordinate inside the block
      size_t ii = i % NB;
      size_t jj = j % MB;
      // calculate global coordinate
      size_t I = bi * (NB * NPROW) + p_row * NB + ii;
      size_t J = bj * (MB * NPCOL) + p_col * MB + jj;
      return {I, J};
    }
    double At(size_t i, size_t j) const {  // get an element at SUB[ (i,j) ]
      return SUB[i + j * SUB_ROWS];
    }
    void Set(size_t i, size_t j, double val) {
      SUB[i + j * SUB_ROWS] = val;
    }
    double* Data() { return SUB.data(); }
    friend std::ostream& operator<<(std::ostream& os, const LMatrix& lm) {
      for (size_t i = 0; i < lm.SUB_ROWS; i++) {
        for (size_t j = 0; j < lm.SUB_COLS; j++) {
          os << lm.At(i, j) << ' ';
        }
        os << "\n";
      }
      return os;
    }

    GMatrix ConstructGlobalMatrix() {
      GMatrix A(N, M);
      for (size_t i = 0; i < SUB_ROWS; i++) {
        for (size_t j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < N && J < M) {
            A.Set(I, J, At(i, j));
          }
        }
      }
      GMatrix AA(N, M);
      MPI_Allreduce(A.Data(), AA.Data(), N*M, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return AA;
    }
  };

  /*
  Scalapack(const std::array<int,2>& proc_grid_size, int matrix_size, int block_size) {
    NPROW = proc_grid_size[0];
    NPCOL = proc_grid_size[1];
    sl_init_(&ICTXT, &NPROW, &NPCOL);
    blacs_gridinfo_(&ICTXT, &NPROW, &NPCOL, &MYROW, &MYCOL);
    N = matrix_size;
    NB = block_size;

    SUB_A_ROWS = (N / (NB * NPROW)) * NB + std::min(N % (NB * NPROW), NB);
    SUB_A_COLS = (N / (NB * NPCOL)) * NB + std::min(N % (NB * NPCOL), NB);
    SUB_B_ROWS = (N / (NB * NPROW)) * NB + std::min(N % (NB * NPROW), NB);

    int RSRC = 0, CSRC = 0, NRHS = 1;
    int INFO;
    descinit_(DESCA, &N, &N, &NB, &NB, &RSRC, &CSRC, &ICTXT, &SUB_A_ROWS, &INFO);
    assert(INFO == 0);
    descinit_(DESCB, &N, &NRHS, &NB, &NRHS, &RSRC, &CSRC, &ICTXT, &SUB_B_ROWS, &INFO);
    assert(INFO == 0);

    SUB_A.resize(SUB_A_ROWS * SUB_A_COLS, 0.0);
    SUB_B.resize(SUB_B_ROWS, 0.0);
    IPIV.resize(SUB_A_ROWS + NB);
  }
  int SUB_A_ROWS, SUB_A_COLS, SUB_B_ROWS;
  int N, NB;
  int DESCA[9], DESCB[9];
  std::vector<double> SUB_A;
  std::vector<double> SUB_B;
  std::vector<int> IPIV;
   */

};

#endif //CPP_SCALAPACK_HPP
