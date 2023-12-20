#include <iostream>
#include <vector>
#include <cassert>

#include "mpi.h"

int GLOB_rank, GLOB_size;

struct Matrix {
  struct Local{};
  struct Shared{};

  Matrix(Shared, int h, int w) : h_{h}, w_{w}, ptr_{nullptr}, win_{} {
    int local_size = GLOB_rank == 0 ? (w * h * sizeof(int)) : 0;
    MPI_Win_allocate_shared(local_size, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &ptr_, &win_);

    MPI_Aint win_size;
    int win_disp;
    MPI_Win_shared_query(win_, 0, &win_size, &win_disp, &ptr_);
  }

  Matrix(Local, int h, int w) : h_{h}, w_{w}, ptr_{new int[w * h]} {}

  int h() { return h_; }
  int w() { return w_; }
  int& operator[](std::pair<size_t, size_t> ij) { 
    return ptr_[ij.first * w_ + ij.second]; 
  }

  void operator+=(const Matrix& o) { 
    for (int i = 0; i < h_ * w_; i++) ptr_[i] += o.ptr_[i]; 
  }

  void fill() { 
    for (int i = 0; i < h_ * w_; i++) ptr_[i] = rand() % 100; 
  }

  void clear() {
    for (int i = 0; i < h_ * w_; i++) ptr_[i] = 0; 
  }

  void fence() { MPI_Win_fence(0, win_); }
  void send(int dest) const { MPI_Send(ptr_, w_ * h_, MPI_INT, dest, 0, MPI_COMM_WORLD); }
  void recv(int src) const { MPI_Recv(ptr_, w_ * h_, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }

  void print() {
    std::cout << "=====" << std::endl;
    for (int i = 0; i < h_; i++) {
      for (int j = 0; j < w_; j++) {
        std::cout << (*this)[{i, j}] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "=====" << std::endl;
  }

private:
  int h_, w_;
  int* ptr_;
  MPI_Win win_;
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &GLOB_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &GLOB_size);

  int n = 2000;
  int r = std::atoi(argv[1]);

  Matrix a{Matrix::Shared{}, n, n}, b{Matrix::Shared{}, n, n};

  b.fence();
  if (GLOB_rank == 0) {
    b.fill();
    a.fill();
  }
  b.fence();

  //std::cout << GLOB_rank << " over fence " << std::endl;

  int kb = a.w() / GLOB_size;
  int ks = GLOB_rank * kb;
  int ke = (GLOB_rank + 1) * kb;

  assert(ks != ke);
  assert(a.w() % GLOB_size == 0);

  Matrix local{Matrix::Local{}, r, n};

  for (int s = 0; s < n; s += r) {
    if (GLOB_rank > 0) {
      //std::cout << GLOB_rank << " waiting for " << s << std::endl;
      local.recv(GLOB_rank - 1);
      //std::cout << GLOB_rank << " got " << s << std::endl;
    } else {
      local.clear();
    }

    for (int i = 0; i < r; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = ks; k < ke; k++) {
          local[{i, j}] += a[{i + s, k}] * b[{j, k}];
        }
      }
    }

    if (GLOB_rank + 1 < GLOB_size) {
      //std::cout << GLOB_rank << " sent part " << s << std::endl;
      local.send(GLOB_rank + 1);
    }
  }

  b.fence();

  MPI_Finalize();
}