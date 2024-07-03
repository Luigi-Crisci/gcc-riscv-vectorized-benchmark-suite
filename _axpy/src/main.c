/*************************************************************************
 * Axpy Kernel
 * Author: Jesus Labarta
 * Barcelona Supercomputing Center
 *************************************************************************/

#include "utils.h"
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../common/riscv_util.h"
#include "axpy.h"
#include "../../common/log.h"
/*************************************************************************
 *GET_TIME
 *returns a long int representing the time
 *************************************************************************/

#include <sys/time.h>
#include <time.h>

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
  return (float)(end_time - start_time) / (1000 * 1000);
}
/*************************************************************************/

// Ref version
template <typename T> void axpy_ref(T a, T *dx, T *dy, int n) {
  // asm("AXPY_REF" "_uniqueid%=:" :::);
  int i;
  for (i = 0; i < n; i++) {
    dy[i] += a * dx[i];
  }
}

template <typename T> void init_vector(T *pv, long n, T value) {
  for (int i = 0; i < n; i++)
    pv[i] = value;
}

template <typename T> void measure(int n) { 
  auto& log = get_logfile();
  T a{1};
  const auto precision = sizeof(T) * 8;

  long long start, end;
  start = get_time();
  /* Allocate the source and result vectors */
  T *dx = (T *)malloc(n * sizeof(T));
  T *dy = (T *)malloc(n * sizeof(T));
  T *dy_ref = (T *)malloc(n * sizeof(T));

  init_vector(dx, n, T{1});
  init_vector(dy, n, T{2});

  end = get_time();
  //printf("init_vector time: %f\n", elapsed_time(start, end));

  // printf("doing reference axpy , vector size %d\n", n);
  start = get_time();

  // Start instruction and cycles count of the region of interest
  // unsigned long cycles1, cycles2, instr2, instr1;
  // instr1 = get_inst_count();
  // cycles1 = get_cycles_count();

  axpy_ref(a, dx, dy, n);

  // End instruction and cycles count of the region of interest
  // instr2 = get_inst_count();
  // cycles2 = get_cycles_count();
  // // Instruction and cycles count of the region of interest
  // printf("-CSR   NUMBER OF EXEC CYCLES :%lu\n", cycles2 - cycles1);
  // printf("-CSR   NUMBER OF INSTRUCTIONS EXECUTED :%lu\n", instr2 - instr1);

  end = get_time();
  //printf("axpy_reference time: %f\n", elapsed_time(start, end));
  log.add_res("reference", precision, 0, elapsed_time(start, end));

  capture_ref_result(dy, dy_ref, n);

  for (int j = 0; j < 4; j++) { // LMUL
    const auto LMUL = 1 << j;
    //std::cout << "Precision: " << precision << " -- LMUL: " << LMUL << "\n";

    init_vector(dx, n, T{1});
    init_vector(dy, n, T{2});

    // printf("doing vector axpy, vector size %d\n", n);
    start = get_time();

    // // Start instruction and cycles count of the region of interest
    // instr1 = get_inst_count();
    // cycles1 = get_cycles_count();

    axpy_dispatch<T>(LMUL, a, dx, dy, n);

    // // End instruction and cycles count of the region of interest
    // instr2 = get_inst_count();
    // cycles2 = get_cycles_count();
    // // Instruction and cycles count of the region of interest
    // printf("-CSR   NUMBER OF EXEC CYCLES :%lu\n", cycles2 - cycles1);
    // printf("-CSR   NUMBER OF INSTRUCTIONS EXECUTED :%lu\n", instr2 - instr1);

    end = get_time();
    //printf("axpy_intrinsics time: %f\n", elapsed_time(start, end));
    log.add_res("intrinsics", precision, LMUL, elapsed_time(start, end));
    // printf("done\n");
    // test_result(dy, dy_ref, n);
  }

  free(dx);
  free(dy);
  free(dy_ref);
}

int main(int argc, char *argv[]) {
  long n;

  if (argc == 2)
    n = 1024 * atol(argv[1]); // input argument: vector size in Ks
  else
    n = (30 * 1024);

  get_logfile("axpy.csv");
  
  measure<float>(n);
  measure<double>(n);

  return 0;
}
