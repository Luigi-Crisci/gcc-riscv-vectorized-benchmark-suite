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

#include "../../common/log.h"
#include "../../common/riscv_util.h"
#include "axpy.h"
/*************************************************************************
 *GET_TIME
 *returns a long int representing the time
 *************************************************************************/

#include <algorithm>
#include <numeric>
#include <sys/time.h>
#include <time.h>
#include <vector>

int num_rep = 5;

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
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
    dy[i] += a * dx[i];
  }
}

template <typename T> void init_vector(T *pv, long n, T value) {
  for (int i = 0; i < n; i++)
    pv[i] = value;
}

template <typename T> void measure(int n) {
  auto &log = get_logfile();
  T a{1};
  const auto precision = sizeof(T) * 8;
  std::vector<double> time_results;

  long long start, end;
  start = get_time();
  /* Allocate the source and result vectors */
  T *dx = (T *)malloc(n * sizeof(T));
  T *dy = (T *)malloc(n * sizeof(T));
  T *dy_ref = (T *)malloc(n * sizeof(T));

  init_vector(dx, n, T{1});
  init_vector(dy, n, T{2});

  end = get_time();
  // printf("init_vector time: %f\n", elapsed_time(start, end));

  // printf("doing reference axpy , vector size %d\n", n);
  time_results.erase(time_results.begin(), time_results.end());
  for (int i = 0; i < num_rep; i++) {

    start = get_time();

    // Start instruction and cycles count of the region of interest
    // unsigned long cycles1, cycles2, instr2, instr1;
    // unsigned long instr1 = get_inst_count();
    // unsigned long cycles1 = get_cycles_count();

    axpy_ref(a, dx, dy, n);

    // End instruction and cycles count of the region of interest
    // unsigned long instr2 = get_inst_count();
    // unsigned long cycles2 = get_cycles_count();
    // // Instruction and cycles count of the region of interest
    // printf("-CSR   NUMBER OF EXEC CYCLES :%lu\n", cycles2 - cycles1);
    // printf("-CSR   NUMBER OF INSTRUCTIONS EXECUTED :%lu\n", instr2 - instr1);

    end = get_time();
    time_results.push_back(elapsed_time(start, end));
    // printf("axpy_reference time: %f\n", elapsed_time(start, end));
  }

  log.add_res("reference", type_t<T>{}, n, precision, 0, median(time_results));

  capture_ref_result(dy, dy_ref, n);

  for (int j = 0; j < 4; j++) { // LMUL
    const auto LMUL = 1 << j;

    init_vector(dx, n, T{1});
    init_vector(dy, n, T{2});

    // printf("doing vector axpy, vector size %d\n", n);
    time_results.erase(time_results.begin(), time_results.end());
    for (int i = 0; i < num_rep; i++) {
      start = get_time();

      // // Start instruction and cycles count of the region of interest
      // unsigned long instr1 = get_inst_count();
      // unsigned long cycles1 = get_cycles_count();

      axpy_dispatch<T>::dispatch(LMUL, a, dx, dy, n);

      // // End instruction and cycles count of the region of interest
      // unsigned long instr2 = get_inst_count();
      // unsigned long cycles2 = get_cycles_count();
      // // Instruction and cycles count of the region of interest
      // printf("-CSR   NUMBER OF EXEC CYCLES :%lu\n", cycles2 - cycles1);
      // printf("-CSR   NUMBER OF INSTRUCTIONS EXECUTED :%lu\n", instr2 - instr1);

      end = get_time();
      time_results.push_back(elapsed_time(start, end));
    }
    // printf("axpy_intrinsics time: %f\n", elapsed_time(start, end));
    log.add_res("intrinsics", type_t<T>{}, n, precision, LMUL,
                median(time_results));
    // printf("done\n");
    // test_result(dy, dy_ref, n);
  }

  free(dx);
  free(dy);
  free(dy_ref);
}

int main(int argc, char *argv[]) {
  size_t n;

  if (argc >= 2) {
    n = atol(argv[1]); // input argument: vector size in Ks
    if (argc == 3) {
      num_rep = strtol(argv[2], NULL, 10);
    }
  } else {
    n = 100;
  }

  get_logfile();

  measure<int32_t>(n);
  measure<int64_t>(n);
  measure<float>(n);
  measure<double>(n);

  return 0;
}
