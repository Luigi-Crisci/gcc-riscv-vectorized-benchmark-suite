/*************************************************************************
 * Vectorized Axpy Kernel
 * Author: Jesus Labarta
 * Barcelona Supercomputing Center
 *************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

#include <riscv_vector.h>

template <int n> struct get_type {};
template <> struct get_type<32> {
  using type = float;
};
template <> struct get_type<64> {
  using type = double;
};

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)

// std::cout << "Calling axpy with FP = " << FP << " and lmul = " << LMUL <<
// "\n"; asm volatile("fence" : :); asm("FP_" STR(FP) "_LMUL" STR(LMUL)
// "_uniqueid%=:" :::);
#define AXPY(FP, LMUL)                                                         \
  template <typename T = get_type<FP>::type>                                   \
  void axpy_intrinsics_fp##FP##lmul##LMUL##_(T a, T *dx, T *dy, int n) {       \
    int i;                                                                     \
    long vl = -1;                                                              \
    vfloat##FP##m##LMUL##_t v_dx;                                              \
    vfloat##FP##m##LMUL##_t v_dy;                                              \
    for (i = 0; i < n;) {                                                      \
      vl = vsetvl_e##FP##m##LMUL(n - i);                                       \
      v_dx = vle##FP##_v_f##FP##m##LMUL(dx, vl);                               \
      dx += vl;                                                                \
      v_dy = vle##FP##_v_f##FP##m##LMUL(dy, vl);                               \
      v_dy = vfmacc_vf_f##FP##m##LMUL(v_dy, a, v_dx, vl);                      \
      vse##FP##_v_f##FP##m##LMUL(dy, v_dy, vl);                                \
      dy += vl;                                                                \
      i += vl;                                                                 \
    }                                                                          \
  }

// Single precision
AXPY(32, 1)
AXPY(32, 2)
AXPY(32, 4)
AXPY(32, 8)

// Double precision
AXPY(64, 1)
AXPY(64, 2)
AXPY(64, 4)
AXPY(64, 8)

template <typename T, typename... Args>
void axpy_dispatch(int lmul, Args &&...args) {
  switch (lmul) {
  case 1: {
    if constexpr (std::is_same_v<T, float>) {
      axpy_intrinsics_fp32lmul1_(std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, double>) {
      axpy_intrinsics_fp64lmul1_(std::forward<Args>(args)...);
    }
    break;
  }
  case 2: {
    if constexpr (std::is_same_v<T, float>) {
      axpy_intrinsics_fp32lmul2_(std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, double>) {
      axpy_intrinsics_fp64lmul2_(std::forward<Args>(args)...);
    }
    break;
  }
  case 4: {
    if constexpr (std::is_same_v<T, float>) {
      axpy_intrinsics_fp32lmul4_(std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, double>) {
      axpy_intrinsics_fp64lmul4_(std::forward<Args>(args)...);
    }
    break;
  }
  case 8: {
    if constexpr (std::is_same_v<T, float>) {
      axpy_intrinsics_fp32lmul8_(std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, double>) {
      axpy_intrinsics_fp64lmul8_(std::forward<Args>(args)...);
    }
    break;
  }
  }
}