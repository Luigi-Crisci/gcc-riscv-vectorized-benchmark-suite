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

#include "../../common/log.h"

#include <riscv_vector.h>

// template <int n> struct get_type {};
// template <> struct get_type<32> {
//   using type = float;
// };
// template <> struct get_type<64> {
//   using type = double;
// };

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)

// std::cout << "Calling axpy with FP = " << FP << " and lmul = " << LMUL <<
// "\n"; asm volatile("fence" : :); asm("FP_" STR(FP) "_LMUL" STR(LMUL)
// "_uniqueid%=:" :::);
#define AXPY(TYPE, TYPE_SHORT, SIGN, LMUL) \
  template <typename T = TYPE>                                 \
  void axpy_intrinsics_##TYPE_SHORT##SIGN##lmul##LMUL##_(T a, T *dx, T *dy, \
                                                         int n) { \
    int i; \
    long vl = -1; \
    v##TYPE##SIGN##m##LMUL##_t v_dx; \
    v##TYPE##SIGN##m##LMUL##_t v_dy; \
    for (i = 0; i < n;) { \
      vl = vsetvl_e##SIGN##m##LMUL(n - i); \
      v_dx = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dx, vl); \
      dx += vl; \
      v_dy = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, vl); \
      if constexpr (std::is_integral_v<T>) { \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
      } else { /* floating points*/ \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
        v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
      } \
      vse##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, v_dy, vl); \
      dy += vl; \
      i += vl; \
    } \
  }

// ONLY FADD
// #define AXPY(TYPE, TYPE_SHORT, SIGN, LMUL)                                     \
//   template <typename T = TYPE>                                                 \
//   void axpy_intrinsics_##TYPE_SHORT##SIGN##lmul##LMUL##_(T a, T *dx, T *dy,    \
//                                                          int n) {              \
//     int i;                                                                     \
//     long vl = -1;                                                              \
//     v##TYPE##SIGN##m##LMUL##_t v_dx;                                           \
//     v##TYPE##SIGN##m##LMUL##_t v_dy;                                           \
//     vl = vsetvl_e##SIGN##m##LMUL(n);                                           \
//     /*std::cout << type_info<T>::name  << " -- " << LMUL << "\n";*/            \
//     v_dx = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dx, vl);              \
//     v_dy = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, vl);                  \
//     for (i = 0; i < n;) {                                                      \
//       vl = vsetvl_e##SIGN##m##LMUL(n - i);                                     \
//       /*std::cout << "\t I = " << i << " - Vector lenght: " << vl << "\n"; */  \
//       if constexpr (std::is_integral_v<T>) {                                   \
//         v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl);                   \
//       } else { /* floating points*/                                            \
//         v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl);                  \
//       }                                                                        \
//       vse##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, v_dy, vl);                 \
//       i += vl;                                                                 \
//     }                                                                          \
//   }

// ONLY LOAD
//  #d ef in e  AXPY(TYPE, TYPE_SHORT, SIGN, LMUL) \
//   template <typename T = TYPE>                                 \
//   void axpy_intrinsics_##TYPE_SHORT##SIGN##lmul##LMUL##_(T a, T *dx, T *dy,
//    \
//                                                          int n) { \
//     int i; \
//     long vl = -1; \
//     v##TYPE##SIGN##m##LMUL##_t v_dy; \
//     v##TYPE##SIGN##m##LMUL##_t v_dx; \
//     /*std::cout << type_info<T>::name  << " -- " << LMUL << "\n";*/ \
//     for (i = 0; i < n;) { \
//       vl = vsetvl_e##SIGN##m##LMUL(n - i); \
//       v_dx = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dx, vl); \
//       /*std::cout << "\t I = " << i << " - Vector lenght: " << vl << "\n";
//        */      \
//       if constexpr (std::is_integral_v<T>) { \
//         v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dx, a, v_dx, vl); \
//       } else { /* floating points*/ \
//         v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dx, a, v_dx, vl); \
//       } \
//       vse##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, v_dy, vl); \
//       i += vl; \
//     } \
//   }

// ONLY STORE
//  #d ef in e  AXPY(TYPE, TYPE_SHORT, SIGN, LMUL) \
//   template <typename T = TYPE>                                 \
//   void axpy_intrinsics_##TYPE_SHORT##SIGN##lmul##LMUL##_(T a, T *dx, T *dy,
//    \
//                                                          int n) { \
//     int i; \
//     long vl = -1; \
//     v##TYPE##SIGN##m##LMUL##_t v_dy; \
//     v##TYPE##SIGN##m##LMUL##_t v_dx; \
//     v_dx = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dx, vl); \
//     v_dy = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, vl); \
//     /*std::cout << type_info<T>::name  << " -- " << LMUL << "\n";*/ \
//     for (i = 0; i < n;) { \
//       vl = vsetvl_e##SIGN##m##LMUL(n - i); \
//       /*std::cout << "\t I = " << i << " - Vector lenght: " << vl << "\n";
//        */      \
//       if constexpr (std::is_integral_v<T>) { \
//         v_dy = vmacc_vx_i##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
//       } else { /* floating points*/ \
//         v_dy = vfmacc_vf_f##SIGN##m##LMUL(v_dy, a, v_dx, vl); \
//       } \
//       vse##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, v_dy, vl); \
//       i += vl; \
//     } \
//   }

// #define AXPY(TYPE, TYPE_SHORT, SIGN, LMUL) \
//   template <typename T = TYPE>                                 \
//   void axpy_intrinsics_##TYPE_SHORT##SIGN##lmul##LMUL##_(T a, T *dx, T *dy, \
//                                                          int n) { \
//     int i; \
//     long vl = -1; \
//     v##TYPE##SIGN##m##LMUL##_t v_dx; \
//     v##TYPE##SIGN##m##LMUL##_t v_dy; \
//     for (i = 0; i < n;) { \
//       vl = vsetvl_e##SIGN##m##LMUL(n - i); \
//       v_dx = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dx, vl); \
//       dx += vl; \
//       v_dy = vle##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, vl); \
//       v_dy = vfadd_vv_f##SIGN##m##LMUL(v_dy, v_dx, vl); \
//       vse##SIGN##_v_##TYPE_SHORT##SIGN##m##LMUL(dy, v_dy, vl); \
//       dy += vl; \
//       i += vl; \
//     } \
//   }

// Single precision
AXPY(float, f, 32, 1)
AXPY(float, f, 32, 2)
AXPY(float, f, 32, 4)
AXPY(float, f, 32, 8)

// Double precision
AXPY(float, f, 64, 1)
AXPY(float, f, 64, 2)
AXPY(float, f, 64, 4)
AXPY(float, f, 64, 8)

// int32 precision
AXPY(int, i, 32, 1)
AXPY(int, i, 32, 2)
AXPY(int, i, 32, 4)
AXPY(int, i, 32, 8)

// int64 precision
AXPY(int, i, 64, 1)
AXPY(int, i, 64, 2)
AXPY(int, i, 64, 4)
AXPY(int, i, 64, 8)

template <typename T> struct axpy_dispatch;

template <> struct axpy_dispatch<float> {
  template <typename... Args> static void dispatch(int lmul, Args &&...args) {
    switch (lmul) {
    case 1: {
      axpy_intrinsics_f32lmul1_(std::forward<Args>(args)...);
      break;
    }
    case 2: {
      axpy_intrinsics_f32lmul2_(std::forward<Args>(args)...);
      break;
    }
    case 4: {
      axpy_intrinsics_f32lmul4_(std::forward<Args>(args)...);
      break;
    }
    case 8: {
      axpy_intrinsics_f32lmul8_(std::forward<Args>(args)...);
      break;
    }
    }
  }
};

template <> struct axpy_dispatch<double> {
  template <typename... Args> static void dispatch(int lmul, Args &&...args) {
    switch (lmul) {
    case 1: {
      axpy_intrinsics_f64lmul1_(std::forward<Args>(args)...);
      break;
    }
    case 2: {
      axpy_intrinsics_f64lmul2_(std::forward<Args>(args)...);
      break;
    }
    case 4: {
      axpy_intrinsics_f64lmul4_(std::forward<Args>(args)...);
      break;
    }
    case 8: {
      axpy_intrinsics_f64lmul8_(std::forward<Args>(args)...);
      break;
    }
    }
  }
};

template <> struct axpy_dispatch<int32_t> {
  template <typename... Args> static void dispatch(int lmul, Args &&...args) {
    switch (lmul) {
    case 1: {
      axpy_intrinsics_i32lmul1_(std::forward<Args>(args)...);
      break;
    }
    case 2: {
      axpy_intrinsics_i32lmul2_(std::forward<Args>(args)...);
      break;
    }
    case 4: {
      axpy_intrinsics_i32lmul4_(std::forward<Args>(args)...);
      break;
    }
    case 8: {
      axpy_intrinsics_i32lmul8_(std::forward<Args>(args)...);
      break;
    }
    }
  }
};

template <> struct axpy_dispatch<int64_t> {
  template <typename... Args> static void dispatch(int lmul, Args &&...args) {
    switch (lmul) {
    case 1: {
      axpy_intrinsics_i64lmul1_(std::forward<Args>(args)...);
      break;
    }
    case 2: {
      axpy_intrinsics_i64lmul2_(std::forward<Args>(args)...);
      break;
    }
    case 4: {
      axpy_intrinsics_i64lmul4_(std::forward<Args>(args)...);
      break;
    }
    case 8: {
      axpy_intrinsics_i64lmul8_(std::forward<Args>(args)...);
      break;
    }
    }
  }
};
