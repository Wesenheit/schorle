#include <cstdio>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#define ACCELERATE_NEW_LAPACK // use the modern BLAS/LAPACK headers
#include <Accelerate/Accelerate.h>

#include "schorle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

constexpr uint32_t M = 4096, N = 4096, K = 4096;
constexpr int SLICES[] = {8, 10, 12}; // S values to benchmark
constexpr int MAX_S = 12;

constexpr int BENCH_WARMUP = 3;
constexpr int BENCH_REPS = 20;

int main() {
  MTL::Device *device = MTL::CreateSystemDefaultDevice();

  schorle::ParametersOzakiI params;
  params.M = M;
  params.N = N;
  params.K = K;
  params.MAX_S = MAX_S;

  schorle::OzakiI ozaki(params, device);

  double *a = ozaki.get_A();
  double *b = ozaki.get_B();
  double *c = ozaki.get_C();

  const size_t elemsA = (size_t)M * K, elemsB = (size_t)K * N,
               elemsC = (size_t)M * N;

  std::vector<double> cRef(elemsC), absA(elemsA), absB(elemsB), absAB(elemsC);
  std::vector<float> phi_arr = {2., 4., 6.};
  std::mt19937_64 rng(42);
  const double flops = 2.0 * M * N * K;

  std::printf("%s  |  C = A*B, M=N=K=%u, fp64\n", device->name()->utf8String(),
              M);
  std::printf("CPU and GPU: median of %d timed runs after %d warmup runs; GPU "
              "= wall time of OzakiI::compute()\n",
              BENCH_REPS, BENCH_WARMUP);
  std::printf("err = max |C - C_dgemm| / (|A||B|)   [2^-53 = %.2e]\n\n",
              std::ldexp(1.0, -53));

  for (auto phi : phi_arr) {
    generate_random_matrix(a, M, K, phi, rng);
    generate_random_matrix(b, K, N, phi, rng);

    // --- CPU reference ------------------------------------------------------
    double cpuSec = time_median_s(
        [&] {
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                      a, K, b, N, 0.0, cRef.data(), N);
        },
        BENCH_REPS, BENCH_WARMUP);

    for (size_t i = 0; i < elemsA; ++i)
      absA[i] = std::fabs(a[i]);
    for (size_t i = 0; i < elemsB; ++i)
      absB[i] = std::fabs(b[i]);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                absA.data(), K, absB.data(), N, 0.0, absAB.data(), N);

    std::printf("phi=%g\n", phi);
    std::printf("  Accelerate dgemm (CPU)  %8.2f ms  %7.1f GFLOPS\n",
                cpuSec * 1e3, flops / cpuSec / 1e9);

    // --- GPU Ozaki, per slice count -----------------------------------------
    for (int S : SLICES) {
      const double gpuSec = time_median_s(
          [&] { ozaki.compute((uint32_t)S); }, BENCH_REPS, BENCH_WARMUP);

      double maxErr = 0.0;
      for (size_t i = 0; i < elemsC; ++i)
        if (absAB[i] > 0)
          maxErr = std::max(maxErr, std::fabs(c[i] - cRef[i]) / absAB[i]);

      std::printf("  Ozaki int8 S=%-2d (GPU)  %8.2f ms  %7.1f GFLOPS  "
                  "x%.2f vs CPU  err %.2e\n",
                  S, gpuSec * 1e3, flops / gpuSec / 1e9, cpuSec / gpuSec,
                  maxErr);
    }
    std::printf("\n");
  }

  device->release(); // ozaki keeps its own reference until it is destroyed
  return 0;
}
