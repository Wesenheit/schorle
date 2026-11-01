#include <cstdio>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "schorle.hpp"

constexpr uint32_t M = 4096, N = 4096, K = 4096;
constexpr uint32_t SLICES[] = {2, 4, 6, 8, 10, 12, 14, 16}; // S values to profile
constexpr float PHI = 2.0f; // exponent spread of the random inputs

constexpr int PROFILE_WARMUP = 2;
constexpr int PROFILE_REPS = 10;

int main() {
  MTL::Device *device = MTL::CreateSystemDefaultDevice();

  schorle::ParametersOzakiI params;
  params.M = M;
  params.N = N;
  params.K = K;
  params.MAX_S = *std::max_element(std::begin(SLICES), std::end(SLICES));

  schorle::OzakiI ozaki(params, device);

  std::mt19937_64 rng(42);
  generate_random_matrix(ozaki.get_A(), M, K, PHI, rng);
  generate_random_matrix(ozaki.get_B(), K, N, PHI, rng);

  using P = schorle::OzakiIProfiler;
  P profiler(ozaki);

  std::vector<P::KernelTimes> times;
  std::vector<double> computeSec;
  for (uint32_t S : SLICES) {
    times.push_back(profiler.profile(S, PROFILE_REPS, PROFILE_WARMUP));
    computeSec.push_back(time_median_s([&] { ozaki.compute(S); },
                                       PROFILE_REPS, PROFILE_WARMUP));
  }

  std::printf("%s  |  C = A*B, M=%u N=%u K=%u, fp64, phi=%g\n",
              device->name()->utf8String(), M, N, K, PHI);
  std::printf("GPU time per stage in ms: median of %d runs after %d warmup "
              "runs, one command buffer per stage\n\n",
              PROFILE_REPS, PROFILE_WARMUP);

  auto rule = [&] {
    std::printf("%s\n", std::string(20 + 10 * times.size(), '-').c_str());
  };

  std::printf("%-20s", "stage \\ S");
  for (uint32_t S : SLICES)
    std::printf("%10u", S);
  std::printf("\n");
  rule();

  for (int st = 0; st < P::NumStages; ++st) {
    std::printf("%-20s", P::KernelTimes::name(P::Stage(st)));
    for (const auto &t : times)
      std::printf("%10.2f", t.seconds[st] * 1e3);
    std::printf("\n");
  }
  rule();

  std::printf("%-20s", "sum");
  for (const auto &t : times)
    std::printf("%10.2f", t.total() * 1e3);
  std::printf("\n");

  std::printf("%-20s", "compute() (wall)");
  for (double s : computeSec)
    std::printf("%10.2f", s * 1e3);
  std::printf("\n");

  std::printf("%-20s", "gemm share [%]");
  for (const auto &t : times)
    std::printf("%10.1f", 100.0 * t[P::Gemm] / t.total());
  std::printf("\n");

  std::printf("%-20s", "GFLOPS (compute)");
  for (double s : computeSec)
    std::printf("%10.1f", 2.0 * M * N * K / s / 1e9);
  std::printf("\n");

  device->release(); // ozaki keeps its own reference until it is destroyed
  return 0;
}
