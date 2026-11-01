
#include <chrono>
#include <random>
#include <vector>

void generate_random_matrix(double *A, int n, int m, float phi,
                            std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  for (int i = 0; i < n * m; i++) {
    A[i] = (uni(rng) - 0.5) * std::exp(phi * gauss(rng));
  }
}
template <class F>
double time_median_s(F &&f, int reps = 10, int warmup = 1) {
  for (int w = 0; w < warmup; ++w)
    f();
  std::vector<double> t(reps);
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    t[r] = std::chrono::duration<double>(t1 - t0).count();
  }
  std::sort(t.begin(), t.end());
  return t[reps / 2];
}
