#pragma once
// Per-kernel GPU timing for OzakiI. Diagnostic only: OzakiI::compute() is
// unaffected and stays the fast path.
//
// Every stage is recorded into its own command buffer on the OzakiI queue and
// timed with GPUEndTime - GPUStartTime. All buffers are committed together and
// waited on only once at the end, so the GPU is not left idle between stages.
//
// The per-stage times do not add up to the time of compute(): separate command
// buffers stop consecutive kernels from overlapping and add a little launch
// overhead (tens of microseconds per stage). Use them to compare kernels, and
// time compute() for the end-to-end number.
#include "ozakiI.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace schorle {

class OzakiIProfiler {
public:
  enum Stage {
    Zero,       // blit: zero colMax and the level buffer
    RowMax,     // strip_max_row(A)
    ColMax,     // strip_max_column(B)
    SplitA,     // split_int8(A, rows)
    SplitB,     // split_int8(B, cols)
    Gemm,       // gemm_level_int8
    Accumulate, // accumulate_fp64
    NumStages
  };

  struct KernelTimes {
    std::array<double, NumStages> seconds{}; // GPU seconds per stage

    double operator[](Stage s) const { return seconds[s]; }
    double total() const {
      return std::accumulate(seconds.begin(), seconds.end(), 0.0);
    }
    static const char *name(Stage s) {
      static constexpr const char *names[NumStages] = {
          "zero (blit)",     "strip_max_row",  "strip_max_column",
          "split_int8 (A)",  "split_int8 (B)", "gemm_level_int8",
          "accumulate_fp64"};
      return names[s];
    }
  };

  explicit OzakiIProfiler(OzakiI &ozaki) : oz(ozaki) {}

  // Runs the full pipeline `warmup + reps` times and returns the median GPU
  // time of every stage over the `reps` timed runs. get_C() holds a valid
  // result afterwards, exactly as after compute(S).
  // Throws like compute(S) does.
  KernelTimes profile(uint32_t S, int reps = 10, int warmup = 2);
  KernelTimes profile(int reps = 10, int warmup = 2) {
    return profile(oz.parameters.MAX_S, reps, warmup);
  }

private:
  KernelTimes runOnce(uint32_t S);

  OzakiI &oz;
};

inline OzakiIProfiler::KernelTimes OzakiIProfiler::runOnce(uint32_t S) {
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  std::array<MTL::CommandBuffer *, NumStages> cbs{};

  // One command buffer per stage. A queue executes its command buffers in
  // commit order and Metal tracks the buffer hazards between them, so each
  // stage still sees the previous stage's writes.
  for (int st = 0; st < NumStages; ++st) {
    MTL::CommandBuffer *cb = oz.queue->commandBuffer();
    cbs[st] = cb;
    if (st == Zero) {
      MTL::BlitCommandEncoder *blit = cb->blitCommandEncoder();
      oz.encodeZero(blit, S);
      blit->endEncoding();
    } else {
      MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
      switch (st) {
      case RowMax: oz.encodeRowMax(enc); break;
      case ColMax: oz.encodeColMax(enc); break;
      case SplitA: oz.encodeSplitA(enc, S); break;
      case SplitB: oz.encodeSplitB(enc, S); break;
      case Gemm: oz.encodeGemm(enc, S); break;
      case Accumulate: oz.encodeAccumulate(enc, S); break;
      default: break;
      }
      enc->endEncoding();
    }
    cb->commit(); // no wait here: keep the GPU busy
  }

  // Wait once; the earlier buffers have finished by the time the last one has,
  // so these waits return immediately.
  KernelTimes t;
  bool ok = true;
  for (int st = 0; st < NumStages; ++st) {
    MTL::CommandBuffer *cb = cbs[st];
    cb->waitUntilCompleted();
    ok = ok && cb->status() == MTL::CommandBufferStatusCompleted;
    t.seconds[st] = cb->GPUEndTime() - cb->GPUStartTime();
  }
  pool->release(); // also releases the command buffers and encoders
  if (!ok)
    throw std::runtime_error("schorle: GPU execution failed");
  return t;
}

inline OzakiIProfiler::KernelTimes
OzakiIProfiler::profile(uint32_t S, int reps, int warmup) {
  oz.checkS(S);
  if (reps < 1 || warmup < 0)
    throw std::invalid_argument("schorle: invalid reps or warmup");

  // Warmup: first touches of fresh buffers fault pages in, and the GPU needs a
  // moment to raise its clocks.
  for (int w = 0; w < warmup; ++w)
    runOnce(S);

  std::array<std::vector<double>, NumStages> samples;
  for (int r = 0; r < reps; ++r) {
    const KernelTimes t = runOnce(S);
    for (int st = 0; st < NumStages; ++st)
      samples[st].push_back(t.seconds[st]);
  }

  KernelTimes med;
  for (int st = 0; st < NumStages; ++st) {
    std::vector<double> &v = samples[st];
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    med.seconds[st] = v[v.size() / 2];
  }
  return med;
}

} // namespace schorle
