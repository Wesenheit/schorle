#pragma once
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifndef SCHORLE_METALLIB_PATH
#error                                                                         \
    "SCHORLE_METALLIB_PATH must be defined (set by CMake via schorle::schorle)"
#endif

namespace schorle {

struct ParametersOzakiI {
  uint32_t N;
  uint32_t M;
  uint32_t K;
  uint32_t MAX_S; // max number of int8 slices; buffers are sized for it

  // Launch hyperparameters.
  uint32_t TG_ROW_MAX = 256;     // strip_max_row threadgroup size (<= 1024)
  uint32_t TG_COL_MAX = 256;     // strip_max_column threadgroup size; N % it == 0
  uint32_t COL_MAX_CHUNKS = 128; // strip_max_column row chunks; K % it == 0
  uint32_t TG_SPLIT = 256;       // split_int8 threadgroup size
  uint32_t SPLIT_PER_THREAD = 4; // split_int8 columns per thread (function
                                 // constant); K, N % (TG_SPLIT * it) == 0
  uint32_t TG_ACCUMULATE = 256;  // accumulate_fp64 threadgroup size
};

class OzakiI {
public:
  explicit OzakiI(ParametersOzakiI params, MTL::Device *device = nullptr,
                  const char *metallibPath = SCHORLE_METALLIB_PATH);
  ~OzakiI();
  OzakiI(const OzakiI &) = delete;
  OzakiI &operator=(const OzakiI &) = delete;
  double *get_A() { return static_cast<double *>(this->bufA->contents()); }
  double *get_B() { return static_cast<double *>(this->bufB->contents()); }
  double *get_C() { return static_cast<double *>(this->bufC->contents()); }
  void compute();
  void compute(uint32_t S);

  static constexpr uint32_t GEMM_TILE_M = 64;
  static constexpr uint32_t GEMM_TILE_N = 128;
  static constexpr uint32_t GEMM_THREADS = 128;

private:
  friend class OzakiIProfiler;

  void checkS(uint32_t S) const;
  void encodeZero(MTL::BlitCommandEncoder *blit, uint32_t S);
  void encodeRowMax(MTL::ComputeCommandEncoder *enc);
  void encodeColMax(MTL::ComputeCommandEncoder *enc);
  void encodeSplitA(MTL::ComputeCommandEncoder *enc, uint32_t S);
  void encodeSplitB(MTL::ComputeCommandEncoder *enc, uint32_t S);
  void encodeGemm(MTL::ComputeCommandEncoder *enc, uint32_t S);
  void encodeAccumulate(MTL::ComputeCommandEncoder *enc, uint32_t S);

  void checkParameters() const;
  void checkThreadgroupLimits() const;
  MTL::Buffer *newShared(size_t bytes);
  void loadLibrary(const char *path);
  MTL::ComputePipelineState *
  makePipeline(const char *name, MTL::FunctionConstantValues *fcv = nullptr);
  void releaseAll();

  ParametersOzakiI parameters;
  MTL::Device *device = nullptr;
  MTL::Library *library = nullptr;
  MTL::CommandQueue *queue = nullptr;

  MTL::ComputePipelineState *psoRowMax = nullptr;
  MTL::ComputePipelineState *psoColMax = nullptr;
  MTL::ComputePipelineState *psoSplitRow = nullptr;
  MTL::ComputePipelineState *psoSplitCol = nullptr;
  MTL::ComputePipelineState *psoGemm = nullptr;
  MTL::ComputePipelineState *psoAcc = nullptr;

  MTL::Buffer *bufA = nullptr;
  MTL::Buffer *bufB = nullptr;
  MTL::Buffer *bufC = nullptr;
  MTL::Buffer *bufRowMax = nullptr;
  MTL::Buffer *bufColMax = nullptr;
  MTL::Buffer *bufQA = nullptr;
  MTL::Buffer *bufQB = nullptr;
  MTL::Buffer *bufLevels = nullptr;
};

inline OzakiI::OzakiI(ParametersOzakiI params, MTL::Device *dev,
                      const char *metallibPath)
    : parameters(params) {
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  try {
    device = dev ? dev->retain() : MTL::CreateSystemDefaultDevice();
    if (!device)
      throw std::runtime_error("schorle: no Metal device available");

    checkParameters();
    loadLibrary(metallibPath);

    queue = device->newCommandQueue();
    if (!queue)
      throw std::runtime_error("schorle: failed to create command queue");

    const size_t M = parameters.M, N = parameters.N, K = parameters.K;
    bufA = newShared(M * K * sizeof(double));
    bufB = newShared(K * N * sizeof(double));
    bufC = newShared(M * N * sizeof(double));
    bufRowMax = newShared(M * sizeof(uint32_t));
    bufColMax = newShared(N * sizeof(uint32_t));
    const size_t S = parameters.MAX_S;
    bufQA = newShared(M * K * S * sizeof(int8_t));
    bufQB = newShared(K * N * S * sizeof(int8_t));
    bufLevels = newShared(M * N * S * sizeof(int32_t));
  } catch (...) {
    releaseAll();
    pool->release();
    throw;
  }
  pool->release();
}

inline OzakiI::~OzakiI() { releaseAll(); }

inline void OzakiI::releaseAll() {
  auto rel = [](auto *&p) {
    if (p) {
      p->release();
      p = nullptr;
    }
  };
  rel(bufA), rel(bufB), rel(bufC), rel(bufRowMax), rel(bufColMax);
  rel(bufQA), rel(bufQB), rel(bufLevels);
  rel(psoRowMax), rel(psoColMax), rel(psoSplitRow), rel(psoSplitCol);
  rel(psoGemm), rel(psoAcc);
  rel(queue);
  rel(library);
  rel(device);
}

inline void OzakiI::checkParameters() const {
  const ParametersOzakiI &p = parameters;
  const uint64_t M = p.M, N = p.N, K = p.K;
  const uint64_t splitTile = (uint64_t)p.TG_SPLIT * p.SPLIT_PER_THREAD;
  const uint64_t maxElems = std::numeric_limits<uint32_t>::max();
  const bool ok =
      M > 0 && N > 0 && K > 0 && p.MAX_S > 0 && p.TG_ROW_MAX > 0 &&
      p.TG_COL_MAX > 0 && p.TG_SPLIT > 0 && p.TG_ACCUMULATE > 0 &&
      p.COL_MAX_CHUNKS > 0 && p.SPLIT_PER_THREAD > 0 &&
      p.TG_ROW_MAX <= 1024 &&               // strip_max_row: sh[32] per simdgroup
      N % p.TG_COL_MAX == 0 && K % p.COL_MAX_CHUNKS == 0 && // strip_max_column
      K % splitTile == 0 && N % splitTile == 0 &&           // split_int8
      M % GEMM_TILE_M == 0 && N % GEMM_TILE_N == 0 &&       // gemm_level_int8
      p.MAX_S * K * 127 * 127 < (uint64_t)std::numeric_limits<int32_t>::max() &&
      M * K <= maxElems && K * N <= maxElems && M * N <= maxElems;
  if (!ok)
    throw std::invalid_argument("schorle: invalid parameters");
}

inline void OzakiI::checkThreadgroupLimits() const {
  const ParametersOzakiI &p = parameters;
  auto fits = [](MTL::ComputePipelineState *pso, NS::UInteger tg) {
    return tg <= pso->maxTotalThreadsPerThreadgroup();
  };
  if (!fits(psoRowMax, p.TG_ROW_MAX) || !fits(psoColMax, p.TG_COL_MAX) ||
      !fits(psoSplitRow, p.TG_SPLIT) || !fits(psoSplitCol, p.TG_SPLIT) ||
      !fits(psoGemm, GEMM_THREADS) || !fits(psoAcc, p.TG_ACCUMULATE))
    throw std::invalid_argument("schorle: threadgroup size too large");
}

inline MTL::Buffer *OzakiI::newShared(size_t bytes) {
  MTL::Buffer *b = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
  if (!b)
    throw std::runtime_error("schorle: buffer allocation failed");
  return b;
}

inline void OzakiI::compute() { compute(parameters.MAX_S); }

inline void OzakiI::checkS(uint32_t S) const {
  if (S == 0 || S > parameters.MAX_S)
    throw std::invalid_argument("schorle: S out of range");
}

inline void OzakiI::compute(uint32_t S) {
  checkS(S);

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::CommandBuffer *cb = queue->commandBuffer();

  MTL::BlitCommandEncoder *blit = cb->blitCommandEncoder();
  encodeZero(blit, S);
  blit->endEncoding();

  MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
  encodeRowMax(enc);
  encodeColMax(enc);
  encodeSplitA(enc, S);
  encodeSplitB(enc, S);
  encodeGemm(enc, S);
  encodeAccumulate(enc, S);
  enc->endEncoding();

  cb->commit();
  cb->waitUntilCompleted();

  const bool ok = cb->status() == MTL::CommandBufferStatusCompleted;
  pool->release(); // also releases cb and the encoders
  if (!ok)
    throw std::runtime_error("schorle: GPU execution failed");
}

// Step 0: zero colMax (atomic max target) and the level buffer
inline void OzakiI::encodeZero(MTL::BlitCommandEncoder *blit, uint32_t S) {
  const size_t N = parameters.N, elemsC = (size_t)parameters.M * N;
  blit->fillBuffer(bufColMax, NS::Range(0, N * sizeof(uint32_t)), 0);
  blit->fillBuffer(bufLevels, NS::Range(0, elemsC * S * sizeof(int32_t)), 0);
}

// Step 1: Get maximum exponent for each row in matrix A
inline void OzakiI::encodeRowMax(MTL::ComputeCommandEncoder *enc) {
  const uint32_t M = parameters.M, K = parameters.K;
  enc->setComputePipelineState(psoRowMax);
  enc->setBuffer(bufA, 0, 0);
  enc->setBuffer(bufRowMax, 0, 1);
  enc->setBytes(&K, sizeof(K), 2);
  enc->dispatchThreadgroups(MTL::Size(M, 1, 1),
                            MTL::Size(parameters.TG_ROW_MAX, 1, 1));
}

// Step 2: Get maximum exponent for each column in matrix B
inline void OzakiI::encodeColMax(MTL::ComputeCommandEncoder *enc) {
  const ParametersOzakiI &p = parameters;
  const uint32_t N = p.N;
  const uint32_t rowsPerChunk = p.K / p.COL_MAX_CHUNKS;
  enc->setComputePipelineState(psoColMax);
  enc->setBuffer(bufB, 0, 0);
  enc->setBuffer(bufColMax, 0, 1);
  enc->setBytes(&N, sizeof(N), 2);
  enc->setBytes(&rowsPerChunk, sizeof(rowsPerChunk), 3);
  enc->dispatchThreadgroups(MTL::Size(N / p.TG_COL_MAX, p.COL_MAX_CHUNKS, 1),
                            MTL::Size(p.TG_COL_MAX, 1, 1));
}

// Step 3: Split matrix A into int8 chunks
inline void OzakiI::encodeSplitA(MTL::ComputeCommandEncoder *enc, uint32_t S) {
  const ParametersOzakiI &p = parameters;
  const uint32_t M = p.M, K = p.K, elemsA = M * K;
  const uint32_t splitTile = p.TG_SPLIT * p.SPLIT_PER_THREAD;
  enc->setComputePipelineState(psoSplitRow);
  enc->setBuffer(bufA, 0, 0);
  enc->setBuffer(bufRowMax, 0, 1);
  enc->setBuffer(bufQA, 0, 2);
  enc->setBytes(&K, sizeof(K), 3);
  enc->setBytes(&elemsA, sizeof(elemsA), 4);
  enc->setBytes(&S, sizeof(S), 5);
  enc->dispatchThreadgroups(MTL::Size(K / splitTile, M, 1),
                            MTL::Size(p.TG_SPLIT, 1, 1));
}

// Step 4: Split matrix B into int8 chunks
inline void OzakiI::encodeSplitB(MTL::ComputeCommandEncoder *enc, uint32_t S) {
  const ParametersOzakiI &p = parameters;
  const uint32_t N = p.N, K = p.K, elemsB = K * N;
  const uint32_t splitTile = p.TG_SPLIT * p.SPLIT_PER_THREAD;
  enc->setComputePipelineState(psoSplitCol);
  enc->setBuffer(bufB, 0, 0);
  enc->setBuffer(bufColMax, 0, 1);
  enc->setBuffer(bufQB, 0, 2);
  enc->setBytes(&N, sizeof(N), 3);
  enc->setBytes(&elemsB, sizeof(elemsB), 4);
  enc->setBytes(&S, sizeof(S), 5);
  enc->dispatchThreadgroups(MTL::Size(N / splitTile, K, 1),
                            MTL::Size(p.TG_SPLIT, 1, 1));
}

// Step 5: Multiply int8 x int8 -> int32 chunks
inline void OzakiI::encodeGemm(MTL::ComputeCommandEncoder *enc, uint32_t S) {
  const uint32_t M = parameters.M, N = parameters.N, K = parameters.K;
  enc->setComputePipelineState(psoGemm);
  enc->setBuffer(bufQA, 0, 0);
  enc->setBuffer(bufQB, 0, 1);
  enc->setBuffer(bufLevels, 0, 2);
  enc->setBytes(&M, sizeof(M), 3);
  enc->setBytes(&N, sizeof(N), 4);
  enc->setBytes(&K, sizeof(K), 5);
  enc->setBytes(&S, sizeof(S), 6);
  enc->dispatchThreadgroups(MTL::Size(N / GEMM_TILE_N, M / GEMM_TILE_M, S),
                            MTL::Size(GEMM_THREADS, 1, 1));
}

// Step 6: Accumulate results into float64 values
inline void OzakiI::encodeAccumulate(MTL::ComputeCommandEncoder *enc,
                                     uint32_t S) {
  const uint32_t M = parameters.M, N = parameters.N, elemsC = M * N;
  enc->setComputePipelineState(psoAcc);
  enc->setBuffer(bufLevels, 0, 0);
  enc->setBuffer(bufRowMax, 0, 1);
  enc->setBuffer(bufColMax, 0, 2);
  enc->setBuffer(bufC, 0, 3);
  enc->setBytes(&N, sizeof(N), 4);
  enc->setBytes(&elemsC, sizeof(elemsC), 5);
  enc->setBytes(&S, sizeof(S), 6);
  enc->dispatchThreads(MTL::Size(N, M, 1),
                       MTL::Size(parameters.TG_ACCUMULATE, 1, 1));
}

inline void OzakiI::loadLibrary(const char *path) {
  library = device->newLibrary(
      NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding)),
      nullptr);
  if (!library)
    throw std::runtime_error("schorle: failed to load metallib");

  psoRowMax = makePipeline("strip_max_row");
  psoColMax = makePipeline("strip_max_column");
  psoGemm = makePipeline("gemm_level_int8");
  psoAcc = makePipeline("accumulate_fp64");

  for (bool splitCols : {false, true}) {
    MTL::FunctionConstantValues *fcv =
        MTL::FunctionConstantValues::alloc()->init();
    fcv->setConstantValue(&splitCols, MTL::DataTypeBool, NS::UInteger(0));
    fcv->setConstantValue(&parameters.SPLIT_PER_THREAD, MTL::DataTypeUInt,
                          NS::UInteger(1));
    try {
      (splitCols ? psoSplitCol : psoSplitRow) = makePipeline("split_int8", fcv);
    } catch (...) {
      fcv->release();
      throw;
    }
    fcv->release();
  }

  checkThreadgroupLimits();
}

inline MTL::ComputePipelineState *
OzakiI::makePipeline(const char *name, MTL::FunctionConstantValues *fcv) {
  NS::Error *err = nullptr; // required by the API; the message is not used
  NS::String *fnName = NS::String::string(name, NS::UTF8StringEncoding);
  MTL::Function *fn = fcv ? library->newFunction(fnName, fcv, &err)
                          : library->newFunction(fnName);
  if (!fn)
    throw std::runtime_error("schorle: kernel not found");
  MTL::ComputePipelineState *pso = device->newComputePipelineState(fn, &err);
  fn->release();
  if (!pso)
    throw std::runtime_error("schorle: pipeline creation failed");
  return pso;
}

} // namespace schorle
