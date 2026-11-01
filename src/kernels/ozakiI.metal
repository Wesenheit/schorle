#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void strip_max_row(device const uint2*  matrix   [[buffer(0)]],
                        device uint*        rowMax [[buffer(1)]],
                        constant uint&      cols   [[buffer(2)]],
                        uint row    [[threadgroup_position_in_grid]],
                        uint tid    [[thread_position_in_threadgroup]],
                        uint tgSize [[threads_per_threadgroup]],
                        uint lane   [[thread_index_in_simdgroup]],
                        uint simdId [[simdgroup_index_in_threadgroup]],
                        uint simdW  [[threads_per_simdgroup]])
{
    device const uint2* M = matrix + (ulong)row * cols;

    uint e = 0;
    for (uint c = tid; c < cols; c += tgSize)
        e = max(e, (M[c].y >> 20) & 0x7FFu);

    e = simd_max(e);

    threadgroup uint sh[32];
    if (lane == 0) sh[simdId] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdId == 0) {
        uint nSimd = (tgSize + simdW - 1) / simdW;
        e = simd_max(lane < nSimd ? sh[lane] : 0u);
        if (lane == 0) rowMax[row] = e;
    }
}
kernel void strip_max_column(device const uint2*  matrix   [[buffer(0)]],
                        device atomic_uint*        colMax [[buffer(1)]],
                        constant uint&      cols   [[buffer(2)]],
                        constant uint&      rows_per_chunk [[buffer(3)]],
                        uint2 tph    [[threadgroup_position_in_grid]],
                        uint2 tid    [[thread_position_in_threadgroup]],
                        uint2 tgSize [[threads_per_threadgroup]])
{
    uint e = 0;
    uint column = tph.x*tgSize.x + tid.x;
    uint row = tph.y * rows_per_chunk;
    uint rows_per_group = row+rows_per_chunk;
    for (uint c = row; c < rows_per_group; c += 1)
        e = max(e, (matrix[cols * c +column].y >> 20) & 0x7FFu);
    if (atomic_load_explicit(&colMax[column], memory_order_relaxed) < e)
        atomic_fetch_max_explicit(&colMax[column], e, memory_order_relaxed);
}

// false: one exponent per row (split A), maxE has `rows` entries.
// true:  one exponent per column (split B), maxE has `cols` entries.
// Set from the host via MTL::FunctionConstantValues; the branch compiles away.
constant bool kSplitCols [[function_constant(0)]];
// Columns handled per thread; set from ParametersOzakiI::SPLIT_PER_THREAD.
constant uint kPerThread [[function_constant(1)]];

kernel void split_int8(
       device const uint2* matrix           [[buffer(0)]],
       device const uint*  maxE        [[buffer(1)]],
       device char*        slices      [[buffer(2)]],
       constant uint&      cols        [[buffer(3)]],
       constant uint&      sliceStride [[buffer(4)]],
       constant uint&      numSlices   [[buffer(5)]],
       uint2 tgid   [[threadgroup_position_in_grid]],
       uint2  tid    [[thread_position_in_threadgroup]],
       uint2  tgSize [[threads_per_threadgroup]])

{
    uint row = tgid.y;
    uint column= (tgid.x * tgSize.x+tid.x)*kPerThread;
    device const uint2* M = matrix + (ulong)row * cols;
    uint max_row_val = kSplitCols ? 0 : maxE[row];
    const int S = numSlices;
    const int BITS = 7;

    for (uint32_t j = column; j < column+kPerThread; ++j) {
        const uint64_t bits = ((ulong)M[j].y << 32) | M[j].x;
        const uint32_t E = (uint32_t)((bits >> 52) & 0x7FFu);
        const bool neg = (bits >> 63) != 0;

        if (E == 0) { // zero or denormal: contributes nothing
          for (int s = 0; s < S; ++s)
            slices[(size_t)s * sliceStride + row*cols + j] = 0;
          continue;
        }

        const uint64_t sig = (1ull << 52) | (bits & 0xFFFFFFFFFFFFFull);
        const uint Eref = kSplitCols ? maxE[j] : max_row_val;
        const int d = (int)Eref - (int)E;

        for (int s = 1; s <= S; ++s) {
          const int sh = 53 + d - BITS * s;
          uint32_t q;
          if (sh >= 64 || sh <= -64)
            q = 0; // window entirely outside sig; also avoids UB shifts
          else if (sh >= 0)
            q = (uint32_t)((sig >> sh) & 0x7Fu);
          else
            q = (uint32_t)((sig << (-sh)) & 0x7Fu);
          slices[(size_t)(s - 1) * sliceStride + row*cols + j] = neg ? (int8_t)(-(int)q) : (int8_t)q;
        }
    }
}
kernel void gemm_level_int8(
       device const char* QA       [[buffer(0)]],  // S slabs of M x K, row-major
       device const char* QB       [[buffer(1)]],  // S slabs of K x N, row-major
       device int*        levels   [[buffer(2)]],  // S slabs of M x N int32; slab k-2 = L_k
       constant int&     M        [[buffer(3)]],
       constant int&     N        [[buffer(4)]],
       constant int&     K        [[buffer(5)]],
       constant int&     numSlices[[buffer(6)]],
       uint3 tgid   [[threadgroup_position_in_grid]])
{
    constexpr int TM = 64;
    constexpr int TN = 128;

    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                              false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op; // dispatch 128 threads/threadgroup

    uint level = tgid.z + 1;

    device int* out = levels + (ulong)tgid.z * M * N;
    tensor<device int32_t, dextents<int32_t,2>, tensor_inline> tC(out, dextents<int32_t,2>(N, M));
    auto mC = tC.template slice<TN, TM>(tgid.x * TN, tgid.y * TM); // fixed-size tile

    for (uint i = 0; i < level; i++){
        uint j = level - i;
        device int8_t* qa = (device int8_t*)QA + (ulong)i       * M * K;
        device int8_t* qb = (device int8_t*)QB + (ulong)(j - 1) * K * N;
        tensor<device int8_t, dextents<int32_t,2>, tensor_inline> tA(qa, dextents<int32_t,2>(K, M));
        tensor<device int8_t, dextents<int32_t,2>, tensor_inline> tB(qb, dextents<int32_t,2>(N, K));

        auto mA = tA.slice(0, tgid.y * TM);
        auto mB = tB.slice(tgid.x * TN, 0);
        op.run(mA, mB, mC);
    }
}

// 128-bit helpers for accumulate_fp64; the value is hi:lo in two's complement.

// bit b (0..127)
static inline ulong bit128(ulong hi, ulong lo, int b)
{
    return b < 64 ? (lo >> b) & 1ul : (hi >> (b - 64)) & 1ul;
}

// true if any of bits 0..b-1 is set
static inline bool any_below128(ulong hi, ulong lo, int b)
{
    if (b <= 0)  return false;
    if (b < 64)  return (lo & ((1ul << b) - 1)) != 0;
    if (b == 64) return lo != 0;
    return lo != 0 || (hi & ((1ul << (b - 64)) - 1)) != 0;
}

// low 64 bits of (hi:lo) >> d, for 1 <= d <= 127
static inline ulong shr128(ulong hi, ulong lo, int d)
{
    return d < 64 ? (lo >> d) | (hi << (64 - d)) : hi >> (d - 64);
}

kernel void accumulate_fp64(
       device const int*  levels      [[buffer(0)]],
       device const uint* rowMaxA     [[buffer(1)]],
       device const uint* colMaxB     [[buffer(2)]],
       device uint2*      C           [[buffer(3)]],  // M x N fp64 bit patterns
       constant uint&     N           [[buffer(4)]],
       constant uint&     levelStride [[buffer(5)]],  // == M*N
       constant uint&     numSlices   [[buffer(6)]],
       uint2 gid [[thread_position_in_grid]])
{
    const int BITS = 7;
    uint x = gid.x; // column
    uint y = gid.y; // row
    ulong idx = (ulong)y * N + x;
    const uint max_row = rowMaxA[y];
    const uint max_col = colMaxB[x];

    ulong hi = 0, lo = 0;
    for (uint s = 0; s < numSlices; ++s)
    {
        const int L = levels[(ulong)s * levelStride + idx];
        hi = (hi << BITS) | (lo >> (64 - BITS));
        lo <<= BITS;
        const ulong v = (ulong)(long)L;                       // sign-extend to 64 bits
        const ulong sum = lo + v;
        hi += (sum < lo ? 1ul : 0ul) + (L < 0 ? ~0ul : 0ul);  // carry + sign extension
        lo = sum;
    }

    if ((hi | lo) == 0) { // exact zero -> +0.0
        C[idx] = uint2(0, 0);
        return;
    }

    // sign / magnitude
    const ulong sign = hi >> 63;
    if (sign) {
        lo = ~lo + 1;
        hi = ~hi + (lo == 0 ? 1ul : 0ul);
    }

    const int m = hi != 0 ? 127 - (int)clz(hi) : 63 - (int)clz(lo);
    int E = m + (int)max_row + (int)max_col - 1021 - BITS * ((int)numSlices + 1);

    ulong mant;
    if (m <= 52) {
        mant = lo << (52 - m); // hi == 0 here; exact
    } else {
        const int d = m - 52;
        mant = shr128(hi, lo, d);
        const bool halfBit = bit128(hi, lo, d - 1) != 0;
        const bool sticky = any_below128(hi, lo, d - 1);
        if (halfBit && (sticky || (mant & 1ul))) {
            ++mant;
            if (mant == (1ul << 53)) { // rounded up to the next power of two
                mant >>= 1;
                ++E;
            }
        }
    }

    const ulong bits = (sign << 63) | ((ulong)E << 52) | (mant & 0xFFFFFFFFFFFFFul);
    C[idx] = uint2((uint)bits, (uint)(bits >> 32));
}
