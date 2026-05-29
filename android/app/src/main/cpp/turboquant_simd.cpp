//
// turboquant_simd.cpp
// ARM NEON–accelerated dequantization for TurboQuant TQ2_0 KV-cache blocks.
//
// Build flags: -O3 -march=armv8-a+simd -ffast-math
//

#include "turboquant_simd.h"

#include <cmath>
#include <cstring>

#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

// ── Half-precision helpers ──────────────────────────────────────────────────
// Convert a raw FP16 bit-pattern to float32 (portable, no hardware f16c needed)
static inline float fp16_to_float(uint16_t h) {
    const uint32_t exp_bias_diff = (127 - 15) << 23;  // exponent re-bias
    const uint32_t mant_shift    = 13;                 // 23 - 10 mantissa bits
    const uint32_t sign          = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp           = (uint32_t)(h & 0x7C00u) << 13;
    const uint32_t mant          = (uint32_t)(h & 0x03FFu) << mant_shift;

    if ((h & 0x7FFFu) == 0u) return 0.0f;  // ±zero

    uint32_t bits;
    if ((h & 0x7C00u) == 0x7C00u) {
        // NaN / Inf
        bits = sign | 0x7F800000u | (mant ? 0x00400000u : 0u);
    } else if ((h & 0x7C00u) == 0u) {
        // Subnormal → normalize
        uint32_t m = (uint32_t)(h & 0x03FFu);
        uint32_t e = 0;
        while (!(m & 0x0400u)) { m <<= 1; e++; }
        bits = sign | (((127 - 15 - e + 1) << 23)) | ((m & 0x03FFu) << mant_shift);
    } else {
        bits = sign | exp + exp_bias_diff | mant;
    }

    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

// ── Ternary lookup: 2-bit code → {0, +1, -1, 0} ───────────────────────────
static const int8_t TQ_LUT[4] = {0, 1, -1, 0};

// ── dequantize_turboquant_simd ──────────────────────────────────────────────
void dequantize_turboquant_simd(
    const uint8_t * __restrict__ src0,
    float         * __restrict__ dst,
    float                        scale,
    int                          ne00
) {
#if defined(__ARM_NEON)
    const float32x4_t vscale = vdupq_n_f32(scale);

    // Unpack TQ2_0 ternary codes into int8, then batch-convert to float32 with NEON.
    // One byte holds 4 elements (2 bits each, LSB-first).

    int i = 0;

    // ── Fast path: process 32 elements (8 input bytes) per iteration ────────
    for (; i + 31 < ne00; i += 32) {
        const uint8_t * q = src0 + (i >> 2);  // byte index (4 elements/byte)

        // Unpack 8 bytes → 32 int8 ternary values
        int8_t tmp[32];
        tmp[ 0] = TQ_LUT[(q[0] >> 0) & 0x03];
        tmp[ 1] = TQ_LUT[(q[0] >> 2) & 0x03];
        tmp[ 2] = TQ_LUT[(q[0] >> 4) & 0x03];
        tmp[ 3] = TQ_LUT[(q[0] >> 6) & 0x03];
        tmp[ 4] = TQ_LUT[(q[1] >> 0) & 0x03];
        tmp[ 5] = TQ_LUT[(q[1] >> 2) & 0x03];
        tmp[ 6] = TQ_LUT[(q[1] >> 4) & 0x03];
        tmp[ 7] = TQ_LUT[(q[1] >> 6) & 0x03];
        tmp[ 8] = TQ_LUT[(q[2] >> 0) & 0x03];
        tmp[ 9] = TQ_LUT[(q[2] >> 2) & 0x03];
        tmp[10] = TQ_LUT[(q[2] >> 4) & 0x03];
        tmp[11] = TQ_LUT[(q[2] >> 6) & 0x03];
        tmp[12] = TQ_LUT[(q[3] >> 0) & 0x03];
        tmp[13] = TQ_LUT[(q[3] >> 2) & 0x03];
        tmp[14] = TQ_LUT[(q[3] >> 4) & 0x03];
        tmp[15] = TQ_LUT[(q[3] >> 6) & 0x03];
        tmp[16] = TQ_LUT[(q[4] >> 0) & 0x03];
        tmp[17] = TQ_LUT[(q[4] >> 2) & 0x03];
        tmp[18] = TQ_LUT[(q[4] >> 4) & 0x03];
        tmp[19] = TQ_LUT[(q[4] >> 6) & 0x03];
        tmp[20] = TQ_LUT[(q[5] >> 0) & 0x03];
        tmp[21] = TQ_LUT[(q[5] >> 2) & 0x03];
        tmp[22] = TQ_LUT[(q[5] >> 4) & 0x03];
        tmp[23] = TQ_LUT[(q[5] >> 6) & 0x03];
        tmp[24] = TQ_LUT[(q[6] >> 0) & 0x03];
        tmp[25] = TQ_LUT[(q[6] >> 2) & 0x03];
        tmp[26] = TQ_LUT[(q[6] >> 4) & 0x03];
        tmp[27] = TQ_LUT[(q[6] >> 6) & 0x03];
        tmp[28] = TQ_LUT[(q[7] >> 0) & 0x03];
        tmp[29] = TQ_LUT[(q[7] >> 2) & 0x03];
        tmp[30] = TQ_LUT[(q[7] >> 4) & 0x03];
        tmp[31] = TQ_LUT[(q[7] >> 6) & 0x03];

        // NEON: int8 → int16 → int32 → float32, 8 elements per iteration
        for (int j = 0; j < 32; j += 8) {
            int8x8_t  s8   = vld1_s8(tmp + j);
            int16x8_t s16  = vmovl_s8(s8);

            int32x4_t s32_lo = vmovl_s16(vget_low_s16(s16));
            int32x4_t s32_hi = vmovl_s16(vget_high_s16(s16));

            float32x4_t f_lo = vmulq_f32(vcvtq_f32_s32(s32_lo), vscale);
            float32x4_t f_hi = vmulq_f32(vcvtq_f32_s32(s32_hi), vscale);

            vst1q_f32(dst + i + j,     f_lo);
            vst1q_f32(dst + i + j + 4, f_hi);
        }
    }

    // ── Scalar tail (remaining elements) ───────────────────────────────────
    for (; i < ne00; i++) {
        uint8_t code = (src0[i >> 2] >> ((i & 0x03) << 1)) & 0x03;
        dst[i] = static_cast<float>(TQ_LUT[code]) * scale;
    }

#else
    // ── Scalar fallback (non-NEON platforms, e.g. x86 emulator) ───────────
    for (int i = 0; i < ne00; i++) {
        uint8_t code = (src0[i >> 2] >> ((i & 0x03) << 1)) & 0x03;
        dst[i] = static_cast<float>(TQ_LUT[code]) * scale;
    }
#endif
}

// ── compute_block_mse ──────────────────────────────────────────────────────
float compute_block_mse(
    const uint16_t * __restrict__ original_fp16,
    const uint8_t  * __restrict__ quantized_tq,
    float                         scale,
    int                           n_elem
) {
    if (n_elem <= 0) return 0.0f;

    float sum_sq_err = 0.0f;

    for (int i = 0; i < n_elem; i++) {
        float orig  = fp16_to_float(original_fp16[i]);
        uint8_t code = (quantized_tq[i >> 2] >> ((i & 0x03) << 1)) & 0x03;
        float recon = static_cast<float>(TQ_LUT[code]) * scale;
        float err   = orig - recon;
        sum_sq_err += err * err;
    }

    return sum_sq_err / static_cast<float>(n_elem);
}
