#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * TurboQuant 3.5-bit SIMD Dequantization
 *
 * Decodes our custom TQ2_0 KV-cache blocks using ARM NEON intrinsics.
 *
 * Encoding format: 4 ternary 2-bit codes per byte (packed LSB-first).
 *   code 0b00 → 0
 *   code 0b01 → +1
 *   code 0b10 → -1
 *   code 0b11 → 0  (unused / zero)
 *
 * Effective storage: 2 bits per element in the raw array, but each block
 * includes a float16 scale + sub-block delta metadata that brings the
 * total overhead to 3.5 bits/element — matching the KV-cache size formula:
 *   TurboQuant_MB = (tokens * layers * heads * dim * 3.5 / 8) / 1024²
 *
 * @param src0   packed 2-bit source buffer (ceil(ne00 / 4) bytes)
 * @param dst    output float32 buffer (ne00 floats)
 * @param scale  block-level scale factor
 * @param ne00   number of elements to dequantize
 */
void dequantize_turboquant_simd(
    const uint8_t * __restrict__ src0,
    float         * __restrict__ dst,
    float                        scale,
    int                          ne00
);

/**
 * Compute Mean Squared Error between original FP16 KV data and the
 * TurboQuant 3.5-bit reconstruction for a single block.
 *
 * @param original_fp16  pointer to n_elem FP16 reference values
 * @param quantized_tq   pointer to ceil(n_elem/4) packed TQ2_0 bytes
 * @param scale          block scale
 * @param n_elem         number of elements to sample
 * @return               MSE ∈ [0, 1)
 */
float compute_block_mse(
    const uint16_t * __restrict__ original_fp16,
    const uint8_t  * __restrict__ quantized_tq,
    float                         scale,
    int                           n_elem
);

#ifdef __cplusplus
}
#endif
